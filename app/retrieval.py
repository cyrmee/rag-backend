import numpy as np
from pgvector import Vector

from app.db import get_connection
from app.embeddings import embed_text
from app.generation import generate_answer

# Reciprocal rank fusion constant - the standard default (Cormack et al.),
# dampens the influence of a single method's #1 hit so neither the vector
# nor keyword ranking dominates the fused score outright.
RRF_K = 60

# MMR trade-off between relevance and novelty - 1.0 would be pure RRF
# ranking, 0.0 would ignore relevance entirely and just maximize spread.
# 0.7 keeps relevance dominant while still demoting near-duplicates.
MMR_LAMBDA = 0.7

# How many neighboring chunks (same file, same side) to pull in around each
# winning "text" chunk - a lone ~500-char chunk is often mid-sentence or
# mid-table; stitching its immediate neighbors back in gives the model the
# surrounding paragraph instead of an isolated fragment.
CONTEXT_WINDOW = 1

# Query decomposition for /ask's single-pass retrieval: a complex question
# often needs more than one angle to answer fully, and a single embedding
# can't represent every angle equally well. Capped at 4 total (including the
# original question verbatim) to bound retrieval latency and prompt size.
MAX_SUB_QUERIES = 4
PER_SUB_QUERY_LIMIT = 8


def _mmr_rerank(candidates: list[dict], limit: int, lambda_mult: float = MMR_LAMBDA) -> list[dict]:
    """Reorders `candidates` (already sorted by RRF `score` desc, each
    carrying an `embedding`) to balance relevance against redundancy.
    Near-duplicate template documents (e.g. many structurally-identical
    per-bank VPN forms, or successive versions of the same TOR) tend to
    cluster tightly in embedding space and, worse, all match the same
    keywords - so pure RRF ranking lets a handful of them monopolize the
    top-k and bury a differently-worded but more relevant result. MMR
    picks the best-scoring remaining candidate each round, penalized by
    its cosine similarity to whatever's already been selected."""
    if not candidates:
        return []

    vectors = np.array([c["embedding"].to_list() for c in candidates], dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    unit_vectors = vectors / norms

    scores = np.array([c["score"] for c in candidates], dtype=np.float32)
    max_score = scores.max() or 1.0
    relevance = scores / max_score

    selected_indices: list[int] = []
    remaining = set(range(len(candidates)))
    max_sim_to_selected = np.zeros(len(candidates), dtype=np.float32)

    while remaining and len(selected_indices) < limit:
        best_idx, best_value = None, -np.inf
        for i in remaining:
            value = lambda_mult * relevance[i] - (1 - lambda_mult) * max_sim_to_selected[i]
            if value > best_value:
                best_idx, best_value = i, value

        selected_indices.append(best_idx)
        remaining.discard(best_idx)

        if remaining:
            sims = unit_vectors[list(remaining)] @ unit_vectors[best_idx]
            for i, sim in zip(remaining, sims):
                if sim > max_sim_to_selected[i]:
                    max_sim_to_selected[i] = sim

    return [candidates[i] for i in selected_indices]


async def hybrid_search(query_vector: list[float], query_text: str, limit: int) -> list[dict]:
    """Combines dense vector similarity with Postgres full-text keyword
    search via reciprocal rank fusion, then applies MMR reranking over a
    wider candidate pool. Pure cosine search underperforms on near-duplicate
    template documents that differ mainly in a proper noun (e.g. many
    near-identical VPN setup forms, one per counterparty bank) - the
    boilerplate dominates the embedding, so the keyword signal is what
    actually distinguishes the right match; MMR then keeps several such
    near-duplicates from all monopolizing the top-k at once."""
    candidate_pool = max(limit * 4, 40)

    async with get_connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                """
                with vector_matches as (
                    select id, row_number() over (order by embedding <=> %(qvec)s) as rnk
                    from documents
                    order by embedding <=> %(qvec)s
                    limit %(pool)s
                ),
                text_matches as (
                    select id, row_number() over (
                        order by ts_rank(content_tsv, websearch_to_tsquery('english', %(qtext)s)) desc
                    ) as rnk
                    from documents
                    where content_tsv @@ websearch_to_tsquery('english', %(qtext)s)
                    limit %(pool)s
                ),
                fused as (
                    select id,
                           coalesce(1.0 / (%(rrf_k)s + v.rnk), 0) + coalesce(1.0 / (%(rrf_k)s + t.rnk), 0) as score
                    from vector_matches v
                    full outer join text_matches t using (id)
                )
                select
                    d.content, d.source_type, d.source_format, d.filename, d.page_number,
                    d.source_image_path, d.chunk_index, d.embedding, fused.score
                from documents d
                join fused on fused.id = d.id
                order by fused.score desc
                limit %(pool)s
                """,
                {
                    "qvec": Vector(query_vector),
                    "qtext": query_text,
                    "pool": candidate_pool,
                    "rrf_k": RRF_K,
                },
            )
            rows = await cur.fetchall()

    candidates = [
        {
            "content": content,
            "source_type": source_type,
            "source_format": source_format,
            "filename": filename,
            "page_number": page_number,
            "source_image_path": source_image_path,
            "chunk_index": chunk_index,
            "embedding": embedding,
            "score": score,
        }
        for content, source_type, source_format, filename, page_number,
            source_image_path, chunk_index, embedding, score in rows
    ]

    reranked = _mmr_rerank(candidates, limit)

    for c in reranked:
        del c["embedding"]
        del c["score"]

    await _expand_context(reranked)
    return reranked


async def _expand_context(results: list[dict]) -> None:
    """Mutates each "text" result's `content` in place to splice in its
    immediate neighboring chunks (same file, adjacent chunk_index) - see
    CONTEXT_WINDOW. chart_data/image_caption rows are left alone; "adjacent
    chunk" isn't a meaningful notion for them."""
    async with get_connection() as conn:
        async with conn.cursor() as cur:
            for r in results:
                if r["source_type"] != "text":
                    continue
                await cur.execute(
                    """
                    select chunk_index, content
                    from documents
                    where filename = %(filename)s
                      and source_type = 'text'
                      and chunk_index between %(lo)s and %(hi)s
                      and chunk_index != %(idx)s
                    order by chunk_index
                    """,
                    {
                        "filename": r["filename"],
                        "lo": r["chunk_index"] - CONTEXT_WINDOW,
                        "hi": r["chunk_index"] + CONTEXT_WINDOW,
                        "idx": r["chunk_index"],
                    },
                )
                neighbors = await cur.fetchall()
                if not neighbors:
                    continue

                before = [content for idx, content in neighbors if idx < r["chunk_index"]]
                after = [content for idx, content in neighbors if idx > r["chunk_index"]]
                r["content"] = "\n".join([*before, r["content"], *after])


async def _decompose_query(question: str) -> list[str]:
    """Asks the chat model to split `question` into a few distinct search
    angles. Always returns the original question first - decomposition is
    meant to augment the direct hit, not replace it - and falls back to
    just the original question if the model call fails or returns nothing
    usable, so a decomposition hiccup degrades gracefully to today's
    single-query behavior rather than failing the whole request."""
    prompt = (
        "Break the following question into 2-4 distinct search queries "
        "that, together, cover every angle needed to answer it fully. "
        "Each query should be short and standalone, suitable for a "
        "document search engine - not a restatement of the whole "
        "question. Reply with ONLY the queries, one per line, no "
        "numbering, no extra commentary.\n\n"
        f"Question: {question}"
    )
    try:
        answer, _ = await generate_answer(prompt)
    except Exception:
        return [question]

    sub_queries = [line.strip(" \t-*") for line in answer.splitlines() if line.strip()]
    queries = [question]
    for q in sub_queries:
        if q.lower() != question.lower() and q not in queries:
            queries.append(q)
        if len(queries) >= MAX_SUB_QUERIES:
            break
    return queries


async def decompose_and_retrieve(question: str, limit: int) -> list[dict]:
    """Multi-angle retrieval for /ask's single-pass path: decomposes
    `question` into a few distinct search queries and merges their results
    (deduped by filename + chunk_index), instead of staking the whole
    answer on however well one embedding captures every angle of a complex
    question. /ask/agentic already has its own multi-hop mechanism (the
    model deciding when to call retrieve() again) - this brings a
    comparable benefit to the single-pass route without an agentic loop."""
    queries = await _decompose_query(question)

    seen: set[tuple[str, int]] = set()
    merged: list[dict] = []
    for q in queries:
        qvec = await embed_text(q)
        for r in await hybrid_search(qvec, q, PER_SUB_QUERY_LIMIT):
            key = (r["filename"], r["chunk_index"])
            if key not in seen:
                seen.add(key)
                merged.append(r)

    return merged[: limit * len(queries)]
