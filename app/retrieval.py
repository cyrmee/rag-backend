import numpy as np
from pgvector import Vector

from app.db import get_connection

# Reciprocal rank fusion constant - the standard default (Cormack et al.),
# dampens the influence of a single method's #1 hit so neither the vector
# nor keyword ranking dominates the fused score outright.
RRF_K = 60

# MMR trade-off between relevance and novelty - 1.0 would be pure RRF
# ranking, 0.0 would ignore relevance entirely and just maximize spread.
# 0.7 keeps relevance dominant while still demoting near-duplicates.
MMR_LAMBDA = 0.7


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
                    d.source_image_path, d.embedding, fused.score
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
            "embedding": embedding,
            "score": score,
        }
        for content, source_type, source_format, filename, page_number, source_image_path, embedding, score in rows
    ]

    reranked = _mmr_rerank(candidates, limit)

    for c in reranked:
        del c["embedding"]
        del c["score"]
    return reranked
