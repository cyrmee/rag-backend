from pgvector import Vector

from app.db import get_connection

# Reciprocal rank fusion constant - the standard default (Cormack et al.),
# dampens the influence of a single method's #1 hit so neither the vector
# nor keyword ranking dominates the fused score outright.
RRF_K = 60


async def hybrid_search(query_vector: list[float], query_text: str, limit: int) -> list[dict]:
    """Combines dense vector similarity with Postgres full-text keyword
    search via reciprocal rank fusion. Pure cosine search underperforms on
    near-duplicate template documents that differ mainly in a proper noun
    (e.g. many near-identical VPN setup forms, one per counterparty bank) -
    the boilerplate dominates the embedding, so the keyword signal is what
    actually distinguishes the right match."""
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
                select d.content, d.source_type, d.source_format, d.filename, d.page_number, d.source_image_path
                from documents d
                join fused on fused.id = d.id
                order by fused.score desc
                limit %(limit)s
                """,
                {
                    "qvec": Vector(query_vector),
                    "qtext": query_text,
                    "pool": candidate_pool,
                    "rrf_k": RRF_K,
                    "limit": limit,
                },
            )
            rows = await cur.fetchall()

    return [
        {
            "content": content,
            "source_type": source_type,
            "source_format": source_format,
            "filename": filename,
            "page_number": page_number,
            "source_image_path": source_image_path,
        }
        for content, source_type, source_format, filename, page_number, source_image_path in rows
    ]
