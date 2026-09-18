-- Additive migration for document-level semantic routing. Chunk-level
-- retrieval (documents table) is unchanged; this adds a second, much
-- smaller table with one row per document - an LLM-written summary plus
-- its embedding - so document_routed_search() can compare a query directly
-- against "what is this whole document about" instead of only inferring
-- document relevance from the best individual matching chunk.

create table if not exists document_summaries (
    filename text primary key,
    summary text not null,
    embedding vector(1024) not null,
    created_at timestamptz not null default now()
);

create index if not exists document_summaries_embedding_idx
    on document_summaries using hnsw (embedding vector_cosine_ops);
