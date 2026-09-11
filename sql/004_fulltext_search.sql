-- Additive migration for hybrid (vector + keyword) retrieval. Pure cosine
-- search struggles to distinguish near-duplicate template documents that
-- differ mainly in a proper noun (e.g. many near-identical VPN setup forms,
-- one per counterparty bank) - a keyword signal fixes exact-term matches
-- that dense embeddings underweight. Generated column stays in sync with
-- `content` automatically, no ingestion-side changes needed.

alter table documents
    add column if not exists content_tsv tsvector
        generated always as (to_tsvector('english', content)) stored;

create index if not exists documents_content_tsv_idx
    on documents using gin (content_tsv);
