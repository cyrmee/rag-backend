-- Knowledge graph for graph retrieval (app/graph.py), after HippoRAG 2.
-- Built at ingestion when GRAPH_RETRIEVAL is on; existing documents:
-- scripts/maintenance/backfill_graph.py. Every table hangs off documents
-- (one row per chunk), so deleting a document's chunks takes its facts
-- with it; phrases and facts nothing points at any more are removed by
-- app/graph.py remove_orphans after a delete or re-upload.

-- When a chunk's two extraction calls last ran (null: never), so a backfill
-- can skip chunks already done, including those that yielded no facts.
alter table documents add column if not exists graph_extracted_at timestamptz;

-- One node per distinct normalised phrase (a triple's subject or object).
create table if not exists graph_phrases (
    id uuid primary key default gen_random_uuid(),
    phrase text not null unique,
    embedding vector(1024) not null,
    created_at timestamptz not null default now()
);

create index if not exists graph_phrases_embedding_idx
    on graph_phrases using hnsw (embedding vector_cosine_ops);

-- One row per distinct (subject, relation, object); its embedding is what a
-- question is matched against first.
create table if not exists graph_facts (
    id uuid primary key default gen_random_uuid(),
    subject_id uuid not null references graph_phrases(id) on delete cascade,
    object_id uuid not null references graph_phrases(id) on delete cascade,
    subject text not null,
    relation text not null,
    object text not null,
    embedding vector(1024) not null,
    created_at timestamptz not null default now(),
    unique (subject, relation, object)
);

create index if not exists graph_facts_subject_idx on graph_facts (subject_id);
create index if not exists graph_facts_object_idx on graph_facts (object_id);
create index if not exists graph_facts_embedding_idx
    on graph_facts using hnsw (embedding vector_cosine_ops);

-- Which chunks state which fact: the passage-phrase edges of the graph.
create table if not exists graph_triples (
    fact_id uuid not null references graph_facts(id) on delete cascade,
    chunk_id uuid not null references documents(id) on delete cascade,
    created_at timestamptz not null default now(),
    primary key (fact_id, chunk_id)
);

create index if not exists graph_triples_chunk_idx on graph_triples (chunk_id);

-- Synonym edges: phrase pairs whose embeddings are at least as similar as
-- app/graph.py SYNONYM_THRESHOLD, stored once (phrase_a < phrase_b).
create table if not exists graph_edges (
    phrase_a uuid not null references graph_phrases(id) on delete cascade,
    phrase_b uuid not null references graph_phrases(id) on delete cascade,
    weight real not null,
    primary key (phrase_a, phrase_b),
    check (phrase_a < phrase_b)
);

create index if not exists graph_edges_b_idx on graph_edges (phrase_b);
