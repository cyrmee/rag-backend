create extension if not exists vector;

create table if not exists documents (
    id uuid primary key default gen_random_uuid(),
    filename text not null,
    chunk_index int not null,
    content text not null,
    embedding vector(1024) not null,
    metadata jsonb not null default '{}'::jsonb,
    created_at timestamptz not null default now(),
    source_type text not null default 'text'
        check (source_type in ('text', 'image_caption', 'chart_data')),
    source_format text not null default 'pdf'
        check (source_format in ('pdf', 'docx', 'pptx', 'xlsx', 'txt', 'md')),
    source_image_path text,
    page_number int,
    bbox jsonb
);

create index if not exists documents_embedding_idx
    on documents using hnsw (embedding vector_cosine_ops);

create index if not exists documents_filename_idx on documents (filename);
