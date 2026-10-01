-- One row per ingested file with the SHA-256 of its bytes. Files were only
-- told apart by filename, so the same file uploaded under a second name
-- ("Copy of X.docx", or the same attachment saved into two folders) was
-- ingested twice and came back from search as two separate sources. The
-- unique hash lets ingestion refuse a byte-identical file under a new name.
-- Existing documents: run scripts/maintenance/backfill_file_hashes.py.

create table if not exists document_files (
    filename text primary key,
    sha256 text not null unique,
    created_at timestamptz not null default now()
);
