-- Caption cache, and a per-file marker for the captioning pass.
--
-- image_captions: one row per distinct image, keyed by the SHA-256 of its
-- bytes, so an image that appears in many documents (a letterhead logo, a
-- slide template's pictures, the same figure in a report and its "Copy of")
-- is captioned once. The caption's embedding is kept with it, so a cached
-- image costs neither model.
create table if not exists image_captions (
    sha256 text primary key,
    caption text not null,
    embedding vector(1024) not null,
    model text not null,
    created_at timestamptz not null default now()
);

-- document_files.captioned_at: when the file's images were captioned. Null
-- means the file was ingested with captioning skipped (ingest_id_drive.py
-- --skip-images); scripts/maintenance/backfill_captions.py picks those up.
-- Every file ingested before this migration was captioned at ingestion.
alter table document_files add column if not exists captioned_at timestamptz;
update document_files set captioned_at = created_at where captioned_at is null;
