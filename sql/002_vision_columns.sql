-- Additive migration for vision captioning / multi-format ingestion.
-- Existing rows remain valid: source_type defaults to 'text', source_format
-- backfills to 'pdf' (best-effort guess for pre-migration rows, all of which
-- came from the original PDF-only /upload path), and the new columns are
-- otherwise nullable.

alter table documents
    add column if not exists source_type text not null default 'text',
    add column if not exists source_format text not null default 'pdf',
    add column if not exists source_image_path text,
    add column if not exists page_number int,
    add column if not exists bbox jsonb;

-- Backfill from the actual filename extension rather than trusting the
-- blanket 'pdf' default, since pre-migration rows may already include
-- other formats ingested before this column existed.
update documents
set source_format = case
    when filename ilike '%.pdf' then 'pdf'
    when filename ilike '%.docx' then 'docx'
    when filename ilike '%.pptx' then 'pptx'
    when filename ilike '%.xlsx' then 'xlsx'
    when filename ilike '%.md' then 'md'
    when filename ilike '%.txt' then 'txt'
    else source_format
end;

alter table documents
    add constraint documents_source_type_check
        check (source_type in ('text', 'image_caption', 'chart_data'));

alter table documents
    add constraint documents_source_format_check
        check (source_format in ('pdf', 'docx', 'pptx', 'xlsx', 'txt', 'md'));
