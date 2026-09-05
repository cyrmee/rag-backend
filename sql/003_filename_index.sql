-- Scaling fix: /upload's delete-then-reinsert (delete_document_chunks) and
-- DELETE /documents/{filename} both filter by filename with no supporting
-- index, forcing a sequential scan that gets linearly slower as the table
-- grows. Purely additive - no behavior change, just an access path.

create index if not exists documents_filename_idx on documents (filename);
