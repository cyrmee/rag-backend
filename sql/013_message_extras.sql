-- Additive migration: what an answer was built from, kept with the answer.
-- Until now only the answer's text was stored, so reopening a conversation
-- lost its sources, citation breakdown, citation-check warnings, the mode
-- that answered and (research mode) what the documents didn't answer - and
-- the [N] markers in the text pointed at nothing. One jsonb column on the
-- assistant message holds them: {mode, sources, citations,
-- citation_warnings, not_found}. Sources are stored without links; GET
-- /conversations/{id} signs fresh ones, since presigned links expire.
-- Null for user messages and for answers saved before this migration.

alter table conversation_messages
    add column if not exists extras jsonb;
