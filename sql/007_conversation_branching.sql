-- Additive migration for conversation titles and branching. Conversations
-- were flat: a title was a client-side (localStorage) guess derived from
-- the first question, and editing/regenerating a past turn silently
-- started a brand-new conversation, discarding the shared history instead
-- of forking from that point.

alter table conversations add column if not exists title text;
alter table conversations add column if not exists active_message_id uuid;

alter table conversation_messages
    add column if not exists parent_message_id uuid references conversation_messages(id) on delete cascade;

create index if not exists conversation_messages_parent_idx
    on conversation_messages (parent_message_id);

do $$
begin
    if not exists (
        select 1 from pg_constraint where conname = 'conversations_active_message_id_fkey'
    ) then
        alter table conversations
            add constraint conversations_active_message_id_fkey
            foreign key (active_message_id) references conversation_messages(id) on delete set null;
    end if;
end $$;
