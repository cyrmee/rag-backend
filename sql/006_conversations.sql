-- Additive migration for persisted multi-turn conversations. /ask was
-- stateless - every call started a brand-new message list with just the
-- system prompt and that one question, so a follow-up like "what about the
-- second one?" had no idea what "the second one" referred to.

create table if not exists conversations (
    id uuid primary key default gen_random_uuid(),
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now()
);

create table if not exists conversation_messages (
    id uuid primary key default gen_random_uuid(),
    conversation_id uuid not null references conversations(id) on delete cascade,
    role text not null check (role in ('user', 'assistant')),
    content text not null,
    created_at timestamptz not null default now()
);

create index if not exists conversation_messages_conversation_id_idx
    on conversation_messages (conversation_id, created_at);
