-- Additive migration for per-user chat history. Every conversation was
-- visible to everyone; now each belongs to the Keycloak user who started
-- it. Users are keyed by the token's "sub" claim (text, not uuid: a
-- federated user's id isn't guaranteed to be one). The row is a local
-- copy of who they are, refreshed as they use the app - Keycloak stays
-- the source of truth. Conversations from before this migration have no
-- owner and are shown to nobody; scripts/maintenance/assign_conversations.py
-- hands them to a user.

create table if not exists users (
    id text primary key,
    username text not null,
    email text,
    name text,
    created_at timestamptz not null default now(),
    last_seen_at timestamptz not null default now()
);

alter table conversations
    add column if not exists owner_id text references users(id) on delete cascade;

create index if not exists conversations_owner_updated_idx
    on conversations (owner_id, updated_at desc);
