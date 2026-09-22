-- Backfill for 007_conversation_branching.sql. That migration added
-- parent_message_id/active_message_id as nullable columns with no
-- backfill, so every message created before it has parent_message_id =
-- NULL and every such conversation has active_message_id = NULL. Once the
-- app starts resolving history via "walk from active_message_id", any of
-- those pre-existing conversations would silently look empty on its next
-- turn - this reconstructs a linear parent chain (in created_at order,
-- matching the old flat-history behavior) so old conversations keep
-- working exactly as before, while anything already given a real tree by
-- the new code (parent_message_id or active_message_id already set) is
-- left untouched.

with ordered as (
    select id, conversation_id,
           lag(id) over (partition by conversation_id order by created_at) as prev_id
    from conversation_messages
    where parent_message_id is null
)
update conversation_messages m
set parent_message_id = o.prev_id
from ordered o
where m.id = o.id and o.prev_id is not null;

with latest as (
    select distinct on (conversation_id) conversation_id, id
    from conversation_messages
    order by conversation_id, created_at desc
)
update conversations c
set active_message_id = latest.id
from latest
where c.id = latest.conversation_id and c.active_message_id is null;
