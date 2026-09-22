import uuid

from app.db import get_connection


async def create_conversation() -> str:
    conversation_id = str(uuid.uuid4())
    async with get_connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute("insert into conversations (id) values (%s)", (conversation_id,))
        await conn.commit()
    return conversation_id


async def conversation_exists(conversation_id: str) -> bool:
    async with get_connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute("select 1 from conversations where id = %s", (conversation_id,))
            return await cur.fetchone() is not None


async def load_messages(conversation_id: str, leaf_message_id: str | None = None) -> list[dict]:
    """Prior turns as plain {role, content} dicts, oldest first - reused
    directly as chat-message history for a new agentic run. Walks the
    message tree from `leaf_message_id` up to the root via parent_message_id
    (one round trip, via a recursive CTE).

    `leaf_message_id` has three meaningful states, since "no branch point
    given" and "branch from the very start" both need to end up meaning
    "no parent" but are NOT the same request: omitted (None) defaults to
    the conversation's current active leaf, the normal "continue this
    conversation" case; the empty string is the explicit root sentinel,
    meaning branch from before the first message, so there's no history at
    all; anything else is an earlier message's id to branch from - an
    edited/regenerated turn should only see what came before the branch
    point, not siblings that came after it on some other branch. Only
    user/assistant text is stored, not the tool-call choreography that
    produced a past answer - the model doesn't need to re-see old
    retrieved chunks, just what was asked and answered."""
    if leaf_message_id == "":
        return []
    async with get_connection() as conn:
        async with conn.cursor() as cur:
            if leaf_message_id is None:
                await cur.execute(
                    "select active_message_id from conversations where id = %s", (conversation_id,)
                )
                row = await cur.fetchone()
                leaf_message_id = str(row[0]) if row and row[0] else None
            if leaf_message_id is None:
                return []

            await cur.execute(
                """
                with recursive path as (
                    select id, parent_message_id, role, content, 0 as depth
                    from conversation_messages
                    where id = %s
                    union all
                    select m.id, m.parent_message_id, m.role, m.content, path.depth + 1
                    from conversation_messages m
                    join path on m.id = path.parent_message_id
                )
                select role, content from path order by depth desc
                """,
                (leaf_message_id,),
            )
            rows = await cur.fetchall()
    return [{"role": role, "content": content} for role, content in rows]


async def append_turn(
    conversation_id: str, question: str, answer: str, parent_message_id: str | None = None,
) -> tuple[str, str]:
    """Inserts one user/assistant turn as a new branch tip and returns
    (user_message_id, assistant_message_id). `parent_message_id` follows
    the same three-state convention as load_messages: omitted (None)
    attaches under the conversation's current active leaf (a normal
    continuation); the empty string explicitly attaches at the root (no
    parent), for editing/regenerating the very first turn; any other value
    is an earlier message's id to fork from. To edit a past question or
    regenerate a past answer, callers pass the *parent* of the message
    being replaced (its grandparent from the edited message's point of
    view, or "" if the edited message was the first turn), so the new turn
    becomes a sibling of the old one rather than a child of it - both
    remain in the tree, and either can become the active branch again
    later."""
    async with get_connection() as conn:
        async with conn.cursor() as cur:
            if parent_message_id is None:
                await cur.execute(
                    "select active_message_id from conversations where id = %s", (conversation_id,)
                )
                row = await cur.fetchone()
                parent_message_id = str(row[0]) if row and row[0] else None
            elif parent_message_id == "":
                parent_message_id = None

            await cur.execute(
                """
                insert into conversation_messages (conversation_id, parent_message_id, role, content)
                values (%s, %s, 'user', %s)
                returning id
                """,
                (conversation_id, parent_message_id, question),
            )
            user_id = (await cur.fetchone())[0]

            await cur.execute(
                """
                insert into conversation_messages (conversation_id, parent_message_id, role, content)
                values (%s, %s, 'assistant', %s)
                returning id
                """,
                (conversation_id, user_id, answer),
            )
            assistant_id = (await cur.fetchone())[0]

            await cur.execute(
                "update conversations set active_message_id = %s, updated_at = now() where id = %s",
                (assistant_id, conversation_id),
            )
        await conn.commit()
    return str(user_id), str(assistant_id)


async def set_title(conversation_id: str, title: str) -> None:
    async with get_connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                "update conversations set title = %s where id = %s", (title, conversation_id)
            )
        await conn.commit()


async def load_tree(conversation_id: str) -> list[dict]:
    """Every message in this conversation, not just the active path - the
    raw material for a branch-switcher UI, which needs to know about
    sibling turns the active path doesn't currently show."""
    async with get_connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                """
                select id, parent_message_id, role, content, created_at
                from conversation_messages
                where conversation_id = %s
                order by created_at
                """,
                (conversation_id,),
            )
            rows = await cur.fetchall()
    return [
        {
            "id": str(r[0]),
            "parent_message_id": str(r[1]) if r[1] else None,
            "role": r[2],
            "content": r[3],
            "created_at": r[4],
        }
        for r in rows
    ]


async def get_conversation_meta(conversation_id: str) -> dict | None:
    async with get_connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                "select title, active_message_id from conversations where id = %s", (conversation_id,)
            )
            row = await cur.fetchone()
    if row is None:
        return None
    return {"title": row[0], "active_message_id": str(row[1]) if row[1] else None}


async def list_conversations() -> list[dict]:
    """Every conversation that has at least one message. _resolve_conversation
    creates the conversations row up front, before the first turn actually
    completes - a request that errors out or gets aborted before its first
    append_turn would otherwise leave a permanent empty, blank-title row
    that has no reason to show up in a conversation list."""
    async with get_connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                """
                select c.id, c.title, c.created_at, c.updated_at,
                       (
                           select content from conversation_messages m
                           where m.conversation_id = c.id and m.role = 'user'
                           order by m.created_at limit 1
                       ) as first_question
                from conversations c
                where exists (select 1 from conversation_messages m where m.conversation_id = c.id)
                order by c.updated_at desc
                """
            )
            rows = await cur.fetchall()
    return [
        {
            "id": str(row[0]),
            "title": row[1],
            "created_at": row[2],
            "updated_at": row[3],
            "first_question": row[4],
        }
        for row in rows
    ]


async def delete_conversation(conversation_id: str) -> bool:
    """Deletes the conversation and (via ON DELETE CASCADE) every message
    in it, all branches included. Returns whether a row actually existed."""
    async with get_connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute("delete from conversations where id = %s", (conversation_id,))
            deleted = cur.rowcount
        await conn.commit()
    return deleted > 0
