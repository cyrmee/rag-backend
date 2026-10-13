import base64
import binascii
import uuid
from datetime import datetime

from psycopg.types.json import Jsonb

from app.db import get_connection


def _is_uuid(value: str) -> bool:
    try:
        uuid.UUID(value)
    except ValueError:
        return False
    return True


async def create_conversation(owner_id: str | None = None) -> str:
    conversation_id = str(uuid.uuid4())
    async with get_connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                "insert into conversations (id, owner_id) values (%s, %s)", (conversation_id, owner_id)
            )
        await conn.commit()
    return conversation_id


async def conversation_exists(conversation_id: str, owner_id: str | None = None) -> bool:
    """With `owner_id`, only true when that user owns it - someone else's
    conversation is indistinguishable from a missing one."""
    if not _is_uuid(conversation_id):
        return False
    query = "select 1 from conversations where id = %s"
    params: tuple = (conversation_id,)
    if owner_id is not None:
        query += " and owner_id = %s"
        params += (owner_id,)
    async with get_connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(query, params)
            return await cur.fetchone() is not None


async def message_in_conversation(conversation_id: str, message_id: str) -> bool:
    if not _is_uuid(message_id):
        return False
    async with get_connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                "select 1 from conversation_messages where id = %s and conversation_id = %s",
                (message_id, conversation_id),
            )
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
                    where id = %s and conversation_id = %s
                    union all
                    select m.id, m.parent_message_id, m.role, m.content, path.depth + 1
                    from conversation_messages m
                    join path on m.id = path.parent_message_id
                )
                select role, content from path order by depth desc
                """,
                (leaf_message_id, conversation_id),
            )
            rows = await cur.fetchall()
    return [{"role": role, "content": content} for role, content in rows]


async def append_turn(
    conversation_id: str,
    question: str,
    answer: str,
    parent_message_id: str | None = None,
    extras: dict | None = None,
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
    later. `extras` is stored on the assistant message: what the answer
    was built from (see sql/013_message_extras.sql), so a reopened
    conversation shows the same sources and citations."""
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
                insert into conversation_messages (conversation_id, parent_message_id, role, content, extras)
                values (%s, %s, 'assistant', %s, %s)
                returning id
                """,
                (conversation_id, user_id, answer, Jsonb(extras) if extras is not None else None),
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


async def rename_conversation(conversation_id: str, owner_id: str, title: str) -> bool:
    """Sets a conversation's title, only if it's this user's. Returns
    whether one was renamed. Leaves updated_at alone: renaming isn't
    activity, so the conversation keeps its place in the list."""
    if not _is_uuid(conversation_id):
        return False
    async with get_connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                "update conversations set title = %s where id = %s and owner_id = %s",
                (title, conversation_id, owner_id),
            )
            renamed = cur.rowcount
        await conn.commit()
    return renamed > 0


def _like_pattern(text: str) -> str:
    """A case-insensitive substring pattern for `ilike ... escape '\\'` that
    matches `text` literally - a typed % or _ is a character, not a
    wildcard."""
    escaped = text.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    return f"%{escaped}%"


async def load_tree(conversation_id: str) -> list[dict]:
    """Every message in this conversation, not just the active path - the
    raw material for a branch-switcher UI, which needs to know about
    sibling turns the active path doesn't currently show."""
    async with get_connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                """
                select id, parent_message_id, role, content, created_at, extras
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
            "extras": r[5],
        }
        for r in rows
    ]


async def get_conversation_meta(conversation_id: str, owner_id: str) -> dict | None:
    if not _is_uuid(conversation_id):
        return None
    async with get_connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                "select title, active_message_id from conversations where id = %s and owner_id = %s",
                (conversation_id, owner_id),
            )
            row = await cur.fetchone()
    if row is None:
        return None
    return {"title": row[0], "active_message_id": str(row[1]) if row[1] else None}


def encode_cursor(updated_at: datetime, conversation_id: str) -> str:
    """Opaque page cursor for list_conversations: the (updated_at, id) of
    the last conversation on a page, which is where the next page starts
    after. Keyset rather than offset because updated_at moves whenever
    the user chats, so an offset would skip or repeat rows between pages."""
    return base64.urlsafe_b64encode(f"{updated_at.isoformat()}|{conversation_id}".encode()).decode()


def decode_cursor(cursor: str) -> tuple[datetime, uuid.UUID]:
    """Inverse of encode_cursor; ValueError for anything it didn't make."""
    try:
        stamp, _, conversation_id = base64.urlsafe_b64decode(cursor.encode()).decode().partition("|")
        updated_at = datetime.fromisoformat(stamp)
    except (binascii.Error, UnicodeDecodeError, ValueError) as exc:
        raise ValueError("malformed cursor") from exc
    if updated_at.tzinfo is None or not conversation_id:
        raise ValueError("malformed cursor")
    return updated_at, uuid.UUID(conversation_id)


async def list_conversations(
    owner_id: str,
    limit: int | None = None,
    before: tuple[datetime, uuid.UUID] | None = None,
    query: str | None = None,
) -> list[dict]:
    """This user's conversations that have at least one message, most
    recently updated first; with `limit` and `before` (a decoded cursor),
    one page of them, strictly after that (updated_at, id) in that order.
    _resolve_conversation creates the conversations row up front, before
    the first turn actually completes - a request that errors out or gets
    aborted before its first append_turn would otherwise leave a permanent
    empty, blank-title row that has no reason to show up in a list.
    `query` keeps conversations whose title or any question in them
    contains it, case-insensitively."""
    params: list = [owner_id]
    search_filter = ""
    if query:
        search_filter = """
                  and (
                      c.title ilike %s escape '\\'
                      or exists (
                          select 1 from conversation_messages m
                          where m.conversation_id = c.id and m.role = 'user'
                            and m.content ilike %s escape '\\'
                      )
                  )"""
        pattern = _like_pattern(query)
        params += [pattern, pattern]
    page_filter = ""
    if before is not None:
        page_filter = " and (c.updated_at, c.id) < (%s, %s)"
        params += list(before)
    page_limit = ""
    if limit is not None:
        page_limit = " limit %s"
        params.append(limit)
    async with get_connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                f"""
                select c.id, c.title, c.created_at, c.updated_at,
                       (
                           select content from conversation_messages m
                           where m.conversation_id = c.id and m.role = 'user'
                           order by m.created_at limit 1
                       ) as first_question
                from conversations c
                where c.owner_id = %s
                  and exists (select 1 from conversation_messages m where m.conversation_id = c.id)
                  {search_filter}
                  {page_filter}
                order by c.updated_at desc, c.id desc
                {page_limit}
                """,
                params,
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


async def delete_conversation(conversation_id: str, owner_id: str | None = None) -> bool:
    """Deletes the conversation and (via ON DELETE CASCADE) every message
    in it, all branches included. Returns whether one was deleted. With
    `owner_id`, only that user's; without, any (in-process check scripts
    cleaning up after themselves)."""
    if not _is_uuid(conversation_id):
        return False
    query = "delete from conversations where id = %s"
    params: tuple = (conversation_id,)
    if owner_id is not None:
        query += " and owner_id = %s"
        params += (owner_id,)
    async with get_connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(query, params)
            deleted = cur.rowcount
        await conn.commit()
    return deleted > 0
