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


async def load_messages(conversation_id: str) -> list[dict]:
    """Prior turns as plain {role, content} dicts, oldest first - reused
    directly as chat-message history for a new agentic run. Only user/
    assistant text is stored, not the tool-call choreography that produced
    a past answer - the model doesn't need to re-see old retrieved chunks,
    just what was asked and answered."""
    async with get_connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                """
                select role, content
                from conversation_messages
                where conversation_id = %s
                order by created_at
                """,
                (conversation_id,),
            )
            rows = await cur.fetchall()
    return [{"role": role, "content": content} for role, content in rows]


async def append_turn(conversation_id: str, question: str, answer: str) -> None:
    async with get_connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                """
                insert into conversation_messages (conversation_id, role, content)
                values (%s, 'user', %s), (%s, 'assistant', %s)
                """,
                (conversation_id, question, conversation_id, answer),
            )
            await cur.execute(
                "update conversations set updated_at = now() where id = %s", (conversation_id,)
            )
        await conn.commit()


async def list_conversations() -> list[dict]:
    async with get_connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                """
                select c.id, c.created_at, c.updated_at,
                       (
                           select content from conversation_messages m
                           where m.conversation_id = c.id and m.role = 'user'
                           order by m.created_at limit 1
                       ) as first_question
                from conversations c
                order by c.updated_at desc
                """
            )
            rows = await cur.fetchall()
    return [
        {"id": str(row[0]), "created_at": row[1], "updated_at": row[2], "first_question": row[3]}
        for row in rows
    ]
