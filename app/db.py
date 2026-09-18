from contextlib import asynccontextmanager

from psycopg_pool import AsyncConnectionPool
from pgvector.psycopg import register_vector_async

from app.config import settings

pool = AsyncConnectionPool(settings.database_url, open=False)


async def open_pool() -> None:
    await pool.open(wait=True)


async def close_pool() -> None:
    await pool.close()


@asynccontextmanager
async def get_connection():
    async with pool.connection() as conn:
        await register_vector_async(conn)
        yield conn


async def delete_document_chunks(filename: str) -> None:
    async with get_connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute("delete from documents where filename = %s", (filename,))
        await conn.commit()


async def list_documents(filename_contains: str | None = None) -> list[tuple[str, int]]:
    """(filename, chunk_count) pairs, optionally filtered by a
    case-insensitive filename substring. Used by both GET /documents (no
    filter, full list) and the agent's list_documents tool (self-limits
    the result it hands the model, doesn't cap here)."""
    async with get_connection() as conn:
        async with conn.cursor() as cur:
            if filename_contains:
                await cur.execute(
                    """
                    select filename, count(*)
                    from documents
                    where filename ilike %s
                    group by filename
                    order by filename
                    """,
                    (f"%{filename_contains}%",),
                )
            else:
                await cur.execute(
                    "select filename, count(*) from documents group by filename order by filename"
                )
            return await cur.fetchall()
