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
