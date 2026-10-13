from contextlib import asynccontextmanager

from psycopg_pool import AsyncConnectionPool
from pgvector.psycopg import register_vector_async

from app.config import settings

# check: every connection is tested before it's handed out, so a database
# restart (e.g. its container being recreated) costs a reconnect instead of
# failing requests on connections that died with the old server.
pool = AsyncConnectionPool(settings.database_url, open=False, check=AsyncConnectionPool.check_connection)


async def open_pool() -> None:
    await pool.open(wait=True)


async def close_pool() -> None:
    await pool.close()


@asynccontextmanager
async def get_connection():
    async with pool.connection() as conn:
        await register_vector_async(conn)
        yield conn


def _filename_filter(filename_contains: str | None) -> tuple[str, list]:
    """where-clause and params for a case-insensitive filename substring
    match; empty when there is no filter."""
    if not filename_contains:
        return "", []
    return " where filename ilike %s", [f"%{filename_contains}%"]


async def list_documents(
    filename_contains: str | None = None, limit: int | None = None, offset: int = 0
) -> list[tuple[str, int]]:
    """(filename, chunk_count) pairs in filename order, optionally
    filtered by a case-insensitive filename substring and cut to one page.
    GET /documents pages; the agent's list_documents tool takes the whole
    match and self-limits what it hands the model."""
    where, params = _filename_filter(filename_contains)
    query = f"select filename, count(*) from documents{where} group by filename order by filename"
    if limit is not None:
        query += " limit %s offset %s"
        params += [limit, offset]
    async with get_connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(query, params)
            return await cur.fetchall()


async def count_documents(filename_contains: str | None = None) -> int:
    """How many distinct filenames list_documents would return for the
    same filter - the total behind a page."""
    where, params = _filename_filter(filename_contains)
    async with get_connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(f"select count(distinct filename) from documents{where}", params)
            return (await cur.fetchone())[0]
