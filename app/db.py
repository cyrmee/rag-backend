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


# The file types GET /documents can filter by, as filename extensions.
DOCUMENT_TYPES = ("pdf", "docx", "pptx", "xlsx", "txt", "md")


def _folder_prefix(folder: str | None) -> str:
    """A folder as the filename prefix it stands for: "ID/Admin" ->
    "ID/Admin/"; empty for the whole library."""
    folder = (folder or "").strip().strip("/")
    return f"{folder}/" if folder else ""


def _filename_filter(
    filename_contains: str | None, folder: str | None = None, file_type: str | None = None
) -> tuple[str, list]:
    """where-clause and params for a case-insensitive filename substring
    match, a folder (everything under it, at any depth) and a file type;
    empty when there is no filter. The folder is compared as a literal
    prefix, not a like pattern - folder names contain _ and %."""
    conditions, params = [], []
    if filename_contains:
        conditions.append("filename ilike %s")
        params.append(f"%{filename_contains}%")
    prefix = _folder_prefix(folder)
    if prefix:
        conditions.append("left(filename, %s) = %s")
        params += [len(prefix), prefix]
    if file_type:
        conditions.append("lower(filename) like %s")
        params.append(f"%.{file_type}")
    return (" where " + " and ".join(conditions), params) if conditions else ("", [])


async def list_documents(
    filename_contains: str | None = None,
    limit: int | None = None,
    offset: int = 0,
    folder: str | None = None,
    file_type: str | None = None,
) -> list[tuple[str, int]]:
    """(filename, chunk_count) pairs in filename order, optionally
    filtered by a case-insensitive filename substring and cut to one page.
    GET /documents pages; the agent's list_documents tool takes the whole
    match and self-limits what it hands the model."""
    where, params = _filename_filter(filename_contains, folder, file_type)
    query = f"select filename, count(*) from documents{where} group by filename order by filename"
    if limit is not None:
        query += " limit %s offset %s"
        params += [limit, offset]
    async with get_connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(query, params)
            return await cur.fetchall()


async def count_documents(
    filename_contains: str | None = None, folder: str | None = None, file_type: str | None = None
) -> int:
    """How many distinct filenames list_documents would return for the
    same filter - the total behind a page."""
    where, params = _filename_filter(filename_contains, folder, file_type)
    async with get_connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(f"select count(distinct filename) from documents{where}", params)
            return (await cur.fetchone())[0]


async def list_folders(parent: str | None = None) -> list[tuple[str, str, int]]:
    """The folders directly inside `parent` (the whole library when
    empty), as (name, path, document count at any depth below it), in
    name order - for browsing the library a level at a time."""
    prefix = _folder_prefix(parent)
    n = len(prefix)
    async with get_connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                """
                select split_part(substr(filename, %(n)s + 1), '/', 1) as name, count(distinct filename)
                from documents
                where left(filename, %(n)s) = %(prefix)s
                  and strpos(substr(filename, %(n)s + 1), '/') > 0
                group by name
                order by name
                """,
                {"n": n, "prefix": prefix},
            )
            rows = await cur.fetchall()
    return [(name, prefix + name, count) for name, count in rows]
