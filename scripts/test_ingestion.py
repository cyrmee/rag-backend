import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import psycopg

from app.config import settings
from app.db import close_pool, open_pool
from app.ingestion import ingest_document

SAMPLE_PATH = str(Path(__file__).resolve().parent / "sample.txt")


async def main() -> None:
    await open_pool()
    try:
        count = await ingest_document(SAMPLE_PATH, "sample.txt", "text/plain")
        print("chunks ingested:", count)
    finally:
        await close_pool()

    conn = psycopg.connect(settings.database_url)
    with conn.cursor() as cur:
        cur.execute("select count(*) from documents where filename = %s", ("sample.txt",))
        print("rows in db for sample.txt:", cur.fetchone()[0])
    conn.close()


if __name__ == "__main__":
    asyncio.run(main())
