"""Generates document_summaries for any already-ingested filename that
doesn't have one yet - a separate pass from the main ingestion batch so the
chat model (needed for summaries) and the vision model (needed for image
captioning) never have to be loaded at once; together with the embedding
model they don't comfortably fit in memory at the same time. Reconstructs
each document's prose from its already-stored chunks, so it never touches
the original files - run after an ingest_id_drive.py --skip-summary pass.
--redo rewrites every document's summary instead, e.g. after a change to
how summaries are written.

Usage:
    python scripts/maintenance/backfill_summaries.py
    python scripts/maintenance/backfill_summaries.py --concurrency 4
    python scripts/maintenance/backfill_summaries.py --redo
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from app.db import close_pool, get_connection, open_pool
from app.embeddings import embed_text
from app.ingestion import _generate_document_summary, _upsert_document_summary, summary_source_chunks

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("backfill_summaries")


async def _pending_filenames(redo: bool) -> list[str]:
    async with get_connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                """
                select distinct d.filename
                from documents d
                left join document_summaries s on s.filename = d.filename
                where d.source_type = 'text' and (%s or s.filename is null)
                order by d.filename
                """,
                (redo,),
            )
            return [row[0] for row in await cur.fetchall()]


async def _load_text(filename: str) -> list[str]:
    """Puts the document's text back together from its stored chunks, in
    order - the same text ingestion would summarize."""
    async with get_connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                """
                select content, page_number, source_format
                from documents
                where filename = %s and source_type = 'text'
                order by chunk_index
                """,
                (filename,),
            )
            rows = await cur.fetchall()
    return summary_source_chunks([(content, page) for content, page, _ in rows], rows[0][2])


async def _backfill_one(filename: str, semaphore: asyncio.Semaphore) -> None:
    async with semaphore:
        summary = await _generate_document_summary(filename, await _load_text(filename))
        if not summary:
            logger.warning("no summary generated for %s", filename)
            return

        embedding = await embed_text(summary)
        async with get_connection() as conn:
            async with conn.cursor() as cur:
                await _upsert_document_summary(cur, filename, summary, embedding)
            await conn.commit()
        logger.info("summarized %s", filename)


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--redo", action="store_true", help="rewrite existing summaries too")
    args = parser.parse_args()

    await open_pool()
    try:
        filenames = await _pending_filenames(args.redo)
        logger.info("%d document(s) need a summary", len(filenames))

        semaphore = asyncio.Semaphore(args.concurrency)
        await asyncio.gather(*[_backfill_one(f, semaphore) for f in filenames])
    finally:
        await close_pool()

    logger.info("done")


if __name__ == "__main__":
    asyncio.run(main())
