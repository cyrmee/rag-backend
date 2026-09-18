"""Generates document_summaries for any already-ingested filename that
doesn't have one yet - a separate pass from the main ingestion batch so the
chat model (needed for summaries) and the vision model (needed for image
captioning) never have to be loaded at once; together with the embedding
model they don't comfortably fit in memory at the same time. Reconstructs
each document's prose from its already-stored chunks, so it never touches
the original files - run after an ingest_id_drive.py --skip-summary pass.

Usage:
    python scripts/maintenance/backfill_summaries.py
    python scripts/maintenance/backfill_summaries.py --concurrency 4
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from app.db import close_pool, get_connection, open_pool
from app.embeddings import embed_text
from app.ingestion import _generate_document_summary, _upsert_document_summary

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("backfill_summaries")


async def _pending_filenames() -> list[str]:
    async with get_connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                """
                select distinct d.filename
                from documents d
                left join document_summaries s on s.filename = d.filename
                where d.source_type = 'text' and s.filename is null
                order by d.filename
                """
            )
            return [row[0] for row in await cur.fetchall()]


async def _load_chunks_with_embeddings(filename: str) -> tuple[list[str], list[list[float]]]:
    """Reads already-embedded chunks straight from the documents table -
    no re-embedding needed, since _generate_document_summary just needs
    the vectors to pick representative chunks, not fresh ones."""
    async with get_connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                """
                select content, embedding
                from documents
                where filename = %s and source_type = 'text'
                order by chunk_index
                """,
                (filename,),
            )
            rows = await cur.fetchall()
    texts = [content for content, _ in rows]
    embeddings = [embedding.to_list() for _, embedding in rows]
    return texts, embeddings


async def _backfill_one(filename: str, semaphore: asyncio.Semaphore) -> None:
    async with semaphore:
        chunk_texts, chunk_embeddings = await _load_chunks_with_embeddings(filename)
        summary = await _generate_document_summary(filename, chunk_texts, chunk_embeddings)
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
    args = parser.parse_args()

    await open_pool()
    try:
        filenames = await _pending_filenames()
        logger.info("%d document(s) need a summary", len(filenames))

        semaphore = asyncio.Semaphore(args.concurrency)
        await asyncio.gather(*[_backfill_one(f, semaphore) for f in filenames])
    finally:
        await close_pool()

    logger.info("done")


if __name__ == "__main__":
    asyncio.run(main())
