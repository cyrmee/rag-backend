"""Captions the images of documents ingested with captioning skipped
(ingest_id_drive.py --skip-images: document_files.captioned_at is null).
The original file comes back from MinIO, its images are extracted again
and captioned through the cache (image_captions - see app/ingestion.py
caption_extracted_images), and the captions are stored as image_caption
chunks after the document's existing ones. Committed per document, so it
can be stopped and resumed. --redo re-captions documents already done,
replacing their old caption chunks, e.g. after a change to the caption
prompt. With GRAPH_RETRIEVAL on, new caption chunks have no graph yet:
run backfill_graph.py afterwards, it picks them up.

Budget: about 9 s per new image on qwen3-vl-caption (3 s on qwen3-vl:8b),
nothing for an image captioned before; the vision model takes one image
at a time whatever --concurrency is (that only overlaps parsing with it).

Usage:
    python scripts/maintenance/backfill_captions.py
    python scripts/maintenance/backfill_captions.py --skip "Training and KBs"
    python scripts/maintenance/backfill_captions.py --only "Program Management" --redo
    python scripts/maintenance/backfill_captions.py --dry-run
"""

import argparse
import asyncio
import logging
import sys
import tempfile
import time
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from app.config import settings
from app.db import close_pool, get_connection, open_pool
from app.graph import remove_orphans
from app.ingestion import (
    _SOURCE_FORMAT_BY_SUFFIX,
    _insert_rows,
    caption_extracted_images,
    caption_rows,
    extract_in_worker,
    upload_images,
)
from app.storage import get_document_bytes

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S",
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger("backfill_captions")


async def _pending(only: list[str], skip: list[str], redo: bool) -> list[str]:
    clauses, params = [], []
    if not redo:
        clauses.append("captioned_at is null")
    for pattern in only:
        clauses.append("filename ilike %s")
        params.append(f"%{pattern}%")
    for pattern in skip:
        clauses.append("filename not ilike %s")
        params.append(f"%{pattern}%")
    where = f"where {' and '.join(clauses)}" if clauses else ""
    async with get_connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(f"select filename from document_files {where} order by filename", params)
            return [row[0] for row in await cur.fetchall()]


async def _document_context(filename: str) -> tuple[dict, str, int]:
    """The metadata and format its existing chunks carry, and the index
    the caption chunks start at (after every chunk that isn't a caption -
    the old captions, if any, are replaced)."""
    async with get_connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                "select metadata, source_format from documents where filename = %s order by chunk_index limit 1",
                (filename,),
            )
            row = await cur.fetchone()
            await cur.execute(
                "select coalesce(max(chunk_index), -1) + 1 from documents "
                "where filename = %s and source_type != 'image_caption'",
                (filename,),
            )
            start_index = (await cur.fetchone())[0]
    if row:
        return row[0], row[1], start_index
    return {}, _SOURCE_FORMAT_BY_SUFFIX.get(Path(filename).suffix.lower(), "pdf"), start_index


async def backfill_document(filename: str) -> dict | None:
    """Captions one document's images and stores them. None if the
    original file isn't in MinIO (it stays pending)."""
    file_bytes = await get_document_bytes(filename)
    if file_bytes is None:
        logger.warning("%s: original file not in MinIO, left pending", filename)
        return None

    with tempfile.NamedTemporaryFile(suffix=Path(filename).suffix, delete=False) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name
    try:
        _, images = await extract_in_worker(tmp_path, filename)
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    captioned, stats = await caption_extracted_images(images)
    metadata, source_format, start_index = await _document_context(filename)
    rows, uploads = caption_rows(filename, str(uuid.uuid4()), captioned, source_format, metadata, start_index)
    await upload_images(uploads)

    async with get_connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                "delete from documents where filename = %s and source_type = 'image_caption'", (filename,)
            )
            await _insert_rows(cur, rows)
            await cur.execute("update document_files set captioned_at = now() where filename = %s", (filename,))
            if settings.graph_retrieval:
                # Replaced caption chunks took their facts with them.
                await remove_orphans(cur)
        await conn.commit()

    stats.update(images=len(images), stored=len(rows))
    return stats


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", action="append", default=[], help="filename substring to include (repeatable)")
    parser.add_argument("--skip", action="append", default=[], help="filename substring to leave out (repeatable)")
    parser.add_argument("--redo", action="store_true", help="re-caption documents already captioned")
    parser.add_argument("--dry-run", action="store_true", help="list the pending documents only")
    parser.add_argument("--concurrency", type=int, default=2, help="documents in flight (parsing overlaps captioning)")
    args = parser.parse_args()

    await open_pool()
    try:
        filenames = await _pending(args.only, args.skip, args.redo)
        logger.info("%d document(s) to caption", len(filenames))
        if args.dry_run:
            for filename in filenames:
                print(filename)
            return

        semaphore = asyncio.Semaphore(args.concurrency)
        start = time.monotonic()
        processed = 0

        async def one(filename: str) -> None:
            nonlocal processed
            async with semaphore:
                doc_start = time.monotonic()
                stats = await backfill_document(filename)
                processed += 1
                elapsed = time.monotonic() - start
                left = (len(filenames) - processed) * elapsed / processed / 60
                if stats is not None:
                    logger.info(
                        "%s: %d images -> %d stored (%d captioned, %d cached, %d skipped%s) in %.0fs | %d/%d, ~%.0f min left",
                        filename, stats["images"], stats["stored"], stats["captioned"], stats["cached"],
                        stats["skipped"], f", {stats['failed']} failed" if stats["failed"] else "",
                        time.monotonic() - doc_start, processed, len(filenames), left,
                    )

        await asyncio.gather(*(one(filename) for filename in filenames))
        logger.info("done in %.1f min", (time.monotonic() - start) / 60)
        if settings.graph_retrieval:
            logger.info("new caption chunks have no graph yet: run scripts/maintenance/backfill_graph.py")
    finally:
        await close_pool()


if __name__ == "__main__":
    asyncio.run(main())
