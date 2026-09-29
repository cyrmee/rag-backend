"""Rebuilds the prose ("text") chunks of already-ingested documents with the
current extractors and chunker, then regenerates each document's summary -
for when chunking changes (e.g. DOCX paragraphs packed into sections, PPTX
text grouped per slide) and existing rows should pick it up.

Only text rows are replaced. Image captions and chart data are left exactly
as they are, so no vision-model captioning is re-run. Originals come from
MinIO (stored at upload time); documents with no stored original are skipped
and listed.

Usage:
    python scripts/maintenance/rechunk_documents.py --dry-run         # old vs new chunk counts
    python scripts/maintenance/rechunk_documents.py                   # all docx + pptx
    python scripts/maintenance/rechunk_documents.py --formats docx
    python scripts/maintenance/rechunk_documents.py --only "ID/Admin/Some file.docx"
"""

import argparse
import asyncio
import logging
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from app.chunking import chunk_text
from app.config import settings
from app.db import close_pool, get_connection, open_pool
from app.dispatcher import extract
from app.embeddings import embed_text
from app.ingestion import _generate_document_summary, _insert_row, _upsert_document_summary
from app.storage import _download_sync, document_key_for

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
for noisy in ("httpx", "httpcore"):
    logging.getLogger(noisy).setLevel(logging.WARNING)
logger = logging.getLogger("rechunk_documents")


async def _documents(formats: list[str], only: list[str] | None) -> list[tuple[str, str]]:
    async with get_connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                """
                select distinct filename, source_format from documents
                where source_type = 'text' and source_format = any(%s)
                order by filename
                """,
                (formats,),
            )
            rows = await cur.fetchall()
    return [(f, fmt) for f, fmt in rows if not only or f in only]


async def _existing(filename: str) -> tuple[int, int | None, dict]:
    """(current text chunk count, lowest chunk_index used by a non-text row
    or None, metadata of the existing text rows)."""
    async with get_connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                """
                select count(*) filter (where source_type = 'text'),
                       min(chunk_index) filter (where source_type <> 'text'),
                       (array_agg(metadata) filter (where source_type = 'text'))[1]
                from documents where filename = %s
                """,
                (filename,),
            )
            count, first_other, metadata = await cur.fetchone()
    return count, first_other, metadata or {}


def _new_chunks(file_bytes: bytes, filename: str) -> list[tuple[str, int | None]]:
    with tempfile.NamedTemporaryFile(suffix=Path(filename).suffix) as tmp:
        tmp.write(file_bytes)
        tmp.flush()
        units, _ = extract(tmp.name, filename)
    return [
        (chunk, unit.page_number)
        for unit in units
        if unit.source_type == "text"
        for chunk in chunk_text(unit.content, settings.chunk_size)
    ]


async def _rechunk(filename: str, source_format: str, dry_run: bool) -> str:
    try:
        file_bytes = await asyncio.to_thread(_download_sync, document_key_for(filename))
    except Exception:
        return "no-original"

    chunks = await asyncio.to_thread(_new_chunks, file_bytes, filename)
    old_count, first_other, metadata = await _existing(filename)
    logger.info("%s: %d -> %d text chunks", filename, old_count, len(chunks))
    if not chunks:
        return "empty"
    # Text rows are numbered 0..n-1 ahead of chart/caption rows; the new
    # set must still fit in front of them so no chunk_index is shared
    # (retrieval dedupes on filename + chunk_index).
    if first_other is not None and len(chunks) > first_other:
        logger.warning("%s: %d new chunks would overlap chunk_index %d, skipped", filename, len(chunks), first_other)
        return "skipped"
    if dry_run:
        return "dry-run"

    texts = [c for c, _ in chunks]
    embeddings = [await embed_text(t) for t in texts]
    summary = await _generate_document_summary(filename, texts, embeddings)
    summary_embedding = await embed_text(summary) if summary else None

    async with get_connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute("delete from documents where filename = %s and source_type = 'text'", (filename,))
            for index, ((content, page_number), embedding) in enumerate(zip(chunks, embeddings)):
                await _insert_row(
                    cur, filename, index, content, embedding, "text", source_format, metadata,
                    page_number=page_number,
                )
            if summary and summary_embedding:
                await _upsert_document_summary(cur, filename, summary, summary_embedding)
        await conn.commit()
    return "rechunked"


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--formats", nargs="+", default=["docx", "pptx"])
    parser.add_argument("--only", nargs="+", help="exact filenames to rechunk")
    parser.add_argument("--dry-run", action="store_true", help="report old vs new chunk counts, change nothing")
    args = parser.parse_args()

    await open_pool()
    outcomes: dict[str, list[str]] = {}
    try:
        for filename, source_format in await _documents(args.formats, args.only):
            outcome = await _rechunk(filename, source_format, args.dry_run)
            outcomes.setdefault(outcome, []).append(filename)
    finally:
        await close_pool()

    for outcome, filenames in sorted(outcomes.items()):
        logger.info("%s: %d", outcome, len(filenames))
        if outcome in ("no-original", "skipped", "empty"):
            for f in filenames:
                logger.info("  %s", f)


if __name__ == "__main__":
    asyncio.run(main())
