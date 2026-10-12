"""Check: image filtering, the caption cache, and the captioning backfill.

1. prepare_image leaves a 64x64 image out, converts a BMP to PNG, and
   passes a PNG through untouched.
2. Ingesting fixtures/sample.pptx (one 600x360 image) fills the cache;
   ingesting it again under the same name is served from the cache - no
   vision call.
3. Ingesting fixtures/sample.docx with captioning skipped leaves
   document_files.captioned_at null and stores no caption; the backfill
   then captions it (from the cache: the fixtures share the image),
   numbers the caption chunk after the text chunks, and sets captioned_at.

Summaries and the graph are left out to keep it to seconds. The test
documents are removed afterwards; the cache rows stay (they're real).
"""

import asyncio
import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "maintenance"))

from PIL import Image

from app.db import close_pool, get_connection, open_pool
from app.graph import remove_orphans
from app.ingestion import ingest_document
from app.vision import prepare_image
from backfill_captions import backfill_document

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"
PPTX_NAME = "caption-cache-check.pptx"
DOCX_NAME = "caption-cache-check.docx"


def _png(size: tuple[int, int], fmt: str = "PNG") -> bytes:
    out = io.BytesIO()
    Image.new("RGB", size, "white").save(out, fmt)
    return out.getvalue()


def test_prepare_image() -> None:
    assert prepare_image(_png((64, 64))) is None, "a 64x64 image should be skipped"
    assert prepare_image(_png((300, 64))) is None, "a 300x64 image should be skipped"
    png = _png((300, 300))
    assert prepare_image(png) == png, "a PNG should pass through unchanged"
    converted = prepare_image(_png((300, 300), "BMP"))
    assert converted and Image.open(io.BytesIO(converted)).format == "PNG", "a BMP should come back as PNG"
    assert prepare_image(b"not an image") is None
    print("OK prepare_image: small images skipped, BMP converted, PNG untouched")


async def _query(sql: str, params: tuple):
    async with get_connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(sql, params)
            return await cur.fetchall()


async def _cleanup() -> None:
    async with get_connection() as conn:
        async with conn.cursor() as cur:
            for name in (PPTX_NAME, DOCX_NAME):
                await cur.execute("delete from documents where filename = %s", (name,))
                await cur.execute("delete from document_files where filename = %s", (name,))
                await cur.execute("delete from document_summaries where filename = %s", (name,))
            await remove_orphans(cur)
        await conn.commit()


async def test_cache_and_backfill() -> None:
    first: dict = {}
    await ingest_document(
        str(FIXTURES / "sample.pptx"), PPTX_NAME, generate_summary=False, extract_graph=False, stats=first,
    )
    captions = first["captions"]
    assert captions["captioned"] + captions["cached"] == 1, f"expected the one image kept: {captions}"
    assert captions["skipped"] == 0, captions
    rows = await _query(
        "select content from documents where filename = %s and source_type = 'image_caption'", (PPTX_NAME,)
    )
    assert len(rows) == 1 and rows[0][0].strip(), "expected one stored caption"
    print(f"OK first ingest: {captions}; caption: {rows[0][0][:90]!r}...")

    second: dict = {}
    await ingest_document(
        str(FIXTURES / "sample.pptx"), PPTX_NAME, generate_summary=False, extract_graph=False, stats=second,
    )
    assert second["captions"]["captioned"] == 0 and second["captions"]["cached"] == 1, second["captions"]
    assert "caption" in second["timings"] and second["timings"]["caption"] < 5, second["timings"]
    print(f"OK second ingest served from the cache: {second['captions']}, {second['timings']['caption']:.2f}s")

    skipped: dict = {}
    await ingest_document(
        str(FIXTURES / "sample.docx"), DOCX_NAME,
        caption_images=False, generate_summary=False, extract_graph=False, stats=skipped,
    )
    assert skipped["captions"] == {}, skipped
    [(captioned_at,)] = await _query("select captioned_at from document_files where filename = %s", (DOCX_NAME,))
    assert captioned_at is None, "captioned_at should be null after --skip-images"
    assert not await _query(
        "select 1 from documents where filename = %s and source_type = 'image_caption'", (DOCX_NAME,)
    )
    print("OK ingest with caption_images=False: no caption rows, captioned_at null")

    stats = await backfill_document(DOCX_NAME)
    assert stats and stats["stored"] == 1, stats
    [(captioned_at,)] = await _query("select captioned_at from document_files where filename = %s", (DOCX_NAME,))
    assert captioned_at is not None, "backfill should set captioned_at"
    indexes = await _query(
        "select source_type, chunk_index from documents where filename = %s order by chunk_index", (DOCX_NAME,)
    )
    assert indexes[-1][0] == "image_caption" and indexes[-1][1] == len(indexes) - 1, indexes
    print(f"OK backfill: {stats}; chunk order {indexes}")

    again = await backfill_document(DOCX_NAME)
    assert again and again["stored"] == 1 and again["captioned"] == 0, again
    count = await _query("select count(*) from documents where filename = %s and source_type = 'image_caption'", (DOCX_NAME,))
    assert count[0][0] == 1, "a second backfill should replace, not add"
    print("OK backfill again replaces the caption chunk")


async def main() -> None:
    test_prepare_image()
    await open_pool()
    try:
        await _cleanup()
        try:
            await test_cache_and_backfill()
        finally:
            await _cleanup()
    finally:
        await close_pool()


if __name__ == "__main__":
    asyncio.run(main())
