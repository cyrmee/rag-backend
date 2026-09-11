import asyncio
import json
import logging
import uuid
from pathlib import Path

from pgvector import Vector

from app.chunking import chunk_text
from app.config import settings
from app.db import get_connection
from app.dispatcher import extract
from app.embeddings import embed_text
from app.storage import upload_document, upload_image
from app.vision import describe_image

logger = logging.getLogger(__name__)

CAPTION_CONCURRENCY = 4

_SOURCE_FORMAT_BY_SUFFIX = {
    ".pdf": "pdf",
    ".docx": "docx",
    ".pptx": "pptx",
    ".xlsx": "xlsx",
    ".md": "md",
    ".txt": "txt",
}


async def _insert_row(
    cur,
    filename: str,
    chunk_index: int,
    content: str,
    embedding: list[float],
    source_type: str,
    source_format: str,
    metadata: dict,
    source_image_path: str | None = None,
    page_number: int | None = None,
    bbox: tuple[float, float, float, float] | None = None,
) -> None:
    await cur.execute(
        """
        insert into documents (
            filename, chunk_index, content, embedding,
            source_type, source_format, metadata, source_image_path, page_number, bbox
        )
        values (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """,
        (
            filename, chunk_index, content, Vector(embedding),
            source_type, source_format, json.dumps(metadata), source_image_path, page_number,
            json.dumps(list(bbox)) if bbox else None,
        ),
    )


async def ingest_document(
    file_path: str,
    filename: str,
    content_type: str | None = None,
    metadata: dict | None = None,
) -> int:
    text_chunks, images = extract(file_path, filename, content_type)
    source_format = _SOURCE_FORMAT_BY_SUFFIX.get(Path(filename).suffix.lower(), "pdf")
    metadata = metadata or {}

    await upload_document(filename, Path(file_path).read_bytes())

    # Chunk each source unit (page/paragraph/slide/sheet) independently
    # rather than joining everything into one string first - that's what
    # lets each output chunk keep an accurate page_number instead of
    # losing page boundaries across a merged blob.
    prose_chunks = [c for c in text_chunks if c.source_type == "text"]
    chart_chunks = [c for c in text_chunks if c.source_type == "chart_data"]

    page_tagged_chunks: list[tuple[str, int | None]] = [
        (sub_chunk, unit.page_number)
        for unit in prose_chunks
        for sub_chunk in chunk_text(unit.content, settings.chunk_size, settings.chunk_overlap)
    ]

    document_id = str(uuid.uuid4())

    # Caption all images concurrently (bounded) before touching the DB -
    # each call is a network round-trip to the remote vision model, so
    # captioning one-at-a-time would dominate ingestion time on a batch.
    semaphore = asyncio.Semaphore(CAPTION_CONCURRENCY)

    async def _caption(image):
        async with semaphore:
            caption = await describe_image(image.image_bytes)
        return image, caption

    captioned = await asyncio.gather(*[_caption(image) for image in images]) if images else []

    inserted = 0
    async with get_connection() as conn:
        async with conn.cursor() as cur:
            for chunk, page_number in page_tagged_chunks:
                embedding = await embed_text(chunk)
                await _insert_row(
                    cur, filename, inserted, chunk, embedding, "text", source_format, metadata,
                    page_number=page_number,
                )
                inserted += 1

            for unit in chart_chunks:
                embedding = await embed_text(unit.content)
                await _insert_row(
                    cur, filename, inserted, unit.content, embedding, "chart_data", source_format, metadata,
                    page_number=unit.page_number,
                )
                inserted += 1

            for image, caption in captioned:
                if caption is None:
                    logger.warning(
                        "skipping image %s (page %s) — captioning failed after retries",
                        image.image_id, image.page_number,
                    )
                    continue

                object_key = f"{document_id}/{image.image_id}.png"
                await upload_image(object_key, image.image_bytes)

                embedding = await embed_text(caption)
                await _insert_row(
                    cur, filename, inserted, caption, embedding, "image_caption", source_format, metadata,
                    source_image_path=object_key, page_number=image.page_number, bbox=image.bbox,
                )
                inserted += 1

        await conn.commit()

    return inserted
