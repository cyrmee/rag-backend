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
from app.vision import describe_image

logger = logging.getLogger(__name__)

IMAGE_STORAGE_DIR = Path("/tmp/extracted_images")
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
    source_image_path: str | None = None,
    page_number: int | None = None,
    bbox: tuple[float, float, float, float] | None = None,
) -> None:
    await cur.execute(
        """
        insert into documents (
            filename, chunk_index, content, embedding,
            source_type, source_format, source_image_path, page_number, bbox
        )
        values (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        """,
        (
            filename, chunk_index, content, Vector(embedding),
            source_type, source_format, source_image_path, page_number,
            json.dumps(list(bbox)) if bbox else None,
        ),
    )


async def ingest_document(file_path: str, filename: str, content_type: str | None = None) -> int:
    text_chunks, images = extract(file_path, filename, content_type)
    source_format = _SOURCE_FORMAT_BY_SUFFIX.get(Path(filename).suffix.lower(), "pdf")

    plain_text = "\n".join(c.content for c in text_chunks if c.source_type == "text")
    chart_chunks = [c.content for c in text_chunks if c.source_type == "chart_data"]
    chunks = chunk_text(plain_text, settings.chunk_size, settings.chunk_overlap) if plain_text else []

    image_dir = None
    if images:
        image_dir = IMAGE_STORAGE_DIR / str(uuid.uuid4())
        image_dir.mkdir(parents=True, exist_ok=True)

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
            for index, chunk in enumerate(chunks):
                embedding = await embed_text(chunk)
                await _insert_row(cur, filename, index, chunk, embedding, "text", source_format)
                inserted += 1

            for chunk in chart_chunks:
                embedding = await embed_text(chunk)
                await _insert_row(cur, filename, inserted, chunk, embedding, "chart_data", source_format)
                inserted += 1

            for image, caption in captioned:
                if caption is None:
                    logger.warning(
                        "skipping image %s (page %s) — captioning failed after retries",
                        image.image_id, image.page_number,
                    )
                    continue

                image_path = str(image_dir / f"{image.image_id}.png")
                Path(image_path).write_bytes(image.image_bytes)

                embedding = await embed_text(caption)
                await _insert_row(
                    cur, filename, inserted, caption, embedding, "image_caption", source_format,
                    source_image_path=image_path, page_number=image.page_number, bbox=image.bbox,
                )
                inserted += 1

        await conn.commit()

    return inserted
