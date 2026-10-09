import asyncio
import hashlib
import json
import logging
import uuid
from pathlib import Path

from pgvector import Vector
from psycopg.errors import UniqueViolation

from app.chunking import chunk_text
from app.config import settings
from app.db import get_connection
from app.dispatcher import extract
from app.embeddings import embed_text, embed_texts
from app.generation import count_text_tokens, generate_answer
from app.parsing import DuplicateDocument
from app.storage import upload_document, upload_image
from app.vision import describe_image

logger = logging.getLogger(__name__)

# The vision model's Ollama backend runs with a single parallel execution
# slot (-np 1) - anything above 1 here just queues client-side and risks
# blowing the describe_image() timeout waiting for a slot, without any real
# throughput gain.
CAPTION_CONCURRENCY = 1

# Document summaries are written from the whole document, not a sample of
# it: the summary exists to tell similar documents apart, and the details
# that do that are often in a section that appears once. Text that fits in
# SUMMARY_INPUT_TOKENS goes to the model in one call; longer documents are
# summarized section by section, then the section summaries are summarized
# (repeatedly, if even those don't fit). Half the context window leaves room
# for the prompt and the answer.
SUMMARY_INPUT_TOKENS = settings.chat_num_ctx // 2

# See summary_source_chunks: a spreadsheet past this many chunks is
# summarized from an even sample of them, keeping it to a few chat calls.
SPREADSHEET_SUMMARY_CHUNKS = 40

# Section summaries for one document run in parallel, bounded - vLLM batches
# concurrent requests, but a huge document shouldn't flood it.
SUMMARY_SECTION_CONCURRENCY = 4

_SUMMARY_INSTRUCTIONS = (
    "Write a concise 2-4 sentence summary of what this document is about, "
    "for use in a semantic search index. Mention specific named entities "
    "(people, organizations, systems, projects, places) explicitly rather "
    "than generic descriptions - a generic summary won't help distinguish "
    "this document from other similar ones. Write names and acronyms "
    "exactly as the document does - never guess what an acronym stands for. "
    "Plain text, no markdown.\n\n"
    "Filename: {filename}\n\n"
)

SUMMARY_PROMPT_TEMPLATE = _SUMMARY_INSTRUCTIONS + "Content:\n{excerpt}\n\nSummary:"

SUMMARY_FROM_SECTIONS_TEMPLATE = (
    _SUMMARY_INSTRUCTIONS
    + "The document was too long to read at once, so here are summaries of "
    "its sections, in order:\n{excerpt}\n\nSummary:"
)

SECTION_SUMMARY_TEMPLATE = (
    "This is part {part} of {parts} of a longer document. Summarize this part "
    "in 4-8 sentences. Keep every specific named entity (people, "
    "organizations, systems, projects, places), identifier, and figure that "
    "would help tell this document apart from similar ones, written exactly "
    "as the document writes it - never guess what an acronym stands for. "
    "Plain text, no markdown.\n\n"
    "Filename: {filename}\n\n"
    "Content:\n{excerpt}\n\n"
    "Summary:"
)

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


def summary_source_chunks(chunks: list[tuple[str, int | None]], source_format: str) -> list[str]:
    """The text a document's summary is written from: every chunk, except
    for large spreadsheets. A sheet with tens of thousands of rows would take
    dozens of chat calls to summarize section by section, to learn little
    more than an even sample of its rows says - so past
    SPREADSHEET_SUMMARY_CHUNKS a workbook contributes each sheet's first
    chunk (sheet name and header; page_number is the sheet index) plus
    chunks spaced evenly through the rest, in order. Only the first chunks
    isn't enough: in a workbook built from a template, those are often the
    template's help and terms-of-use sheets, not the data."""
    texts = [chunk for chunk, _ in chunks]
    if source_format != "xlsx" or len(chunks) <= SPREADSHEET_SUMMARY_CHUNKS:
        return texts
    firsts = {}
    for index, (_, page_number) in enumerate(chunks):
        firsts.setdefault(page_number, index)
    keep = set(firsts.values())
    rest = [i for i in range(len(chunks)) if i not in keep]
    slots = max(SPREADSHEET_SUMMARY_CHUNKS - len(keep), 0)
    if slots:
        step = len(rest) / slots
        keep.update(rest[int(n * step)] for n in range(slots))
    return [texts[i] for i in sorted(keep)]


def _group_into_sections(texts: list[str], token_counts: list[int]) -> list[str]:
    """Packs consecutive texts into sections of at most SUMMARY_INPUT_TOKENS
    each, in order. A single text over the limit becomes its own section
    rather than being split - chunks are far smaller than the limit, so
    this only guards against the loop never making progress."""
    sections: list[list[str]] = [[]]
    used = 0
    for text, tokens in zip(texts, token_counts):
        if sections[-1] and used + tokens > SUMMARY_INPUT_TOKENS:
            sections.append([])
            used = 0
        sections[-1].append(text)
        used += tokens
    return ["\n\n".join(section) for section in sections]


async def _summarize_sections(filename: str, sections: list[str]) -> list[str]:
    semaphore = asyncio.Semaphore(SUMMARY_SECTION_CONCURRENCY)

    async def one(index: int, section: str) -> str:
        prompt = SECTION_SUMMARY_TEMPLATE.format(
            part=index, parts=len(sections), filename=filename, excerpt=section,
        )
        async with semaphore:
            summary, _ = await generate_answer(prompt)
        return f"Part {index}: {summary.strip()}"

    return list(await asyncio.gather(*(one(i, s) for i, s in enumerate(sections, start=1))))


async def _generate_document_summary(filename: str, texts: list[str]) -> str | None:
    """Summarizes the whole of `texts` (see summary_source_chunks). Best-
    effort: a failed or empty summary just means this document won't get a
    document-level routing signal (document_routed_search falls back to its
    chunk-scan signal alone), not an ingestion failure."""
    texts = [t for t in texts if t.strip()]
    if not texts:
        return None
    try:
        from_sections = False
        while True:
            sections = _group_into_sections(texts, await count_text_tokens(texts))
            if len(sections) == 1:
                break
            logger.info("summarizing %s in %d sections", filename, len(sections))
            texts = await _summarize_sections(filename, sections)
            from_sections = True
        template = SUMMARY_FROM_SECTIONS_TEMPLATE if from_sections else SUMMARY_PROMPT_TEMPLATE
        summary, _ = await generate_answer(template.format(filename=filename, excerpt=sections[0]))
    except Exception:
        logger.warning("summary generation failed for %s", filename, exc_info=True)
        return None
    return summary.strip() or None


async def _upsert_document_summary(cur, filename: str, summary: str, embedding: list[float]) -> None:
    await cur.execute(
        """
        insert into document_summaries (filename, summary, embedding)
        values (%s, %s, %s)
        on conflict (filename) do update
        set summary = excluded.summary, embedding = excluded.embedding, created_at = now()
        """,
        (filename, summary, Vector(embedding)),
    )


async def find_identical_document(file_hash: str, filename: str) -> str | None:
    """The filename an identical file (same SHA-256) was already ingested
    under, or None. The same filename doesn't count - re-uploading a file
    under its own name re-ingests it."""
    async with get_connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                "select filename from document_files where sha256 = %s and filename != %s",
                (file_hash, filename),
            )
            row = await cur.fetchone()
    return row[0] if row else None


async def ingest_document(
    file_path: str,
    filename: str,
    content_type: str | None = None,
    metadata: dict | None = None,
    caption_images: bool = True,
    generate_summary: bool = True,
) -> int:
    """Ingests `file_path` as `filename`, replacing whatever was ingested
    under that filename before (old rows are deleted in the same
    transaction the new ones are inserted in, so a failed ingest leaves the
    previous version searchable). Raises DuplicateDocument, before doing
    any work, if identical bytes were already ingested under another
    filename."""
    file_bytes = Path(file_path).read_bytes()
    file_hash = hashlib.sha256(file_bytes).hexdigest()
    existing = await find_identical_document(file_hash, filename)
    if existing:
        raise DuplicateDocument(filename, existing)

    text_chunks, images = extract(file_path, filename, content_type)
    source_format = _SOURCE_FORMAT_BY_SUFFIX.get(Path(filename).suffix.lower(), "pdf")
    metadata = metadata or {}

    await upload_document(filename, file_bytes)

    # Chunk each source unit (page/paragraph/slide/sheet) independently
    # rather than joining everything into one string first - that's what
    # lets each output chunk keep an accurate page_number instead of
    # losing page boundaries across a merged blob.
    prose_chunks = [c for c in text_chunks if c.source_type == "text"]
    chart_chunks = [c for c in text_chunks if c.source_type == "chart_data"]

    page_tagged_chunks: list[tuple[str, int | None]] = [
        (sub_chunk, unit.page_number)
        for unit in prose_chunks
        for sub_chunk in chunk_text(unit.content, settings.chunk_size)
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

    captioned = (
        await asyncio.gather(*[_caption(image) for image in images])
        if images and caption_images
        else []
    )

    # Embed everything up front, in batches (see app/embeddings.py), so the
    # DB loop below only inserts.
    chunk_texts = [chunk for chunk, _ in page_tagged_chunks]
    chunk_embeddings = await embed_texts(chunk_texts)
    chart_embeddings = await embed_texts([unit.content for unit in chart_chunks])

    for image, caption in captioned:
        if caption is None:
            logger.warning(
                "skipping image %s (page %s) — captioning failed after retries",
                image.image_id, image.page_number,
            )
    captioned = [(image, caption) for image, caption in captioned if caption is not None]
    caption_embeddings = await embed_texts([caption for _, caption in captioned])

    summary = (
        await _generate_document_summary(filename, summary_source_chunks(page_tagged_chunks, source_format))
        if generate_summary else None
    )
    summary_embedding = await embed_text(summary) if summary else None

    inserted = 0
    async with get_connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute("delete from documents where filename = %s", (filename,))
            try:
                await cur.execute(
                    """
                    insert into document_files (filename, sha256) values (%s, %s)
                    on conflict (filename) do update set sha256 = excluded.sha256, created_at = now()
                    """,
                    (filename, file_hash),
                )
            except UniqueViolation:
                # The same bytes were committed under another name while
                # this one was being processed (two concurrent uploads).
                # Leaving the block rolls this transaction back.
                existing = await find_identical_document(file_hash, filename)
                raise DuplicateDocument(filename, existing or "another file") from None
            if summary and summary_embedding:
                await _upsert_document_summary(cur, filename, summary, summary_embedding)

            for (chunk, page_number), embedding in zip(page_tagged_chunks, chunk_embeddings):
                await _insert_row(
                    cur, filename, inserted, chunk, embedding, "text", source_format, metadata,
                    page_number=page_number,
                )
                inserted += 1

            for unit, embedding in zip(chart_chunks, chart_embeddings):
                await _insert_row(
                    cur, filename, inserted, unit.content, embedding, "chart_data", source_format, metadata,
                    page_number=unit.page_number,
                )
                inserted += 1

            for (image, caption), embedding in zip(captioned, caption_embeddings):
                object_key = f"{document_id}/{image.image_id}.png"
                await upload_image(object_key, image.image_bytes)

                await _insert_row(
                    cur, filename, inserted, caption, embedding, "image_caption", source_format, metadata,
                    source_image_path=object_key, page_number=image.page_number, bbox=image.bbox,
                )
                inserted += 1

        await conn.commit()

    return inserted
