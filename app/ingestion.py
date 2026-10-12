import asyncio
import hashlib
import json
import logging
import time
import uuid
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from pathlib import Path

from pgvector import Vector
from psycopg.errors import UniqueViolation

from app.chunking import chunk_text
from app.config import settings
from app.db import get_connection
from app.dispatcher import extract
from app.embeddings import embed_text, embed_texts
from app.extractors.types import ExtractedImage, ExtractionResult
from app.generation import count_text_tokens, generate_answer
from app.graph import passage_text, prepare_graph, remove_orphans, store_graph
from app.parsing import DuplicateDocument
from app.storage import upload_document, upload_image
from app.vision import describe_image, image_hash, prepare_image

logger = logging.getLogger(__name__)

# Parsing a document (pymupdf, python-docx, openpyxl, LibreOffice for
# charts) is CPU work that takes seconds for a report and minutes for a
# thousand-page book. Run on the event loop it stalled every other
# document in flight, model calls included; it runs in worker processes
# instead (threads won't do: pymupdf isn't thread-safe). The pool starts
# on first use, so importing this module costs nothing.
_extract_pool: ProcessPoolExecutor | None = None


def _pool() -> ProcessPoolExecutor:
    global _extract_pool
    if _extract_pool is None:
        _extract_pool = ProcessPoolExecutor(max_workers=settings.extract_workers)
    return _extract_pool


async def extract_in_worker(file_path: str, filename: str, content_type: str | None = None) -> ExtractionResult:
    """app.dispatcher.extract, in a worker process."""
    return await asyncio.get_running_loop().run_in_executor(_pool(), extract, file_path, filename, content_type)


@contextmanager
def _timed(timings: dict | None, stage: str):
    """Adds the block's wall-clock seconds to timings[stage]."""
    start = time.monotonic()
    try:
        yield
    finally:
        if timings is not None:
            timings[stage] = timings.get(stage, 0.0) + time.monotonic() - start

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


def _row(
    filename: str,
    chunk_index: int,
    content: str,
    embedding,
    source_type: str,
    source_format: str,
    metadata: dict,
    source_image_path: str | None = None,
    page_number: int | None = None,
    bbox: tuple[float, float, float, float] | None = None,
) -> tuple:
    """One documents row, with its id chosen here so rows can be inserted
    in one batch and still be referred to (the graph links facts to chunk
    ids) without a round trip per chunk."""
    return (
        str(uuid.uuid4()), filename, chunk_index, content, Vector(embedding),
        source_type, source_format, json.dumps(metadata), source_image_path, page_number,
        json.dumps(list(bbox)) if bbox else None,
    )


async def _insert_rows(cur, rows: list[tuple]) -> None:
    if not rows:
        return
    await cur.executemany(
        """
        insert into documents (
            id, filename, chunk_index, content, embedding,
            source_type, source_format, metadata, source_image_path, page_number, bbox
        )
        values (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """,
        rows,
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


async def _cached_captions(hashes: list[str]) -> dict[str, tuple[str, list[float]]]:
    async with get_connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                "select sha256, caption, embedding from image_captions where sha256 = any(%s)", (hashes,)
            )
            # pgvector reads the column back as its own Vector type; the
            # rows are built with Vector(list) like everything else.
            return {
                sha256: (caption, embedding.to_list()) for sha256, caption, embedding in await cur.fetchall()
            }


async def _store_captions(captions: list[tuple[str, str]], embeddings: list[list[float]]) -> None:
    """Adds (sha256, caption) pairs to the cache, in their own transaction:
    a caption is worth keeping even if the document it came from fails
    later on."""
    async with get_connection() as conn:
        async with conn.cursor() as cur:
            await cur.executemany(
                """
                insert into image_captions (sha256, caption, embedding, model)
                values (%s, %s, %s, %s)
                on conflict (sha256) do nothing
                """,
                [
                    (sha256, caption, Vector(embedding), settings.vision_model)
                    for (sha256, caption), embedding in zip(captions, embeddings)
                ],
            )
        await conn.commit()


async def caption_extracted_images(
    images: list[ExtractedImage],
) -> tuple[list[tuple[ExtractedImage, str, list[float]]], dict]:
    """Captions the images worth captioning (app.vision.prepare_image,
    which also replaces each image's bytes with what the model and the
    store get), through the cache in image_captions: an image captioned
    before - in another document, an earlier version of this one, or
    earlier in this one - comes from the table, caption and embedding
    both. Returns (image, caption, embedding) for every kept image, in the
    order given, and counts: skipped (too small, or unreadable), cached
    (occurrences served from the cache), captioned and failed (distinct
    new images). Vision calls are serialized by app.vision."""
    stats = {"skipped": 0, "cached": 0, "captioned": 0, "failed": 0}
    kept: list[tuple[ExtractedImage, str]] = []
    for image in images:
        prepared = prepare_image(image.image_bytes)
        if prepared is None:
            stats["skipped"] += 1
            continue
        image.image_bytes = prepared
        kept.append((image, image_hash(prepared)))
    if not kept:
        return [], stats

    hashes = sorted({sha256 for _, sha256 in kept})
    cached = await _cached_captions(hashes)
    stats["cached"] = sum(1 for _, sha256 in kept if sha256 in cached)

    missing = [sha256 for sha256 in hashes if sha256 not in cached]
    if missing:
        first_image = {}
        for image, sha256 in kept:
            first_image.setdefault(sha256, image)
        captions = await asyncio.gather(*(describe_image(first_image[sha256].image_bytes) for sha256 in missing))
        new = [(sha256, caption) for sha256, caption in zip(missing, captions) if caption]
        stats["captioned"] = len(new)
        stats["failed"] = len(missing) - len(new)
        if stats["failed"]:
            logger.warning("%d image(s) skipped: captioning failed after retries", stats["failed"])
        if new:
            embeddings = await embed_texts([caption for _, caption in new])
            await _store_captions(new, embeddings)
            cached.update({sha256: (caption, embedding) for (sha256, caption), embedding in zip(new, embeddings)})

    return [(image, *cached[sha256]) for image, sha256 in kept if sha256 in cached], stats


def caption_rows(
    filename: str,
    document_id: str,
    captioned: list[tuple[ExtractedImage, str, list[float]]],
    source_format: str,
    metadata: dict,
    start_index: int,
) -> tuple[list[tuple], list[tuple[str, bytes]]]:
    """The documents rows for captioned images, numbered from start_index,
    and the (object key, bytes) pairs to put in MinIO for them."""
    rows, uploads = [], []
    for index, (image, caption, embedding) in enumerate(captioned, start=start_index):
        object_key = f"{document_id}/{image.image_id}.png"
        uploads.append((object_key, image.image_bytes))
        rows.append(_row(
            filename, index, caption, embedding, "image_caption", source_format, metadata,
            source_image_path=object_key, page_number=image.page_number, bbox=image.bbox,
        ))
    return rows, uploads


async def upload_images(uploads: list[tuple[str, bytes]]) -> None:
    await asyncio.gather(*(upload_image(object_key, image_bytes) for object_key, image_bytes in uploads))


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
    extract_graph: bool = True,
    stats: dict | None = None,
) -> int:
    """Ingests `file_path` as `filename`, replacing whatever was ingested
    under that filename before (old rows are deleted in the same
    transaction the new ones are inserted in, so a failed ingest leaves the
    previous version searchable). Raises DuplicateDocument, before doing
    any work, if identical bytes were already ingested under another
    filename.

    The three enrichment steps - image captions, the document summary, the
    knowledge graph (the last only with settings.graph_retrieval) - can
    each be left out and done later by the matching backfill script; a
    bulk run does extract/embed/store only, as that's the pass the others
    depend on. `stats`, if given, receives "timings" (seconds per stage),
    "captions" (see caption_extracted_images) and "chunks"."""
    timings: dict[str, float] = {}
    file_bytes = Path(file_path).read_bytes()
    file_hash = hashlib.sha256(file_bytes).hexdigest()
    existing = await find_identical_document(file_hash, filename)
    if existing:
        raise DuplicateDocument(filename, existing)

    with _timed(timings, "extract"):
        text_chunks, images = await extract_in_worker(file_path, filename, content_type)
    source_format = _SOURCE_FORMAT_BY_SUFFIX.get(Path(filename).suffix.lower(), "pdf")
    metadata = metadata or {}

    with _timed(timings, "store"):
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

    captioned, caption_stats = [], {}
    if images and caption_images:
        with _timed(timings, "caption"):
            captioned, caption_stats = await caption_extracted_images(images)

    # Embed everything up front, in batches (see app/embeddings.py), so the
    # DB transaction below only inserts.
    with _timed(timings, "embed"):
        chunk_texts = [chunk for chunk, _ in page_tagged_chunks]
        chunk_embeddings = await embed_texts(chunk_texts)
        chart_embeddings = await embed_texts([unit.content for unit in chart_chunks])

    summary = summary_embedding = None
    if generate_summary:
        with _timed(timings, "summary"):
            summary = await _generate_document_summary(
                filename, summary_source_chunks(page_tagged_chunks, source_format)
            )
            summary_embedding = await embed_text(summary) if summary else None

    rows: list[tuple] = []
    for (chunk, page_number), embedding in zip(page_tagged_chunks, chunk_embeddings):
        rows.append(_row(filename, len(rows), chunk, embedding, "text", source_format, metadata, page_number=page_number))
    for unit, embedding in zip(chart_chunks, chart_embeddings):
        rows.append(_row(
            filename, len(rows), unit.content, embedding, "chart_data", source_format, metadata,
            page_number=unit.page_number,
        ))
    caption_chunk_rows, uploads = caption_rows(filename, document_id, captioned, source_format, metadata, len(rows))
    rows.extend(caption_chunk_rows)
    # The graph links facts to chunk ids; the backfill passes the ids it
    # already has, so the keys are whatever the caller used.
    chunk_ids = {index: row[0] for index, row in enumerate(rows)}

    # Knowledge-graph extraction (app/graph.py). Model calls happen here,
    # before the transaction; only the writes are in it.
    graph_batch = None
    if settings.graph_retrieval and extract_graph:
        with _timed(timings, "graph"):
            texts: dict[int, str] = {}
            for chunk, page_number in page_tagged_chunks:
                texts[len(texts)] = passage_text(filename, page_number, "text", chunk)
            for unit in chart_chunks:
                texts[len(texts)] = passage_text(filename, unit.page_number, "chart_data", unit.content)
            for image, caption, _ in captioned:
                texts[len(texts)] = passage_text(filename, image.page_number, "image_caption", caption)
            graph_batch = await prepare_graph(texts)

    with _timed(timings, "store"):
        await upload_images(uploads)
        async with get_connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute("delete from documents where filename = %s", (filename,))
                try:
                    await cur.execute(
                        """
                        insert into document_files (filename, sha256, captioned_at)
                        values (%s, %s, case when %s then now() end)
                        on conflict (filename) do update
                        set sha256 = excluded.sha256, created_at = now(), captioned_at = excluded.captioned_at
                        """,
                        (filename, file_hash, caption_images),
                    )
                except UniqueViolation:
                    # The same bytes were committed under another name while
                    # this one was being processed (two concurrent uploads).
                    # Leaving the block rolls this transaction back.
                    existing = await find_identical_document(file_hash, filename)
                    raise DuplicateDocument(filename, existing or "another file") from None
                if summary and summary_embedding:
                    await _upsert_document_summary(cur, filename, summary, summary_embedding)

                await _insert_rows(cur, rows)

                if graph_batch is not None:
                    graph_stats = await store_graph(cur, graph_batch, chunk_ids)
                    logger.info("graph for %s: %s", filename, graph_stats)
                # The previous version's facts went with its rows; whatever
                # phrases and facts only it used go too.
                await remove_orphans(cur)

            await conn.commit()

    if stats is not None:
        stats.update(timings=timings, captions=caption_stats, chunks=len(rows))
    return len(rows)
