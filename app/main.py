import json
import logging
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
from fastapi import FastAPI, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.requests import Request
from fastapi.responses import JSONResponse, StreamingResponse

from pgvector import Vector

from app.agent import run_agentic_ask, run_agentic_ask_stream
from app.config import settings
from app.db import close_pool, delete_document_chunks, get_connection, open_pool
from app.embeddings import embed_text
from app.generation import generate_answer, stream_generate
from app.ingestion import ingest_document
from app.parsing import UnsupportedFileType
from app.schemas import AskRequest, AskResponse, DocumentInfo, SourceInfo, UploadResponse
from app.storage import ensure_bucket, get_document_url

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logging.getLogger("app.agent").setLevel(logging.INFO)


@asynccontextmanager
async def lifespan(app: FastAPI):
    await open_pool()
    async with get_connection() as conn:
        await conn.execute("select 1")
    await ensure_bucket()
    yield
    await close_pool()


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(httpx.HTTPError)
async def ollama_unreachable_handler(request: Request, exc: httpx.HTTPError):
    return JSONResponse(
        status_code=503,
        content={"detail": f"Ollama is unreachable or returned an error: {exc}"},
    )


def sse_event(event: str, data: str) -> str:
    """Formats one Server-Sent Event. `data` is split on newlines since
    each line of an SSE data field needs its own "data: " prefix."""
    data_lines = "\n".join(f"data: {line}" for line in data.split("\n"))
    return f"event: {event}\n{data_lines}\n\n"


async def build_source_infos(rows: list[dict]) -> list[SourceInfo]:
    """Turns raw (content, source_type, source_format, filename, page_number)
    dicts into SourceInfo objects, generating one presigned document link per
    unique filename (not per row) to avoid redundant MinIO round-trips."""
    url_cache: dict[str, str | None] = {}
    infos = []
    for row in rows:
        filename = row["filename"]
        if filename not in url_cache:
            url_cache[filename] = await get_document_url(filename)
        infos.append(SourceInfo(
            content=row["content"],
            filename=filename,
            source_type=row["source_type"],
            source_format=row["source_format"],
            page_number=row.get("page_number"),
            document_url=url_cache[filename],
        ))
    return infos


@app.post("/upload", response_model=UploadResponse)
async def upload(file: UploadFile):
    suffix = Path(file.filename or "").suffix
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        await delete_document_chunks(file.filename or "unknown")
        chunk_count = await ingest_document(tmp_path, file.filename or "unknown", file.content_type)
    except UnsupportedFileType as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    return UploadResponse(filename=file.filename or "unknown", chunks_ingested=chunk_count)


async def _retrieve_rows(question: str) -> list[dict]:
    query_vector = Vector(await embed_text(question))

    async with get_connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                """
                select content, source_type, source_format, filename, page_number
                from documents
                order by embedding <=> %s
                limit %s
                """,
                (query_vector, settings.top_k),
            )
            rows = await cur.fetchall()

    return [
        {
            "content": content, "source_type": source_type, "source_format": source_format,
            "filename": filename, "page_number": page_number,
        }
        for content, source_type, source_format, filename, page_number in rows
    ]


@app.post("/ask", response_model=AskResponse)
async def ask(request: AskRequest):
    rows = await _retrieve_rows(request.question)

    if not rows:
        return AskResponse(
            answer="No documents have been uploaded yet, so I have nothing to answer from.",
            sources=[],
        )

    context = "\n\n".join(f"[{i+1}] {row['content']}" for i, row in enumerate(rows))
    prompt = (
        "Answer the question using only the context below. "
        "If the answer isn't present in the context, say you don't know.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {request.question}\n"
        "Answer:"
    )

    answer, _thinking = await generate_answer(prompt)
    sources = await build_source_infos(rows)
    return AskResponse(answer=answer, sources=sources)


@app.post("/ask/stream")
async def ask_stream(request: AskRequest):
    rows = await _retrieve_rows(request.question)

    async def event_stream():
        if not rows:
            yield sse_event(
                "answer",
                "No documents have been uploaded yet, so I have nothing to answer from.",
            )
            yield sse_event("done", json.dumps({"sources": []}))
            return

        context = "\n\n".join(f"[{i+1}] {row['content']}" for i, row in enumerate(rows))
        prompt = (
            "Answer the question using only the context below. "
            "If the answer isn't present in the context, say you don't know.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {request.question}\n"
            "Answer:"
        )

        try:
            async for chunk in stream_generate(prompt):
                if chunk.get("thinking"):
                    yield sse_event("thinking", chunk["thinking"])
                if chunk.get("response"):
                    yield sse_event("answer", chunk["response"])
        except httpx.HTTPError as exc:
            yield sse_event("error", f"Ollama is unreachable or returned an error: {exc}")
            return

        sources = await build_source_infos(rows)
        yield sse_event("done", json.dumps({"sources": [s.model_dump() for s in sources]}))

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.post("/ask/agentic", response_model=AskResponse)
async def ask_agentic(
    request: AskRequest,
    max_iterations: int | None = Query(
        default=None,
        ge=1,
        le=10,
        description="Override the default retrieval-loop cap for this request (1-10).",
    ),
):
    result = await run_agentic_ask(request.question, max_iterations=max_iterations)
    sources = await build_source_infos(result["sources"])
    return AskResponse(answer=result["answer"], sources=sources)


@app.post("/ask/agentic/stream")
async def ask_agentic_stream(
    request: AskRequest,
    max_iterations: int | None = Query(default=None, ge=1, le=10),
):
    async def event_stream():
        try:
            async for event in run_agentic_ask_stream(request.question, max_iterations=max_iterations):
                event_type = event["type"]
                if event_type in ("thinking", "answer"):
                    yield sse_event(event_type, event["text"])
                elif event_type in ("tool_call", "tool_result"):
                    yield sse_event(event_type, json.dumps(event))
                elif event_type == "done":
                    sources = await build_source_infos(event["sources"])
                    yield sse_event("done", json.dumps({"sources": [s.model_dump() for s in sources]}))
        except httpx.HTTPError as exc:
            yield sse_event("error", f"Ollama is unreachable or returned an error: {exc}")

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.get("/documents", response_model=list[DocumentInfo])
async def list_documents():
    async with get_connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                """
                select filename, count(*)
                from documents
                group by filename
                order by filename
                """
            )
            rows = await cur.fetchall()

    return [DocumentInfo(filename=row[0], chunk_count=row[1]) for row in rows]


@app.delete("/documents/{filename}")
async def delete_document(filename: str):
    async with get_connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute("delete from documents where filename = %s", (filename,))
            deleted = cur.rowcount
        await conn.commit()

    if deleted == 0:
        raise HTTPException(status_code=404, detail=f"No document found with filename '{filename}'")

    return {"filename": filename, "chunks_deleted": deleted}
