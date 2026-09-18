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

from app.agent import run_agentic_ask_stream
from app.db import close_pool, delete_document_chunks, get_connection, list_documents as db_list_documents, open_pool
from app.ingestion import ingest_document
from app.parsing import UnsupportedFileType
from app.schemas import AskRequest, DocumentInfo, SourceInfo, UploadResponse
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


@app.post("/ask")
async def ask(
    request: AskRequest,
    max_iterations: int | None = Query(
        default=None,
        ge=1,
        le=10,
        description="Override the default retrieval-loop cap for this request (1-10).",
    ),
):
    """Always agentic, always streamed: the chat model decides when and how
    many times to retrieve (each retrieve call is itself document-routed and
    multi-angle-decomposed - see app/retrieval.py), and can call
    describe_image on a specific figure for a deeper look. SSE events:
    thinking/answer (tokens), tool_call/tool_result (around each retrieve or
    describe_image call), then one final done with {"sources": [...]}."""
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
async def list_documents_route():
    rows = await db_list_documents()
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
