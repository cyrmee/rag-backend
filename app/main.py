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
from app.attachments import (
    MAX_MESSAGE_TOKENS,
    AttachmentTooLarge,
    extract_attachment_text,
    store_attachment,
    total_tokens,
)
from app.config import settings
from app.conversations import delete_conversation, get_conversation_meta, list_conversations, load_tree
from app.db import close_pool, get_connection, list_documents as db_list_documents, open_pool
from app.generation import count_text_tokens
from app.ingestion import ingest_document
from app.parsing import DuplicateDocument, UnsupportedFileType
from app.schemas import (
    AskRequest,
    AttachmentInfo,
    ConversationDetail,
    ConversationMessageNode,
    ConversationSummary,
    DocumentInfo,
    SourceInfo,
    UploadResponse,
)
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


def _backend_name(exc: httpx.HTTPError) -> str:
    """Chat runs on its own server while embeddings/vision stay on Ollama,
    so name whichever one the failed request was actually aimed at."""
    try:
        url = str(exc.request.url)
    except RuntimeError:
        return "Ollama"
    return "chat model server" if url.startswith(settings.chat_base_url) else "Ollama"


@app.exception_handler(httpx.HTTPError)
async def model_server_unreachable_handler(request: Request, exc: httpx.HTTPError):
    return JSONResponse(
        status_code=503,
        content={"detail": f"{_backend_name(exc)} is unreachable or returned an error: {exc}"},
    )


def sse_event(event: str, data: str) -> str:
    """Formats one Server-Sent Event. `data` is split on newlines since
    each line of an SSE data field needs its own "data: " prefix."""
    data_lines = "\n".join(f"data: {line}" for line in data.split("\n"))
    return f"event: {event}\n{data_lines}\n\n"


async def build_source_infos(rows: list[dict]) -> list[SourceInfo]:
    """Turns raw (content, source_type, source_format, filename, page_number)
    dicts into SourceInfo objects, generating one presigned document link per
    unique filename (not per row) to avoid redundant MinIO round-trips.
    A web_search result (source_type "web") already carries its own real
    `url` - it's not a MinIO object, so it skips the document-link lookup
    entirely and uses that url as-is."""
    url_cache: dict[str, str | None] = {}
    infos = []
    for row in rows:
        filename = row["filename"]
        if row["source_type"] == "web":
            document_url = row.get("url")
        else:
            if filename not in url_cache:
                url_cache[filename] = await get_document_url(filename)
            document_url = url_cache[filename]
        infos.append(SourceInfo(
            content=row["content"],
            filename=filename,
            source_type=row["source_type"],
            source_format=row["source_format"],
            page_number=row.get("page_number"),
            document_url=document_url,
        ))
    return infos


@app.get("/health")
async def health():
    """Liveness for the container healthcheck: the process is up and the
    database answers. Model servers aren't checked - /ask reports those
    itself (503 or an error event) when they're down."""
    async with get_connection() as conn:
        await conn.execute("select 1")
    return {"status": "ok"}


@app.post("/upload", response_model=UploadResponse)
async def upload(file: UploadFile):
    suffix = Path(file.filename or "").suffix
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        chunk_count = await ingest_document(tmp_path, file.filename or "unknown", file.content_type)
    except UnsupportedFileType as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except DuplicateDocument as exc:
        raise HTTPException(
            status_code=409,
            detail={"message": str(exc), "existing_filename": exc.existing_filename},
        )
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    return UploadResponse(filename=file.filename or "unknown", chunks_ingested=chunk_count)


@app.post("/attachments", response_model=AttachmentInfo)
async def upload_attachment(file: UploadFile):
    """Extracts plain text from a file to attach to a chat message - not
    added to the searchable document corpus (see /upload for that), just
    held in memory until referenced by AskRequest.attachment_ids. The
    combined-attachments budget (MAX_MESSAGE_TOKENS) is enforced at ask
    time, not here, since it's a limit on one message's total attached
    content, not on any single file."""
    file_bytes = await file.read()
    try:
        text = await extract_attachment_text(file.filename or "unknown", file_bytes, file.content_type)
    except UnsupportedFileType as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except AttachmentTooLarge as exc:
        raise HTTPException(status_code=413, detail=str(exc))

    [token_count] = await count_text_tokens([text])
    return AttachmentInfo(**store_attachment(file.filename or "unknown", text, token_count))


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
    thinking/answer (tokens - the raw answer text may transiently contain
    inline [N] citation markers as the model generates them, before the
    structured view below is available), tool_call/tool_result (around
    each retrieve or describe_image call), then one final done with
    {"sources": [...], "conversation_id": ..., "citations": [...],
    "user_message_id": ..., "assistant_message_id": ..., "title": ...}.
    `citations` is the clean, structured citation breakdown - a list of
    {text, source_indices} segments (source_indices are 1-based positions
    into `sources`) with the [N] markers already stripped out, so callers
    never need to parse citation syntax out of prose themselves. `title` is
    only set (non-null) the first time a brand-new conversation completes -
    a short, generated label for it; every later turn returns null, meaning
    "unchanged".

    Pass conversation_id (from a prior response's done event) to continue
    that conversation - the model sees the prior turns as history. Omit it
    (or pass a stale/unknown one) to start a new conversation; its id comes
    back in the done event either way. Pass parent_message_id to edit an
    earlier question or regenerate an earlier answer instead of continuing
    from the conversation's current tip - see AskRequest."""
    [question_tokens] = await count_text_tokens([request.question])
    message_tokens = question_tokens + total_tokens(request.attachment_ids)
    if message_tokens > MAX_MESSAGE_TOKENS:
        if request.attachment_ids:
            what, fix = "Your message and attached files are", "Shorten it, remove a file, or split them"
        else:
            what, fix = "Your message is", "Shorten it or split it"
        raise HTTPException(
            status_code=413,
            detail=(
                f"{what} too long - {message_tokens / MAX_MESSAGE_TOKENS:.0%} of what "
                f"fits in one message. {fix} across separate questions."
            ),
        )

    async def event_stream():
        try:
            async for event in run_agentic_ask_stream(
                request.question,
                max_iterations=max_iterations,
                conversation_id=request.conversation_id,
                parent_message_id=request.parent_message_id,
                web_search=request.web_search,
                attachment_ids=request.attachment_ids,
                council_mode=request.council,
            ):
                event_type = event["type"]
                if event_type in ("thinking", "answer"):
                    yield sse_event(event_type, event["text"])
                elif event_type in ("tool_call", "tool_result"):
                    yield sse_event(event_type, json.dumps(event))
                elif event_type == "done":
                    sources = await build_source_infos(event["sources"])
                    yield sse_event("done", json.dumps({
                        "sources": [s.model_dump() for s in sources],
                        "conversation_id": event["conversation_id"],
                        "citations": event["citations"],
                        "user_message_id": event["user_message_id"],
                        "assistant_message_id": event["assistant_message_id"],
                        "title": event["title"],
                        "citation_warnings": event["citation_warnings"],
                    }))
        except httpx.HTTPError as exc:
            yield sse_event("error", f"{_backend_name(exc)} is unreachable or returned an error: {exc}")

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.get("/conversations", response_model=list[ConversationSummary])
async def list_conversations_route():
    rows = await list_conversations()
    return [ConversationSummary(**row) for row in rows]


@app.get("/conversations/{conversation_id}", response_model=ConversationDetail)
async def get_conversation(conversation_id: str):
    """Every message in every branch, not just the currently active path -
    see ConversationDetail. A client walks parent_message_id from
    active_message_id to render the current transcript, and can use
    whatever else is in `messages` to offer switching to a sibling
    branch."""
    meta = await get_conversation_meta(conversation_id)
    if meta is None:
        raise HTTPException(status_code=404, detail=f"No conversation found with id '{conversation_id}'")
    tree = await load_tree(conversation_id)
    return ConversationDetail(
        id=conversation_id,
        title=meta["title"],
        active_message_id=meta["active_message_id"],
        messages=[ConversationMessageNode(**m) for m in tree],
    )


@app.delete("/conversations/{conversation_id}")
async def delete_conversation_route(conversation_id: str):
    deleted = await delete_conversation(conversation_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"No conversation found with id '{conversation_id}'")
    return {"id": conversation_id}


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
            # Its summary too - a leftover row would keep document-routed
            # search steering queries toward a file with no chunks left.
            await cur.execute("delete from document_summaries where filename = %s", (filename,))
            # And its hash, so the same file can be uploaded again later.
            await cur.execute("delete from document_files where filename = %s", (filename,))
        await conn.commit()

    if deleted == 0:
        raise HTTPException(status_code=404, detail=f"No document found with filename '{filename}'")

    return {"filename": filename, "chunks_deleted": deleted}
