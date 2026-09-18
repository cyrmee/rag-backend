import logging

from app.config import settings
from app.db import list_documents as db_list_documents
from app.generation import (
    DESCRIBE_IMAGE_TOOL,
    LIST_DOCUMENTS_TOOL,
    RETRIEVE_TOOL,
    chat_with_tools,
    stream_chat_with_tools,
)
from app.retrieval import decompose_and_retrieve
from app.storage import get_image_bytes
from app.vision import describe_image as vision_describe_image

logger = logging.getLogger(__name__)

NO_RESULTS_MESSAGE = "No results found for this query."
AGENT_TOOLS = [RETRIEVE_TOOL, DESCRIBE_IMAGE_TOOL, LIST_DOCUMENTS_TOOL]
LIST_DOCUMENTS_SAMPLE_LIMIT = 50


async def retrieve(query: str, top_k: int | None = None) -> list[str]:
    """Same document-routed, multi-angle retrieval as /ask: decompose the
    query into a few distinct angles, rank whole documents before diving
    into chunks, return the content strings of the top matches."""
    limit = top_k if top_k is not None else settings.top_k
    rows = await decompose_and_retrieve(query, limit)
    return [row["content"] for row in rows]


async def _retrieve_for_agent(query: str, top_k: int | None = None) -> tuple[list[dict], list[str]]:
    """Like retrieve(), but also returns filename/source_format/page_number
    metadata for citing sources back to their origin, and tags
    image_caption chunks with their source_image_path so the agent can cite
    it in a follow-up describe_image call (Phase 10). Returns
    (source_infos, tagged_for_model)."""
    limit = top_k if top_k is not None else settings.top_k
    rows = await decompose_and_retrieve(query, limit)

    source_infos = [
        {
            "content": row["content"],
            "source_type": row["source_type"],
            "source_format": row["source_format"],
            "filename": row["filename"],
            "page_number": row["page_number"],
        }
        for row in rows
    ]
    tagged = [
        f"[image_path={row['source_image_path']}] {row['content']}"
        if row["source_type"] == "image_caption" and row["source_image_path"]
        else row["content"]
        for row in rows
    ]
    return source_infos, tagged


SYSTEM_PROMPT = (
    "You are a retrieval-augmented assistant. Answer only using information "
    "returned by your tools - you have no other knowledge of the documents. "
    "Pick the right tool for the question: use `list_documents` for "
    "questions about the document corpus itself (how many documents exist, "
    "what documents/files there are, how many match a name/folder pattern) "
    "- it returns a count plus filenames, not document content. Use "
    "`retrieve` for questions about what's actually inside the documents. "
    "For questions with multiple parts, call `retrieve` once per part (or "
    "with refined queries) rather than relying on a single search. "
    "Some retrieved chunks are pre-computed captions of a chart/figure image, "
    "tagged like [image_path=...]. If such a caption doesn't have the detail "
    "you need (e.g. an exact axis value it summarized loosely), call "
    "`describe_image` with that exact image_path for a fresh, deeper look - "
    "don't call it with a path that wasn't given to you. If, after retrying "
    "with different queries or a deeper image look, the context still "
    "doesn't contain the answer, say so honestly instead of guessing."
)


async def _handle_retrieve_call(call: dict, iteration: int, all_sources: list[dict]) -> str:
    query = call.get("function", {}).get("arguments", {}).get("query")
    if not isinstance(query, str) or not query.strip():
        logger.warning(
            "iteration %d: skipping malformed retrieve call (no usable query): %r",
            iteration, call,
        )
        return "Error: no valid query argument was provided for this call."

    plain_results, tagged_results = await _retrieve_for_agent(query)
    all_sources.extend(plain_results)
    logger.info(
        "iteration %d: retrieve(%r) -> %d result(s), running total %d",
        iteration, query, len(plain_results), len(all_sources),
    )
    return "\n---\n".join(tagged_results) if tagged_results else NO_RESULTS_MESSAGE


async def _handle_describe_image_call(call: dict, iteration: int) -> str:
    image_path = call.get("function", {}).get("arguments", {}).get("image_path")
    if not isinstance(image_path, str) or not image_path.strip():
        logger.warning(
            "iteration %d: skipping malformed describe_image call (no usable image_path): %r",
            iteration, call,
        )
        return "Error: no valid image_path argument was provided for this call."

    try:
        image_bytes = await get_image_bytes(image_path)
    except Exception:
        logger.warning("iteration %d: describe_image object not found: %r", iteration, image_path)
        return f"Error: no image found at path {image_path!r}."

    caption = await vision_describe_image(image_bytes)
    if caption is None:
        logger.warning("iteration %d: describe_image failed for %r", iteration, image_path)
        return f"Error: could not generate a description for {image_path!r} right now."

    logger.info("iteration %d: describe_image(%r) -> fresh caption generated", iteration, image_path)
    return caption


async def _handle_list_documents_call(call: dict, iteration: int) -> str:
    args = call.get("function", {}).get("arguments", {}) or {}
    filename_contains = args.get("filename_contains")
    if filename_contains is not None and not isinstance(filename_contains, str):
        filename_contains = None

    rows = await db_list_documents(filename_contains)
    logger.info(
        "iteration %d: list_documents(filename_contains=%r) -> %d document(s)",
        iteration, filename_contains, len(rows),
    )

    if not rows:
        return "No documents match that filter." if filename_contains else "No documents have been ingested yet."

    header = (
        f"{len(rows)} document(s) match \"{filename_contains}\"."
        if filename_contains else f"{len(rows)} document(s) total."
    )
    sample = rows[:LIST_DOCUMENTS_SAMPLE_LIMIT]
    lines = [header, f"Filenames (showing {len(sample)} of {len(rows)}):"]
    lines.extend(f"- {filename} ({chunk_count} chunks)" for filename, chunk_count in sample)
    return "\n".join(lines)


TOOL_HANDLERS = {
    "describe_image": _handle_describe_image_call,
    "list_documents": _handle_list_documents_call,
}


async def _dispatch_tool_call(call: dict, iteration: int, all_sources: list[dict]) -> str:
    name = call.get("function", {}).get("name")
    handler = TOOL_HANDLERS.get(name)
    if handler is not None:
        return await handler(call, iteration)
    return await _handle_retrieve_call(call, iteration, all_sources)


async def run_agentic_ask(question: str, max_iterations: int | None = None) -> dict:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    all_sources: list[dict] = []
    iterations = max_iterations if max_iterations is not None else settings.max_agent_iterations

    message: dict = {}
    for iteration in range(iterations):
        message = await chat_with_tools(messages, tools=AGENT_TOOLS)
        messages.append(message)

        if message.get("tool_calls"):
            for call in message["tool_calls"]:
                tool_content = await _dispatch_tool_call(call, iteration, all_sources)
                messages.append({"role": "tool", "content": tool_content})
        else:
            return {"answer": message["content"], "sources": all_sources}

    # Ran out of iterations. If the last turn was a tool call, its message
    # has no answer content — force one final, tool-less turn so the model
    # summarizes whatever it already retrieved instead of returning empty.
    if message.get("tool_calls"):
        messages.append({
            "role": "user",
            "content": (
                "You're out of retrieval attempts. Answer now using only "
                "what you've already retrieved above."
            ),
        })
        message = await chat_with_tools(messages, allow_tools=False)

    return {"answer": message.get("content", ""), "sources": all_sources}


async def _stream_turn(messages: list[dict], tools: list[dict] | None = None, allow_tools: bool = True):
    """Streams one chat turn, yielding {"type": "thinking"|"answer", "text": ...}
    events as tokens arrive, then a final {"type": "_message", "message": {...}}
    carrying the assembled message (appended to `messages` here). Content
    stays empty while a tool_calls chunk is being formed, so no bogus
    "answer" events fire during tool-call turns - confirmed against the
    live model before relying on it here."""
    content_parts: list[str] = []
    tool_calls = None

    async for chunk in stream_chat_with_tools(messages, tools=tools, allow_tools=allow_tools):
        msg = chunk.get("message", {})
        if msg.get("thinking"):
            yield {"type": "thinking", "text": msg["thinking"]}
        if msg.get("content"):
            content_parts.append(msg["content"])
            yield {"type": "answer", "text": msg["content"]}
        if msg.get("tool_calls"):
            tool_calls = msg["tool_calls"]
        if chunk.get("done"):
            break

    message = {"role": "assistant", "content": "".join(content_parts), "tool_calls": tool_calls}
    messages.append(message)
    yield {"type": "_message", "message": message}


async def run_agentic_ask_stream(question: str, max_iterations: int | None = None):
    """Streaming counterpart to run_agentic_ask. Yields typed events:
    {"type": "thinking"|"answer", "text": ...} as tokens arrive,
    {"type": "tool_call", "name": ..., "args": {...}} when a tool is invoked,
    {"type": "tool_result", "name": ..., "preview": ...} once it returns,
    {"type": "done", "sources": [...]} exactly once at the end."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    all_sources: list[dict] = []
    iterations = max_iterations if max_iterations is not None else settings.max_agent_iterations

    for iteration in range(iterations):
        message = None
        async for event in _stream_turn(messages, tools=AGENT_TOOLS):
            if event["type"] == "_message":
                message = event["message"]
            else:
                yield event

        if message.get("tool_calls"):
            for call in message["tool_calls"]:
                name = call.get("function", {}).get("name")
                args = call.get("function", {}).get("arguments", {})
                yield {"type": "tool_call", "name": name, "args": args}

                tool_content = await _dispatch_tool_call(call, iteration, all_sources)

                messages.append({"role": "tool", "content": tool_content})
                yield {"type": "tool_result", "name": name, "preview": tool_content[:200]}
        else:
            yield {"type": "done", "sources": all_sources}
            return

    # Ran out of iterations — force one final, tool-less streamed turn so
    # the model summarizes whatever it already retrieved.
    messages.append({
        "role": "user",
        "content": (
            "You're out of retrieval attempts. Answer now using only "
            "what you've already retrieved above."
        ),
    })
    async for event in _stream_turn(messages, allow_tools=False):
        if event["type"] != "_message":
            yield event

    yield {"type": "done", "sources": all_sources}
