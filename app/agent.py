import logging
import re

from app.config import settings
from app.conversations import append_turn, conversation_exists, create_conversation, load_messages
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

_CITATION_MARKER = re.compile(r"\[(\d+)\]")
# Split each line into sentence-ish units before extracting citations, so a
# multi-sentence paragraph doesn't collapse into one segment carrying every
# citation in it - a lookahead on the next unit starting with a capital
# letter, digit, or markdown bullet keeps this reasonably safe against
# false splits on abbreviations/decimals without needing real NLP.
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?:])\s+(?=[A-Z0-9*])")


def _segment_citations(answer: str) -> list[dict]:
    """Turns the model's raw citation-annotated answer (inline [N] markers,
    per SYSTEM_PROMPT) into a clean, structured breakdown: a list of
    {text, source_indices} segments with the bracket markers stripped out
    of the text entirely - callers get plain prose plus which sources
    (1-based indices into `sources`) back each segment, never raw '[7]'
    syntax to parse themselves. Only meaningful once the full answer is
    known (i.e. at done, not mid-stream) - the raw streamed `answer`
    tokens still carry the markers as the model generates them; this is
    the cleaned-up structural view for after generation completes.

    A blank line in the original (a paragraph break) becomes its own
    empty segment ({"text": "", "source_indices": []}) rather than being
    dropped - a renderer that rejoins segment text with "\n" then
    reconstructs the original paragraph/list structure correctly (a
    single "\n" between real segments is just a soft wrap within one
    block per CommonMark; an empty segment between two real ones produces
    the blank line a markdown renderer needs to start a new paragraph)."""
    if not answer:
        return []
    segments: list[dict] = []
    for line in answer.strip().splitlines():
        line = line.strip()
        if not line:
            if segments and segments[-1]["text"] != "":
                segments.append({"text": "", "source_indices": []})
            continue
        for unit in _SENTENCE_SPLIT.split(line):
            indices = [int(n) for n in _CITATION_MARKER.findall(unit)]
            text = _CITATION_MARKER.sub("", unit).strip()
            # Marker removal can leave a space stranded before trailing
            # punctuation (e.g. "device [1]." -> "device .") - close it up.
            text = re.sub(r"\s+([.!?,:;])", r"\1", text)
            text = re.sub(r"\s{2,}", " ", text)
            if text:
                segments.append({"text": text, "source_indices": indices})
    while segments and segments[-1]["text"] == "":
        segments.pop()
    return segments


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
    "questions about the document corpus itself - how many documents "
    "exist, what documents/files there are, how many match a name/folder "
    "pattern, or 'how many X integrations/instances/forms/reports do we "
    "have' when X is the kind of thing a document is typically named/filed "
    "under (e.g. a VPN setup form per counterparty - call list_documents "
    "with filename_contains='VPN' and count/categorize the filenames it "
    "returns, don't guess from a handful of retrieved chunks). It returns "
    "a count plus filenames, not document content. Use `retrieve` for "
    "questions about content inside the documents - specific configuration "
    "values, what a report says, details only visible in the text itself. "
    "When unsure which fits, prefer list_documents first for anything "
    "shaped like a count. "
    "For questions with multiple parts, call `retrieve` once per part (or "
    "with refined queries) rather than relying on a single search. "
    "Some retrieved chunks are pre-computed captions of a chart/figure image, "
    "tagged like [image_path=...]. If such a caption doesn't have the detail "
    "you need (e.g. an exact axis value it summarized loosely), call "
    "`describe_image` with that exact image_path for a fresh, deeper look - "
    "don't call it with a path that wasn't given to you. If, after retrying "
    "with different queries or a deeper image look, the context still "
    "doesn't contain the answer, say so honestly instead of guessing. "
    "Every chunk a `retrieve` call returns is prefixed with its citation "
    "number, like '[7] <chunk text>'. Cite that exact number immediately "
    "after every sentence or claim in your answer that uses it, e.g. "
    "'BUNNA Bank uses AES-256 encryption [7].' - a sentence drawing on "
    "more than one chunk gets more than one number, e.g. '...[2][5].' "
    "Cite only chunks you actually used for that specific sentence, never "
    "every chunk you saw, and never invent or renumber - use exactly the "
    "number shown. This applies per sentence throughout the whole answer, "
    "not just once at the end. Content from `list_documents` or "
    "`describe_image` has no citation number - don't invent one for it. "
    "Give a complete, detailed answer using everything relevant the "
    "retrieved context actually contains - not just the minimum needed to "
    "address the question. Your reasoning/thinking is a private scratchpad "
    "the user does not read as the answer - any analysis, breakdown, "
    "categorization, or deduplication you work out there must be written "
    "out again in full in the final answer text itself. Never do the real "
    "work in reasoning and then hand back a short conclusion that just "
    "refers to it - the final answer must stand on its own with the same "
    "level of detail. If the context is genuinely thin (e.g. a short "
    "exam-question snippet with no surrounding explanation), say so rather "
    "than padding the answer with invented detail. "
    "Write like a knowledgeable person explaining it, not a templated "
    "report: no bolded section headers (e.g. 'Purpose and Scope:', "
    "'Technical Specifications:'), no labeled sections, nothing that reads "
    "like it was filled into a fixed template. Plain flowing prose - "
    "paragraphs, and a plain list only where a list is genuinely the "
    "clearest way to present it (e.g. enumerating many named entities), "
    "never a list as the whole answer's structure. Never open with a "
    "hedging preamble like 'Based on the available documentation,' "
    "'According to the retrieved context,' or similar - start directly "
    "with the answer itself. "
    "When a question asks for a real-world count (e.g. how many "
    "partners/entities/integrations) and you had to deduplicate "
    "documents to get it, give the final count and the actual analysis "
    "(e.g. the list of distinct entities) directly - don't narrate the "
    "bookkeeping you used to get there (document totals, how many were "
    "templates/duplicates/version updates, arithmetic like '51 minus 1 "
    "is 50'). That process is for your own reasoning only, never the "
    "visible answer. "
    "If the retrieved context supports it, fold 2-3 specific follow-up "
    "questions into the answer's closing - grounded in what's actually in "
    "that content, not generic prompts like 'would you like to know "
    "more?'. Never put them under a heading like 'Suggested follow-up "
    "questions' or set them apart as their own section - a natural closing "
    "sentence or two, same as the rest of the answer. Skip them entirely "
    "if the context was too thin to produce genuinely specific ones, or "
    "the question was simple factual (e.g. a document count) with nothing "
    "meaningful to follow up on."
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
    # Citation numbers are 1-based positions in the final `sources` array
    # returned to the caller - assigned here (before extending) so a chunk
    # numbered [7] in what the model reads is exactly sources[6] in the
    # done event, even when retrieve is called more than once in a turn.
    start_index = len(all_sources) + 1
    all_sources.extend(plain_results)
    logger.info(
        "iteration %d: retrieve(%r) -> %d result(s), running total %d",
        iteration, query, len(plain_results), len(all_sources),
    )
    numbered = [f"[{start_index + i}] {chunk}" for i, chunk in enumerate(tagged_results)]
    return "\n---\n".join(numbered) if numbered else NO_RESULTS_MESSAGE


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


async def _resolve_conversation(conversation_id: str | None) -> tuple[str, list[dict]]:
    """Returns (conversation_id, prior_messages). Starts a new conversation
    if none was given, or if the given id doesn't exist (a stale/bad id
    from a client shouldn't error the request - it should just start a
    fresh conversation instead)."""
    if conversation_id and await conversation_exists(conversation_id):
        return conversation_id, await load_messages(conversation_id)
    return await create_conversation(), []


async def run_agentic_ask(
    question: str, max_iterations: int | None = None, conversation_id: str | None = None,
) -> dict:
    conversation_id, history = await _resolve_conversation(conversation_id)
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, *history, {"role": "user", "content": question}]
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
            await append_turn(conversation_id, question, message["content"])
            return {
                "answer": message["content"],
                "sources": all_sources,
                "conversation_id": conversation_id,
                "citations": _segment_citations(message["content"]),
            }

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

    answer = message.get("content", "")
    await append_turn(conversation_id, question, answer)
    return {
        "answer": answer,
        "sources": all_sources,
        "conversation_id": conversation_id,
        "citations": _segment_citations(answer),
    }


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


async def run_agentic_ask_stream(question: str, max_iterations: int | None = None, conversation_id: str | None = None):
    """Streaming counterpart to run_agentic_ask. Yields typed events:
    {"type": "thinking"|"answer", "text": ...} as tokens arrive,
    {"type": "tool_call", "name": ..., "args": {...}} when a tool is invoked,
    {"type": "tool_result", "name": ..., "preview": ...} once it returns,
    {"type": "done", "sources": [...], "conversation_id": ...} exactly once
    at the end - persisting this question and the final answer as a new
    turn in that conversation first."""
    conversation_id, history = await _resolve_conversation(conversation_id)
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, *history, {"role": "user", "content": question}]
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
            answer = message.get("content", "")
            await append_turn(conversation_id, question, answer)
            yield {
                "type": "done",
                "sources": all_sources,
                "conversation_id": conversation_id,
                "citations": _segment_citations(answer),
            }
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
    final_message: dict = {}
    async for event in _stream_turn(messages, allow_tools=False):
        if event["type"] == "_message":
            final_message = event["message"]
        else:
            yield event

    final_answer = final_message.get("content", "")
    await append_turn(conversation_id, question, final_answer)
    yield {
        "type": "done",
        "sources": all_sources,
        "conversation_id": conversation_id,
        "citations": _segment_citations(final_answer),
    }
