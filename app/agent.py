import asyncio
import json
import logging
import re
import uuid

from app import council
from app.attachments import get_attachment
from app.config import settings
from app.conversations import append_turn, conversation_exists, create_conversation, load_messages, set_title
from app.db import list_documents as db_list_documents
from app.generation import (
    DESCRIBE_IMAGE_TOOL,
    LIST_DOCUMENTS_TOOL,
    RETRIEVE_TOOL,
    WEB_SEARCH_TOOL,
    chat_with_tools,
    count_prompt_tokens,
    count_text_tokens,
    force_tool,
    generate_title,
    stream_chat_with_tools,
)
from app.retrieval import decompose_and_retrieve
from app.storage import get_image_bytes
from app.vision import describe_image as vision_describe_image
from app.web_search import search_web

logger = logging.getLogger(__name__)

NO_RESULTS_MESSAGE = "No results found for this query."

# Room kept free in the context window for the model's own output (its
# thinking plus the answer) when fitting tool results in - see
# _context_room. vLLM rejects any request whose prompt doesn't fit, so tool
# results are trimmed to what's left rather than sent and failed.
RESERVED_OUTPUT_TOKENS = 6000
_SEPARATOR_TOKENS = 3  # "\n---\n" between numbered results
LIST_DOCUMENTS_SAMPLE_LIMIT = 50


def _agent_tools(web_search: bool) -> list[dict]:
    """web_search is opt-in per request (the frontend's toggle) rather than
    always available - keeping it out of the tool list entirely when off
    means the model can't reach for it even if it wanted to, not just that
    it's discouraged from doing so."""
    tools = [RETRIEVE_TOOL, DESCRIBE_IMAGE_TOOL, LIST_DOCUMENTS_TOOL]
    if web_search:
        tools.append(WEB_SEARCH_TOOL)
    return tools


FORCE_WEB_SEARCH = force_tool("web_search")


def _tool_choice(web_search: bool, iteration: int) -> dict | None:
    """With the web_search toggle on, the first model turn is forced to
    call web_search (and _add_companion_retrieve pairs it with a retrieve)
    - left to its own judgment the model would sometimes skip it and answer
    from memory instead, verified live, which isn't what turning the toggle
    on asks for. Every other turn is "auto": the model decides whether it
    needs a tool at all (a greeting doesn't)."""
    return FORCE_WEB_SEARCH if web_search and iteration == 0 else None


def _add_companion_retrieve(message: dict, tool_choice) -> None:
    """When web_search was forced, also run retrieve with the same query
    the model wrote for it - so the organization's own documents are always
    checked alongside the web, without spending another model turn asking
    for it. Added to the message's tool_calls (with its own id) so history
    shows the model a normal two-call turn with two results."""
    if tool_choice != FORCE_WEB_SEARCH:
        return
    companions = []
    for call in message.get("tool_calls") or []:
        query = call["function"].get("arguments", {}).get("query")
        if call["function"]["name"] == "web_search" and isinstance(query, str) and query.strip():
            companions.append({
                "id": f"call_{uuid.uuid4().hex[:24]}",
                "type": "function",
                "function": {"name": "retrieve", "arguments": {"query": query}},
            })
    if companions:
        message["tool_calls"] = message["tool_calls"] + companions


def _augment_with_attachments(question: str, attachment_ids: list[str]) -> str:
    """Folds attached files' extracted text into what the model sees for
    this turn, right after the question itself. Only the plain `question`
    (the caller's original, unmodified) gets persisted via append_turn -
    like retrieved chunks, attachment text is available for reasoning about
    *this* turn but isn't re-shown to the model on a later turn; the
    question text alone (and the answer it produced) is enough history to
    continue the conversation from."""
    if not attachment_ids:
        return question
    parts = [question, "", "--- Attached files ---"]
    for attachment_id in attachment_ids:
        attachment = get_attachment(attachment_id)
        if attachment is None:
            continue
        parts.append(f"\n[{attachment['filename']}]\n{attachment['text']}")
    return "\n".join(parts)


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
    return _rows_for_agent(await decompose_and_retrieve(query, limit))


def _rows_for_agent(rows: list[dict]) -> tuple[list[dict], list[str]]:
    """Retrieval rows -> (source_infos for the caller's `sources`, text for
    the model, image captions tagged with their image_path)."""
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
    "That description is for your own understanding, not for the user - "
    "the people asking you questions are not technical, and terms like "
    "'retrieval-augmented', 'chunk', 'tool call', 'citation', or 'agentic' "
    "mean nothing to them. Never use this kind of implementation jargon in "
    "an answer, including when asked who you are or what you can do - "
    "describe yourself and your capabilities in plain, everyday language "
    "instead (e.g. 'I search through the organization's documents to "
    "answer your question and show you exactly which document it came "
    "from' rather than 'I am a retrieval-augmented assistant that cites "
    "source chunks'). This is only about how you talk about yourself and "
    "your own workings - actual technical content that exists in the "
    "documents themselves (encryption settings, API details, and so on) "
    "should stay precise and unsimplified; don't dumb down the documents' "
    "real content, just don't describe your own machinery in jargon. "
    "You have not searched anything in this turn until you call a tool and "
    "see what it returns - never say you searched, looked, checked the "
    "documents, or found nothing unless a tool call in this turn actually "
    "returned that. Any question asking for a fact, figure, name, date, or "
    "anything about the organization, its documents, or its work needs a "
    "`retrieve` (or `list_documents`) call before you answer, even if you "
    "think you already know, and even if the answer seems obvious - only "
    "small talk (a greeting, thanks) or a question about what you can do "
    "may be answered without a tool. "
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
    "shaped like a count. If `web_search` is available to you, use it only "
    "for things outside the document archive entirely - current events, "
    "today's rates/regulations, or anything the retrieved documents don't "
    "cover - never for questions about the organization's own documents, "
    "even if a web search might turn up something related; retrieve stays "
    "the source of truth for that. "
    "For questions with multiple parts, call `retrieve` once per part (or "
    "with refined queries) rather than relying on a single search. "
    "Some retrieved chunks are pre-computed captions of a chart/figure image, "
    "tagged like [image_path=...]. If such a caption doesn't have the detail "
    "you need (e.g. an exact axis value it summarized loosely), call "
    "`describe_image` with that exact image_path for a fresh, deeper look - "
    "don't call it with a path that wasn't given to you. Before asserting "
    "any specific fact - a number, date, name, or setting - confirm you "
    "can point to a specific retrieved chunk that actually states it; if "
    "you can't, don't assert it. If two or more retrieved chunks "
    "disagree on a fact (e.g. two versions of the same form list "
    "different values), don't silently pick one - say so explicitly and "
    "cite both, e.g. 'One version lists <value A> [4] while a more recent "
    "form lists <value B> [9].' If retrieved chunks come back empty or "
    "clearly irrelevant to the question (e.g. about something else "
    "entirely), that's not evidence the answer doesn't exist - it means "
    "this particular document archive doesn't cover it. When `web_search` "
    "is available to you, try it before giving up in that situation, even "
    "for a question you're not sure is 'the right kind' for it (a person's "
    "name, a current fact, anything the archive plainly has nothing on) - "
    "a wrong guess that it's out of scope costs nothing, an unnecessary "
    "refusal does. Only after retrying retrieve with different queries "
    "(or a deeper image look) and, if available, a web search, still "
    "turns up nothing, say exactly: \"I don't have that in the documents "
    "I can search.\" Don't soften this with a longer explanation or guess "
    "at a partial answer. "
    "Every chunk a `retrieve` call returns, and every result a `web_search` "
    "call returns, is prefixed with its citation number, like "
    "'[7] <chunk text>'. Cite that exact number immediately "
    "after every sentence or claim in your answer that uses it, e.g. "
    "'<Organization> uses <setting> [7].' - a sentence drawing on "
    "more than one chunk gets more than one number, e.g. '...[2][5].' "
    "Cite only chunks you actually used for that specific sentence, never "
    "every chunk you saw, and never invent or renumber - use exactly the "
    "number shown. Citation numbers only exist for results returned in "
    "this turn - never reuse a number from an earlier answer in the "
    "conversation; if you answer without new results, write no [N] "
    "markers at all. This applies per sentence throughout the whole answer, "
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


async def _handle_retrieve_call(call: dict, iteration: int, all_sources: list[dict], room: int) -> str:
    query = call.get("function", {}).get("arguments", {}).get("query")
    if not isinstance(query, str) or not query.strip():
        logger.warning(
            "iteration %d: skipping malformed retrieve call (no usable query): %r",
            iteration, call,
        )
        return "Error: no valid query argument was provided for this call."

    plain_results, tagged_results = await _retrieve_for_agent(query)
    numbered, _ = await _number_into(all_sources, plain_results, tagged_results, room)
    logger.info(
        "iteration %d: retrieve(%r) -> %d result(s), running total %d",
        iteration, query, len(plain_results), len(all_sources),
    )
    return "\n---\n".join(numbered) if numbered else NO_RESULTS_MESSAGE


async def _context_room(messages: list[dict], tools: list[dict] | None) -> int:
    """Tokens still free for tool results in the next request, measured
    exactly by the chat server (the same conversation + tool definitions it
    will be sent), minus RESERVED_OUTPUT_TOKENS."""
    used, limit = await count_prompt_tokens(messages, tools)
    return max(0, limit - RESERVED_OUTPUT_TOKENS - used)


def _source_key(info: dict) -> tuple:
    """What makes two sources the same passage: same web page, or same
    document and same text."""
    if info.get("url"):
        return ("web", info["url"])
    return (info["filename"], " ".join(info["content"].split()))


async def _number_into(
    all_sources: list[dict], infos: list[dict], texts: list[str], room: int,
) -> tuple[list[str], int]:
    """Prefixes `texts` with their citation numbers and appends the matching
    `infos` to `all_sources`, keeping only as many results (whole ones, in
    rank order) as fit in `room` tokens. Returns (numbered texts, tokens
    used). Numbers are 1-based positions in the final `sources` array
    returned to the caller, so a chunk numbered [7] in what the model reads
    is exactly sources[6] in the done event, however many tool calls a turn
    makes - results that don't fit never enter `all_sources`, so that holds
    after trimming too. The model is told how many were left out.

    A result already in `all_sources` (a later retrieve call in the same
    turn finding the same passage again) isn't added a second time - it
    comes back as a one-line pointer to its existing number, so the same
    passage never gets two citation numbers."""
    existing = {_source_key(info): n for n, info in enumerate(all_sources, start=1)}
    numbered: list[str] = []
    new_infos: list[dict | None] = []
    next_index = len(all_sources) + 1
    for info, text in zip(infos, texts):
        key = _source_key(info)
        if key in existing:
            numbered.append(f"[{existing[key]}] (same passage as [{existing[key]}] above)")
            new_infos.append(None)
            continue
        existing[key] = next_index
        numbered.append(f"[{next_index}] {text}")
        new_infos.append(info)
        next_index += 1

    kept: list[str] = []
    used = 0
    for text, tokens in zip(numbered, await count_text_tokens(numbered)):
        if used + tokens + _SEPARATOR_TOKENS > room:
            break
        kept.append(text)
        used += tokens + _SEPARATOR_TOKENS
    all_sources.extend(info for info in new_infos[: len(kept)] if info is not None)
    if len(kept) < len(numbered):
        omitted = len(numbered) - len(kept)
        logger.warning("context full: kept %d of %d results (%d tokens of room)", len(kept), len(numbered), room)
        kept.append(f"({omitted} more results didn't fit in the remaining space and were left out.)")
    return kept, used


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


async def _handle_web_search_call(call: dict, iteration: int, all_sources: list[dict], room: int) -> str:
    query = call.get("function", {}).get("arguments", {}).get("query")
    if not isinstance(query, str) or not query.strip():
        logger.warning(
            "iteration %d: skipping malformed web_search call (no usable query): %r",
            iteration, call,
        )
        return "Error: no valid query argument was provided for this call."

    results = await search_web(query)
    logger.info("iteration %d: web_search(%r) -> %d result(s)", iteration, query, len(results))
    if not results:
        return "No web results found for this query."

    numbered, _ = await _number_into(all_sources, *_web_sources(results), room)
    return "\n---\n".join(numbered)


def _web_sources(results: list[dict]) -> tuple[list[dict], list[str]]:
    """Web results -> (source_infos, text for the model). `url` (not a
    MinIO filename) is what tells build_source_infos to link straight to
    the page instead of trying to generate a presigned document URL."""
    infos = [
        {
            "content": row["content"],
            "source_type": "web",
            "source_format": "web",
            "filename": row["title"] or row["url"],
            "page_number": None,
            "url": row["url"],
        }
        for row in results
    ]
    texts = [f"{row['title']}\n{row['url']}\n{row['content']}" for row in results]
    return infos, texts


TOOL_HANDLERS = {
    "describe_image": _handle_describe_image_call,
    "list_documents": _handle_list_documents_call,
}


async def _dispatch_tool_call(call: dict, iteration: int, all_sources: list[dict], room: int) -> str:
    """`room` is the token space left for this result (_context_room);
    retrieve and web_search trim to it. list_documents and describe_image
    output is small and bounded, and comes out of RESERVED_OUTPUT_TOKENS."""
    name = call.get("function", {}).get("name")
    if name == "web_search":
        return await _handle_web_search_call(call, iteration, all_sources, room)
    handler = TOOL_HANDLERS.get(name)
    if handler is not None:
        return await handler(call, iteration)
    return await _handle_retrieve_call(call, iteration, all_sources, room)


async def _resolve_conversation(
    conversation_id: str | None, parent_message_id: str | None = None
) -> tuple[str, list[dict]]:
    """Returns (conversation_id, prior_messages). Starts a new conversation
    if none was given, or if the given id doesn't exist (a stale/bad id
    from a client shouldn't error the request - it should just start a
    fresh conversation instead). `parent_message_id`, when given, loads
    history only up to that message instead of the conversation's current
    tip - this is what makes branching work: editing or regenerating an
    earlier turn should see the conversation as it stood at that point, not
    pull in sibling turns from a different branch."""
    if conversation_id and await conversation_exists(conversation_id):
        return conversation_id, await load_messages(conversation_id, leaf_message_id=parent_message_id)
    return await create_conversation(), []


async def _maybe_generate_title(conversation_id: str, question: str, is_new: bool) -> str | None:
    """Generates and persists a title once, right after a brand-new
    conversation's first turn. Returns the title so callers can hand it
    back to the client in the same response instead of requiring a
    separate round trip; returns None for every later turn, meaning
    "title unchanged" - the client should keep whatever it already has."""
    if not is_new:
        return None
    title = await generate_title(question)
    await set_title(conversation_id, title)
    return title


async def _council_evidence(
    question: str, history: list[dict], web_search: bool, all_sources: list[dict], room: int,
) -> tuple[dict, list[dict]]:
    """Council mode's search stage (see app/council.py): planners pick the
    angles, every search runs in parallel, and the results come back as one
    already-completed assistant turn - a tool call per kind of search, plus
    its result - so the main model starts with the evidence in its history
    exactly as if it had called the tools itself. Document results get first
    claim on `room` tokens, web results what's left. Returns (assistant
    message in internal shape, tool result messages)."""
    plan = await council.plan(question, history, web_search, settings.council_angles)
    doc_queries = plan.content_queries + plan.figure_queries
    logger.info(
        "council plan: %d content, %d figure, corpus=%r, web=%r",
        len(plan.content_queries), len(plan.figure_queries), plan.corpus_filters, plan.web_queries,
    )

    async def nothing() -> list:
        return []

    listing_calls = [
        {
            "id": f"call_{uuid.uuid4().hex[:24]}",
            "type": "function",
            "function": {"name": "list_documents", "arguments": {"filename_contains": f} if f else {}},
        }
        for f in plan.corpus_filters
    ]
    doc_rows, web_rows, *listings = await asyncio.gather(
        council.search_documents(doc_queries, settings.council_max_chunks) if doc_queries else nothing(),
        council.search_web_many(plan.web_queries) if plan.web_queries else nothing(),
        *(_handle_list_documents_call(call, 0) for call in listing_calls),
    )

    calls: list[dict] = []
    results: list[str] = []
    # Documents numbered first, then web - same order every time.
    if doc_queries:
        numbered, used = await _number_into(all_sources, *_rows_for_agent(doc_rows), room)
        room -= used
        calls.append({
            "id": f"call_{uuid.uuid4().hex[:24]}",
            "type": "function",
            "function": {"name": "retrieve", "arguments": {"query": question, "angles": doc_queries}},
        })
        results.append("\n---\n".join(numbered) if numbered else NO_RESULTS_MESSAGE)
    calls.extend(listing_calls)
    results.extend(listings)
    if plan.web_queries:
        numbered, _ = await _number_into(all_sources, *_web_sources(web_rows), room)
        calls.append({
            "id": f"call_{uuid.uuid4().hex[:24]}",
            "type": "function",
            "function": {"name": "web_search", "arguments": {"query": question, "angles": plan.web_queries}},
        })
        results.append("\n---\n".join(numbered) if numbered else "No web results found for this query.")

    message = {"role": "assistant", "content": "", "tool_calls": calls}
    tool_messages = [
        {"role": "tool", "tool_call_id": call["id"], "content": result} for call, result in zip(calls, results)
    ]
    return message, tool_messages


def _drop_dangling_citations(citations: list[dict], source_count: int) -> list[dict]:
    dangling = sorted({i for seg in citations for i in seg["source_indices"] if not 1 <= i <= source_count})
    if not dangling:
        return citations
    logger.warning("answer cites %s but this turn has only %d source(s); dropped", dangling, source_count)
    return [
        {**seg, "source_indices": [i for i in seg["source_indices"] if 1 <= i <= source_count]}
        for seg in citations
    ]


async def _finish_turn(
    conversation_id: str,
    question: str,
    answer: str,
    parent_message_id: str | None,
    is_new: bool,
    all_sources: list[dict],
    verify: bool,
) -> dict:
    """Persists the turn and builds the done payload shared by both agent
    loops. `citation_warnings` is only populated in council mode."""
    user_id, assistant_id = await append_turn(conversation_id, question, answer, parent_message_id)
    title = await _maybe_generate_title(conversation_id, question, is_new)
    citations = _segment_citations(answer)
    # Council mode's check reports citation numbers with no source as a
    # warning first; then, in every mode, they're dropped so the client
    # never gets a citation pointing at nothing (e.g. a follow-up answered
    # from history reusing an earlier turn's numbers).
    warnings = await council.verify_citations(citations, all_sources) if verify else []
    citations = _drop_dangling_citations(citations, len(all_sources))
    return {
        "sources": all_sources,
        "conversation_id": conversation_id,
        "citations": citations,
        "citation_warnings": warnings,
        "user_message_id": user_id,
        "assistant_message_id": assistant_id,
        "title": title,
    }


async def run_agentic_ask(
    question: str,
    max_iterations: int | None = None,
    conversation_id: str | None = None,
    parent_message_id: str | None = None,
    web_search: bool = False,
    attachment_ids: list[str] | None = None,
    council_mode: bool = False,
) -> dict:
    conversation_id, history = await _resolve_conversation(conversation_id, parent_message_id)
    is_new = not history
    augmented_question = _augment_with_attachments(question, attachment_ids or [])
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT}, *history, {"role": "user", "content": augmented_question}
    ]
    all_sources: list[dict] = []
    iterations = max_iterations if max_iterations is not None else settings.max_agent_iterations
    tools = _agent_tools(web_search)

    if council_mode:
        room = await _context_room(messages, tools)
        evidence, tool_messages = await _council_evidence(question, history, web_search, all_sources, room)
        messages.append(_history_message(evidence))
        messages.extend(tool_messages)

    message: dict = {}
    for iteration in range(iterations):
        # Council mode has already run the web search itself.
        tool_choice = _tool_choice(web_search and not council_mode, iteration)
        message = await chat_with_tools(messages, tools=tools, tool_choice=tool_choice)
        _add_companion_retrieve(message, tool_choice)
        messages.append(_history_message(message))

        if message.get("tool_calls"):
            for call in message["tool_calls"]:
                room = await _context_room(messages, tools)
                tool_content = await _dispatch_tool_call(call, iteration, all_sources, room)
                messages.append({"role": "tool", "tool_call_id": call["id"], "content": tool_content})
        else:
            done = await _finish_turn(
                conversation_id, question, message["content"], parent_message_id, is_new, all_sources, council_mode,
            )
            return {"answer": message["content"], **done}

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
    done = await _finish_turn(conversation_id, question, answer, parent_message_id, is_new, all_sources, council_mode)
    return {"answer": answer, **done}


def _history_message(message: dict) -> dict:
    """The assistant message as it goes back into `messages` for the next
    request - OpenAI format, so tool calls keep their id/type and carry
    arguments re-serialized as a JSON string (generation.py hands them to
    us parsed, which is what the tool handlers want, not the server)."""
    entry = {"role": "assistant", "content": message.get("content") or ""}
    if message.get("tool_calls"):
        entry["tool_calls"] = [
            {
                "id": call["id"],
                "type": "function",
                "function": {
                    "name": call["function"]["name"],
                    "arguments": json.dumps(call["function"].get("arguments") or {}),
                },
            }
            for call in message["tool_calls"]
        ]
    return entry


async def _stream_turn(
    messages: list[dict],
    tools: list[dict] | None = None,
    allow_tools: bool = True,
    tool_choice: dict | None = None,
):
    """Streams one chat turn, yielding {"type": "thinking"|"answer", "text": ...}
    events as tokens arrive, then a final {"type": "_message", "message": {...}}
    carrying the assembled message (appended to `messages` here, in
    OpenAI history format). Tool calls never arrive mid-stream:
    generation.py accumulates the server's tool-call fragments and hands
    them over fully-formed, arguments parsed, only on the final done chunk."""
    content_parts: list[str] = []
    tool_calls = None

    async for chunk in stream_chat_with_tools(messages, tools=tools, allow_tools=allow_tools, tool_choice=tool_choice):
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
    _add_companion_retrieve(message, tool_choice)
    messages.append(_history_message(message))
    yield {"type": "_message", "message": message}


async def run_agentic_ask_stream(
    question: str,
    max_iterations: int | None = None,
    conversation_id: str | None = None,
    parent_message_id: str | None = None,
    web_search: bool = False,
    attachment_ids: list[str] | None = None,
    council_mode: bool = False,
):
    """Streaming counterpart to run_agentic_ask. Yields typed events:
    {"type": "thinking"|"answer", "text": ...} as tokens arrive,
    {"type": "tool_call", "name": ..., "args": {...}} when a tool is invoked,
    {"type": "tool_result", "name": ..., "preview": ...} once it returns,
    {"type": "done", "sources": [...], "conversation_id": ..., "title": ...,
    "user_message_id": ..., "assistant_message_id": ...,
    "citation_warnings": [...]} exactly once at the end - persisting this question and the final answer as a new turn in
    that conversation first.

    `parent_message_id`, when given, forks the conversation from that
    message instead of continuing from its current tip - pass the id of an
    earlier message's parent to edit that turn's question or regenerate its
    answer; the old turn stays in the tree as a sibling, still reachable by
    resending a later ask with the same parent_message_id and the old
    question/answer's ids.

    `attachment_ids` folds those attachments' extracted text into what the
    model sees this turn (see _augment_with_attachments) - persisted
    history still stores just the plain question, not the attached text.

    `council_mode` runs app/council.py's parallel search stage before the
    model's first turn (reported as ordinary tool_call/tool_result events)
    and checks the answer's citations before done."""
    conversation_id, history = await _resolve_conversation(conversation_id, parent_message_id)
    is_new = not history
    augmented_question = _augment_with_attachments(question, attachment_ids or [])
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT}, *history, {"role": "user", "content": augmented_question}
    ]
    all_sources: list[dict] = []
    iterations = max_iterations if max_iterations is not None else settings.max_agent_iterations
    tools = _agent_tools(web_search)

    if council_mode:
        room = await _context_room(messages, tools)
        evidence, tool_messages = await _council_evidence(question, history, web_search, all_sources, room)
        messages.append(_history_message(evidence))
        messages.extend(tool_messages)
        for call, result in zip(evidence["tool_calls"], tool_messages):
            yield {"type": "tool_call", "name": call["function"]["name"], "args": call["function"]["arguments"]}
            yield {"type": "tool_result", "name": call["function"]["name"], "preview": result["content"][:200]}

    for iteration in range(iterations):
        message = None
        # Council mode has already run the web search itself.
        tool_choice = _tool_choice(web_search and not council_mode, iteration)
        async for event in _stream_turn(messages, tools=tools, tool_choice=tool_choice):
            if event["type"] == "_message":
                message = event["message"]
            else:
                yield event

        if message.get("tool_calls"):
            for call in message["tool_calls"]:
                name = call.get("function", {}).get("name")
                args = call.get("function", {}).get("arguments", {})
                yield {"type": "tool_call", "name": name, "args": args}

                room = await _context_room(messages, tools)
                tool_content = await _dispatch_tool_call(call, iteration, all_sources, room)

                messages.append({"role": "tool", "tool_call_id": call["id"], "content": tool_content})
                yield {"type": "tool_result", "name": name, "preview": tool_content[:200]}
        else:
            answer = message.get("content", "")
            done = await _finish_turn(
                conversation_id, question, answer, parent_message_id, is_new, all_sources, council_mode,
            )
            yield {"type": "done", **done}
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
    done = await _finish_turn(
        conversation_id, question, final_answer, parent_message_id, is_new, all_sources, council_mode,
    )
    yield {"type": "done", **done}
