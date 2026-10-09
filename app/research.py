"""Research mode for /ask (mode="research"): the slow, thorough path for
complex stakeholder questions. Council mode plans once, searches once and
answers from the best ~20 chunks in a single prompt; research mode reads
far more than one prompt holds by working in stages, each a cheap model
call, and only the compact result of the reading reaches the answer prompt:

  plan    - a thinking-on planner turns the question into sub-questions and
            picks the relevant documents from every document's summary (the
            whole corpus at document level fits in one prompt)
  gather  - every sub-question is searched corpus-wide and inside each
            picked document; picked documents small enough are read whole
  read    - the excerpts are packed into batches of ~RESEARCH_BATCH_TOKENS
            and read in parallel, thinking off, into one-line notes that
            each point at the excerpt they came from
  gaps    - the notes are checked against the sub-questions; whatever is
            still unanswered becomes new searches for the next round
  write   - (app/agent.py) the model answers from the notes, citing the
            excerpts behind them, so the citation check reads the real
            source text rather than the notes

Hundreds of excerpts' worth of evidence fits in the answer prompt this way,
where council mode fits twenty chunks."""

import asyncio
import logging
import re
from dataclasses import dataclass, field

from app.config import settings
from app.council import search_web_many
from app.db import get_connection
from app.embeddings import embed_text, embed_texts
from app.generation import chat_with_tools, count_text_tokens, generate_answer
from app.retrieval import (
    CONTEXT_WINDOW,
    RRF_K,
    document_routed_search,
    expand_context,
    search_within_document,
    summary_ranked_filenames,
)

logger = logging.getLogger(__name__)

MIN_SUB_QUESTIONS = 6
MAX_SUB_QUESTIONS = 12
MAX_WEB_QUERIES = 4
WEB_RESULT_LIMIT = 12

# The planner's document list: every summary, most relevant first, cut to
# this many tokens if the corpus outgrows it (each summary trimmed first).
PLAN_SUMMARY_TOKENS = 16000
SUMMARY_CHARS = 400

# Corpus-wide search per sub-question: documents considered and chunks per
# document - wider than the agent's 3x6, since nothing here goes straight
# into a prompt.
TOP_DOCUMENTS = 8
PER_DOCUMENT = 6
# Inside each planner-picked document, per sub-question. Those lists are
# fused at a lower weight: a chunk that tops a search within one small
# document shouldn't outrank the corpus-wide winner on its own.
PICKED_PER_DOCUMENT = 3
PICKED_WEIGHT = 0.5
# Picked documents up to this size are read in full instead of searched,
# WHOLE_DOCUMENT_GROUP consecutive chunks per excerpt.
WHOLE_DOCUMENT_CHARS = 16000
WHOLE_DOCUMENT_GROUP = 3
EXCERPT_MAX_CHARS = 6000

MAX_NOTES_PER_BATCH = 25
READ_CONCURRENCY = 4
CHECK_CONCURRENCY = 4
MAX_GAP_QUERIES = 6

# Reply caps (see generation._chat_payload): the planner thinks before its
# ~400-token plan; a batch's notes are at most MAX_NOTES_PER_BATCH lines;
# a note check is a list of numbers; the gap check is a verdict a line.
PLAN_MAX_TOKENS = 6000
READ_MAX_TOKENS = 1500
CHECK_MAX_TOKENS = 60
GAPS_MAX_TOKENS = 1000

HISTORY_MESSAGES = 4
HISTORY_CHARS = 500

_NONE = "NONE"
_BULLET = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s*")
_BULLET_ONLY = re.compile(r"^\s*[-*•]\s*")
_NOTE_LINE = re.compile(r"^\s*(?:[-*•]|\d+[.)])?\s*\[#(\d+)\]\s*(.+?)\s*$")
# A "note" that records an absence isn't a fact from the excerpt.
_META_NOTE = re.compile(
    r"^(no|nothing|none)\b|\b(not|n't|no)\s+(relevant|mention|state|specif|provid|includ|contain|address|discuss|"
    r"indicat|give|available|found)",
    re.IGNORECASE,
)
_VERDICT = re.compile(
    r"^\s*\**\s*(?:Q|#|sub-question\s*)?(\d+)\**\W+\**\s*(ANSWERED|PARTIAL|MISSING)\b", re.IGNORECASE,
)


@dataclass
class Excerpt:
    """One passage the reader sees: a retrieved chunk (with its neighbours),
    a slice of a whole-read document, or a web result. `info` is what the
    done event's `sources` gets if a note cites it."""

    key: tuple
    filename: str
    label: str
    content: str
    info: dict
    position: int


@dataclass
class Note:
    claim: str
    excerpt: Excerpt


@dataclass
class ResearchPlan:
    sub_questions: list[str]
    documents: list[str]
    web_queries: list[str]


@dataclass
class Coverage:
    """What earlier rounds already read, so a later round only reads new
    text: chunk indexes per file, files read whole, web pages."""

    chunks: dict[str, set[int]] = field(default_factory=dict)
    whole: set[str] = field(default_factory=set)
    urls: set[str] = field(default_factory=set)

    def covers(self, filename: str, chunk_index: int) -> bool:
        return filename in self.whole or chunk_index in self.chunks.get(filename, ())

    def add(self, filename: str, chunk_index: int, source_type: str) -> None:
        indexes = self.chunks.setdefault(filename, set())
        # A text chunk is read with its neighbours (expand_context), so
        # those count as read too.
        span = CONTEXT_WINDOW if source_type == "text" else 0
        indexes.update(range(chunk_index - span, chunk_index + span + 1))


WRITER_SYSTEM_PROMPT = (
    "You are answering a question for a non-technical reader at an "
    "organization, using research notes gathered from its documents (and, "
    "if any are included, from web pages). The notes are everything you "
    "know - add nothing from memory. Write a thorough, detailed, long "
    "answer: cover every part of the question and every sub-question the "
    "research set out to answer, give exact figures, dates, names, amounts "
    "and conditions as they appear in the notes, and explain the context "
    "and reasons where the notes give them. Organize it with short "
    "headings or paragraphs per part of the question; use lists or a "
    "table where they make the detail clearer. Every sentence that states "
    "a fact must end with the number(s) of the note(s) it comes from, in "
    "square brackets like [3] or [3][7], exactly as numbered in the notes. "
    "Where notes from different documents disagree on a fact, don't pick "
    "one silently - say so and cite both, e.g. 'One form lists <A> [4] "
    "while a later version lists <B> [9].' Where the notes are silent on a "
    "part of the question, say plainly that the documents don't cover it - "
    "never fill the gap with a guess, and don't speculate about reasons or "
    "motives the notes don't give (no 'likely', 'probably', 'may have been "
    "chosen to'). Don't add a section describing the documents or their "
    "pages as a list of sources - the citations do that. Refer to documents "
    "by name when that helps the reader, but don't describe your own "
    "workings: no mention of notes, excerpts, searches, retrieval or tools - "
    "to the reader, you read the organization's documents."
)


def _context_block(question: str, history: list[dict]) -> str:
    lines = []
    for msg in history[-HISTORY_MESSAGES:]:
        content = msg["content"].strip()
        if len(content) > HISTORY_CHARS:
            content = content[:HISTORY_CHARS] + "..."
        lines.append(f"{msg['role']}: {content}")
    convo = "Conversation so far:\n" + "\n".join(lines) + "\n\n" if lines else ""
    return f"{convo}Current question: {question}"


def _parse_sections(answer: str, names: list[str], keep_numbering: tuple[str, ...] = ()) -> dict[str, list[str]]:
    """Items under `NAME:` headers, one per line, bullets/numbering
    stripped, NONE dropped. Text before the first header is ignored. In
    the `keep_numbering` sections only bullets are stripped - a leading
    "3." there is the item's own number (a verdict's sub-question), not
    list numbering."""
    sections: dict[str, list[str]] = {name: [] for name in names}
    header = re.compile(rf"^\s*(?:[-*#]+\s*)?({'|'.join(re.escape(n) for n in names)})\s*:\s*(.*)$", re.IGNORECASE)
    current: str | None = None
    for line in answer.splitlines():
        match = header.match(line)
        if match:
            current = match.group(1).upper()
            line = match.group(2)
        if current is None:
            continue
        strip = _BULLET_ONLY if current in keep_numbering else _BULLET
        item = strip.sub("", line).strip().strip("\"'`").strip()
        if not item or item.upper().rstrip(".") == _NONE:
            continue
        if item.lower() not in (i.lower() for i in sections[current]):
            sections[current].append(item)
    return sections


async def _load_summaries() -> dict[str, str]:
    async with get_connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute("select filename, summary from document_summaries")
            return {filename: summary for filename, summary in await cur.fetchall()}


async def plan(question: str, history: list[dict], web_search: bool) -> ResearchPlan:
    """One thinking-on call that sees the question and every document's
    summary (most relevant first) and replies with the sub-questions to
    research and the documents to read. Falls back to the bare question
    and no picks if the call fails - the corpus-wide searches still run."""
    summaries = await _load_summaries()
    ranked = await summary_ranked_filenames(await embed_text(question), len(summaries)) if summaries else []
    ordered = [f for f in ranked if f in summaries] + [f for f in summaries if f not in ranked]
    lines = [f"- {f}: {' '.join(summaries[f].split())[:SUMMARY_CHARS]}" for f in ordered]
    kept: list[str] = []
    used = 0
    for line, tokens in zip(lines, await count_text_tokens(lines)):
        if used + tokens > PLAN_SUMMARY_TOKENS:
            break
        kept.append(line)
        used += tokens
    if len(kept) < len(lines):
        logger.info("research planner sees %d of %d document summaries", len(kept), len(lines))

    web_step = (
        f"\n3. Write up to {MAX_WEB_QUERIES} public web search queries for anything the question "
        "needs from outside the archive (public facts, regulations, background on named "
        "organizations). NONE if nothing.\n"
        if web_search else ""
    )
    prompt = (
        f"{_context_block(question, history)}\n\n"
        "You are planning research into an organization's internal document archive to answer "
        "the current question thoroughly.\n\n"
        "1. Break the question into the sub-questions a careful researcher would need answered to "
        "cover it completely - every part of the question, every named item, and the background "
        "facts, dates, amounts, parties and conditions involved. Write between "
        f"{MIN_SUB_QUESTIONS} and {MAX_SUB_QUESTIONS} sub-questions, each a standalone search "
        "query using the specific names, numbers and terms involved (resolve 'it', 'that year' "
        "and the like from the conversation). Every sub-question must be needed to answer the "
        "question as asked - don't add topics it doesn't ask about.\n"
        "2. From the document list below, pick the documents most likely to hold the answers - "
        f"up to {settings.research_max_documents}, most relevant first. Copy each filename "
        "exactly as listed. NONE if none look relevant.\n"
        f"{web_step}\n"
        "Documents (filename: what it is about):\n"
        f"{chr(10).join(kept) or '(no document summaries available)'}\n\n"
        "Reply with ONLY this format, nothing before or after it:\n"
        "SUB-QUESTIONS:\n<one per line>\n"
        "DOCUMENTS:\n<one filename per line, or NONE>\n"
        f"{'WEB:' + chr(10) + '<one query per line, or NONE>' if web_search else ''}"
    )
    try:
        message = await chat_with_tools(
            [{"role": "user", "content": prompt}], allow_tools=False, max_tokens=PLAN_MAX_TOKENS,
        )
        sections = _parse_sections(message["content"], ["SUB-QUESTIONS", "DOCUMENTS", "WEB"])
    except Exception:
        logger.warning("research planner failed; researching the question as asked", exc_info=True)
        sections = {"SUB-QUESTIONS": [], "DOCUMENTS": [], "WEB": []}

    sub_questions = sections["SUB-QUESTIONS"][:MAX_SUB_QUESTIONS]
    # The question verbatim stays one of the searches (as in council mode).
    if question.lower() not in (q.lower() for q in sub_questions):
        sub_questions = [question, *sub_questions]

    known = {f.lower(): f for f in summaries}
    documents: list[str] = []
    for item in sections["DOCUMENTS"]:
        name = item.lower()
        match = known.get(name) or next((f for low, f in known.items() if low.endswith(name) or name in low), None)
        if match and match not in documents:
            documents.append(match)
    documents = documents[: settings.research_max_documents]
    web_queries = sections["WEB"][:MAX_WEB_QUERIES] if web_search else []
    return ResearchPlan(sub_questions, documents, web_queries)


async def _document_sizes(filenames: list[str]) -> dict[str, int]:
    if not filenames:
        return {}
    async with get_connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                """
                select filename, sum(length(content))
                from documents
                where source_type = 'text' and filename = any(%s)
                group by filename
                """,
                (filenames,),
            )
            return {filename: int(size) for filename, size in await cur.fetchall()}


def _label(row: dict) -> str:
    page = f", page {row['page_number']}" if row.get("page_number") is not None else ""
    if row["source_type"] == "image_caption":
        return f"description of a figure{page}"
    if row["source_type"] == "chart_data":
        return f"data behind a chart{page}"
    return page[2:] if page else "text"


def _excerpt(row: dict, position: int) -> Excerpt:
    content = row["content"][:EXCERPT_MAX_CHARS]
    return Excerpt(
        key=(row["filename"], row["chunk_index"]),
        filename=row["filename"],
        label=_label(row),
        content=content,
        info={
            "content": content,
            "source_type": row["source_type"],
            "source_format": row["source_format"],
            "filename": row["filename"],
            "page_number": row.get("page_number"),
        },
        position=position,
    )


async def _read_whole(filename: str) -> list[Excerpt]:
    """Every text chunk of one document, in order, WHOLE_DOCUMENT_GROUP
    chunks to an excerpt so a citation points at a paragraph or two."""
    async with get_connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                """
                select chunk_index, content, page_number, source_format
                from documents
                where filename = %s and source_type = 'text'
                order by chunk_index
                """,
                (filename,),
            )
            rows = await cur.fetchall()
    excerpts: list[Excerpt] = []
    for start in range(0, len(rows), WHOLE_DOCUMENT_GROUP):
        group = rows[start : start + WHOLE_DOCUMENT_GROUP]
        chunk_index, _, page_number, source_format = group[0]
        row = {
            "filename": filename,
            "chunk_index": chunk_index,
            "content": "\n".join(content for _, content, _, _ in group),
            "page_number": page_number,
            "source_format": source_format,
            "source_type": "text",
        }
        excerpts.append(_excerpt(row, chunk_index))
    return excerpts


async def gather(queries: list[str], picked: list[str], coverage: Coverage, budget: int) -> list[Excerpt]:
    """One round of searching: every query corpus-wide (TOP_DOCUMENTS x
    PER_DOCUMENT) and inside every picked document that isn't read whole,
    all in parallel, fused by reciprocal rank and cut to `budget` chunks;
    plus the full text of picked documents up to WHOLE_DOCUMENT_CHARS that
    no earlier round read. Text already covered is skipped. Picked
    documents' excerpts come first, then documents by fused score."""
    vectors = await embed_texts(queries)
    sizes = await _document_sizes(picked)
    whole = [f for f in picked if f not in coverage.whole and sizes.get(f, 0) <= WHOLE_DOCUMENT_CHARS]
    searched = [f for f in picked if f not in whole and f not in coverage.whole]

    tasks = [
        document_routed_search(v, q, settings.research_chunks_per_query, TOP_DOCUMENTS, PER_DOCUMENT)
        for q, v in zip(queries, vectors)
    ]
    weights = [1.0] * len(tasks)
    for q, v in zip(queries, vectors):
        for filename in searched:
            tasks.append(search_within_document(v, q, filename, PICKED_PER_DOCUMENT))
            weights.append(PICKED_WEIGHT)
    outcomes = await asyncio.gather(*tasks, return_exceptions=True)
    failures = [o for o in outcomes if isinstance(o, BaseException)]
    if failures:
        logger.warning("%d of %d research searches failed: %r", len(failures), len(outcomes), failures[0])
        if len(failures) == len(outcomes) and not whole:
            raise failures[0]

    scores: dict[tuple[str, int], float] = {}
    rows: dict[tuple[str, int], dict] = {}
    for outcome, weight in zip(outcomes, weights):
        if isinstance(outcome, BaseException):
            continue
        for rank, row in enumerate(outcome, start=1):
            if row["filename"] in whole or coverage.covers(row["filename"], row["chunk_index"]):
                continue
            key = (row["filename"], row["chunk_index"])
            scores[key] = scores.get(key, 0.0) + weight / (RRF_K + rank)
            rows.setdefault(key, row)
    ranked = sorted(scores, key=scores.get, reverse=True)[:budget]

    doc_order: dict[str, int] = {f: i for i, f in enumerate(whole)}
    doc_scores: dict[str, float] = {}
    for key in ranked:
        doc_scores[key[0]] = doc_scores.get(key[0], 0.0) + scores[key]
    for filename in sorted(doc_scores, key=doc_scores.get, reverse=True):
        doc_order.setdefault(filename, len(doc_order))

    excerpts: list[Excerpt] = []
    for filename in whole:
        excerpts.extend(await _read_whole(filename))
        coverage.whole.add(filename)
    for key in ranked:
        coverage.add(key[0], key[1], rows[key]["source_type"])
    for row in await expand_context([rows[key] for key in ranked]):
        excerpts.append(_excerpt(row, row["chunk_index"]))
    excerpts.sort(key=lambda e: (doc_order[e.filename], e.position))
    logger.info(
        "research gather: %d queries, %d picked documents (%d read whole) -> %d excerpts",
        len(queries), len(picked), len(whole), len(excerpts),
    )
    return excerpts


async def gather_web(queries: list[str], coverage: Coverage) -> list[Excerpt]:
    excerpts: list[Excerpt] = []
    for i, row in enumerate(await search_web_many(queries, WEB_RESULT_LIMIT)):
        if row["url"] in coverage.urls:
            continue
        coverage.urls.add(row["url"])
        content = row["content"][:EXCERPT_MAX_CHARS]
        excerpts.append(Excerpt(
            key=("web", row["url"]),
            filename=row["title"] or row["url"],
            label=f"web page, {row['url']}",
            content=content,
            info={
                "content": content,
                "source_type": "web",
                "source_format": "web",
                "filename": row["title"] or row["url"],
                "page_number": None,
                "url": row["url"],
            },
            position=i,
        ))
    return excerpts


async def build_batches(excerpts: list[Excerpt], batch_tokens: int) -> list[list[Excerpt]]:
    """Packs excerpts, in order, into batches of at most `batch_tokens`
    (an excerpt bigger than that gets a batch of its own)."""
    batches: list[list[Excerpt]] = []
    current: list[Excerpt] = []
    used = 0
    for excerpt, tokens in zip(excerpts, await count_text_tokens([e.content for e in excerpts])):
        tokens += 20  # the [#n] filename (label) header
        if current and used + tokens > batch_tokens:
            batches.append(current)
            current, used = [], 0
        current.append(excerpt)
        used += tokens
    if current:
        batches.append(current)
    return batches


async def read_batch(batch: list[Excerpt], sub_questions: list[str], semaphore: asyncio.Semaphore) -> list[Note]:
    """One thinking-off read of one batch into notes. A failed call is
    logged and contributes nothing rather than failing the research."""
    questions = "\n".join(f"{i}. {q}" for i, q in enumerate(sub_questions, start=1))
    excerpts = "\n\n".join(f"[#{i}] {e.filename} ({e.label})\n{e.content}" for i, e in enumerate(batch, start=1))
    prompt = (
        f"Research questions:\n{questions}\n\n"
        "Below are numbered excerpts from an organization's documents. Write down every fact in "
        "them that helps answer any of the research questions - figures, names, dates, amounts, "
        "parties, conditions, obligations, decisions and the reasons given. Copy numbers, units, "
        "currencies and dates exactly as written in the excerpt. Each note is one self-contained "
        "sentence that makes sense on its own (name the subject: 'TECH5's 2022 net revenue was "
        "USD 6,320,815', not 'it was 6,320,815') and starts with the number of the excerpt it "
        "comes from, like: [#3] <fact>. A note may only state what its own excerpt states - never "
        "write that something is not mentioned, not stated or unclear; if an excerpt has nothing "
        "on a question, write nothing for it. Skip anything unrelated to the questions. Most "
        "excerpts have nothing relevant: NONE is the usual, correct reply, and an invented fact is "
        "far worse than a missing one. If nothing in the excerpts is relevant, reply with exactly "
        "NONE. Reply with ONLY the notes, one per line, no preamble.\n\n"
        f"EXCERPTS:\n{excerpts}"
    )
    async with semaphore:
        try:
            answer, _ = await generate_answer(prompt, max_tokens=READ_MAX_TOKENS)
        except Exception:
            logger.warning("research read failed for a batch of %d excerpts", len(batch), exc_info=True)
            return []
    notes: list[Note] = []
    for line in answer.splitlines():
        match = _NOTE_LINE.match(line)
        if not match:
            continue
        number, claim = int(match.group(1)), match.group(2)
        if 1 <= number <= len(batch) and claim.upper().rstrip(".") != _NONE and not _META_NOTE.search(claim):
            notes.append(Note(claim, batch[number - 1]))
        if len(notes) >= MAX_NOTES_PER_BATCH:
            break
    return notes


async def _check_excerpt_notes(excerpt: Excerpt, notes: list[Note], semaphore: asyncio.Semaphore) -> list[Note]:
    """One thinking-off check of one excerpt's notes against only that
    excerpt - the reader, asked to answer the research questions, will
    sometimes invent a plausible fact and pin it on an unrelated excerpt;
    asked instead whether the excerpt states a given note, with nothing
    else in the prompt, the same model says no. A failed call keeps the
    notes (logged) - the answer's own citation check still runs."""
    numbered = "\n".join(f"{i}. {note.claim}" for i, note in enumerate(notes, start=1))
    prompt = (
        "Below is an excerpt from a document and some notes written from it. For each note, decide "
        "whether the excerpt itself states it: its specific facts (numbers, names, dates, amounts, "
        "conditions) appear in the excerpt, even if worded differently. A note whose facts are not "
        "in the excerpt, or that records something as absent, is NOT stated by it.\n\n"
        f"EXCERPT:\n{excerpt.content}\n\nNOTES:\n{numbered}\n\n"
        "Reply with ONLY the numbers of the notes the excerpt states, comma-separated, or NONE."
    )
    async with semaphore:
        try:
            answer, _ = await generate_answer(prompt, max_tokens=CHECK_MAX_TOKENS)
        except Exception:
            logger.warning("research note check failed for %r; keeping %d notes", excerpt.key, len(notes), exc_info=True)
            return notes
    if answer.strip().upper().startswith(_NONE):
        return []
    stated = {int(n) for n in re.findall(r"\d+", answer)}
    return [note for i, note in enumerate(notes, start=1) if i in stated]


async def check_notes(notes: list[Note]) -> list[Note]:
    """Keeps only the notes their own excerpt states (see
    _check_excerpt_notes), CHECK_CONCURRENCY excerpts at a time."""
    by_excerpt: dict[tuple, list[Note]] = {}
    for note in notes:
        by_excerpt.setdefault(note.excerpt.key, []).append(note)
    semaphore = asyncio.Semaphore(CHECK_CONCURRENCY)
    checked = await asyncio.gather(*(
        _check_excerpt_notes(group[0].excerpt, group, semaphore) for group in by_excerpt.values()
    ))
    kept = [note for group in checked for note in group]
    logger.info("research note check: kept %d of %d notes across %d excerpts", len(kept), len(notes), len(by_excerpt))
    return kept


async def read_all(batches: list[list[Excerpt]], sub_questions: list[str]):
    """Reads every batch, READ_CONCURRENCY at a time, yielding (batches
    done so far, that batch's notes) as each finishes."""
    semaphore = asyncio.Semaphore(READ_CONCURRENCY)
    tasks = [asyncio.create_task(read_batch(batch, sub_questions, semaphore)) for batch in batches]
    done = 0
    try:
        for future in asyncio.as_completed(tasks):
            notes = await future
            done += 1
            yield done, notes
    finally:
        for task in tasks:
            task.cancel()


async def render_notes(notes: list[Note], budget: int) -> tuple[str, list[dict]]:
    """The notes as the model reads them - grouped by document, documents
    with the most notes first, each note prefixed with its source number -
    cut to `budget` tokens by dropping whole documents from the end.
    Returns (text, sources): sources[n-1] is the excerpt behind every [n]
    in the text, so citations in the answer resolve exactly as in the
    other modes."""
    by_document: dict[str, list[Note]] = {}
    for note in notes:
        by_document.setdefault(note.excerpt.filename, []).append(note)
    documents = sorted(by_document, key=lambda f: -len(by_document[f]))

    blocks: list[str] = []
    block_sources: list[list[dict]] = []
    numbers: dict[tuple, int] = {}
    sources: list[dict] = []
    for filename in documents:
        lines = [f"## {filename}"]
        own: list[dict] = []
        for note in sorted(by_document[filename], key=lambda n: n.excerpt.position):
            if note.excerpt.key not in numbers:
                sources.append(note.excerpt.info)
                own.append(note.excerpt.info)
                numbers[note.excerpt.key] = len(sources)
            lines.append(f"- [{numbers[note.excerpt.key]}] ({note.excerpt.label}) {note.claim}")
        blocks.append("\n".join(lines))
        block_sources.append(own)

    kept: list[str] = []
    kept_sources: list[dict] = []
    used = 0
    for block, own, tokens in zip(blocks, block_sources, await count_text_tokens(blocks)):
        if used + tokens > budget:
            break
        kept.append(block)
        kept_sources.extend(own)
        used += tokens
    if len(kept) < len(blocks):
        left_out = len(blocks) - len(kept)
        logger.warning("research notes trimmed: %d of %d documents fit in %d tokens", len(kept), len(blocks), budget)
        kept.append(f"(Notes from {left_out} more document(s) didn't fit and were left out.)")
    return "\n\n".join(kept), kept_sources


async def find_gaps(question: str, sub_questions: list[str], notes_text: str) -> tuple[list[str], list[str]]:
    """One thinking-off check of the notes against the sub-questions: a
    verdict per sub-question (ANSWERED, PARTIAL or MISSING) and a new
    search query for each that isn't answered. Returns (sub-questions the
    notes have nothing on, new queries). Asked for a verdict on every
    sub-question rather than a list of the missing ones: asked for a
    list, the model tended to list most of them. An unclear or failed
    reply counts as nothing missing."""
    numbered = "\n".join(f"{i}. {q}" for i, q in enumerate(sub_questions, start=1))
    prompt = (
        f"Question being researched: {question}\n\n"
        f"Sub-questions the research set out to answer:\n{numbered}\n\n"
        f"Notes gathered so far, grouped by document:\n{notes_text or '(none)'}\n\n"
        "Judge each sub-question against the notes: ANSWERED if the notes give what it asks for, "
        "PARTIAL if they give some of it but a figure, date, party or condition it asks for is "
        "still missing, MISSING if the notes have nothing on it. Then, for each PARTIAL or MISSING "
        "sub-question, write one new search query for the document archive worded differently from "
        "the sub-question - other names, synonyms, or the kind of form, report or letter the fact "
        "would appear in. Reply with ONLY this format:\n"
        "VERDICTS:\n<sub-question number: ANSWERED, PARTIAL or MISSING - a few words why, one line "
        "per sub-question>\n"
        f"QUERIES:\n<one per line, up to {MAX_GAP_QUERIES}, or NONE if every sub-question is ANSWERED>"
    )
    try:
        answer, _ = await generate_answer(prompt, max_tokens=GAPS_MAX_TOKENS)
    except Exception:
        logger.warning("research gap check failed", exc_info=True)
        return [], []
    sections = _parse_sections(answer, ["VERDICTS", "QUERIES"], keep_numbering=("VERDICTS",))
    verdicts: dict[int, str] = {}
    for item in sections["VERDICTS"]:
        match = _VERDICT.match(item)
        if match and 1 <= int(match.group(1)) <= len(sub_questions):
            verdicts.setdefault(int(match.group(1)), match.group(2).upper())
    if not verdicts:
        logger.warning("research gap check gave no verdicts; reply began %r", answer[:300])
        return [], []
    counts = {v: sum(1 for x in verdicts.values() if x == v) for v in ("ANSWERED", "PARTIAL", "MISSING")}
    logger.info(
        "research gap check: %d answered, %d partial, %d missing of %d; %d queries",
        counts["ANSWERED"], counts["PARTIAL"], counts["MISSING"], len(sub_questions), len(sections["QUERIES"]),
    )
    missing = [sub_questions[n - 1] for n in sorted(verdicts) if verdicts[n] == "MISSING"]
    if all(v == "ANSWERED" for v in verdicts.values()):
        return missing, []
    known = {q.lower() for q in sub_questions}
    queries = [q for q in sections["QUERIES"] if q.lower() not in known][:MAX_GAP_QUERIES]
    return missing, queries


def notes_message(question: str, sub_questions: list[str], not_found: list[str], notes_text: str) -> str:
    """The research result as the answering model receives it."""
    numbered = "\n".join(f"{i}. {q}" for i, q in enumerate(sub_questions, start=1))
    if not_found:
        coverage = "Not covered by the documents (say so in the answer):\n" + "\n".join(f"- {q}" for q in not_found)
    else:
        coverage = "Every sub-question found at least some evidence."
    return (
        f"Research notes for: {question}\n\n"
        f"Sub-questions researched:\n{numbered}\n\n"
        f"{coverage}\n\n"
        "Notes, grouped by document. The number in brackets at the start of each note is the "
        "source to cite for it.\n\n"
        f"{notes_text or '(no relevant notes were found in the documents)'}\n\n"
        "Now write the full answer. Cite the number of the specific note each fact comes from - "
        "a fact from a note numbered [12] is cited [12], never the number of a neighbouring note "
        "or of another document's notes - and end every factual sentence with its citation."
    )
