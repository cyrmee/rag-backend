"""Council mode for /ask: several fast planner calls decide *what* to search
for, code runs every one of those searches in parallel, and only then does
the main (thinking) model see the question - with the evidence already in
front of it. Searching is no longer something the model can skip, and the
expensive thinking pass happens once instead of once per search round.

The searches are wide but the evidence handed to the model is not: every
angle's results are pooled, fused by rank (a chunk several angles agree on
beats one only a single angle found), and cut to a fixed budget - so adding
angles broadens recall without growing the prompt.

After the answer, verify_citations() does a cheap second read to flag
sentences whose [N] citation doesn't actually back them up."""

import asyncio
import json
import logging
import re
from dataclasses import dataclass

from app.embeddings import embed_text
from app.generation import generate_answer
from app.retrieval import PER_SUB_QUERY_LIMIT, RRF_K, document_routed_search
from app.web_search import search_web

logger = logging.getLogger(__name__)

MAX_FIGURE_QUERIES = 2
MAX_CORPUS_FILTERS = 3
MAX_WEB_QUERIES = 3
WEB_RESULT_LIMIT = 8

# Planners see the tail of the conversation so a follow-up like "and in
# 2021?" can be turned into a standalone query.
HISTORY_MESSAGES = 4
HISTORY_CHARS = 500

# Verifier prompt budget for source text - each cited source gets an equal
# share, capped per source.
VERIFY_SOURCE_CHARS = 48000
VERIFY_PER_SOURCE_MAX = 2000

_NONE = "NONE"
_PREAMBLE = re.compile(r"^(yes|no)\b[\s,.:!-]", re.IGNORECASE)


@dataclass
class CouncilPlan:
    content_queries: list[str]
    figure_queries: list[str]
    # "" means every document (a plain count/listing of the whole corpus).
    corpus_filters: list[str]
    web_queries: list[str]


def _context_block(question: str, history: list[dict]) -> str:
    lines = []
    for msg in history[-HISTORY_MESSAGES:]:
        content = msg["content"].strip()
        if len(content) > HISTORY_CHARS:
            content = content[:HISTORY_CHARS] + "..."
        lines.append(f"{msg['role']}: {content}")
    convo = "Conversation so far:\n" + "\n".join(lines) + "\n\n" if lines else ""
    return f"{convo}Current question: {question}"


def _parse_lines(answer: str, limit: int) -> list[str]:
    """One item per line, stripped of bullets/numbering/quotes; a bare NONE
    (or nothing) means the planner found nothing to do."""
    items: list[str] = []
    for line in answer.splitlines():
        item = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", line).strip().strip("\"'`").strip()
        if not item or item.upper().rstrip(".") == _NONE:
            continue
        # Planners sometimes answer their yes/no framing question before
        # the queries ("Yes, this could use a chart:") - that's not a query.
        if _PREAMBLE.match(item) or item.endswith(":"):
            continue
        if item.lower() not in (i.lower() for i in items):
            items.append(item)
        if len(items) >= limit:
            break
    return items


async def _ask_planner(prompt: str, limit: int) -> list[str]:
    """A planner failing just means that angle goes unsearched - the other
    planners' results still stand."""
    try:
        answer, _ = await generate_answer(prompt)
    except Exception:
        logger.warning("council planner call failed", exc_info=True)
        return []
    return _parse_lines(answer, limit)


async def plan(question: str, history: list[dict], web_search: bool, angles: int) -> CouncilPlan:
    """Runs the planners in parallel (thinking off, so each is ~a second).
    Each one only proposes searches - none of them answers the question."""
    context = _context_block(question, history)
    prompts = [
        (
            f"{context}\n\n"
            f"Write up to {angles} distinct search queries for an organization's "
            "internal document archive that, together, find everything needed to "
            "answer the current question - different angles, synonyms, the "
            "specific names/numbers/terms involved. Each query short and "
            "standalone (resolve words like 'it' or 'that year' using the "
            "conversation). If the question is pure small talk (a greeting, "
            "thanks) reply NONE. Reply with ONLY the queries, one per line.",
            angles,
        ),
        (
            f"{context}\n\n"
            "Could answering this use numbers or content from a chart, graph, "
            "table, or figure (e.g. a statistic, trend, breakdown, or dashboard)? "
            f"If yes, write up to {MAX_FIGURE_QUERIES} search queries phrased the "
            "way a description of such a chart would be worded. If no, reply "
            "with exactly NONE. Reply with ONLY the queries (one per line, no "
            "preamble, don't restate yes/no) or NONE.",
            MAX_FIGURE_QUERIES,
        ),
        (
            f"{context}\n\n"
            "Is the current question about the document collection itself - how "
            "many documents/files there are, which files exist, or files in a "
            "given folder or with a given name? If yes, reply with up to "
            f"{MAX_CORPUS_FILTERS} short filename/folder substrings to filter by, "
            "one per line, or ALL to cover every document - no preamble, don't "
            "restate yes/no. If the question is about what documents *say* "
            "rather than which documents exist, reply with exactly NONE.",
            MAX_CORPUS_FILTERS,
        ),
    ]
    if web_search:
        prompts.append((
            f"{context}\n\n"
            f"Write up to {MAX_WEB_QUERIES} public web search queries that would "
            "help answer the current question. Reply with ONLY the queries, one "
            "per line.",
            MAX_WEB_QUERIES,
        ))

    results = await asyncio.gather(*(_ask_planner(p, limit) for p, limit in prompts))
    content, figures, corpus = results[0], results[1], results[2]
    web = results[3] if web_search else []

    # The question verbatim stays one of the angles (as in
    # decompose_and_retrieve) unless the planner judged it small talk.
    if content and question.lower() not in (q.lower() for q in content):
        content = [question, *content]
    corpus_filters = ["" if f.upper() == "ALL" else f for f in corpus]
    return CouncilPlan(content, figures, corpus_filters, web)


async def search_documents(queries: list[str], budget: int) -> list[dict]:
    """Runs every query's document-routed search in parallel, then fuses
    the result lists by reciprocal rank - a chunk ranked well by several
    angles outranks one a single angle happened to rank first - and keeps
    the top `budget`. Individual angle failures are skipped; if every angle
    fails (e.g. the embedding server is down) that error is raised."""

    async def one(query: str) -> list[dict]:
        return await document_routed_search(await embed_text(query), query, PER_SUB_QUERY_LIMIT)

    outcomes = await asyncio.gather(*(one(q) for q in queries), return_exceptions=True)
    failures = [o for o in outcomes if isinstance(o, BaseException)]
    for query, outcome in zip(queries, outcomes):
        if isinstance(outcome, BaseException):
            logger.warning("council search failed for %r: %r", query, outcome)
    if failures and len(failures) == len(outcomes):
        raise failures[0]

    scores: dict[tuple[str, int], float] = {}
    rows: dict[tuple[str, int], dict] = {}
    for outcome in outcomes:
        if isinstance(outcome, BaseException):
            continue
        for rank, row in enumerate(outcome, start=1):
            key = (row["filename"], row["chunk_index"])
            scores[key] = scores.get(key, 0.0) + 1.0 / (RRF_K + rank)
            rows.setdefault(key, row)

    ranked = sorted(scores, key=scores.get, reverse=True)[:budget]
    return [rows[key] for key in ranked]


async def search_web_many(queries: list[str], limit: int = WEB_RESULT_LIMIT) -> list[dict]:
    """All web queries in parallel, deduped by URL and interleaved
    round-robin so one query's results can't crowd out the others'."""
    result_lists = await asyncio.gather(*(search_web(q) for q in queries))
    merged: list[dict] = []
    seen: set[str] = set()
    for i in range(max((len(r) for r in result_lists), default=0)):
        for results in result_lists:
            if i < len(results) and results[i]["url"] not in seen:
                seen.add(results[i]["url"])
                merged.append(results[i])
    return merged[:limit]


async def verify_citations(segments: list[dict], sources: list[dict]) -> list[dict]:
    """Flags answer sentences whose citations don't support them: any
    citation number with no matching source outright, and - via one
    thinking-off model pass over the cited sources - sentences whose cited
    sources don't state what the sentence claims. Returns
    [{text, source_indices, reason}]; an empty list means nothing was
    flagged (or the check itself couldn't run, which is logged)."""
    warnings: list[dict] = []
    claims: list[dict] = []
    for seg in segments:
        if not seg["source_indices"]:
            continue
        missing = [i for i in seg["source_indices"] if not 1 <= i <= len(sources)]
        if missing:
            warnings.append({**seg, "reason": f"cites source(s) {missing}, which don't exist"})
        else:
            claims.append(seg)
    if not claims:
        return warnings

    cited = sorted({i for seg in claims for i in seg["source_indices"]})
    per_source = min(VERIFY_PER_SOURCE_MAX, VERIFY_SOURCE_CHARS // len(cited))
    source_block = "\n\n".join(f"[{i}] {sources[i - 1]['content'][:per_source]}" for i in cited)
    claim_block = "\n".join(
        f"{n}. {seg['text']} (cites {''.join(f'[{i}]' for i in seg['source_indices'])})"
        for n, seg in enumerate(claims, start=1)
    )
    prompt = (
        "Below are numbered source excerpts, then numbered claims from an "
        "answer, each citing some of the sources. For each claim, decide "
        "whether its cited sources actually state or directly support it. A "
        "claim is supported if its specific facts (numbers, names, dates, "
        "settings) appear in at least one of its cited sources, even if worded "
        "differently. A claim that makes no specific factual assertion counts "
        "as supported.\n\n"
        f"SOURCES:\n{source_block}\n\nCLAIMS:\n{claim_block}\n\n"
        'Reply with ONLY a JSON object: {"unsupported": [claim numbers]}'
    )
    try:
        answer, _ = await generate_answer(prompt)
        match = re.search(r"\{.*\}", answer, re.DOTALL)
        unsupported = json.loads(match.group(0))["unsupported"] if match else []
    except Exception:
        logger.warning("citation verification failed", exc_info=True)
        return warnings

    for n in unsupported:
        if isinstance(n, int) and 1 <= n <= len(claims):
            warnings.append({**claims[n - 1], "reason": "cited source(s) don't support this"})
    return warnings
