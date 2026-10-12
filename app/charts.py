"""Charts in answers. The model draws a line, bar, pie or donut chart by
writing a fenced ```chart block holding one JSON spec (CHART_INSTRUCTIONS,
appended to every answer-writing system prompt), and the client renders
it. This module is everything in between:

- ChartSpec, the schema a block must satisfy;
- ChartStreamFilter, which holds each block back from the streamed answer
  until it is complete, then passes it on normalized - or drops it when it
  is malformed - so a client never sees half a spec and never has to
  decide whether a spec is drawable;
- a grounding check: every value must be a number the cited sources
  actually state. A chart with values that aren't is still passed on, with
  the offending values listed under "unverified" in the spec and a
  warning in the done event's citation_warnings, the same way an
  unsupported sentence is flagged rather than deleted.

filter_answer() runs the same filter over a whole answer at once, so the
answer persisted with the turn matches what was streamed.
"""

import json
import logging
import math
import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from app.generation import generate_answer

logger = logging.getLogger(__name__)

MAX_POINTS = 60
MAX_SERIES = 8

CHART_INSTRUCTIONS = (
    "Charts: add a chart to the answer when the user asks for a chart, "
    "graph, plot or visualization, and also, unasked, when the substance "
    "of the answer is a series of three or more comparable numbers that a "
    "chart conveys better than prose - a value over several years, "
    "quarters or months, amounts compared across categories, or the parts "
    "of one total. The concrete test: if the answer lists or tabulates "
    "three or more values of the same measure (a figure per quarter, per "
    "year, per region, per item), it gets a chart of those values as "
    "well - whether or not the user used the word chart. Never add one "
    "for a question answered by a figure or two, for numbers from "
    "contexts that aren't comparable, or when you would have to guess any "
    "value; and never refer to a chart ('as the chart shows') unless you "
    "have written the block in this very answer. Write it as a fenced "
    "block tagged "
    "`chart` holding one JSON object, exactly like this:\n"
    "```chart\n"
    '{"type": "line", "title": "Staff headcount by year", '
    '"labels": ["2021", "2022", "2023"], '
    '"series": [{"name": "Headcount", "values": [118, 131, 127]}], '
    '"unit": "people", "sources": [3]}\n'
    "```\n"
    "Choose `type` by what the data is, not by what sounds nice: line for "
    "a value over time (the labels are years, quarters, months or dates); "
    "bar for amounts compared across categories, or over time when there "
    "are only a few points; pie or donut only for the shares of one whole "
    "at one moment (categories that add up to a total, such as a budget "
    "split by region) - never for a value over time, never for figures "
    "that don't add up to anything, and never with fewer than two or more "
    "than eight slices. If the user asks for a type that doesn't fit the "
    "data, draw the type that fits and say so in a sentence. `labels` "
    "names the points or categories; each entry of `series` has a `name` "
    "and exactly as many numeric `values` as there are labels, and pie and "
    "donut charts have exactly one series; `unit` is optional; `sources` "
    "lists the citation numbers of the chunks the values come from. Every "
    "value must be a number stated in those chunks, written as a plain "
    "JSON number (no units, currency signs, thousands separators or "
    "quotes) - never estimate, round or compute a value yourself; for a "
    "pie or donut give the stated amounts, not percentages you worked out. "
    "The chart is part of a written answer, never the whole of it: before "
    "the block, state in a sentence or two what the figures are, with "
    "their citations, and after it say in a sentence what the chart shows "
    "(the high and low points, the trend); put the block on its own lines "
    "between those sentences. Never mention the block, JSON or the chart's "
    "format. If the chunks don't state the numbers needed, say so instead "
    "of drawing a chart with guessed values. A chart's numbers must come "
    "from chunks returned in this turn: when a follow-up asks for a chart "
    "of figures from an earlier answer, call `retrieve` for them again "
    "first, so the chart can cite them - never chart from memory of the "
    "conversation."
)


class ChartSeries(BaseModel):
    model_config = ConfigDict(extra="ignore")

    name: str = Field(min_length=1, max_length=80)
    values: list[float] = Field(min_length=1, max_length=MAX_POINTS)

    @field_validator("values", mode="before")
    @classmethod
    def _no_bools(cls, values):
        if isinstance(values, list) and any(isinstance(v, bool) for v in values):
            raise ValueError("values must be numbers")
        return values

    @field_validator("values")
    @classmethod
    def _finite(cls, values: list[float]) -> list[float]:
        if any(not math.isfinite(v) for v in values):
            raise ValueError("values must be finite numbers")
        return values


class ChartSpec(BaseModel):
    """One ```chart block. Unknown keys are ignored rather than rejected -
    an extra key the model made up is no reason to lose the chart."""

    model_config = ConfigDict(extra="ignore")

    type: Literal["line", "bar", "pie", "donut"]
    title: str = Field(min_length=1, max_length=200)
    labels: list[str] = Field(min_length=1, max_length=MAX_POINTS)
    series: list[ChartSeries] = Field(min_length=1, max_length=MAX_SERIES)
    unit: str | None = Field(default=None, max_length=40)
    sources: list[int] = []

    @field_validator("labels", mode="before")
    @classmethod
    def _labels_to_str(cls, labels):
        # Years and other numeric categories arrive as numbers.
        if isinstance(labels, list):
            return [str(x) if isinstance(x, (int, float)) and not isinstance(x, bool) else x for x in labels]
        return labels

    @field_validator("type", mode="before")
    @classmethod
    def _type_lower(cls, value):
        return value.strip().lower() if isinstance(value, str) else value

    @model_validator(mode="after")
    def _consistent(self) -> "ChartSpec":
        for s in self.series:
            if len(s.values) != len(self.labels):
                raise ValueError(
                    f"series {s.name!r} has {len(s.values)} values for {len(self.labels)} labels"
                )
        if self.type in ("pie", "donut") and len(self.series) != 1:
            raise ValueError(f"a {self.type} chart takes exactly one series, got {len(self.series)}")
        return self


# Slips the model makes that have one obvious reading: citation-style
# sources ("sources": [1][3]) and a trailing comma before a closing bracket.
_CITATION_STYLE_SOURCES = re.compile(r'("sources"\s*:\s*)((?:\[\s*\d+\s*\]\s*){2,})')
_TRAILING_COMMA = re.compile(r",\s*([}\]])")


def _repair(body: str) -> str:
    body = _CITATION_STYLE_SOURCES.sub(
        lambda m: m.group(1) + "[" + ", ".join(re.findall(r"\d+", m.group(2))) + "]", body,
    )
    return _TRAILING_COMMA.sub(r"\1", body)


# Labels that read as points in time: years (2019, FY2023, 2023/24),
# quarters and halves (Q1, Q3 2024, H2), months (Jan, March 2024,
# 2024-03) and dates. A pie or donut over them shows a trend as shares of
# nothing, so such a chart is drawn as bars instead.
_MONTHS = r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*"
_TIME_LABEL = re.compile(
    r"^\s*(?:"
    r"(?:fy\s*)?(?:19|20)\d\d(?:\s*[/-]\s*(?:19|20)?\d\d)?"  # 2023, FY2023, 2023/24, 2023-2024
    r"|[qh][1-4](?:\s*(?:fy)?\s*(?:19|20)?\d\d)?"               # Q1, Q3 2024, H2 FY23
    r"|(?:19|20)\d\d\s*[qh][1-4]"                                   # 2024 Q3
    r"|" + _MONTHS + r"(?:\s*(?:19|20)?\d\d)?"                      # Jan, March 2024
    r"|(?:19|20)\d\d-\d\d(?:-\d\d)?"                             # 2024-03, 2024-03-15
    r"|\d\d?[/.-]\d\d?[/.-](?:19|20)?\d\d"                       # 15/03/2024
    r")\s*$",
    re.IGNORECASE,
)


def is_time_series(labels: list[str]) -> bool:
    """True when at least two thirds of the labels read as points in time."""
    if len(labels) < 2:
        return False
    hits = sum(1 for label in labels if _TIME_LABEL.match(label))
    return hits * 3 >= len(labels) * 2


def fit_type(spec: ChartSpec) -> ChartSpec:
    """A pie or donut that can't show shares of a whole - labels that are
    points in time, a negative value, a single slice or more than eight -
    becomes a bar chart of the same numbers (logged). The model is asked
    to pick the right form; this makes sure the reader never gets the
    wrong one."""
    if spec.type not in ("pie", "donut"):
        return spec
    values = spec.series[0].values
    reason = None
    if is_time_series(spec.labels):
        reason = "labels are points in time"
    elif any(v < 0 for v in values):
        reason = "a value is negative"
    elif len(values) < 2:
        reason = "only one slice"
    elif len(values) > 8:
        reason = f"{len(values)} slices"
    if reason is None:
        return spec
    logger.warning("%s chart %r drawn as bars: %s", spec.type, spec.title, reason)
    return spec.model_copy(update={"type": "bar"})


def parse_chart(body: str) -> ChartSpec | None:
    """The spec in a block's body, or None (logged) when it isn't one."""
    try:
        data = json.loads(body)
    except ValueError as exc:
        try:
            data = json.loads(_repair(body))
        except ValueError:
            logger.warning("chart block is not JSON (%s): %s", exc, body[:200])
            return None
        logger.info("chart block repaired (%s): %s", exc, body[:200])
    if not isinstance(data, dict):
        logger.warning("chart block is not a JSON object: %s", body[:200])
        return None
    try:
        return ChartSpec.model_validate(data)
    except ValidationError as exc:
        logger.warning("chart block rejected: %s; block: %s", "; ".join(e["msg"] for e in exc.errors()), body[:200])
        return None


_NUMBER = re.compile(r"-?\d[\d,]*(?:\.\d+)?|-?\.\d+")


def _numbers_in(text: str) -> set[float]:
    found: set[float] = set()
    for token in _NUMBER.findall(text):
        try:
            found.add(float(token.replace(",", "")))
        except ValueError:
            continue
    return found


def _close(a: float, b: float) -> bool:
    return abs(a - b) <= 1e-9 * max(1.0, abs(a), abs(b))


def unverified_values(spec: ChartSpec, sources: list[dict]) -> list[float]:
    """Values no cited source states (every source, when the spec cites
    none). Numbers are compared after stripping thousands separators, so
    "$6,320,815" in a source backs 6320815 in a chart - and nothing else:
    a value the model derived, scaled or rounded counts as unverified."""
    cited = [sources[i - 1] for i in spec.sources if 1 <= i <= len(sources)] or sources
    stated: set[float] = set()
    for source in cited:
        stated |= _numbers_in(source.get("content") or "")
    missing: list[float] = []
    for s in spec.series:
        for v in s.values:
            if not any(_close(v, n) for n in stated) and v not in missing:
                missing.append(v)
    return missing


def _plain(value: float) -> int | float:
    return int(value) if value.is_integer() else value


def render_spec(spec: ChartSpec, unverified: list[float]) -> dict:
    """The spec as the client gets it: integral values as ints, dangling
    sources already removed by the caller, `unverified` only when
    non-empty."""
    data = {
        "type": spec.type,
        "title": spec.title,
        "labels": spec.labels,
        "series": [{"name": s.name, "values": [_plain(v) for v in s.values]} for s in spec.series],
    }
    if spec.unit:
        data["unit"] = spec.unit
    data["sources"] = spec.sources
    if unverified:
        data["unverified"] = [_plain(v) for v in unverified]
    return data


# A chart block opens with ``` at the start of a line (indentation allowed,
# as inside a list item), the tag "chart", and a newline; it closes with
# ``` alone on a line. Mid-stream, a buffer may end inside the opener
# ("``", "```ch"), so there's a prefix pattern to know when to wait.
_FENCE = re.compile(r"```")
_OPENER = re.compile(r"```[ \t]*chart[ \t]*\r?\n", re.IGNORECASE)
_OPENER_PREFIX = re.compile(r"```[ \t]*(?:c|ch|cha|char|chart)?[ \t]*\r?", re.IGNORECASE)
_CLOSER = re.compile(r"^[ \t]*```[ \t]*\r?\n", re.MULTILINE)
_CLOSER_AT_END = re.compile(r"^[ \t]*```[ \t]*\Z", re.MULTILINE)


class ChartStreamFilter:
    """Feed the answer's text as it streams; get back what to pass on.
    Outside a chart block, text passes through as it arrives (less a
    trailing backtick or two that might start a fence). From a chart
    block's opening fence on, everything is held until the closing fence,
    then the whole block is replaced by its normalized form - or by
    nothing, when the block isn't a valid spec. flush() at the end of the
    stream returns whatever is still held.

    `sources` is this turn's source list (1-based citation numbers index
    into it) for the grounding check and for dropping sources the spec
    cites that don't exist; it's read when a block completes, so the list
    the agent loop is still filling may be passed as is."""

    def __init__(self, sources: list[dict] | None = None):
        self._sources = sources if sources is not None else []
        self._buffer = ""
        self._in_block = False
        self._line_start = True  # does the buffer begin at the start of a line?
        self._indent = ""
        self.charts: list[dict] = []
        self.warnings: list[dict] = []
        self.dropped = 0

    def feed(self, text: str) -> str:
        self._buffer += text
        return self._drain(final=False)

    def flush(self) -> str:
        return self._drain(final=True)

    def _drain(self, final: bool) -> str:
        out: list[str] = []
        while self._buffer:
            if self._in_block:
                match = _CLOSER.search(self._buffer) or (final and _CLOSER_AT_END.search(self._buffer))
                if not match:
                    if final:
                        self.dropped += 1
                        logger.warning("chart block never closed; dropped: %s", self._buffer[:200])
                        self._buffer = ""
                    break
                out.append(self._complete_block(self._buffer[: match.start()]))
                self._buffer = self._buffer[match.end():]
                self._in_block = False
                self._line_start = True
                continue

            opened = False
            for fence in _FENCE.finditer(self._buffer):
                line_begin = self._line_begin(fence.start())
                if line_begin is None:
                    continue
                tail = self._buffer[fence.start():]
                opener = _OPENER.match(tail)
                if opener:
                    # The block takes the fence's indentation (inside a
                    # list item, say) on every line it's replaced with.
                    self._indent = self._buffer[line_begin:fence.start()]
                    out.append(self._emit(self._buffer[:line_begin]))
                    self._buffer = tail[opener.end():]
                    self._in_block = True
                    self._line_start = True
                    opened = True
                    break
                if not final and _OPENER_PREFIX.fullmatch(tail):
                    # Might still become ```chart - wait for more.
                    out.append(self._emit(self._buffer[:line_begin]))
                    self._buffer = self._buffer[line_begin:]
                    return "".join(out)
            if opened:
                continue
            keep = 0
            if not final:
                # A line's leading whitespace and up to two backticks could
                # still be the start of a fence - hold them back.
                partial = re.search(r"[ \t]*`{0,2}$", self._buffer)
                if self._line_begin(partial.start()) is not None:
                    keep = len(self._buffer) - partial.start()
            out.append(self._emit(self._buffer[: len(self._buffer) - keep]))
            self._buffer = self._buffer[len(self._buffer) - keep:]
            break
        return "".join(out)

    def _line_begin(self, position: int) -> int | None:
        """Where the line holding `position` starts, when only whitespace
        precedes `position` on it (so a fence there opens a block); None
        otherwise."""
        line_begin = self._buffer.rfind("\n", 0, position) + 1
        if line_begin == 0 and not self._line_start:
            return None
        if self._buffer[line_begin:position].strip(" \t") != "":
            return None
        return line_begin

    def _emit(self, text: str) -> str:
        if text:
            self._line_start = text.endswith("\n")
        return text

    def _complete_block(self, body: str) -> str:
        spec = parse_chart(body)
        if spec is None:
            self.dropped += 1
            return ""
        source_count = len(self._sources)
        dangling = [i for i in spec.sources if not 1 <= i <= source_count]
        if dangling:
            logger.warning("chart cites %s but this turn has only %d source(s); dropped", dangling, source_count)
        spec.sources = sorted({i for i in spec.sources if 1 <= i <= source_count})
        spec = fit_type(spec)
        unverified = unverified_values(spec, self._sources)
        if unverified:
            self.warnings.append({
                "text": spec.title,
                "source_indices": spec.sources,
                "reason": "chart values not stated in the cited sources: "
                          + ", ".join(str(_plain(v)) for v in unverified),
            })
        data = render_spec(spec, unverified)
        self.charts.append(data)
        return "".join(
            self._indent + line + "\n" for line in ("```chart", json.dumps(data, ensure_ascii=False), "```")
        )


def filter_answer(answer: str, sources: list[dict] | None = None) -> tuple[str, list[dict], list[dict]]:
    """(answer with every chart block normalized or dropped, the charts in
    order, grounding warnings) - the whole-answer form of ChartStreamFilter,
    byte-for-byte what streaming the same text would have passed on."""
    stream = ChartStreamFilter(sources)
    text = stream.feed(answer) + stream.flush()
    return text, stream.charts, stream.warnings


# ---------------------------------------------------------------------------
# The safety net: the answering model is asked to chart a comparable series
# itself, but doesn't always (measured: it drew for "how did revenue change
# across the quarters" in 2 of 3 runs and for "describe the revenue trend"
# in 0 of 3, while listing the four figures every time). So an answer that
# ends without a chart, yet states three or more numbers, gets one
# thinking-off call that either writes the block from the retrieved chunks
# or says NONE. The block goes through the same validation, grounding and
# type fitting as one the model wrote itself, and is appended to the answer.

_ANSWER_NUMBER = re.compile(r"(?<![\[\w])-?\d[\d,]*(?:\.\d+)?%?(?!\])")
PROPOSAL_MIN_NUMBERS = 3
PROPOSAL_MAX_SOURCES = 25
PROPOSAL_SOURCE_CHARS = 1200
PROPOSAL_MAX_TOKENS = 600

PROPOSAL_PROMPT = (
    "An assistant answered a question from numbered document chunks. Decide "
    "whether the answer should carry a chart, and if so write it.\n\n"
    "Write a chart only when the answer's substance is a series of three "
    "or more comparable numbers that a chart conveys better than prose: a "
    "value over several years, quarters or months, amounts compared across "
    "categories, or the parts of one total. Write NONE when the question "
    "asks for a single value (one period, one item, one setting), even if "
    "the answer mentions other figures for context; when the answer "
    "gives a figure or two; when its numbers come from contexts that "
    "aren't comparable (different measures, units or documents that don't "
    "belong together), when they are identifiers, dates or page numbers "
    "rather than measures, or when the chunks don't state every value.\n\n"
    "The chart is a fenced block tagged `chart` holding one JSON object, "
    "exactly like this:\n"
    "```chart\n"
    '{"type": "line", "title": "Staff headcount by year", '
    '"labels": ["2021", "2022", "2023"], '
    '"series": [{"name": "Headcount", "values": [118, 131, 127]}], '
    '"unit": "people", "sources": [3]}\n'
    "```\n"
    "`type`: line for a value over time (labels are years, quarters, "
    "months or dates); bar for amounts compared across categories, or over "
    "time with only a few points; pie or donut only for shares of one "
    "whole at one moment, never for a value over time. `labels` names the "
    "points or categories; each entry of `series` has a `name` and exactly "
    "as many numeric `values` as there are labels; pie and donut have one "
    "series; `unit` is optional; `sources` lists the numbers of the chunks "
    "the values come from. Every value must be a number stated in those "
    "chunks, as a plain JSON number - never estimated, rounded, scaled or "
    "computed; for a pie or donut, the stated amounts, not percentages.\n\n"
    "Reply with only the block, or only the word NONE.\n\n"
    "Question: <<question>>\n\nAnswer:\n<<answer>>\n\nChunks:\n<<chunks>>"
)


def wants_proposal(answer: str) -> bool:
    """An answer that has no chart but states three or more distinct
    numbers (citation markers like [7] don't count)."""
    if "```chart" in answer:
        return False
    numbers = {m.group(0).replace(",", "") for m in _ANSWER_NUMBER.finditer(answer)}
    return len(numbers) >= PROPOSAL_MIN_NUMBERS


def _chunks_for_proposal(sources: list[dict]) -> str:
    lines = []
    for n, source in enumerate(sources[:PROPOSAL_MAX_SOURCES], start=1):
        content = re.sub(r"\s+", " ", source.get("content") or "").strip()
        lines.append(f"[{n}] {content[:PROPOSAL_SOURCE_CHARS]}")
    return "\n\n".join(lines)


_PROPOSED_BLOCK = re.compile(r"```[ \t]*chart[ \t]*\n.*?\n[ \t]*```", re.DOTALL | re.IGNORECASE)


def build_proposal_prompt(question: str, answer: str, sources: list[dict]) -> str:
    # Plain replacement, not str.format: the example block has braces.
    return (
        PROPOSAL_PROMPT.replace("<<question>>", question)
        .replace("<<answer>>", answer.strip())
        .replace("<<chunks>>", _chunks_for_proposal(sources))
    )


async def propose_chart(question: str, answer: str, sources: list[dict]) -> str | None:
    """One thinking-off call: the ```chart block the answer should have
    carried, as text for filter_answer to validate, or None. Anything
    that goes wrong is logged and counts as NONE - the net never breaks
    an answer."""
    try:
        if not sources or not wants_proposal(answer):
            return None
        reply, _ = await generate_answer(build_proposal_prompt(question, answer, sources), max_tokens=PROPOSAL_MAX_TOKENS)
        match = _PROPOSED_BLOCK.search(reply)
        if not match:
            if "NONE" not in reply.upper():
                logger.info("chart proposal was neither a block nor NONE: %s", reply[:200])
            return None
        logger.info("chart proposed for %r", question[:80])
        return match.group(0)
    except Exception as exc:  # noqa: BLE001
        logger.warning("chart proposal failed: %s", exc)
        return None
