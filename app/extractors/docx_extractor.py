import re
import uuid

import docx
from docx.oxml.ns import qn
from docx.text.paragraph import Paragraph

from app.extractors.types import ExtractedImage, ExtractionResult, TextChunk

# Word paragraphs are often a single line - an email's "To:" line, a
# signature, one form field - so one TextChunk per paragraph meant one tiny,
# context-free chunk each (e.g. a chunk that's just "to"). Consecutive
# paragraphs are packed into sections instead: a new section starts at each
# heading, or once the current one reaches SECTION_WORDS.
SECTION_WORDS = 150
_BLANK_LINES = re.compile(r"\n\s*\n+")


_T, _TAB, _BR, _CR = qn("w:t"), qn("w:tab"), qn("w:br"), qn("w:cr")


def _paragraph_text(para) -> str:
    """The paragraph's text with tracked changes accepted: inserted text
    (<w:ins>) included, deleted text (<w:del>, stored as <w:delText>) left
    out. python-docx's `paragraph.text` skips tracked insertions entirely,
    so a revised document came out as neither version - e.g. a contract's
    "In the event of clear contradiction between..." read "In the event of
    between...", and first letters inserted as revisions went missing
    ("rincipal" for "Principal")."""
    parts = []
    for el in para._p.iter(_T, _TAB, _BR, _CR):
        if el.tag == _T:
            parts.append(el.text or "")
        elif el.tag == _TAB:
            parts.append("\t")
        else:
            parts.append("\n")
    return "".join(parts)


_ROMAN = [(1000, "m"), (900, "cm"), (500, "d"), (400, "cd"), (100, "c"), (90, "xc"),
          (50, "l"), (40, "xl"), (10, "x"), (9, "ix"), (5, "v"), (4, "iv"), (1, "i")]


def _roman(n: int) -> str:
    out = ""
    for value, letters in _ROMAN:
        while n >= value:
            out, n = out + letters, n - value
    return out


def _letters(n: int) -> str:
    # Word's letter lists go a..z, then aa..zz, aaa..zzz
    return chr(ord("a") + (n - 1) % 26) * ((n - 1) // 26 + 1)


def _format_number(n: int, num_fmt: str) -> str:
    if num_fmt == "lowerLetter":
        return _letters(n)
    if num_fmt == "upperLetter":
        return _letters(n).upper()
    if num_fmt == "lowerRoman":
        return _roman(n)
    if num_fmt == "upperRoman":
        return _roman(n).upper()
    if num_fmt == "decimalZero":
        return f"{n:02d}"
    return str(n)


def _val(el, tag: str) -> str | None:
    child = el.find(qn(tag)) if el is not None else None
    return child.get(qn("w:val")) if child is not None else None


class _ListNumbering:
    """The numbers and bullets Word displays in front of list paragraphs
    and numbered headings. They aren't part of the paragraph text - Word
    computes them from numbering.xml when it renders - so without this a
    contract's "5.2 The Supplier shall..." was indexed as "The Supplier
    shall...", and a question about clause 5.2 had nothing to match.

    labels() walks every paragraph in document order - table cells
    included, since a numbered paragraph inside a table advances the same
    counters as one outside it. Counters advance as Word's do: per list
    definition, and a level's counter resets whenever a higher level
    advances."""

    def __init__(self, document):
        self._levels: dict[str, dict[int, tuple[str, str, int, bool]]] = {}
        self._nums: dict[str, tuple[str, dict[int, int]]] = {}
        self._counters: dict[str, dict[int, int]] = {}
        self._started: set[str] = set()
        try:
            root = document.part.numbering_part.element
        except (KeyError, NotImplementedError):
            return
        for abstract in root.findall(qn("w:abstractNum")):
            levels = {}
            for lvl in abstract.findall(qn("w:lvl")):
                levels[int(lvl.get(qn("w:ilvl")))] = (
                    _val(lvl, "w:numFmt") or "decimal",
                    _val(lvl, "w:lvlText") or "",
                    int(_val(lvl, "w:start") or 1),
                    lvl.find(qn("w:isLgl")) is not None,
                )
            self._levels[abstract.get(qn("w:abstractNumId"))] = levels
        for num in root.findall(qn("w:num")):
            overrides = {
                int(o.get(qn("w:ilvl"))): int(_val(o, "w:startOverride"))
                for o in num.findall(qn("w:lvlOverride"))
                if _val(o, "w:startOverride") is not None
            }
            self._nums[num.get(qn("w:numId"))] = (_val(num, "w:abstractNumId"), overrides)

    def _num_pr(self, para) -> tuple[str | None, int]:
        """(numId, level) from the paragraph's own properties, else from
        its style chain (numbered "Heading 1"/"List Number" styles)."""
        p_pr = para._p.pPr
        num_pr = p_pr.numPr if p_pr is not None else None
        if num_pr is not None and num_pr.numId is not None:
            return str(num_pr.numId.val), num_pr.ilvl.val if num_pr.ilvl is not None else 0
        style = para.style
        while style is not None:
            style_num_pr = style.element.pPr.numPr if style.element.pPr is not None else None
            if style_num_pr is not None and style_num_pr.numId is not None:
                level = style_num_pr.ilvl.val if style_num_pr.ilvl is not None else 0
                return str(style_num_pr.numId.val), level
            style = style.base_style
        return None, 0

    def labels(self, document) -> dict:
        """{paragraph XML element: label} for every numbered paragraph."""
        out = {}
        for p in document.element.body.iter(qn("w:p")):
            label = self._label(Paragraph(p, document._body))
            if label:
                out[p] = label
        return out

    def _label(self, para) -> str:
        num_id, level = self._num_pr(para)
        if num_id is None or num_id == "0" or num_id not in self._nums:
            return ""
        abstract_id, overrides = self._nums[num_id]
        levels = self._levels.get(abstract_id, {})
        if level not in levels:
            return ""
        counters = self._counters.setdefault(abstract_id, {})
        if num_id not in self._started:
            self._started.add(num_id)
            for lvl, start in overrides.items():
                counters[lvl] = start - 1
        # Numbering a paragraph puts every higher level that hasn't been used
        # yet at its start value (a "1.1" with no "1" before it shows "1"),
        # and from then on that level counts as used - the next paragraph at
        # it is start + 1, as in Word.
        for higher in range(level):
            if higher in levels:
                counters.setdefault(higher, levels[higher][2])
        counters[level] = counters.get(level, levels[level][2] - 1) + 1
        for deeper in [lvl for lvl in counters if lvl > level]:
            del counters[deeper]

        num_fmt, lvl_text, _, is_legal = levels[level]
        if num_fmt == "bullet":
            return "-"
        if num_fmt == "none":
            return lvl_text.strip()

        def number(match: re.Match) -> str:
            lvl = int(match.group(1)) - 1
            fmt, _, start, _ = levels.get(lvl, ("decimal", "", 1, False))
            return _format_number(counters.get(lvl, start), "decimal" if is_legal else fmt)

        return re.sub(r"%(\d)", number, lvl_text).strip()


def _is_heading(para) -> bool:
    style = para.style.name.lower() if para.style is not None and para.style.name else ""
    return style.startswith(("heading", "title"))


def _labeled_text(para, labels: dict) -> str:
    """The paragraph's text with its list number/bullet in front, if any
    (see _ListNumbering)."""
    text = _BLANK_LINES.sub("\n", _paragraph_text(para)).strip()
    label = labels.get(para._p)
    return f"{label} {text}" if label and text else text


def _table_to_markdown(table: docx.table.Table, labels: dict) -> str:
    rows = [
        ["\n".join(_labeled_text(p, labels) for p in cell.paragraphs).strip() for cell in row.cells]
        for row in table.rows
    ]
    if not rows:
        return ""

    lines = ["| " + " | ".join(rows[0]) + " |"]
    lines.append("| " + " | ".join("---" for _ in rows[0]) + " |")
    for row in rows[1:]:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def extract_docx(file_path: str) -> ExtractionResult:
    document = docx.Document(file_path)
    text_chunks: list[TextChunk] = []

    # docx has no true page concept at the XML level, so paragraph/table
    # index is used as a synthetic "page_number" throughout - a section
    # takes its first paragraph's index.
    labels = _ListNumbering(document).labels(document)
    section: list[str] = []
    section_words = 0
    section_start = 0
    for index, para in enumerate(document.paragraphs):
        text = _labeled_text(para, labels)
        if not text:
            continue
        if section and (_is_heading(para) or section_words >= SECTION_WORDS):
            # Single newlines keep the section one paragraph to chunk_text.
            text_chunks.append(TextChunk(content="\n".join(section), page_number=section_start))
            section, section_words = [], 0
        if not section:
            section_start = index
        section.append(text)
        section_words += len(text.split())
    if section:
        text_chunks.append(TextChunk(content="\n".join(section), page_number=section_start))

    for index, table in enumerate(document.tables):
        markdown = _table_to_markdown(table, labels)
        if markdown:
            text_chunks.append(TextChunk(content=markdown, page_number=index))

    images: list[ExtractedImage] = []
    for index, rel in enumerate(document.part.related_parts.values()):
        if "image" in rel.content_type:
            images.append(
                ExtractedImage(
                    image_id=str(uuid.uuid4()),
                    image_bytes=rel.blob,
                    page_number=index,  # synthetic: relationship order, not a real page
                    bbox=None,
                )
            )

    return text_chunks, images
