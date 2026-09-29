import re
import uuid

import docx

from app.extractors.types import ExtractedImage, ExtractionResult, TextChunk

# Word paragraphs are often a single line - an email's "To:" line, a
# signature, one form field - so one TextChunk per paragraph meant one tiny,
# context-free chunk each (e.g. a chunk that's just "to"). Consecutive
# paragraphs are packed into sections instead: a new section starts at each
# heading, or once the current one reaches SECTION_WORDS.
SECTION_WORDS = 150
_BLANK_LINES = re.compile(r"\n\s*\n+")


def _is_heading(para) -> bool:
    style = para.style.name.lower() if para.style is not None and para.style.name else ""
    return style.startswith(("heading", "title"))


def _table_to_markdown(table: docx.table.Table) -> str:
    rows = [[cell.text.strip() for cell in row.cells] for row in table.rows]
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
    section: list[str] = []
    section_words = 0
    section_start = 0
    for index, para in enumerate(document.paragraphs):
        text = _BLANK_LINES.sub("\n", para.text).strip()
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
        markdown = _table_to_markdown(table)
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
