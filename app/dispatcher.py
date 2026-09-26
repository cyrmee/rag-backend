from pathlib import Path

from app.extractors.docx_extractor import extract_docx
from app.extractors.pdf_extractor import extract_pdf
from app.extractors.pptx_extractor import extract_pptx
from app.extractors.types import ExtractionResult, text_only
from app.extractors.xlsx_extractor import extract_xlsx
from app.parsing import UnsupportedFileType

_MIME_TO_SUFFIX = {
    "application/pdf": ".pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ".docx",
    "application/vnd.openxmlformats-officedocument.presentationml.presentation": ".pptx",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": ".xlsx",
}


def _read_text_file(file_path: str) -> str:
    # Plenty of older Windows-authored .txt files are cp1252, not UTF-8
    # (e.g. a \x96 en dash) - cp1252 decodes nearly any byte, so it's a
    # safe second attempt before giving up.
    raw = Path(file_path).read_bytes()
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return raw.decode("cp1252", errors="replace")


def _extract_raw(file_path: str, suffix: str) -> ExtractionResult:
    if suffix == ".pdf":
        return extract_pdf(file_path)
    if suffix == ".docx":
        return extract_docx(file_path)
    if suffix == ".pptx":
        return extract_pptx(file_path)
    if suffix == ".xlsx":
        return extract_xlsx(file_path)
    if suffix in (".txt", ".md"):
        return text_only([_read_text_file(file_path)])
    raise UnsupportedFileType(f"Unsupported file type: {suffix or 'unknown'}")


def extract(file_path: str, filename: str, content_type: str | None = None) -> ExtractionResult:
    suffix = Path(filename).suffix.lower()
    if not suffix and content_type:
        suffix = _MIME_TO_SUFFIX.get(content_type, "")
    if not suffix:
        raise UnsupportedFileType(f"Unsupported file type: {content_type or 'unknown'}")

    chunks, images = _extract_raw(file_path, suffix)
    # Some PDFs' text layers contain NUL bytes, which Postgres text columns
    # reject outright - stripped here so every consumer gets clean text.
    for chunk in chunks:
        chunk.content = chunk.content.replace("\x00", "")
    return chunks, images
