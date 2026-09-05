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


def extract(file_path: str, filename: str, content_type: str | None = None) -> ExtractionResult:
    suffix = Path(filename).suffix.lower()
    if not suffix and content_type:
        suffix = _MIME_TO_SUFFIX.get(content_type, "")

    if suffix == ".pdf":
        return extract_pdf(file_path)
    if suffix == ".docx":
        return extract_docx(file_path)
    if suffix == ".pptx":
        return extract_pptx(file_path)
    if suffix == ".xlsx":
        return extract_xlsx(file_path)
    if suffix in (".txt", ".md"):
        return text_only([Path(file_path).read_text(encoding="utf-8")])

    raise UnsupportedFileType(
        f"Unsupported file type: {suffix or content_type or 'unknown'}"
    )
