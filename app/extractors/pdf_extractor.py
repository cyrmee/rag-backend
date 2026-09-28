import uuid

import pymupdf as fitz

from app.extractors.types import ExtractedImage, ExtractionResult, TextChunk
from app.parsing import UnsupportedFileType


def extract_pdf(file_path: str) -> ExtractionResult:
    text_chunks: list[TextChunk] = []
    images: list[ExtractedImage] = []

    try:
        doc = fitz.open(file_path)
    except fitz.FileDataError as exc:
        raise UnsupportedFileType("PDF is damaged or isn't really a PDF, so it can't be read") from exc
    # Owner-password-only PDFs (copy/print restrictions) open with an empty
    # password; ones that need a user password to view can't be read at all,
    # and pymupdf would otherwise fail later with "document closed or encrypted".
    if doc.needs_pass and not doc.authenticate(""):
        doc.close()
        raise UnsupportedFileType("PDF is password-protected and can't be read without its password")
    try:
        for page_index, page in enumerate(doc):
            page_text = page.get_text()
            if page_text.strip():
                text_chunks.append(TextChunk(content=page_text, page_number=page_index + 1))

            for img in page.get_images(full=True):
                xref = img[0]
                try:
                    bbox = page.get_image_bbox(img)
                    bbox_tuple = (bbox.x0, bbox.y0, bbox.x1, bbox.y1)
                except ValueError:
                    bbox_tuple = None

                base_image = doc.extract_image(xref)
                images.append(
                    ExtractedImage(
                        image_id=str(uuid.uuid4()),
                        image_bytes=base_image["image"],
                        page_number=page_index + 1,
                        bbox=bbox_tuple,
                    )
                )
    finally:
        doc.close()

    return text_chunks, images
