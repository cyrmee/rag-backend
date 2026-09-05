import subprocess
import tempfile
import uuid
from pathlib import Path

import openpyxl

from app.extractors.types import ExtractedImage, ExtractionResult, TextChunk


def _sheet_to_markdown(ws) -> str | None:
    rows = [[("" if c.value is None else str(c.value)) for c in row] for row in ws.iter_rows()]
    rows = [r for r in rows if any(cell.strip() for cell in r)]
    if not rows:
        return None

    header = "| " + " | ".join(rows[0]) + " |"
    sep = "| " + " | ".join("---" for _ in rows[0]) + " |"
    body = ["| " + " | ".join(r) + " |" for r in rows[1:]]
    return "\n".join([header, sep] + body)


def _chart_data_to_markdown(ws, chart) -> str | None:
    """Resolve a chart's referenced cell ranges back to real values and
    build a markdown table, preferring exact numbers over a rendered
    screenshot (same judgment call as PPTX)."""
    try:
        series_tables = []
        categories: list[str] | None = None

        for series in chart.series:
            if series.cat is not None and categories is None:
                ref = series.cat.numRef or series.cat.strRef
                if ref is not None:
                    categories = [pt.v for pt in ref.numCache.pt] if ref.numCache else \
                        [pt.v for pt in ref.strCache.pt]

            if series.val is not None and series.val.numRef and series.val.numRef.numCache:
                name = series.tx.strRef.strCache.pt[0].v if series.tx and series.tx.strRef else "Series"
                values = [pt.v for pt in series.val.numRef.numCache.pt]
                series_tables.append((name, values))

        if not categories or not series_tables:
            return None

        header = "| Category | " + " | ".join(name for name, _ in series_tables) + " |"
        sep = "| --- | " + " | ".join("---" for _ in series_tables) + " |"
        rows = []
        for i, cat in enumerate(categories):
            values = [str(vals[i]) if i < len(vals) else "" for _, vals in series_tables]
            rows.append(f"| {cat} | " + " | ".join(values) + " |")
        return "\n".join([header, sep] + rows)
    except Exception:
        return None


def _render_sheet_images(file_path: str) -> list[bytes]:
    """Fallback when chart data can't be resolved from the workbook:
    render the workbook to images via LibreOffice so the vision model
    sees the chart as it visually appears."""
    with tempfile.TemporaryDirectory() as tmp:
        # Force each sheet onto a single page before rendering - otherwise
        # LibreOffice paginates by print area and can split a chart across
        # two rendered pages, cropping it.
        fit_to_page_path = Path(tmp) / Path(file_path).name
        wb = openpyxl.load_workbook(file_path)
        for ws in wb.worksheets:
            ws.page_setup.fitToWidth = 1
            ws.page_setup.fitToHeight = 1
            ws.sheet_properties.pageSetUpPr.fitToPage = True
        wb.save(fit_to_page_path)

        subprocess.run(
            [
                "libreoffice", "--headless", "--convert-to", "pdf",
                "--outdir", tmp, str(fit_to_page_path),
            ],
            check=True, capture_output=True, timeout=120,
        )
        converted = next(p for p in Path(tmp).glob("*.pdf"))

        import pymupdf as fitz
        doc = fitz.open(str(converted))
        try:
            return [page.get_pixmap(dpi=150).tobytes("png") for page in doc]
        finally:
            doc.close()


def extract_xlsx(file_path: str) -> ExtractionResult:
    wb = openpyxl.load_workbook(file_path, data_only=True)
    text_chunks: list[TextChunk] = []
    needs_render = False

    for ws in wb.worksheets:
        markdown = _sheet_to_markdown(ws)
        if markdown:
            text_chunks.append(TextChunk(content=f"Sheet: {ws.title}\n{markdown}"))

        for chart in getattr(ws, "_charts", []):
            chart_markdown = _chart_data_to_markdown(ws, chart)
            if chart_markdown:
                text_chunks.append(TextChunk(content=chart_markdown, source_type="chart_data"))
            else:
                needs_render = True

    images: list[ExtractedImage] = []
    if needs_render:
        for page_number, image_bytes in enumerate(_render_sheet_images(file_path), start=1):
            images.append(
                ExtractedImage(
                    image_id=str(uuid.uuid4()),
                    image_bytes=image_bytes,
                    page_number=page_number,
                    bbox=None,
                )
            )

    return text_chunks, images
