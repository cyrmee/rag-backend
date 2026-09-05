from dataclasses import dataclass
from typing import Literal

SourceType = Literal["text", "chart_data"]


@dataclass
class TextChunk:
    """A unit of extracted text. `source_type` distinguishes prose (embedded
    and stored as source_type='text') from a structured chart/table data
    extract (source_type='chart_data') per the PPTX/XLSX judgment call:
    prefer exact extracted numbers over a vision model's read of a chart."""

    content: str
    source_type: SourceType = "text"


@dataclass
class ExtractedImage:
    """A bitmap image pulled from (or rendered from) a source document,
    normalized the same way regardless of source format."""

    image_id: str
    image_bytes: bytes
    page_number: int | None = None
    bbox: tuple[float, float, float, float] | None = None


ExtractionResult = tuple[list[TextChunk], list[ExtractedImage]]


def text_only(strings: list[str]) -> ExtractionResult:
    return [TextChunk(content=s) for s in strings if s.strip()], []
