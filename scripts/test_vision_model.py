"""Phase 4 check: send real extracted chart images (pulled from PDF, DOCX,
and the LibreOffice-rendered PPTX/XLSX fallback paths) to qwen3-vl-caption
and manually review whether captions are genuinely specific (axis labels,
numbers, trend) rather than vague ("this is a bar chart").
"""

import asyncio
import base64
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx

from app.config import settings

PROMPT = (
    "Describe this image in detail for someone who cannot see it. If it is "
    "a chart or graph, state the chart type, axis labels, the approximate "
    "or exact value for each data point/category, and the overall trend. "
    "If it contains visible text, transcribe the key text."
)


async def caption(image_bytes: bytes) -> str:
    b64 = base64.b64encode(image_bytes).decode("ascii")
    async with httpx.AsyncClient(base_url=settings.ollama_base_url, timeout=180.0) as client:
        resp = await client.post(
            "/api/generate",
            json={
                "model": settings.vision_model,
                "prompt": PROMPT,
                "images": [b64],
                "stream": False,
            },
        )
        resp.raise_for_status()
        return resp.json()["response"]


async def main() -> None:
    image_paths = sorted(Path("/tmp/extracted_images").rglob("*.png"))
    if not image_paths:
        print("No extracted images found — run /upload on the fixtures first.")
        return

    for path in image_paths:
        print(f"\n=== {path} ===")
        image_bytes = path.read_bytes()
        result = await caption(image_bytes)
        print(result)


if __name__ == "__main__":
    asyncio.run(main())
