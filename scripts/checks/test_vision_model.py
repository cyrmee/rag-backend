"""Phase 4 check: send real extracted chart images (pulled from PDF, DOCX,
and the LibreOffice-rendered PPTX/XLSX fallback paths) to qwen3-vl-caption
and manually review whether captions are genuinely specific (axis labels,
numbers, trend) rather than vague ("this is a bar chart").
"""

import asyncio
import base64
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import httpx

from app.config import settings
from app.storage import get_image_bytes, list_image_keys

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
    keys = sorted(await list_image_keys())
    if not keys:
        print("No extracted images found in MinIO — run /upload on the fixtures first.")
        return

    for key in keys:
        print(f"\n=== {key} ===")
        image_bytes = await get_image_bytes(key)
        result = await caption(image_bytes)
        print(result)


if __name__ == "__main__":
    asyncio.run(main())
