import asyncio
import base64
import logging

import httpx

from app.config import settings

logger = logging.getLogger(__name__)

CAPTION_PROMPT = (
    "Describe this image in detail for someone who cannot see it. If it is "
    "a chart or graph, state the chart type, axis labels, the approximate "
    "or exact value for each data point/category, and the overall trend. "
    "If it contains visible text, transcribe the key text."
)

MAX_RETRIES = 3
BACKOFF_SECONDS = 2.0


async def describe_image(image_bytes: bytes) -> str | None:
    """Captions an image via the vision model. Returns None (and logs a
    warning) rather than raising after exhausting retries - a single bad
    image shouldn't crash an entire ingestion run."""
    b64 = base64.b64encode(image_bytes).decode("ascii")

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            async with httpx.AsyncClient(base_url=settings.ollama_base_url, timeout=180.0) as client:
                resp = await client.post(
                    "/api/generate",
                    json={
                        "model": settings.vision_model,
                        "prompt": CAPTION_PROMPT,
                        "images": [b64],
                        "stream": False,
                    },
                )
                resp.raise_for_status()
                return resp.json()["response"]
        except httpx.HTTPError as exc:
            logger.warning(
                "describe_image attempt %d/%d failed: %s", attempt, MAX_RETRIES, exc
            )
            if attempt < MAX_RETRIES:
                await asyncio.sleep(BACKOFF_SECONDS * attempt)

    logger.error("describe_image: giving up after %d attempts", MAX_RETRIES)
    return None
