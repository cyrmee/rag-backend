import asyncio
import logging

import httpx

from app.config import settings
from app.vision import BACKOFF_SECONDS, MAX_RETRIES

logger = logging.getLogger(__name__)


async def embed_text(text: str) -> list[float]:
    """Embeds `text` via Ollama, retrying transient failures the same way
    describe_image() does. Unlike a skippable image caption, there's no
    fallback embedding to substitute, so exhausting retries raises rather
    than returning None - a failed chunk embedding should surface as a
    real ingestion failure for that document."""
    last_exc: httpx.HTTPError | None = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            async with httpx.AsyncClient(base_url=settings.ollama_base_url, timeout=60.0) as client:
                resp = await client.post(
                    "/api/embed",
                    json={
                        "model": settings.embed_model,
                        "input": text,
                        "dimensions": settings.embed_dim,
                    },
                )
                resp.raise_for_status()
                data = resp.json()
                return data["embeddings"][0]
        except httpx.HTTPError as exc:
            last_exc = exc
            logger.warning(
                "embed_text attempt %d/%d failed: %s", attempt, MAX_RETRIES, exc
            )
            if attempt < MAX_RETRIES:
                await asyncio.sleep(BACKOFF_SECONDS * attempt)

    logger.error("embed_text: giving up after %d attempts", MAX_RETRIES)
    raise last_exc
