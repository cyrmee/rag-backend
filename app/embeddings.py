import asyncio
import logging

import httpx

from app.config import settings
from app.vision import BACKOFF_SECONDS, MAX_RETRIES

logger = logging.getLogger(__name__)

# Ollama's /api/embed takes a list of inputs and embeds them in one request,
# so ingestion sends chunks in batches instead of one round-trip per chunk -
# per-chunk calls made a sheet with tens of thousands of rows take hours.
# Kept moderate so one request stays well inside the timeout below.
EMBED_BATCH_SIZE = 32


async def _embed_batch(texts: list[str]) -> list[list[float]]:
    """Embeds one batch via Ollama, retrying transient failures the same way
    describe_image() does. Unlike a skippable image caption, there's no
    fallback embedding to substitute, so exhausting retries raises rather
    than returning None - a failed chunk embedding should surface as a
    real ingestion failure for that document."""
    last_exc: httpx.HTTPError | None = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            async with httpx.AsyncClient(base_url=settings.ollama_base_url, timeout=300.0) as client:
                resp = await client.post(
                    "/api/embed",
                    json={
                        "model": settings.embed_model,
                        "input": texts,
                        "dimensions": settings.embed_dim,
                    },
                )
                resp.raise_for_status()
                embeddings = resp.json()["embeddings"]
                if len(embeddings) != len(texts):
                    raise ValueError(f"asked for {len(texts)} embeddings, got {len(embeddings)}")
                return embeddings
        except httpx.HTTPError as exc:
            last_exc = exc
            logger.warning(
                "embed attempt %d/%d failed (batch of %d): %s", attempt, MAX_RETRIES, len(texts), exc
            )
            if attempt < MAX_RETRIES:
                await asyncio.sleep(BACKOFF_SECONDS * attempt)

    logger.error("embed: giving up after %d attempts", MAX_RETRIES)
    raise last_exc


async def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embeds `texts` in order, EMBED_BATCH_SIZE per request."""
    embeddings: list[list[float]] = []
    for start in range(0, len(texts), EMBED_BATCH_SIZE):
        embeddings.extend(await _embed_batch(texts[start : start + EMBED_BATCH_SIZE]))
    return embeddings


async def embed_text(text: str) -> list[float]:
    return (await _embed_batch([text]))[0]
