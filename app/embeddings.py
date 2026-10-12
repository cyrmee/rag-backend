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


class ZeroEmbedding(Exception):
    """The embedding server returned an all-zero vector."""


async def _embed_batch(texts: list[str], allow_zero: bool = False) -> list[list[float]]:
    """Embeds one batch via Ollama, retrying transient failures the same way
    describe_image() does. Unlike a skippable image caption, there's no
    fallback embedding to substitute, so exhausting retries raises rather
    than returning None - a failed chunk embedding should surface as a
    real ingestion failure for that document. `allow_zero` returns the
    zero vectors instead when they persist, for callers that can drop
    the affected texts (the graph's phrases)."""
    last_exc: Exception | None = None

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
                # Ollama now and then answers with an all-zero vector for
                # an input it embeds fine a moment later (seen with short
                # phrases, October 2026). A zero vector has no direction -
                # every cosine distance to it is NaN - so it's a failed
                # attempt, not an embedding.
                zeros = [i for i, e in enumerate(embeddings) if not any(e)]
                if zeros and not (allow_zero and attempt == MAX_RETRIES):
                    raise ZeroEmbedding(f"{len(zeros)} of {len(texts)} embeddings came back all zero")
                if zeros:
                    logger.warning("embed: %d of %d texts stay all zero, e.g. %r", len(zeros), len(texts), texts[zeros[0]])
                return embeddings
        except (httpx.HTTPError, ZeroEmbedding) as exc:
            last_exc = exc
            logger.warning(
                "embed attempt %d/%d failed (batch of %d): %s", attempt, MAX_RETRIES, len(texts), exc
            )
            if attempt < MAX_RETRIES:
                await asyncio.sleep(BACKOFF_SECONDS * attempt)

    logger.error("embed: giving up after %d attempts", MAX_RETRIES)
    raise last_exc


async def embed_texts(texts: list[str], allow_zero: bool = False) -> list[list[float]]:
    """Embeds `texts` in order, EMBED_BATCH_SIZE per request."""
    embeddings: list[list[float]] = []
    for start in range(0, len(texts), EMBED_BATCH_SIZE):
        embeddings.extend(await _embed_batch(texts[start : start + EMBED_BATCH_SIZE], allow_zero))
    return embeddings


async def embed_text(text: str) -> list[float]:
    return (await _embed_batch([text]))[0]
