import asyncio
import base64
import hashlib
import io
import logging

import httpx
from PIL import Image, UnidentifiedImageError

from app.config import settings

logger = logging.getLogger(__name__)

# A caption is search-index text, not a description for someone who can't
# see the image: enough to find the image for a question about what it
# shows, with its key text and values transcribed. The first prompt asked
# for detail and got 1,700 characters per image on average - about 1,200
# generated tokens, 45 s each on the 33B model, one image at a time.
CAPTION_PROMPT = (
    "Describe this image for a search index in at most 100 words. Say what "
    "it shows; for a chart, table or diagram give its type, labels and the "
    "key values; transcribe the important visible text exactly. No preamble."
)
CAPTION_MAX_TOKENS = 220

# Qwen3-VL's chat template with the assistant turn opened on an empty
# think block - how Qwen3 models are switched out of thinking. The Ollama
# tags this app uses (qwen3-vl, qwen3-vl-caption) carry a thinking
# renderer that ignores the API's "think": false, and left to think the
# model spends 300-1,000 tokens reasoning before every caption. Sent as a
# raw prompt, so Ollama applies no template of its own; [img-0] is where
# it places the image. settings.vision_no_think_prefill off sends the
# plain prompt, for a vision model with another template.
_NO_THINK_PROMPT = (
    "<|im_start|>user\n[img-0]\n{prompt}<|im_end|>\n"
    "<|im_start|>assistant\n<think>\n\n</think>\n\n"
)

# The vision model's Ollama runner has one execution slot, so captions are
# serialized here, process-wide, for every caller - ingestion, the caption
# backfill and the agent's describe_image tool. A second request in flight
# would only queue server-side, and time out while it waits (which is what
# happened with one semaphore per document and four documents in flight).
CAPTION_CONCURRENCY = 1
_caption_slots = asyncio.Semaphore(CAPTION_CONCURRENCY)

MAX_RETRIES = 3
BACKOFF_SECONDS = 2.0

# An image smaller than this on either side is an icon, a bullet, a rule
# or a letterhead logo - a third of the images in a sample of the ID Drive
# - and a caption of "a small red square" helps no search.
MIN_IMAGE_SIDE = 120

# What Ollama's image decoder takes. Anything else PIL can open (BMP,
# TIFF, GIF) is converted to PNG; what it can't (EMF/WMF drawings from
# Word, JBIG2 scans) is left out rather than sent to fail three times.
_NATIVE_FORMATS = {"PNG", "JPEG", "WEBP"}


def prepare_image(image_bytes: bytes) -> bytes | None:
    """The bytes to caption and store for an extracted image, or None if
    it isn't worth captioning (too small) or can't be (a format the
    vision model doesn't take and PIL can't convert). Only the header is
    read unless a conversion is needed."""
    try:
        with Image.open(io.BytesIO(image_bytes)) as image:
            width, height = image.size
            if width < MIN_IMAGE_SIDE or height < MIN_IMAGE_SIDE:
                return None
            if image.format in _NATIVE_FORMATS:
                return image_bytes
            if image.mode not in ("1", "L", "LA", "P", "RGB", "RGBA"):
                image = image.convert("RGB")
            out = io.BytesIO()
            image.save(out, "PNG")
            return out.getvalue()
    except (UnidentifiedImageError, OSError, ValueError):
        return None


def image_hash(image_bytes: bytes) -> str:
    """Key of the caption cache (image_captions.sha256)."""
    return hashlib.sha256(image_bytes).hexdigest()


def _payload(image_bytes: bytes) -> dict:
    payload = {
        "model": settings.vision_model,
        "images": [base64.b64encode(image_bytes).decode("ascii")],
        "stream": False,
        # Low temperature: the same image should get much the same caption
        # whichever document it turns up in.
        "options": {"num_predict": CAPTION_MAX_TOKENS, "temperature": 0.2},
    }
    if settings.vision_no_think_prefill:
        payload.update(prompt=_NO_THINK_PROMPT.format(prompt=CAPTION_PROMPT), raw=True)
    else:
        payload["prompt"] = CAPTION_PROMPT
    return payload


async def describe_image(image_bytes: bytes) -> str | None:
    """Captions an image via the vision model. Returns None (and logs a
    warning) rather than raising after exhausting retries - a single bad
    image shouldn't crash an entire ingestion run."""
    payload = _payload(image_bytes)

    async with _caption_slots:
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                async with httpx.AsyncClient(base_url=settings.ollama_base_url, timeout=180.0) as client:
                    resp = await client.post("/api/generate", json=payload)
                    resp.raise_for_status()
                    caption = resp.json()["response"].strip()
                if caption:
                    return caption
                logger.warning("describe_image attempt %d/%d: empty caption", attempt, MAX_RETRIES)
            except httpx.HTTPError as exc:
                logger.warning(
                    "describe_image attempt %d/%d failed: %s", attempt, MAX_RETRIES, exc or type(exc).__name__
                )
            if attempt < MAX_RETRIES:
                await asyncio.sleep(BACKOFF_SECONDS * attempt)

    logger.error("describe_image: giving up after %d attempts", MAX_RETRIES)
    return None
