"""Phase 8 check: caption a batch of 20+ images concurrently and measure
throughput (images/minute). Reuses the real extracted chart images from
/tmp/extracted_images, repeated to reach batch size, since the current
corpus doesn't yet have 20+ distinct real charts.
"""

import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.ingestion import CAPTION_CONCURRENCY
from app.vision import describe_image

BATCH_SIZE = 24


async def main() -> None:
    real_images = sorted(Path("/tmp/extracted_images").rglob("*.png"))
    if not real_images:
        print("No extracted images found — run /upload on the fixtures first.")
        return

    batch = [real_images[i % len(real_images)] for i in range(BATCH_SIZE)]
    image_bytes = [p.read_bytes() for p in batch]

    semaphore = asyncio.Semaphore(CAPTION_CONCURRENCY)

    async def captioned(b: bytes):
        async with semaphore:
            return await describe_image(b)

    start = time.monotonic()
    results = await asyncio.gather(*[captioned(b) for b in image_bytes])
    elapsed = time.monotonic() - start

    ok = sum(1 for r in results if r is not None)
    images_per_minute = (ok / elapsed) * 60 if elapsed > 0 else 0

    print(f"captioned {ok}/{len(batch)} images in {elapsed:.1f}s "
          f"(concurrency={CAPTION_CONCURRENCY})")
    print(f"throughput: {images_per_minute:.1f} images/minute")


if __name__ == "__main__":
    asyncio.run(main())
