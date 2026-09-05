"""Phase 8 check: caption a batch of 20+ images concurrently and measure
throughput (images/minute). Reuses the real extracted chart images stored
in MinIO, repeated to reach batch size, since the current corpus doesn't
yet have 20+ distinct real charts.
"""

import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from app.ingestion import CAPTION_CONCURRENCY
from app.storage import get_image_bytes, list_image_keys
from app.vision import describe_image

BATCH_SIZE = 24


async def main() -> None:
    keys = sorted(await list_image_keys())
    if not keys:
        print("No extracted images found in MinIO — run /upload on the fixtures first.")
        return

    batch_keys = [keys[i % len(keys)] for i in range(BATCH_SIZE)]
    image_bytes = [await get_image_bytes(k) for k in batch_keys]

    semaphore = asyncio.Semaphore(CAPTION_CONCURRENCY)

    async def captioned(b: bytes):
        async with semaphore:
            return await describe_image(b)

    start = time.monotonic()
    results = await asyncio.gather(*[captioned(b) for b in image_bytes])
    elapsed = time.monotonic() - start

    ok = sum(1 for r in results if r is not None)
    images_per_minute = (ok / elapsed) * 60 if elapsed > 0 else 0

    print(f"captioned {ok}/{len(batch_keys)} images in {elapsed:.1f}s "
          f"(concurrency={CAPTION_CONCURRENCY})")
    print(f"throughput: {images_per_minute:.1f} images/minute")


if __name__ == "__main__":
    asyncio.run(main())
