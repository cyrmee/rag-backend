"""Phase 5 check: describe_image() works normally, and survives (rather
than crashing on) a simulated network failure via retry/backoff."""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx

import app.vision as vision_module
from app.vision import describe_image

TEST_IMAGE = next(Path("/tmp/extracted_images").rglob("*.png"))


async def test_normal() -> None:
    caption = await describe_image(TEST_IMAGE.read_bytes())
    assert caption, "expected a non-empty caption"
    print("OK normal case, caption:", caption[:120], "...")


async def test_network_blip() -> None:
    call_count = 0
    real_post = httpx.AsyncClient.post

    async def flaky_post(self, *args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count < 2:
            raise httpx.ConnectError("simulated network blip")
        return await real_post(self, *args, **kwargs)

    vision_module.BACKOFF_SECONDS = 0.1  # speed up the test
    httpx.AsyncClient.post = flaky_post
    try:
        caption = await describe_image(TEST_IMAGE.read_bytes())
        assert caption, "expected describe_image to recover after retry"
        print(f"OK survived simulated blip after {call_count} attempt(s), no crash")
    finally:
        httpx.AsyncClient.post = real_post


async def test_permanent_failure() -> None:
    async def always_fail(self, *args, **kwargs):
        raise httpx.ConnectError("simulated permanent outage")

    real_post = httpx.AsyncClient.post
    httpx.AsyncClient.post = always_fail
    try:
        caption = await describe_image(TEST_IMAGE.read_bytes())
        assert caption is None, "expected None after exhausting retries, not a crash"
        print("OK permanent failure returns None instead of raising")
    finally:
        httpx.AsyncClient.post = real_post


async def main() -> None:
    await test_normal()
    await test_network_blip()
    await test_permanent_failure()


if __name__ == "__main__":
    asyncio.run(main())
