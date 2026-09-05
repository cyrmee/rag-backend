"""Phase 10 check: the agent's describe_image tool-call handler actually
works end-to-end - given a real image_path (as would appear in a retrieved
chunk's [image_path=...] tag), it re-captions the image fresh. Also checks
the graceful-failure paths (missing path, malformed call).
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from app.agent import _handle_describe_image_call, _retrieve_for_agent
from app.db import close_pool, open_pool


async def main() -> None:
    await open_pool()
    try:
        # 1. Retrieve should tag an image_caption chunk with its path.
        _, tagged = await _retrieve_for_agent("Kestrel revenue chart bar values by quarter", top_k=5)
        tagged_with_image = [t for t in tagged if t.startswith("[image_path=")]
        assert tagged_with_image, "expected at least one retrieved chunk tagged with an image_path"
        sample = tagged_with_image[0]
        image_path = sample.split("[image_path=", 1)[1].split("]", 1)[0]
        print("found tagged chunk, image_path =", image_path)

        # 2. A real describe_image call against that path should succeed.
        call = {"function": {"name": "describe_image", "arguments": {"image_path": image_path}}}
        result = await _handle_describe_image_call(call, iteration=0)
        assert not result.startswith("Error:"), f"expected a fresh caption, got: {result}"
        print("OK real describe_image call succeeded:", result[:150], "...")

        # 3. Missing path should fail gracefully, not raise.
        bad_call = {"function": {"name": "describe_image", "arguments": {"image_path": "/nonexistent/x.png"}}}
        bad_result = await _handle_describe_image_call(bad_call, iteration=0)
        assert bad_result.startswith("Error:"), "expected a graceful error for a missing file"
        print("OK missing-path case handled gracefully:", bad_result)

        # 4. Malformed call (no image_path) should fail gracefully.
        malformed_call = {"function": {"name": "describe_image", "arguments": {}}}
        malformed_result = await _handle_describe_image_call(malformed_call, iteration=0)
        assert malformed_result.startswith("Error:"), "expected a graceful error for a malformed call"
        print("OK malformed-call case handled gracefully:", malformed_result)
    finally:
        await close_pool()


if __name__ == "__main__":
    asyncio.run(main())
