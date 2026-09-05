"""Phase 2 check: retrieve() in app/agent.py returns the same kind of
results as the existing /ask route's retrieval step."""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from app.agent import retrieve
from app.db import close_pool, open_pool


async def main() -> None:
    await open_pool()
    try:
        results = await retrieve("shard-splitting cap incident")
        print(f"got {len(results)} results")
        for r in results:
            print("-", r[:100])
        assert results, "expected at least one result"
    finally:
        await close_pool()


if __name__ == "__main__":
    asyncio.run(main())
