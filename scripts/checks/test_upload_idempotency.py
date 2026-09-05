"""Regression test: uploading the same file twice via /upload must not
duplicate its chunks in `documents`, and /ask must not return duplicate
sources for it.

Requires the API server to be running (default http://127.0.0.1:8001).
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import httpx
import psycopg

from app.config import settings

BASE_URL = "http://127.0.0.1:8001"
SAMPLE_PATH = Path(__file__).resolve().parent.parent / "fixtures" / "sample.txt"
FILENAME = "sample.txt"


async def upload_once(client: httpx.AsyncClient) -> int:
    with open(SAMPLE_PATH, "rb") as f:
        resp = await client.post(
            "/upload",
            files={"file": (FILENAME, f, "text/plain")},
        )
    resp.raise_for_status()
    return resp.json()["chunks_ingested"]


async def main() -> None:
    async with httpx.AsyncClient(base_url=BASE_URL, timeout=60.0) as client:
        first_count = await upload_once(client)
        second_count = await upload_once(client)

        assert first_count == second_count, (
            f"chunk count changed between uploads: {first_count} vs {second_count}"
        )

        conn = psycopg.connect(settings.database_url)
        with conn.cursor() as cur:
            cur.execute(
                "select count(*) from documents where filename = %s", (FILENAME,)
            )
            row_count = cur.fetchone()[0]
        conn.close()

        assert row_count == first_count, (
            f"expected {first_count} rows for {FILENAME}, found {row_count} "
            "(duplicate rows were not replaced)"
        )
        print(f"OK: {row_count} rows for {FILENAME} after two uploads (no duplication)")

        ask_resp = await client.post(
            "/ask", json={"question": "What is Project Zephyr?"}
        )
        ask_resp.raise_for_status()
        sources = ask_resp.json()["sources"]
        assert len(sources) == len(set(sources)), (
            "duplicate entries found in /ask sources"
        )
        print(f"OK: {len(sources)} sources returned from /ask, all unique")


if __name__ == "__main__":
    asyncio.run(main())
