"""Phase 9 check: chart/table-specific queries should surface the right
row type (image_caption or chart_data, not just body text), with enough
metadata (source_image_path/page_number/source_format) to trace back to
the original figure.
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from pgvector import Vector

from app.config import settings
from app.db import close_pool, get_connection, open_pool
from app.embeddings import embed_text

# One query per format, phrased so it's only answerable from chart/table
# content (not surrounding body prose).
CASES = [
    ("pdf", "Kestrel revenue chart bar values by quarter"),
    ("docx", "Kestrel revenue chart bar values by quarter"),
    ("pptx", "Kestrel revenue chart data by quarter"),
    ("xlsx", "Kestrel revenue chart bar values by quarter"),
]


async def top_match(query: str, source_format: str):
    query_vector = Vector(await embed_text(query))
    async with get_connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                """
                select content, source_type, source_format, source_image_path, page_number
                from documents
                where source_format = %s
                order by embedding <=> %s
                limit 1
                """,
                (source_format, query_vector),
            )
            return await cur.fetchone()


async def main() -> None:
    await open_pool()
    try:
        all_ok = True
        for source_format, query in CASES:
            row = await top_match(query, source_format)
            if row is None:
                print(f"[{source_format}] NO MATCH for {query!r}")
                all_ok = False
                continue

            content, source_type, fmt, image_path, page_number = row
            traceable = source_type == "chart_data" or (source_type == "image_caption" and image_path)
            status = "OK" if source_type in ("chart_data", "image_caption") and traceable else "FAIL"
            all_ok = all_ok and status == "OK"

            print(f"[{source_format}] query={query!r}")
            print(f"  -> source_type={source_type} page={page_number} image_path={image_path}")
            print(f"  -> content preview: {content[:100]!r}")
            print(f"  -> {status}")

        print("\nALL OK" if all_ok else "\nSOME CASES FAILED")
    finally:
        await close_pool()


if __name__ == "__main__":
    asyncio.run(main())
