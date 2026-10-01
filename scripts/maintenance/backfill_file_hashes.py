"""Fills document_files (sql/009_document_files.sql) for documents ingested
before file hashing existed: downloads each document's original from MinIO,
hashes it, and records it. Files with identical bytes under different names
are reported as duplicate groups; only one name per group gets a hash row
(the shortest filename - "X.docx" over "Copy of X.docx" or "X (1).docx" -
then the earliest ingested).

With --delete, the other copies in each group are removed from search
(their chunks, summary, and hash row - the same as DELETE /documents). The
original files stay in MinIO.

Usage:
    python scripts/maintenance/backfill_file_hashes.py           # record hashes, report duplicates
    python scripts/maintenance/backfill_file_hashes.py --delete  # also remove the duplicate copies
"""

import argparse
import asyncio
import hashlib
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from app.db import close_pool, get_connection, open_pool
from app.storage import _download_sync, document_key_for


async def _hash_original(filename: str) -> str | None:
    try:
        data = await asyncio.to_thread(_download_sync, document_key_for(filename))
    except Exception:
        return None
    return hashlib.sha256(data).hexdigest()


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--delete", action="store_true", help="Remove the duplicate copies from search")
    args = parser.parse_args()

    await open_pool()
    try:
        async with get_connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "select filename, min(created_at) from documents group by filename order by filename"
                )
                documents = await cur.fetchall()
                await cur.execute("select filename, sha256 from document_files")
                recorded = dict(await cur.fetchall())

        by_hash: dict[str, list[tuple[str, object]]] = defaultdict(list)
        missing: list[str] = []
        for filename, first_ingested in documents:
            file_hash = recorded.get(filename) or await _hash_original(filename)
            if file_hash is None:
                missing.append(filename)
            else:
                by_hash[file_hash].append((filename, first_ingested))

        duplicates: list[tuple[str, list[str]]] = []
        async with get_connection() as conn:
            async with conn.cursor() as cur:
                for file_hash, files in by_hash.items():
                    holder = next((f for f, _ in files if recorded.get(f) == file_hash), None)
                    keep = holder or min(files, key=lambda f: (len(f[0]), f[1]))[0]
                    if keep not in recorded:
                        await cur.execute(
                            "insert into document_files (filename, sha256) values (%s, %s)", (keep, file_hash),
                        )
                    others = [f for f, _ in files if f != keep]
                    if others:
                        duplicates.append((keep, others))
                    if args.delete:
                        for other in others:
                            await cur.execute("delete from documents where filename = %s", (other,))
                            await cur.execute("delete from document_summaries where filename = %s", (other,))
                            await cur.execute("delete from document_files where filename = %s", (other,))
            await conn.commit()

        print(f"{len(documents)} documents, {len(by_hash)} distinct files")
        if missing:
            print(f"\n{len(missing)} without an original in storage (not hashed):")
            for f in missing:
                print(f"  {f}")
        if duplicates:
            action = "removed" if args.delete else "would be removed with --delete"
            print(f"\n{len(duplicates)} duplicate groups (kept -> copies {action}):")
            for keep, others in duplicates:
                print(f"  {keep}")
                for other in others:
                    print(f"    = {other}")
        else:
            print("\nno duplicates")
    finally:
        await close_pool()


if __name__ == "__main__":
    asyncio.run(main())
