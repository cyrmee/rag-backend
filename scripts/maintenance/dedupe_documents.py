"""One-time cleanup: remove duplicate rows from `documents` left over from
before /upload was made idempotent. Keeps one row per
(filename, chunk_index, content), preferring the lowest ctid.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import psycopg

from app.config import settings

DEDUPE_SQL = """
delete from documents a
using documents b
where a.ctid > b.ctid
  and a.filename = b.filename
  and a.chunk_index = b.chunk_index
  and a.content = b.content
"""


def main() -> None:
    conn = psycopg.connect(settings.database_url)
    try:
        with conn.cursor() as cur:
            cur.execute(DEDUPE_SQL)
            print("duplicate rows deleted:", cur.rowcount)
        conn.commit()
    finally:
        conn.close()


if __name__ == "__main__":
    main()
