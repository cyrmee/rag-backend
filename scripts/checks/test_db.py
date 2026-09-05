import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import psycopg
from pgvector import Vector
from pgvector.psycopg import register_vector

from app.config import settings


def main() -> None:
    conn = psycopg.connect(settings.database_url)
    register_vector(conn)

    vec = Vector([random.random() for _ in range(settings.embed_dim)])

    with conn.cursor() as cur:
        cur.execute(
            """
            insert into documents (filename, chunk_index, content, embedding)
            values (%s, %s, %s, %s)
            returning id
            """,
            ("test.txt", 0, "hello world test chunk", vec),
        )
        inserted_id = cur.fetchone()[0]
        conn.commit()

        cur.execute(
            """
            select id, filename, content, embedding <=> %s as distance
            from documents
            order by embedding <=> %s
            limit 1
            """,
            (vec, vec),
        )
        row = cur.fetchone()
        print("inserted id:", inserted_id)
        print("retrieved row:", row)

    conn.close()


if __name__ == "__main__":
    main()
