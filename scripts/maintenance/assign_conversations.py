"""Gives every conversation without an owner - those from before sign-in
was added (sql/011_users.sql) - to one user, so they show up in that
user's history again. The user must have signed in once, so their row
exists in `users`.

    python scripts/maintenance/assign_conversations.py alice
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import psycopg

from app.config import settings


def main(username: str) -> None:
    with psycopg.connect(settings.database_url) as conn, conn.cursor() as cur:
        cur.execute("select id from users where username = %s", (username,))
        row = cur.fetchone()
        if row is None:
            sys.exit(f"No user '{username}' yet - they need to sign in to the app once first.")
        cur.execute("update conversations set owner_id = %s where owner_id is null", (row[0],))
        print(f"conversations assigned to {username}:", cur.rowcount)
        conn.commit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("username", help="Keycloak username, as shown in the users table")
    main(parser.parse_args().username)
