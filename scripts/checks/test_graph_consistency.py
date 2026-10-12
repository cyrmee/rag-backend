"""Knowledge graph (app/graph.py) end to end, against the live API and
database: upload a fixture with GRAPH_RETRIEVAL on -> its chunks have
facts and phrases -> re-upload under the same name leaves one copy of
everything -> graph_search finds the fixture's chunk for a question only
it answers -> delete removes its facts and the phrases only it used.
Needs the API server (default http://127.0.0.1:8001) with GRAPH_RETRIEVAL
on, and the DB; takes a minute or two (two extraction calls per chunk).

    python scripts/checks/test_graph_consistency.py
    python scripts/checks/test_graph_consistency.py --base-url http://10.0.0.5:8001
"""

import argparse
import asyncio
import sys
from pathlib import Path

import httpx
from keycloak_token import KeycloakAuth

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from app.db import close_pool, get_connection, open_pool
from app.embeddings import embed_text
from app.graph import graph_search

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"
UPLOAD_NAME = "zz-graph-kestrel.docx"
DOCX = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"

failures: list[str] = []


def check(ok: bool, name: str, detail: str = "") -> bool:
    print(f"{'PASS' if ok else 'FAIL'} {name}{f' - {detail}' if detail and not ok else ''}", flush=True)
    if not ok:
        failures.append(name)
    return ok


async def _graph_rows(filename: str) -> tuple[int, int, int]:
    """(chunks extracted, chunk-fact links, distinct phrases) of a document."""
    async with get_connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                """
                select (select count(*) from documents where filename = %(f)s and graph_extracted_at is not null),
                       (select count(*) from graph_triples t join documents d on d.id = t.chunk_id where d.filename = %(f)s),
                       (select count(distinct p.id) from graph_phrases p
                          join graph_facts f on f.subject_id = p.id or f.object_id = p.id
                          join graph_triples t on t.fact_id = f.id
                          join documents d on d.id = t.chunk_id where d.filename = %(f)s)
                """,
                {"f": filename},
            )
            return await cur.fetchone()


async def _orphans() -> tuple[int, int]:
    """Facts no chunk states, phrases no fact uses - should always be 0."""
    async with get_connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                """
                select (select count(*) from graph_facts f
                         where not exists (select 1 from graph_triples t where t.fact_id = f.id)),
                       (select count(*) from graph_phrases p
                         where not exists (select 1 from graph_facts f where f.subject_id = p.id)
                           and not exists (select 1 from graph_facts f where f.object_id = p.id))
                """
            )
            return await cur.fetchone()


async def _kestrel_phrases() -> set[str]:
    """Phrase nodes naming the fixture's company. Other documents may use
    some too (a fixture copy in the corpus does), so the checks compare
    this set before the upload and after the delete rather than expect
    it empty."""
    async with get_connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute("select phrase from graph_phrases where phrase like '%kestrel%'")
            return {row[0] for row in await cur.fetchall()}


async def main(base_url: str) -> None:
    data = (FIXTURES / "sample.docx").read_bytes()
    await open_pool()
    try:
        async with httpx.AsyncClient(base_url=base_url, timeout=900, auth=KeycloakAuth()) as client:
            phrases_before = await _kestrel_phrases()
            resp = await client.post("/upload", files={"file": (UPLOAD_NAME, data, DOCX)})
            check(resp.status_code == 200, "upload", resp.text[:200])
            extracted, links, phrases = await _graph_rows(UPLOAD_NAME)
            check(extracted == resp.json().get("chunks_ingested"), "every chunk went through extraction",
                  f"{extracted} of {resp.json().get('chunks_ingested')}")
            check(links > 0 and phrases > 0, "upload has facts and phrases", f"{links} links, {phrases} phrases")
            check(await _kestrel_phrases() > phrases_before, "the upload added a 'kestrel' phrase node")

            resp = await client.post("/upload", files={"file": (UPLOAD_NAME, data, DOCX)})
            check(resp.status_code == 200, "re-upload", resp.text[:200])
            again = await _graph_rows(UPLOAD_NAME)
            check(again[0] == extracted and again[1] > 0, "re-upload re-extracts the new rows",
                  f"{again} vs {(extracted, links, phrases)}")
            check((await _orphans()) == (0, 0), "no orphan facts or phrases after re-upload", str(await _orphans()))

            question = "What was Kestrel's revenue in Q3?"
            rows = await graph_search(await embed_text(question), question, 5)
            check(any(r["filename"] == UPLOAD_NAME for r in rows), "graph_search finds the upload for its question",
                  "; ".join(f"{r['filename']}#{r['chunk_index']}" for r in rows))

            resp = await client.delete(f"/documents/{UPLOAD_NAME}")
            check(resp.status_code == 200, "delete", resp.text[:200])
            check((await _graph_rows(UPLOAD_NAME)) == (0, 0, 0), "delete removed the graph rows")
            check((await _orphans()) == (0, 0), "no orphan facts or phrases after delete", str(await _orphans()))
            check(await _kestrel_phrases() == phrases_before, "phrases only the upload used are gone",
                  str(sorted(await _kestrel_phrases() - phrases_before)))
    finally:
        await close_pool()

    print()
    if failures:
        print(f"{len(failures)} check(s) failed: {', '.join(failures)}")
        sys.exit(1)
    print("all checks passed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-url", default="http://127.0.0.1:8001")
    args = parser.parse_args()
    asyncio.run(main(args.base_url))
