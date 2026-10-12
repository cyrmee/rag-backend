"""Builds the knowledge graph (app/graph.py) for chunks that don't have
one yet: everything ingested before graph retrieval was switched on, and
any chunk whose extraction failed at ingestion. One document at a time,
its chunks' extraction calls in parallel, committed per document, so it
can be stopped and resumed at any point. Order and filters let the
documents that matter go first - e.g. --skip "Training and KBs" to leave
the training books for a second run.

Budget: about 3.7 s and 2,300 chat tokens per chunk on the dev machine
(measured October 2026), so the whole corpus is a matter of hours; run it
in the background.

Usage:
    python scripts/maintenance/backfill_graph.py
    python scripts/maintenance/backfill_graph.py --skip "Training and KBs"
    python scripts/maintenance/backfill_graph.py --only "TECH5" --redo
    python scripts/maintenance/backfill_graph.py --dry-run
    python scripts/maintenance/backfill_graph.py --concurrency 32
"""

import argparse
import asyncio
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import app.graph as graph
from app.db import close_pool, get_connection, open_pool
from app.graph import passage_text, prepare_graph, remove_orphans, store_graph

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("backfill_graph")


async def _pending(only: list[str], skip: list[str], redo: bool) -> dict[str, list[tuple]]:
    """Chunks to extract, grouped by document in filename order."""
    clauses, params = [], []
    if not redo:
        clauses.append("graph_extracted_at is null")
    for pattern in only:
        clauses.append("filename ilike %s")
        params.append(f"%{pattern}%")
    for pattern in skip:
        clauses.append("filename not ilike %s")
        params.append(f"%{pattern}%")
    where = f"where {' and '.join(clauses)}" if clauses else ""
    async with get_connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                f"""
                select id, filename, page_number, source_type, content
                from documents {where}
                order by filename, chunk_index
                """,
                params,
            )
            rows = await cur.fetchall()
    by_document: dict[str, list[tuple]] = {}
    for row in rows:
        by_document.setdefault(row[1], []).append(row)
    return by_document


async def _backfill_document(filename: str, rows: list[tuple], redo: bool) -> dict:
    texts = {row[0]: passage_text(filename, row[2], row[3], row[4]) for row in rows}
    batch = await prepare_graph(texts)
    async with get_connection() as conn:
        async with conn.cursor() as cur:
            if redo:
                await cur.execute("delete from graph_triples where chunk_id = any(%s)", (list(texts),))
            stats = await store_graph(cur, batch, {chunk_id: chunk_id for chunk_id in texts})
            if redo:
                await remove_orphans(cur)
        await conn.commit()
    stats["failed"] = len(texts) - len(batch.triples)
    return stats


async def _totals() -> str:
    async with get_connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                """
                select (select count(*) from graph_phrases), (select count(*) from graph_facts),
                       (select count(*) from graph_triples), (select count(*) from graph_edges),
                       (select count(*) from documents where graph_extracted_at is not null),
                       (select count(*) from documents)
                """
            )
            phrases, facts, triples, edges, done, chunks = await cur.fetchone()
    return (f"{phrases} phrases, {facts} facts, {triples} chunk-fact links, {edges} synonym edges; "
            f"{done} of {chunks} chunks extracted")


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", action="append", default=[], help="filename substring to include (repeatable)")
    parser.add_argument("--skip", action="append", default=[], help="filename substring to leave out (repeatable)")
    parser.add_argument("--redo", action="store_true", help="re-extract chunks that already have a graph")
    parser.add_argument("--dry-run", action="store_true", help="list the documents and chunk counts only")
    parser.add_argument(
        "--concurrency", type=int, default=16,
        help="extraction calls in flight at once (ingestion uses %d; the chat server batches them, so "
             "more means more chunks per minute until it saturates)" % graph.EXTRACTION_CONCURRENCY,
    )
    args = parser.parse_args()
    graph.EXTRACTION_CONCURRENCY = args.concurrency

    await open_pool()
    try:
        pending = await _pending(args.only, args.skip, args.redo)
        total = sum(len(rows) for rows in pending.values())
        logger.info("%d chunk(s) in %d document(s) to extract", total, len(pending))
        if args.dry_run:
            for filename, rows in pending.items():
                print(f"{len(rows):5}  {filename}")
            return

        done = 0
        start = time.time()
        for filename, rows in pending.items():
            doc_start = time.time()
            stats = await _backfill_document(filename, rows, args.redo)
            done += len(rows)
            elapsed = time.time() - start
            remaining = (total - done) * elapsed / done if done else 0
            logger.info(
                "%s: %d chunks in %.0fs -> %d facts, %d new phrases, %d synonym edges%s | %d/%d chunks, ~%.0f min left",
                filename, len(rows), time.time() - doc_start, stats["facts"], stats["new_phrases"],
                stats["synonym_edges"], f", {stats['failed']} failed" if stats["failed"] else "",
                done, total, remaining / 60,
            )
        logger.info("done: %s", await _totals())
    finally:
        await close_pool()


if __name__ == "__main__":
    asyncio.run(main())
