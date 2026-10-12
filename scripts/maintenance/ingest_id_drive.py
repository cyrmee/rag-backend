"""Batch-ingests a directory of Google-Drive-export zip files (e.g. a
Takeout-style split archive) into the RAG backend, without ever unpacking a
whole zip to disk - each entry is streamed out, ingested, and deleted before
moving to the next, so disk usage stays at "one file at a time" instead of
the full archive size.

Only formats the pipeline actually supports (.pdf/.docx/.pptx/.xlsx/.txt/.md)
are ingested; everything else (raw images, scripts, keys, Office lock files,
folder markers, ...) is skipped and recorded in ingest_logs/skipped.jsonl
with a reason, never ingested.

Resumable: every successfully ingested (zip, internal path) is appended to
ingest_logs/checkpoint.jsonl as it completes, and a re-run skips anything
already marked done there - safe to Ctrl-C and restart.

Passes. A bulk run does the pass everything else depends on - parse, chunk,
embed, store - and leaves image captions, document summaries and the
knowledge graph to the backfill scripts, each of which runs at the pace of
its own model server instead of holding up the others:

    python scripts/maintenance/ingest_id_drive.py --skip-images --skip-summary --skip-graph
    python scripts/maintenance/backfill_captions.py
    python scripts/maintenance/backfill_summaries.py
    python scripts/maintenance/backfill_graph.py

Without the --skip flags, each document is fully enriched as it is
ingested, the way /upload does it. Every log line carries the time, and
each document's line says how long each stage took and how many images
were captioned, served from the caption cache or skipped as too small;
the last line says where the run's time went.

Usage:
    python scripts/maintenance/ingest_id_drive.py                    # all zips in ~/ID Drive
    python scripts/maintenance/ingest_id_drive.py --dry-run          # just report counts, no ingestion
    python scripts/maintenance/ingest_id_drive.py --zip ID-...-001.zip  # one archive only
    python scripts/maintenance/ingest_id_drive.py --skip "Training and KBs"   # path substring, case-insensitive
    python scripts/maintenance/ingest_id_drive.py --only "ID/Program Management" --limit 20   # a trial run
    python scripts/maintenance/ingest_id_drive.py --concurrency 8    # more in-flight documents
"""

import argparse
import asyncio
import json
import logging
import sys
import tempfile
import time
import zipfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from app.db import close_pool, open_pool
from app.ingestion import ingest_document
from app.parsing import DuplicateDocument, UnsupportedFileType

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S",
)
# One line per embedding batch and model call says nothing a stage timing
# doesn't; a full run of them was 12,000 lines.
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger("ingest_id_drive")

DEFAULT_ID_DRIVE_DIR = Path.home() / "ID Drive"
LOG_DIR = Path(__file__).resolve().parents[2] / "ingest_logs"
CHECKPOINT_PATH = LOG_DIR / "checkpoint.jsonl"
SKIPPED_PATH = LOG_DIR / "skipped.jsonl"

SUPPORTED_SUFFIXES = {".pdf", ".docx", ".pptx", ".xlsx", ".txt", ".md"}


@dataclass
class Entry:
    zip_path: Path
    info: zipfile.ZipInfo


@dataclass
class Options:
    caption_images: bool
    generate_summary: bool
    extract_graph: bool


def _skip_reason(zip_info: zipfile.ZipInfo) -> str | None:
    """None means "ingest this entry"; otherwise the reason it's skipped."""
    name = zip_info.filename
    if zip_info.is_dir():
        return None  # not logged - a directory marker isn't a skipped file
    basename = Path(name).name
    if name.startswith("__MACOSX/") or basename == ".DS_Store":
        return "macos_junk"
    if basename.startswith("~$"):
        return "office_lock_file"
    suffix = Path(name).suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        return f"unsupported_extension:{suffix or 'none'}"
    return None


def _selected(internal_path: str, only: list[str], skip: list[str]) -> bool:
    lowered = internal_path.lower()
    if only and not any(pattern.lower() in lowered for pattern in only):
        return False
    return not any(pattern.lower() in lowered for pattern in skip)


def _load_checkpoint() -> set[tuple[str, str]]:
    if not CHECKPOINT_PATH.exists():
        return set()
    done = set()
    with CHECKPOINT_PATH.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if record.get("status") in ("done", "duplicate"):
                done.add((record["zip"], record["path"]))
    return done


def _append_jsonl(path: Path, record: dict) -> None:
    with path.open("a") as f:
        f.write(json.dumps(record) + "\n")


def _plan(
    zip_paths: list[Path], already_done: set[tuple[str, str]], only: list[str], skip: list[str],
    dry_run: bool, stats: Counter,
) -> list[Entry]:
    """Every entry to ingest, in zip order, from the zips' central
    directories alone (no entry is read). Knowing the total up front is
    what makes the progress lines' estimate possible."""
    entries = []
    for zip_path in zip_paths:
        with zipfile.ZipFile(zip_path) as zf:
            for info in zf.infolist():
                if info.is_dir():
                    continue
                reason = _skip_reason(info)
                if reason is not None:
                    stats["skipped"] += 1
                    if not dry_run:
                        _append_jsonl(SKIPPED_PATH, {"zip": zip_path.name, "path": info.filename, "reason": reason})
                    continue
                if (zip_path.name, info.filename) in already_done:
                    stats["already_done"] += 1
                    continue
                if not _selected(info.filename, only, skip):
                    stats["filtered_out"] += 1
                    continue
                entries.append(Entry(zip_path, info))
    return entries


class Progress:
    """Documents processed so far, the rate, and where the time went."""

    def __init__(self, total: int):
        self.total = total
        self.processed = 0
        self.start = time.monotonic()
        self.stage_seconds: Counter = Counter()

    def record(self, stats: dict | None) -> str:
        self.processed += 1
        if stats:
            self.stage_seconds.update(stats.get("timings", {}))
        elapsed = time.monotonic() - self.start
        per_hour = self.processed / elapsed * 3600 if elapsed else 0
        left = (self.total - self.processed) / per_hour * 60 if per_hour else 0
        return f"{self.processed}/{self.total}, {per_hour:.0f}/h, ~{left:.0f} min left"

    def summary(self) -> str:
        elapsed = time.monotonic() - self.start
        stages = sum(self.stage_seconds.values())
        where = ", ".join(
            f"{stage} {seconds / stages:.0%}" for stage, seconds in self.stage_seconds.most_common()
        ) if stages else "nothing ingested"
        return f"{self.processed} documents in {elapsed / 60:.1f} min; time by stage: {where}"


def _describe(stats: dict) -> str:
    """\"caption 12.3s embed 4.0s ...; 3 images captioned, 2 cached, 1 skipped\"."""
    timings = " ".join(f"{stage} {seconds:.1f}s" for stage, seconds in stats.get("timings", {}).items())
    captions = stats.get("captions") or {}
    if captions:
        timings += (
            f"; images: {captions['captioned']} captioned, {captions['cached']} cached, "
            f"{captions['skipped']} skipped" + (f", {captions['failed']} failed" if captions["failed"] else "")
        )
    return timings


async def _ingest_one(
    zf: zipfile.ZipFile, entry: Entry, semaphore: asyncio.Semaphore, options: Options, progress: Progress,
) -> None:
    zip_name = entry.zip_path.name
    internal_path = entry.info.filename
    suffix = Path(internal_path).suffix
    async with semaphore:
        # Read from the zip only now, so what's in memory is bounded by
        # --concurrency, not by the zip's size.
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(zf.read(entry.info))
            tmp_path = tmp.name

        stats: dict = {}
        started = time.monotonic()
        try:
            # Upsert, not append: a filename re-ingested (e.g. a later
            # backfill pass, or a forced re-run) replaces its old rows
            # rather than duplicating them - same semantics as /upload.
            chunk_count = await ingest_document(
                tmp_path,
                internal_path,
                metadata={
                    "source_zip": zip_name,
                    "internal_path": internal_path,
                    "folder": str(Path(internal_path).parent),
                },
                caption_images=options.caption_images,
                generate_summary=options.generate_summary,
                extract_graph=options.extract_graph,
                stats=stats,
            )
            _append_jsonl(CHECKPOINT_PATH, {
                "zip": zip_name, "path": internal_path, "status": "done", "chunks": chunk_count,
                "seconds": round(time.monotonic() - started, 1), "timings": stats.get("timings"),
            })
            logger.info(
                "ingested %s (%d chunks) in %.0fs: %s | %s",
                internal_path, chunk_count, time.monotonic() - started, _describe(stats), progress.record(stats),
            )
        except DuplicateDocument as exc:
            # Marked "duplicate", which a re-run treats like "done" - the
            # content is already searchable under the other name.
            _append_jsonl(CHECKPOINT_PATH, {
                "zip": zip_name, "path": internal_path, "status": "duplicate",
                "existing": exc.existing_filename,
            })
            logger.info("skipped %s: identical to %s | %s", internal_path, exc.existing_filename, progress.record(None))
        except UnsupportedFileType as exc:
            _append_jsonl(CHECKPOINT_PATH, {
                "zip": zip_name, "path": internal_path, "status": "error", "error": str(exc),
            })
            logger.warning("unsupported %s: %s | %s", internal_path, exc, progress.record(None))
        except Exception as exc:
            _append_jsonl(CHECKPOINT_PATH, {
                "zip": zip_name, "path": internal_path, "status": "error", "error": repr(exc),
            })
            logger.exception("failed %s | %s", internal_path, progress.record(None))
        finally:
            Path(tmp_path).unlink(missing_ok=True)


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--id-drive-dir", type=Path, default=DEFAULT_ID_DRIVE_DIR)
    parser.add_argument("--zip", action="append", dest="zips", help="Only process this zip filename (repeatable)")
    parser.add_argument("--only", action="append", default=[], help="path substring to include (repeatable)")
    parser.add_argument("--skip", action="append", default=[], help="path substring to leave out (repeatable)")
    parser.add_argument("--limit", type=int, help="Stop after this many new documents")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--dry-run", action="store_true", help="Report counts only, ingest nothing")
    parser.add_argument(
        "--skip-images", action="store_true",
        help="Skip vision captioning of embedded images; scripts/maintenance/backfill_captions.py does it later",
    )
    parser.add_argument(
        "--skip-summary", action="store_true",
        help="Skip LLM document-summary generation; scripts/maintenance/backfill_summaries.py does it later",
    )
    parser.add_argument(
        "--skip-graph", action="store_true",
        help="Skip knowledge-graph extraction (two chat calls per chunk); "
             "scripts/maintenance/backfill_graph.py does it later",
    )
    args = parser.parse_args()

    LOG_DIR.mkdir(exist_ok=True)

    if args.zips:
        zip_paths = [args.id_drive_dir / name for name in args.zips]
        missing = [p for p in zip_paths if not p.exists()]
        if missing:
            raise SystemExit(f"zip file(s) not found: {missing}")
    else:
        zip_paths = sorted(args.id_drive_dir.glob("*.zip"))
        if not zip_paths:
            raise SystemExit(f"no .zip files found in {args.id_drive_dir}")

    already_done = _load_checkpoint()
    logger.info("%d entries already ingested per checkpoint", len(already_done))

    stats: Counter = Counter()
    entries = _plan(zip_paths, already_done, args.only, args.skip, args.dry_run, stats)
    if args.limit is not None:
        entries = entries[: args.limit]
    logger.info(
        "to_ingest=%d skipped=%d already_done=%d filtered_out=%d",
        len(entries), stats["skipped"], stats["already_done"], stats["filtered_out"],
    )
    if args.dry_run:
        return

    options = Options(
        caption_images=not args.skip_images,
        generate_summary=not args.skip_summary,
        extract_graph=not args.skip_graph,
    )
    semaphore = asyncio.Semaphore(args.concurrency)
    progress = Progress(len(entries))

    # Every zip stays open for the run (a handle holds its central
    # directory, nothing more) and the entries go through one semaphore
    # in zip order, so the last documents of one zip overlap the first of
    # the next; waiting for each zip to drain kept only a document or two
    # in flight at the end of every archive.
    zips = {path: zipfile.ZipFile(path) for path in dict.fromkeys(entry.zip_path for entry in entries)}
    logger.info("%d documents in %d zip(s)", len(entries), len(zips))
    await open_pool()
    try:
        await asyncio.gather(*(
            _ingest_one(zips[entry.zip_path], entry, semaphore, options, progress) for entry in entries
        ))
    finally:
        await close_pool()
        for zf in zips.values():
            zf.close()

    logger.info("done: %s (see %s and %s)", progress.summary(), CHECKPOINT_PATH, SKIPPED_PATH)


if __name__ == "__main__":
    asyncio.run(main())
