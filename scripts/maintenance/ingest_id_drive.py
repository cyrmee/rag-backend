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

Usage:
    python scripts/maintenance/ingest_id_drive.py                    # all zips in ~/ID Drive
    python scripts/maintenance/ingest_id_drive.py --dry-run          # just report counts, no ingestion
    python scripts/maintenance/ingest_id_drive.py --zip ID-...-001.zip  # one archive only
    python scripts/maintenance/ingest_id_drive.py --concurrency 8    # more in-flight documents
"""

import argparse
import asyncio
import json
import logging
import sys
import tempfile
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from app.db import close_pool, delete_document_chunks, open_pool
from app.ingestion import ingest_document
from app.parsing import UnsupportedFileType

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("ingest_id_drive")

DEFAULT_ID_DRIVE_DIR = Path.home() / "ID Drive"
LOG_DIR = Path(__file__).resolve().parents[2] / "ingest_logs"
CHECKPOINT_PATH = LOG_DIR / "checkpoint.jsonl"
SKIPPED_PATH = LOG_DIR / "skipped.jsonl"

SUPPORTED_SUFFIXES = {".pdf", ".docx", ".pptx", ".xlsx", ".txt", ".md"}


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
            if record.get("status") == "done":
                done.add((record["zip"], record["path"]))
    return done


def _append_jsonl(path: Path, record: dict) -> None:
    with path.open("a") as f:
        f.write(json.dumps(record) + "\n")


async def _ingest_one(
    zip_name: str,
    zip_info: zipfile.ZipInfo,
    file_bytes: bytes,
    semaphore: asyncio.Semaphore,
    caption_images: bool,
    generate_summary: bool,
) -> None:
    internal_path = zip_info.filename
    suffix = Path(internal_path).suffix
    async with semaphore:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name

        try:
            # Upsert, not append: a filename re-ingested (e.g. a later
            # backfill pass, or a forced re-run) replaces its old rows
            # rather than duplicating them - same semantics as /upload.
            await delete_document_chunks(internal_path)
            chunk_count = await ingest_document(
                tmp_path,
                internal_path,
                metadata={
                    "source_zip": zip_name,
                    "internal_path": internal_path,
                    "folder": str(Path(internal_path).parent),
                },
                caption_images=caption_images,
                generate_summary=generate_summary,
            )
            _append_jsonl(CHECKPOINT_PATH, {
                "zip": zip_name, "path": internal_path, "status": "done", "chunks": chunk_count,
            })
            logger.info("ingested %s -> %s (%d chunks)", zip_name, internal_path, chunk_count)
        except UnsupportedFileType as exc:
            _append_jsonl(CHECKPOINT_PATH, {
                "zip": zip_name, "path": internal_path, "status": "error", "error": str(exc),
            })
            logger.warning("unsupported %s -> %s: %s", zip_name, internal_path, exc)
        except Exception as exc:
            _append_jsonl(CHECKPOINT_PATH, {
                "zip": zip_name, "path": internal_path, "status": "error", "error": repr(exc),
            })
            logger.exception("failed %s -> %s", zip_name, internal_path)
        finally:
            Path(tmp_path).unlink(missing_ok=True)


async def _process_zip(
    zip_path: Path,
    already_done: set[tuple[str, str]],
    semaphore: asyncio.Semaphore,
    dry_run: bool,
    stats: dict,
    caption_images: bool,
    generate_summary: bool,
) -> None:
    zip_name = zip_path.name
    logger.info("opening %s", zip_name)

    with zipfile.ZipFile(zip_path) as zf:
        tasks = []
        for info in zf.infolist():
            reason = _skip_reason(info)
            if info.is_dir():
                continue
            if reason is not None:
                stats["skipped"] += 1
                if not dry_run:
                    _append_jsonl(SKIPPED_PATH, {"zip": zip_name, "path": info.filename, "reason": reason})
                continue

            if (zip_name, info.filename) in already_done:
                stats["already_done"] += 1
                continue

            stats["to_ingest"] += 1
            if dry_run:
                continue

            file_bytes = zf.read(info)
            tasks.append(asyncio.create_task(
                _ingest_one(zip_name, info, file_bytes, semaphore, caption_images, generate_summary)
            ))

        if tasks:
            await asyncio.gather(*tasks)

    logger.info("finished %s", zip_name)


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--id-drive-dir", type=Path, default=DEFAULT_ID_DRIVE_DIR)
    parser.add_argument("--zip", action="append", dest="zips", help="Only process this zip filename (repeatable)")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--dry-run", action="store_true", help="Report counts only, ingest nothing")
    parser.add_argument(
        "--skip-images", action="store_true",
        help="Skip vision captioning of embedded images - text/chart_data only, much faster",
    )
    parser.add_argument(
        "--skip-summary", action="store_true",
        help=(
            "Skip LLM document-summary generation - avoids needing the chat model loaded "
            "alongside the vision model (they don't comfortably coexist in memory with it); "
            "run scripts/maintenance/backfill_summaries.py afterward instead"
        ),
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

    stats = {"to_ingest": 0, "skipped": 0, "already_done": 0}
    semaphore = asyncio.Semaphore(args.concurrency)

    if not args.dry_run:
        await open_pool()
    try:
        for zip_path in zip_paths:
            await _process_zip(
                zip_path, already_done, semaphore, args.dry_run, stats,
                caption_images=not args.skip_images,
                generate_summary=not args.skip_summary,
            )
    finally:
        if not args.dry_run:
            await close_pool()

    logger.info(
        "done. to_ingest=%d skipped=%d already_done=%d (see %s and %s)",
        stats["to_ingest"], stats["skipped"], stats["already_done"], CHECKPOINT_PATH, SKIPPED_PATH,
    )


if __name__ == "__main__":
    asyncio.run(main())
