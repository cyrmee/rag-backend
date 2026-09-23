import tempfile
import uuid
from pathlib import Path

from app.dispatcher import extract
from app.generation import CHAT_NUM_CTX

# The limit is on total attached content for one turn, not file count - one
# file or five, what matters is how much of the model's context window it
# eats. Budgeted as a slice of CHAT_NUM_CTX (~4 chars/token is a standard
# rough estimate for English text): system prompt, retrieved chunks,
# conversation history, and the answer itself all need room in the same
# window, so attachments get well under half of it.
MAX_ATTACHMENT_CHARS = CHAT_NUM_CTX * 4 // 3

# Raw upload size cap, checked before parsing even starts - independent of
# the character budget above, this just guards against spending time
# extracting text from an absurdly large file before finding out it would
# have blown the budget anyway.
MAX_UPLOAD_BYTES = 25 * 1024 * 1024

# Ephemeral, in-memory, per-process store: an attachment's extracted text
# only needs to survive from "user picks a file in the composer" to "user
# sends the message it's attached to," not across restarts. Capped so a
# long-running server with many attach-then-abandon actions doesn't grow
# without bound - oldest entries are evicted first.
_MAX_STORED = 500
_store: dict[str, dict] = {}


class AttachmentTooLarge(Exception):
    pass


async def extract_attachment_text(filename: str, file_bytes: bytes, content_type: str | None) -> str:
    """Reuses the same per-format extractors ingestion uses (pdf/docx/pptx/
    xlsx/txt/md - see app/dispatcher.py) to pull plain text out of an
    attached file, without chunking, embedding, or persisting it anywhere -
    an attachment is scoped to the one message it's sent with, not added to
    the searchable document corpus. Raises UnsupportedFileType for a format
    ingestion couldn't handle either."""
    if len(file_bytes) > MAX_UPLOAD_BYTES:
        raise AttachmentTooLarge(
            f"File is {len(file_bytes) / 1_000_000:.1f} MB; the limit is "
            f"{MAX_UPLOAD_BYTES / 1_000_000:.0f} MB."
        )

    suffix = Path(filename).suffix
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    try:
        chunks, _images = extract(tmp_path, filename, content_type)
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    return "\n\n".join(c.content for c in chunks if c.content.strip())


def store_attachment(filename: str, text: str) -> dict:
    if len(_store) >= _MAX_STORED:
        oldest_id = next(iter(_store))
        del _store[oldest_id]

    attachment_id = str(uuid.uuid4())
    _store[attachment_id] = {"filename": filename, "text": text, "char_count": len(text)}
    return {"id": attachment_id, "filename": filename, "char_count": len(text)}


def get_attachment(attachment_id: str) -> dict | None:
    return _store.get(attachment_id)


def total_chars(attachment_ids: list[str]) -> int:
    return sum(len(att["text"]) for aid in attachment_ids if (att := get_attachment(aid)) is not None)
