import re

_PARAGRAPH_SEPARATOR = re.compile(r"\n\s*\n+")


def chunk_text(text: str, max_chunk_size: int) -> list[str]:
    """Splits on paragraph boundaries (blank lines) - the unit the author
    actually wrote - rather than a blind word-count sliding window. A
    paragraph is only split further if it exceeds max_chunk_size words, and
    even then at word boundaries (never mid-word); no overlap is needed
    since each chunk is already a coherent unit, not an arbitrary slice."""
    paragraphs = _PARAGRAPH_SEPARATOR.split(text.strip())
    chunks = []
    for para in paragraphs:
        words = para.split()
        if not words:
            continue
        if len(words) <= max_chunk_size:
            chunks.append(" ".join(words))
            continue
        for start in range(0, len(words), max_chunk_size):
            sub = " ".join(words[start : start + max_chunk_size]).strip()
            if sub:
                chunks.append(sub)
    return chunks
