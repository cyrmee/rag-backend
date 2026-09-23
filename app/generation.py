import json
import re
from typing import AsyncIterator

import httpx

from app.config import settings

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


async def generate_answer(prompt: str) -> tuple[str, str]:
    """Returns (answer, thinking). `thinking` is Ollama's separate reasoning
    field when the model/template supports it; otherwise it's whatever was
    stripped out of an inline <think>...</think> block, if any."""
    async with httpx.AsyncClient(base_url=settings.ollama_base_url, timeout=300.0) as client:
        resp = await client.post(
            "/api/generate",
            json={
                "model": settings.chat_model,
                "prompt": prompt,
                "stream": False,
            },
        )
        resp.raise_for_status()
        data = resp.json()

    raw_answer = data.get("response", "")
    thinking = data.get("thinking", "") or ""

    answer = _THINK_RE.sub("", raw_answer).strip()
    return answer, thinking


_TITLE_PROMPT = (
    "Summarize this question as a short conversation title: 3-6 words, "
    "plain text, no quotes, no trailing punctuation, no preamble like "
    '"Title:".\n\nQuestion: {question}\n\nTitle:'
)


async def generate_title(question: str) -> str:
    """A short label for a conversation list, generated once from its first
    question. Falls back to a truncated version of the question itself if
    the model wraps its answer in quotes/preamble it ignored, or returns
    something clearly not a short title."""
    raw, _ = await generate_answer(_TITLE_PROMPT.format(question=question))
    title = raw.strip().strip('"').strip("'").strip()
    if not title or len(title) > 80:
        title = question.strip()[:60]
    return title


RETRIEVE_TOOL = {
    "type": "function",
    "function": {
        "name": "retrieve",
        "description": "Search the document database for relevant chunks",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The search query"}
            },
            "required": ["query"],
        },
    },
}

DESCRIBE_IMAGE_TOOL = {
    "type": "function",
    "function": {
        "name": "describe_image",
        "description": (
            "Get a fresh, detailed vision-model description of a specific "
            "figure/chart, for a deeper look than the pre-computed caption "
            "already in a retrieved chunk. Only call this with an "
            "image_path that appeared in a retrieved chunk's "
            "[image_path=...] tag - not a made-up path."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "image_path": {
                    "type": "string",
                    "description": "The image_path from a retrieved chunk's [image_path=...] tag",
                }
            },
            "required": ["image_path"],
        },
    },
}

LIST_DOCUMENTS_TOOL = {
    "type": "function",
    "function": {
        "name": "list_documents",
        "description": (
            "Lists/counts ingested documents by filename - for questions "
            "about the document corpus itself (e.g. 'how many documents do "
            "we have', 'what documents do we have', 'how many files "
            "mention X in their name/folder'). Do NOT use this for "
            "questions about what's inside a document's content - use "
            "retrieve for that. Returns a total count plus up to 50 "
            "matching filenames with their chunk counts."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "filename_contains": {
                    "type": "string",
                    "description": (
                        "Optional case-insensitive substring to filter filenames/folder "
                        "paths by. Omit to list/count every document."
                    ),
                }
            },
        },
    },
}

WEB_SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": (
            "Searches the public internet - for anything current or "
            "outside the document archive (today's news, current "
            "regulations/exchange rates, a fact the retrieved documents "
            "don't cover). Do NOT use this for questions about the "
            "organization's own documents/policies/configuration - use "
            "retrieve for that, even if a web search might also turn up "
            "something related. Returns a short list of "
            "{title, url, content} results, not full pages."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The web search query"}
            },
            "required": ["query"],
        },
    },
}

DEFAULT_TOOLS = [RETRIEVE_TOOL]

# Ollama falls back to its own conservative default (historically 4096)
# when a request doesn't set this, regardless of what the model itself
# supports (gemma4 here goes up to 262144) - too small to comfortably hold
# a system prompt, TOP_K retrieved chunks, conversation history, and any
# attached-file text together. 32768 is a deliberate middle ground: ample
# room for all of that plus MAX_ATTACHMENT_CHARS worth of attachments
# (see app/attachments.py), without the much larger KV-cache memory
# footprint a very large context window would cost on this machine.
CHAT_NUM_CTX = 32768


async def chat_with_tools(
    messages: list[dict], tools: list[dict] | None = None, allow_tools: bool = True
) -> dict:
    """Sends `messages` to Ollama's /api/chat with `tools` exposed (defaults
    to just the retrieve tool). Returns the assistant message dict (may
    contain "tool_calls"). Pass allow_tools=False to force a direct text
    answer (e.g. when the caller's iteration budget is exhausted and it
    needs a final response)."""
    async with httpx.AsyncClient(base_url=settings.ollama_base_url, timeout=300.0) as client:
        resp = await client.post(
            "/api/chat",
            json={
                "model": settings.chat_model,
                "messages": messages,
                "tools": (tools if tools is not None else DEFAULT_TOOLS) if allow_tools else [],
                "stream": False,
                "options": {"num_ctx": CHAT_NUM_CTX},
            },
        )
        resp.raise_for_status()
        data = resp.json()

    return data["message"]


async def stream_chat_with_tools(
    messages: list[dict], tools: list[dict] | None = None, allow_tools: bool = True
) -> AsyncIterator[dict]:
    """Streaming counterpart to chat_with_tools. Yields raw Ollama /api/chat
    stream chunks - each has a "message" dict with incremental "content"
    and/or "thinking" deltas; tool_calls (when the model decides to call
    one) arrive fully-formed in a single chunk with empty content, not
    token-by-token, since partial tool-call JSON can't be validated."""
    async with httpx.AsyncClient(base_url=settings.ollama_base_url, timeout=300.0) as client:
        async with client.stream(
            "POST",
            "/api/chat",
            json={
                "model": settings.chat_model,
                "messages": messages,
                "tools": (tools if tools is not None else DEFAULT_TOOLS) if allow_tools else [],
                "stream": True,
                "options": {"num_ctx": CHAT_NUM_CTX},
            },
        ) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if line.strip():
                    yield json.loads(line)
