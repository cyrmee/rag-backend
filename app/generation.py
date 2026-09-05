import re

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

DEFAULT_TOOLS = [RETRIEVE_TOOL]


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
            },
        )
        resp.raise_for_status()
        data = resp.json()

    return data["message"]
