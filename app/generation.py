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


async def chat_with_tools(messages: list[dict], allow_tools: bool = True) -> dict:
    """Sends `messages` to Ollama's /api/chat with the retrieve tool exposed.
    Returns the assistant message dict (may contain "tool_calls"). Pass
    allow_tools=False to force a direct text answer (e.g. when the caller's
    iteration budget is exhausted and it needs a final response)."""
    async with httpx.AsyncClient(base_url=settings.ollama_base_url, timeout=300.0) as client:
        resp = await client.post(
            "/api/chat",
            json={
                "model": settings.chat_model,
                "messages": messages,
                "tools": [RETRIEVE_TOOL] if allow_tools else [],
                "stream": False,
            },
        )
        resp.raise_for_status()
        data = resp.json()

    return data["message"]
