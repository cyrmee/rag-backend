"""Phase 1 check: confirm the chat model actually invokes the `retrieve`
tool through the chat server's OpenAI-compatible tool-calling API (vLLM,
CHAT_BASE_URL), rather than just answering directly or returning
malformed output - both the non-streaming and streaming paths, since
app/generation.py normalizes each one separately.
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from app.generation import chat_with_tools, stream_chat_with_tools

QUESTION = "What's in the documents about the shard-splitting cap incident?"

SYSTEM_PROMPT = (
    "You answer questions using only information retrieved from a document "
    "database via the `retrieve` tool. You have no built-in knowledge of "
    "the documents, so you must call `retrieve` before answering."
)


def check_retrieve_call(label: str, message: dict) -> None:
    tool_calls = message.get("tool_calls")
    assert tool_calls, f"{label}: expected tool_calls to be populated, got none"

    call = tool_calls[0]
    assert call.get("id"), f"{label}: expected a tool call id (needed for tool_call_id)"
    assert call.get("type") == "function", f"{label}: expected type 'function', got {call.get('type')!r}"
    fn = call["function"]
    assert fn["name"] == "retrieve", f"{label}: expected 'retrieve', got {fn['name']!r}"
    assert isinstance(fn["arguments"], dict), f"{label}: expected arguments parsed to a dict"
    query = fn["arguments"].get("query")
    assert isinstance(query, str) and query.strip(), f"{label}: expected a non-empty query string"

    print(f"OK ({label}): model called retrieve with query:", repr(query))


async def main() -> None:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": QUESTION},
    ]
    message = await chat_with_tools(messages)
    print("raw message:", message)
    check_retrieve_call("non-streaming", message)

    final = None
    async for chunk in stream_chat_with_tools(messages):
        if chunk.get("done"):
            final = chunk["message"]
    print("final stream message:", final)
    assert final is not None, "streaming: stream ended without a done chunk"
    check_retrieve_call("streaming", final)


if __name__ == "__main__":
    asyncio.run(main())
