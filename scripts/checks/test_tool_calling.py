"""Phase 1 check: confirm the chat model actually invokes the `retrieve`
tool through Ollama's native tool-calling API, rather than just answering
directly or returning malformed output.
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from app.generation import chat_with_tools

QUESTION = "What's in the documents about the shard-splitting cap incident?"

SYSTEM_PROMPT = (
    "You answer questions using only information retrieved from a document "
    "database via the `retrieve` tool. You have no built-in knowledge of "
    "the documents, so you must call `retrieve` before answering."
)


async def main() -> None:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": QUESTION},
    ]
    message = await chat_with_tools(messages)
    print("raw message:", message)

    tool_calls = message.get("tool_calls")
    assert tool_calls, "expected tool_calls to be populated, got none"

    call = tool_calls[0]
    fn = call["function"]
    assert fn["name"] == "retrieve", f"expected 'retrieve', got {fn['name']!r}"
    query = fn["arguments"]["query"]
    assert isinstance(query, str) and query.strip(), "expected a non-empty query string"

    print("OK: model called retrieve with query:", repr(query))


if __name__ == "__main__":
    asyncio.run(main())
