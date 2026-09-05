"""Phase 3 check: run_agentic_ask on a multi-fact question should trigger
2+ tool calls before a final answer, not just one."""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import app.agent as agent_module
from app.agent import run_agentic_ask
from app.db import close_pool, open_pool

_tool_call_count = 0
_original_chat_with_tools = agent_module.chat_with_tools


async def _counting_chat_with_tools(messages, **kwargs):
    message = await _original_chat_with_tools(messages, **kwargs)
    global _tool_call_count
    _tool_call_count += len(message.get("tool_calls") or [])
    return message


agent_module.chat_with_tools = _counting_chat_with_tools

QUESTION = (
    "Compare the root causes of incidents K-114 and K-129, and explain how "
    "each led to a process change."
)


async def main() -> None:
    await open_pool()
    try:
        result = await run_agentic_ask(QUESTION)
        print("answer:", result["answer"])
        print("num sources:", len(result["sources"]))
        print("tool calls made:", _tool_call_count)
        assert _tool_call_count >= 2, (
            f"expected 2+ tool calls for a multi-fact question, got {_tool_call_count}"
        )
        print("OK: agent made multiple retrieve calls")
    finally:
        await close_pool()


if __name__ == "__main__":
    asyncio.run(main())
