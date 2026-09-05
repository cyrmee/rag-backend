"""Phase 5 validation: run the known-good single-hop question and a new
multi-hop question against both /ask (single-pass) and /ask/agentic,
logging latency and (for the agentic route) tool-call/iteration count.
"""

import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import httpx

import app.agent as agent_module
from app.agent import run_agentic_ask
from app.db import close_pool, open_pool

BASE_URL = "http://127.0.0.1:8001"

SINGLE_HOP = (
    "What was the shard-splitting cap before it changed, and what incident "
    "caused the change?"
)
MULTI_HOP = (
    "Compare the root causes of incidents K-114 and K-129, and explain how "
    "each led to a process change."
)

_original_chat_with_tools = agent_module.chat_with_tools


async def run_single_pass(client: httpx.AsyncClient, question: str) -> tuple[str, float]:
    start = time.monotonic()
    resp = await client.post("/ask", json={"question": question})
    resp.raise_for_status()
    elapsed = time.monotonic() - start
    return resp.json()["answer"], elapsed


async def run_agentic_instrumented(question: str) -> tuple[str, float, int]:
    call_count = 0

    async def counting_chat_with_tools(messages, **kwargs):
        nonlocal call_count
        message = await _original_chat_with_tools(messages, **kwargs)
        call_count += len(message.get("tool_calls") or [])
        return message

    agent_module.chat_with_tools = counting_chat_with_tools
    try:
        start = time.monotonic()
        result = await run_agentic_ask(question)
        elapsed = time.monotonic() - start
    finally:
        agent_module.chat_with_tools = _original_chat_with_tools

    return result["answer"], elapsed, call_count


async def compare(client: httpx.AsyncClient, label: str, question: str) -> None:
    print(f"\n=== {label} ===")
    print(f"Q: {question}\n")

    single_answer, single_elapsed = await run_single_pass(client, question)
    print(f"[/ask] {single_elapsed:.2f}s")
    print(single_answer)

    agentic_answer, agentic_elapsed, tool_calls = await run_agentic_instrumented(question)
    print(f"\n[/ask/agentic] {agentic_elapsed:.2f}s, {tool_calls} retrieve call(s)")
    print(agentic_answer)


async def main() -> None:
    await open_pool()
    try:
        async with httpx.AsyncClient(base_url=BASE_URL, timeout=300.0) as client:
            await compare(client, "single-hop (known-good)", SINGLE_HOP)
            await compare(client, "multi-hop", MULTI_HOP)
    finally:
        await close_pool()


if __name__ == "__main__":
    asyncio.run(main())
