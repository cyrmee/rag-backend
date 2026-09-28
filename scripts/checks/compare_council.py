"""Compares the regular agent against council mode (app/council.py) at a few
angle counts, on questions with known answers from the current corpus.

For each question and mode it records: whether the answer contains the
expected facts, latency, source count, how many tools the model called
itself, and how many sentences the citation check flagged (council only).
Every conversation it creates is deleted afterwards. Needs the chat (vLLM),
embedding (Ollama) and DB servers running; takes ~20-30 minutes.

    python scripts/checks/compare_council.py
    python scripts/checks/compare_council.py --modes agent council:8 --runs 2
    python scripts/checks/compare_council.py --only laxton admin-count
"""

import argparse
import asyncio
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import app.agent as agent_module
from app.agent import run_agentic_ask
from app.config import settings
from app.conversations import delete_conversation
from app.db import close_pool, open_pool

REFUSAL = ["don't have", "do not have", "not in the documents", "couldn't find", "could not find", "no information"]

# (label, [turns], expected) - only the last turn's answer is scored.
# `expected` entries are all required; a list inside means any-of.
QUESTIONS = [
    ("tech5-2022", ["What was TECH5's net revenue in 2022?"], ["6320815"]),
    ("tech5-multi", ["What was TECH5's net revenue in 2020, and what was its result for 2022?"], ["3442556", "2721957"]),
    ("tech5-avg", ["What is TECH5's 3-year average net revenue?"], ["4091326"]),
    ("follow-up", ["What was TECH5's net revenue in 2022?", "And in 2021?"], ["2510606"]),
    ("bunna-vpn", ["Which Diffie-Hellman group and phase 1 lifetime does Bunna Bank's VPN to NID use?"], [r"\b19\b", "43200"]),
    ("laxton", ["When was Laxton Group's financial proposal going to be opened, per their technical evaluation result?"], ["february 16"]),
    ("furniture", ["In the furniture list, what is the unit price of an Esol table and the overall total?"], ["120000", "2040000"]),
    ("admin-count", ["How many documents are in the Admin folder?"], [[r"\b3\b", "three"]]),
    ("not-in-corpus", ["Who is the executive sponsor of Project Falcon?"], [REFUSAL]),
]


def score(answer: str, expected: list) -> bool:
    text = answer.lower().replace(",", "")
    for item in expected:
        options = item if isinstance(item, list) else [item]
        if not any(re.search(opt, text) for opt in options):
            return False
    return True


_model_tool_calls = 0
_original_chat = agent_module.chat_with_tools


async def _counting_chat(messages, **kwargs):
    global _model_tool_calls
    message = await _original_chat(messages, **kwargs)
    _model_tool_calls += len(message.get("tool_calls") or [])
    return message


agent_module.chat_with_tools = _counting_chat


async def run_case(turns: list[str], mode: str) -> dict:
    global _model_tool_calls
    council_mode = mode.startswith("council")
    if council_mode:
        settings.council_angles = int(mode.split(":")[1])
    conversation_id = None
    try:
        for i, question in enumerate(turns):
            _model_tool_calls = 0
            start = time.time()
            result = await run_agentic_ask(question, conversation_id=conversation_id, council_mode=council_mode)
            elapsed = time.time() - start
            conversation_id = result["conversation_id"]
        return {
            "answer": result["answer"].strip(),
            "seconds": elapsed,
            "sources": len(result["sources"]),
            "model_tools": _model_tool_calls,
            "warnings": len(result["citation_warnings"]),
        }
    finally:
        if conversation_id:
            await delete_conversation(conversation_id)


async def main(modes: list[str], runs: int, only: list[str] | None) -> None:
    await open_pool()
    totals = {m: {"correct": 0, "seconds": 0.0, "warnings": 0, "n": 0} for m in modes}
    try:
        for label, turns, expected in QUESTIONS:
            if only and label not in only:
                continue
            for mode in modes:
                for _ in range(runs):
                    try:
                        r = await run_case(turns, mode)
                    except Exception as exc:  # one broken case shouldn't sink the whole comparison
                        r = {"answer": f"ERROR {type(exc).__name__}: {exc}", "seconds": 0.0, "sources": 0,
                             "model_tools": 0, "warnings": 0}
                    ok = score(r["answer"], expected) and not r["answer"].startswith("ERROR")
                    t = totals[mode]
                    t["correct"] += ok
                    t["seconds"] += r["seconds"]
                    t["warnings"] += r["warnings"]
                    t["n"] += 1
                    print(
                        f"{'PASS' if ok else 'FAIL'} {label:14} {mode:10} {r['seconds']:5.1f}s "
                        f"sources={r['sources']:2} model_tools={r['model_tools']} flagged={r['warnings']} "
                        f"| {r['answer'][:110]!r}",
                        flush=True,
                    )
    finally:
        await close_pool()

    print("\nmode        correct   avg latency   flagged sentences")
    for mode, t in totals.items():
        print(f"{mode:10}  {t['correct']:2}/{t['n']:<2}     {t['seconds'] / t['n']:6.1f}s      {t['warnings']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--modes", nargs="+", default=["agent", "council:4", "council:8", "council:12"])
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--only", nargs="+", help="question labels to run (default: all)")
    args = parser.parse_args()
    asyncio.run(main(args.modes, args.runs, args.only))
