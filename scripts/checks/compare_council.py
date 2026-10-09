"""Compares the regular agent against council mode (app/council.py) at a few
angle counts, on questions with known answers from the current corpus.

For each question and mode it records: whether the answer is correct,
latency, source count, how many tools the model called itself, and how many
sentences the citation check flagged (council only). "Correct" means the
expected facts appear in the answer's opening (first HEAD_CHARS characters)
and no known wrong value does - a right number buried after a wrong
headline doesn't count. --save writes every full answer to a JSONL file for
manual review.
Every conversation it creates is deleted afterwards. Needs the chat (vLLM),
embedding (Ollama) and DB servers running; takes ~20-30 minutes.

    python scripts/checks/compare_council.py
    python scripts/checks/compare_council.py --modes agent council:8 --runs 2
    python scripts/checks/compare_council.py --only laxton admin-count
    python scripts/checks/compare_council.py --save answers.jsonl
    python scripts/checks/compare_council.py --set complex --runs 3   # multi-part questions
    python scripts/checks/compare_council.py --set all --modes auto  # the default routing
"""

import argparse
import asyncio
import json
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

REFUSAL = [
    "don't have", "do not have", "not in the documents", "couldn't find", "could not find", "no information",
    r"do(es)? not (explicitly )?mention", "not mentioned",
]

HEAD_CHARS = 350

# (label, [turns], expected, rejected) - only the last turn's answer is
# scored, lowercased with thousands separators removed. Every `expected`
# entry must appear in the opening (a list inside means any-of); any
# `rejected` pattern there fails it. TECH5's 2020/2022 figures legitimately
# appear two ways across documents (exact, and rounded in thousands).
QUESTIONS = [
    ("tech5-2022", ["What was TECH5's net revenue in 2022?"], ["6320815"], ["billion"]),
    (
        "tech5-multi",
        ["What was TECH5's net revenue in 2020, and what was its result for 2022?"],
        [["3442556", "3443000", r"3\.44\d? million"], ["2721957", "2722000", r"2\.72\d? million"]],
        ["billion"],
    ),
    ("tech5-avg", ["What is TECH5's 3-year average net revenue?"], ["4091326"], ["billion"]),
    ("follow-up", ["What was TECH5's net revenue in 2022?", "And in 2021?"], ["2510606"], ["5178432"]),
    (
        "bunna-vpn",
        ["Which Diffie-Hellman group and phase 1 lifetime does Bunna Bank's VPN to NID use?"],
        [r"\b19\b", "43200"],
        # A note that *other* banks use group 2 is fine; only a wrong lifetime fails it.
        ["28800"],
    ),
    (
        "laxton",
        ["When was Laxton Group's financial proposal going to be opened, per their technical evaluation result?"],
        ["february 16"],
        [],
    ),
    (
        "furniture",
        ["In the furniture list, what is the unit price of an Esol table and the overall total?"],
        # The sheet has a "Total" of 2,040,000 and a further 2,090,000 line
        # (+50,000), so either reading of "overall total" counts.
        ["120000", ["2040000", "2090000"]],
        [],
    ),
    ("admin-count", ["How many documents are in the Admin folder?"], [[r"\b3\b", "three"]], [r"\b1[45]\d documents"]),
    ("not-in-corpus", ["Who is the executive sponsor of Project Falcon?"], [REFUSAL], []),
]

# Multi-part questions of the kind stakeholders ask - facts spread through a
# long answer, so these are scored on the whole answer, not its opening.
# Drafts built from the current corpus; swap in real stakeholder questions.
COMPLEX_QUESTIONS = [
    (
        "deep-ambassador",
        ["Who was proposed as Fayda's brand ambassador, through what selection method, for how long, "
         "and what grounds were given for that choice?"],
        ["kenenisa", "single.?source", r"(two|2) (consecutive )?years", [r"trust", r"credib"]],
        [],
    ),
    (
        "deep-supplementary",
        ["Under the supplementary agreement between NIDP and Ethio Telecom, what additional obligations does "
         "NIDP take on, who bears damages caused by registration personnel, and which agreement prevails if "
         "the two conflict?"],
        ["dashboard", "direct damages", [r"supplementary agreement.{0,80}prevail", r"agreement (one|1).{0,80}prevail",
                                         r"prevail.{0,80}supplementary"]],
        [],
    ),
    (
        "deep-kits-procurement",
        ["Summarize the procurement of the 1000 biometric registration kits: what kind of tender it was, how "
         "Laxton Group fared, and the key dates."],
        [
            "1000",
            [r"international competitive bidding", r"\bicb\b"],
            [r"passed", r"successful(ly)?", r"qualified", r"cleared"],
            [r"february 16", r"16 february"],
            [r"february 29", r"29 february", r"february 19", r"19 february"],
        ],
        [],
    ),
    (
        "deep-tech5-trend",
        ["How did TECH5's net revenue change from 2020 through 2022, and was the company profitable in 2022?"],
        [
            ["3442556", "3443000", r"3\.44\d? million"],
            ["2510606", "2511000", r"2\.51\d? million"],
            ["6320815", "6321000", r"6\.32\d? million"],
            [r"loss", r"not profitable", r"unprofitable"],
        ],
        ["billion"],
    ),
    (
        "deep-telecom-obligations",
        ["In the registration partnership between NIDP and Ethio Telecom, what does NIDP provide to Ethio "
         "Telecom, and what is Ethio Telecom accountable for?"],
        ["dashboard", [r"strategic plan", r"registration targets"], [r"damages", r"accountab"]],
        [],
    ),
]


def score(answer: str, expected: list, rejected: list, whole: bool = False) -> bool:
    # Whitespace collapsed so a pattern can match across a line break
    # (long answers put a heading between "prevails" and the agreement).
    head = " ".join(answer.lower().replace(",", "").split())
    if not whole:
        head = head[:HEAD_CHARS]
    for item in expected:
        options = item if isinstance(item, list) else [item]
        if not any(re.search(opt, head) for opt in options):
            return False
    return not any(re.search(pattern, head) for pattern in rejected)


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
    # "auto" leaves the mode to question-complexity routing; "research" is
    # app/research.py's multi-round mode (minutes per question).
    council_mode = None if mode in ("auto", "research") else mode.startswith("council")
    if council_mode:
        settings.council_angles = int(mode.split(":")[1])
    conversation_id = None
    try:
        for i, question in enumerate(turns):
            _model_tool_calls = 0
            start = time.time()
            result = await run_agentic_ask(
                question, conversation_id=conversation_id, council_mode=council_mode,
                mode="research" if mode == "research" else None,
            )
            elapsed = time.time() - start
            conversation_id = result["conversation_id"]
        return {
            "answer": result["answer"].strip(),
            "seconds": elapsed,
            "sources": len(result["sources"]),
            "model_tools": _model_tool_calls,
            "warnings": len(result["citation_warnings"]),
            "flagged": [f"{w['text'][:160]} -> {w['reason']}" for w in result["citation_warnings"]],
            "council": result["council"],
            "answered_by": result["mode"],
        }
    finally:
        if conversation_id:
            await delete_conversation(conversation_id)


async def main(modes: list[str], runs: int, only: list[str] | None, save: str | None, question_set: str) -> None:
    await open_pool()
    totals = {m: {"correct": 0, "seconds": 0.0, "warnings": 0, "n": 0} for m in modes}
    saved = open(save, "w") if save else None
    try:
        questions = {
            "simple": [(*q, False) for q in QUESTIONS],
            "complex": [(*q, True) for q in COMPLEX_QUESTIONS],
            "all": [(*q, False) for q in QUESTIONS] + [(*q, True) for q in COMPLEX_QUESTIONS],
        }[question_set]
        for label, turns, expected, rejected, whole in questions:
            if only and label not in only:
                continue
            for mode in modes:
                for _ in range(runs):
                    try:
                        r = await run_case(turns, mode)
                    except Exception as exc:  # one broken case shouldn't sink the whole comparison
                        r = {"answer": f"ERROR {type(exc).__name__}: {exc}", "seconds": 0.0, "sources": 0,
                             "model_tools": 0, "warnings": 0, "council": None, "answered_by": None}
                    ok = score(r["answer"], expected, rejected, whole) and not r["answer"].startswith("ERROR")
                    if saved:
                        saved.write(json.dumps({"label": label, "mode": mode, "correct": ok, **r}) + "\n")
                        saved.flush()
                    t = totals[mode]
                    t["correct"] += ok
                    t["seconds"] += r["seconds"]
                    t["warnings"] += r["warnings"]
                    t["n"] += 1
                    print(
                        f"{'PASS' if ok else 'FAIL'} {label:14} {mode:10} {r['seconds']:5.1f}s "
                        f"sources={r['sources']:2} model_tools={r['model_tools']} flagged={r['warnings']} "
                        f"{r.get('answered_by') or 'error':8} "
                        f"| {r['answer'][:110]!r}",
                        flush=True,
                    )
    finally:
        await close_pool()
        if saved:
            saved.close()

    print("\nmode        correct   avg latency   flagged sentences")
    for mode, t in totals.items():
        print(f"{mode:10}  {t['correct']:2}/{t['n']:<2}     {t['seconds'] / t['n']:6.1f}s      {t['warnings']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--modes", nargs="+", default=["agent", "council:4", "council:8", "council:12"])
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--only", nargs="+", help="question labels to run (default: all)")
    parser.add_argument("--save", help="write every full answer to this JSONL file")
    parser.add_argument("--set", dest="question_set", choices=["simple", "complex", "all"], default="simple")
    args = parser.parse_args()
    asyncio.run(main(args.modes, args.runs, args.only, args.save, args.question_set))
