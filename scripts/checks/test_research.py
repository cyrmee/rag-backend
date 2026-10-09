"""Runs one question through research mode (app/research.py) in-process,
printing every plan/progress event with its timing, then the answer, the
sources it cited, the citation check's warnings and what the documents
didn't cover. For watching a research turn work and timing its stages;
scoring against known answers is compare_council.py --modes research.
Deletes the conversation it creates. Needs the chat (vLLM), embedding
(Ollama) and DB servers running; takes minutes.

    python scripts/checks/test_research.py
    python scripts/checks/test_research.py --web "What does the 2024 procurement law require of NIDP?"
    python scripts/checks/test_research.py --debug   # also prints the notes the writer received, and the gap verdicts
"""

import argparse
import asyncio
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from app.agent import run_agentic_ask_stream
from app.conversations import delete_conversation
from app.db import close_pool, open_pool

DEFAULT_QUESTION = (
    "Summarize the procurement of the 1000 biometric registration kits: what kind of tender it was, "
    "how Laxton Group fared, and the key dates."
)


async def main(question: str, web_search: bool) -> None:
    await open_pool()
    start = time.time()
    conversation_id = None
    answer: list[str] = []
    done: dict = {}
    try:
        async for event in run_agentic_ask_stream(question, web_search=web_search, mode="research"):
            elapsed = time.time() - start
            kind = event["type"]
            if kind == "research_plan":
                print(f"[{elapsed:6.1f}s] plan: {len(event['sub_questions'])} sub-questions, "
                      f"{len(event['documents'])} documents, web={event['web_queries']}")
                for q in event["sub_questions"]:
                    print(f"           - {q}")
                for d in event["documents"]:
                    print(f"           * {d}")
            elif kind == "research_progress":
                print(f"[{elapsed:6.1f}s] {event['stage']:6} round {event['round']}: {event['message']}", flush=True)
            elif kind == "answer":
                answer.append(event["text"])
            elif kind == "done":
                done = event
                conversation_id = event["conversation_id"]
        elapsed = time.time() - start
        print(f"\n[{elapsed:6.1f}s] done: {len(done['sources'])} sources, "
              f"{len(done['citation_warnings'])} flagged sentences, mode={done['mode']}")
        print("\nANSWER:\n" + "".join(answer))
        if done["not_found"]:
            print("\nNOT FOUND:")
            for q in done["not_found"]:
                print(f"- {q}")
        if done["citation_warnings"]:
            print("\nFLAGGED:")
            for w in done["citation_warnings"]:
                print(f"- {w['text'][:120]!r} -> {w['reason']}")
        cited = sorted({i for seg in done["citations"] for i in seg["source_indices"]})
        print(f"\nCITED {len(cited)} of {len(done['sources'])} sources:")
        for i in cited:
            s = done["sources"][i - 1]
            print(f"[{i}] {s['filename']} (page {s['page_number']}) {s['content'][:80]!r}")
    finally:
        if conversation_id:
            await delete_conversation(conversation_id)
        await close_pool()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("question", nargs="?", default=DEFAULT_QUESTION)
    parser.add_argument("--web", action="store_true", help="let the planner add web searches")
    parser.add_argument("--debug", action="store_true", help="show the app's research logging, notes included")
    args = parser.parse_args()
    if args.debug:
        logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
        for name in ("app.research", "app.agent"):
            logging.getLogger(name).setLevel(logging.DEBUG)
    asyncio.run(main(args.question, args.web))
