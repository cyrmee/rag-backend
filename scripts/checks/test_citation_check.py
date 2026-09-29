"""Measures council mode's citation check (app/council.py verify_citations)
on claims with known answers: correct claims that must pass, and wrong
numbers / wrong cited source that must be flagged. Pulls real sources from
the live corpus, so it needs the chat (vLLM), embedding (Ollama) and DB
servers running.

    python scripts/checks/test_citation_check.py
    python scripts/checks/test_citation_check.py --runs 5
"""

import argparse
import asyncio
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from app import council
from app.db import close_pool, open_pool


def _contains(source: dict, text: str) -> bool:
    return text.replace(",", "") in re.sub(r"\s+", " ", source["content"]).replace(",", "")


def _first(sources: list[dict], predicate) -> int:
    return next(i for i, s in enumerate(sources, start=1) if predicate(s))


async def main(runs: int) -> None:
    await open_pool()
    try:
        rows = await council.search_documents(
            ["TECH5 net revenue 2022", "TECH5 income statement", "Bunna Bank IPsec VPN phase 1"], 20,
        )
    finally:
        await close_pool()
    sources = [{"content": r["content"]} for r in rows]

    revenue = _first(sources, lambda s: _contains(s, "6,320,815"))
    vpn = _first(sources, lambda s: "diffie" in s["content"].lower() and _contains(s, "43200"))
    unrelated = _first(sources, lambda s: not _contains(s, "6,320,815") and "diffie" not in s["content"].lower())

    # (claim, cited sources, should be flagged)
    cases = [
        ("TECH5's net revenue in 2022 was $6,320,815.", [revenue], False),
        ("Bunna Bank's VPN uses Diffie-Hellman group 19 with a phase 1 lifetime of 43200 seconds.", [vpn], False),
        ("This figure is stated in the company's financial documents.", [revenue], False),
        ("TECH5's net revenue in 2022 was $6,320,815, according to the income statement.", [revenue, unrelated], False),
        ("The 2022 figure of $6,320,815 is higher than the 2021 figure.", [revenue], False),
        ("TECH5's net revenue in 2022 was $7,480,200.", [revenue], True),
        ("Bunna Bank's VPN uses Diffie-Hellman group 14 with a phase 1 lifetime of 86400 seconds.", [vpn], True),
        ("TECH5's net revenue in 2022 was $6,320,815.", [unrelated], True),
        ("Bunna Bank's VPN uses Diffie-Hellman group 19.", [revenue], True),
    ]
    segments = [{"text": claim, "source_indices": cited} for claim, cited, _ in cases]

    flag_counts = [0] * len(cases)
    for _ in range(runs):
        flagged = {(w["text"], tuple(w["source_indices"])) for w in await council.verify_citations(segments, sources)}
        for k, (claim, cited, _) in enumerate(cases):
            flag_counts[k] += (claim, tuple(cited)) in flagged

    for (claim, cited, should_flag), count in zip(cases, flag_counts):
        expectation = "should FLAG" if should_flag else "should pass"
        print(f"{expectation:11} flagged {count}/{runs}  cites {cited}  {claim[:70]!r}")

    caught = sum(c for (_, _, s), c in zip(cases, flag_counts) if s)
    false_alarms = sum(c for (_, _, s), c in zip(cases, flag_counts) if not s)
    n_bad = sum(1 for *_, s in cases if s) * runs
    n_good = sum(1 for *_, s in cases if not s) * runs
    print(f"\ncaught {caught}/{n_bad} bad citations; false alarms {false_alarms}/{n_good}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=3)
    asyncio.run(main(parser.parse_args().runs))
