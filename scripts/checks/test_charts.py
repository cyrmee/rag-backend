"""Checks charts in answers (app/charts.py): the ```chart block the model
writes, the stream filter that holds each block back until it is complete
and valid, the grounding of its values in the sources, and the fence-aware
citation segmenter.

The offline part needs no servers: it streams answers through
ChartStreamFilter in every token size from 1 to 9 characters and requires
the output to match the whole-answer filter byte for byte, and checks the
malformed, ungrounded and unclosed cases. --live adds an end-to-end run
against the API: it uploads sample.pptx (a native chart of Kestrel's
quarterly revenue, Q1-Q4: 42, 58, 51, 73) under a throwaway name, asks for
a chart and for a single figure, checks the stream and the done event, and
deletes what it created.

    python scripts/checks/test_charts.py
    python scripts/checks/test_charts.py --live --base-url http://127.0.0.1:8001
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from keycloak_token import KeycloakAuth
from test_ask_regression import FIXTURES, ask, check, check_done, failures

from app.charts import ChartStreamFilter, filter_answer

SPEC = {
    "type": "bar", "title": "Kestrel revenue by quarter ($M)", "labels": ["Q1", "Q2", "Q3", "Q4"],
    "series": [{"name": "Revenue", "values": [42, 58, 51, 73]}], "unit": "$M", "sources": [2],
}
SOURCES = [
    {"content": "Project Zephyr relay uses the Corvid-9 transceiver."},
    {"content": "| Quarter | Revenue ($M) |\n| --- | --- |\n| Q1 | 42 |\n| Q2 | 58 |\n| Q3 | 51 |\n| Q4 | 73 |"},
]
UPLOAD_NAME = "zz-charts-kestrel.pptx"


def block(spec: dict, tag: str = "chart") -> str:
    return f"```{tag}\n{json.dumps(spec)}\n```\n"


def stream(answer: str, size: int, sources: list[dict]) -> tuple[str, ChartStreamFilter]:
    f = ChartStreamFilter(sources)
    out = "".join(f.feed(answer[i:i + size]) for i in range(0, len(answer), size))
    return out + f.flush(), f


def offline() -> None:
    answers = {
        "chart in the flow": "Revenue rose over the year [2].\n\n" + block(SPEC) + "\nQ4 was the best quarter [2].",
        "chart at the end": "Here it is [2]:\n" + block(SPEC).rstrip("\n"),
        "chart at the start": block(SPEC) + "That is the whole year [2].",
        "indented in a list": "- first point\n  " + block(SPEC).replace("\n```", "\n  ```") + "- next point",
        "tag variants": "a\n``` Chart \n" + json.dumps(SPEC) + "\n```   \nb",
        "two charts": "a\n" + block(SPEC) + "b\n" + block({**SPEC, "type": "line"}) + "c",
        "other fence untouched": "see\n```json\n{\"type\": \"bar\"}\n```\nand `inline` and ``double``",
        "backticks in prose": "use ``` to open a fence, or ` alone",
        "no chart": "Kestrel's Q4 revenue was 73 [2].",
    }
    for name, answer in answers.items():
        whole, charts, warnings = filter_answer(answer, SOURCES)
        same = all(stream(answer, n, SOURCES)[0] == whole for n in range(1, 10))
        check(same, f"offline: {name}: every token size streams the same text", repr(whole[:120]))
        check(not warnings, f"offline: {name}: no warnings", str(warnings))
    whole, charts, _ = filter_answer(answers["two charts"], SOURCES)
    check([c["type"] for c in charts] == ["bar", "line"], "offline: two charts both reported", str(charts))
    check(whole.count("```chart") == 2, "offline: two charts both in the text", whole)
    whole, charts, _ = filter_answer(answers["other fence untouched"], SOURCES)
    check(whole == answers["other fence untouched"] and not charts, "offline: json fence passes through", whole)
    whole, charts, _ = filter_answer(answers["tag variants"], SOURCES)
    check(whole == "a\n" + block(SPEC) + "b" and charts, "offline: ' Chart ' tag normalized", whole)
    whole, charts, _ = filter_answer(answers["indented in a list"], SOURCES)
    check(len(charts) == 1 and whole.count("\n  ```") == 2 and "\n  {" in whole,
          "offline: indented block keeps the list item's indentation on every line", whole)

    # Normalization: ints stay ints, labels given as numbers become strings,
    # unknown keys are dropped, type case is folded.
    spec = {**SPEC, "labels": [2021, 2022, 2023, 2024], "type": "Line", "extra": True,
            "series": [{"name": "Revenue", "values": [42.0, 58, 51.5, 73]}]}
    whole, charts, _ = filter_answer(block(spec), [{"content": "42 58 51.5 73"}])
    check(charts and charts[0]["labels"] == ["2021", "2022", "2023", "2024"], "offline: numeric labels -> strings", str(charts))
    check(charts and charts[0]["type"] == "line" and "extra" not in charts[0], "offline: type folded, extra key dropped", str(charts))
    check(charts and charts[0]["series"][0]["values"] == [42, 58, 51.5, 73], "offline: integral floats emitted as ints", str(charts))
    check(charts and charts[0]["sources"] == [], "offline: dangling source dropped", str(charts))

    # Slips with one obvious reading are repaired.
    repaired = {
        "citation-style sources": json.dumps(SPEC).replace("[2]", "[1][2]"),
        "trailing commas": json.dumps(SPEC).replace("73]", "73,]").replace("[2]}", "[2],}"),
    }
    for name, body in repaired.items():
        _, charts, _ = filter_answer(f"```chart\n{body}\n```\n", SOURCES)
        check(len(charts) == 1 and charts[0]["series"][0]["values"] == [42, 58, 51, 73],
              f"offline: {name}: repaired", str(charts))
    check(charts and charts[0]["sources"] == [2], "offline: trailing commas: sources intact", str(charts))
    _, charts, _ = filter_answer("```chart\n" + repaired["citation-style sources"] + "\n```\n", SOURCES)
    check(charts and charts[0]["sources"] == [1, 2], "offline: citation-style sources: both kept", str(charts))

    # Malformed blocks are dropped from the text entirely.
    bad = {
        "not json": "```chart\n{type: bar}\n```\n",
        "not an object": "```chart\n[1, 2, 3]\n```\n",
        "unknown type": block({**SPEC, "type": "scatter"}),
        "values/labels mismatch": block({**SPEC, "series": [{"name": "R", "values": [1, 2]}]}),
        "pie with two series": block({**SPEC, "type": "pie", "series": [SPEC["series"][0], SPEC["series"][0]]}),
        "string values": block({**SPEC, "series": [{"name": "R", "values": ["42", "58", "n/a", "73"]}]}),
        "null value": block({**SPEC, "series": [{"name": "R", "values": [42, None, 51, 73]}]}),
        "empty labels": block({**SPEC, "labels": [], "series": [{"name": "R", "values": []}]}),
        "never closed": "intro\n```chart\n" + json.dumps(SPEC),
    }
    for name, text in bad.items():
        answer = "before\n" + text + "after"
        whole, charts, warnings = filter_answer(answer, SOURCES)
        streamed, f = stream(answer, 3, SOURCES)
        check(not charts and "```" not in whole, f"offline: {name}: dropped", repr(whole))
        check(streamed == whole and f.dropped == 1, f"offline: {name}: streamed the same, counted", repr(streamed))
    whole, _, _ = filter_answer("before\n" + bad["never closed"], SOURCES)
    check(whole == "before\nintro\n", "offline: never closed: prose before it kept", repr(whole))

    # The wrong form for the data is corrected: a pie over time, with a
    # negative value, with one slice or with too many becomes bars; a pie
    # of shares stays a pie.
    from app.charts import is_time_series
    for labels in (["2020", "2021", "2022"], ["Q1", "Q2", "Q3", "Q4"], ["Jan", "Feb", "Mar"], ["FY2023", "FY2024"],
                   ["2024-01", "2024-02"], ["Q3 2024", "Q4 2024", "Q1 2025"], ["2023/24", "2024/25"], ["March 2024", "April 2024"]):
        check(is_time_series(labels), f"offline: time labels recognised: {labels}")
    for labels in (["Amhara", "Tigray", "Afar"], ["Engineering", "Legal"], ["Esol table", "Chair", "Desk"], ["1", "2", "3"]):
        check(not is_time_series(labels), f"offline: category labels not time: {labels}")
    pie_over_time = {**SPEC, "type": "pie", "labels": ["2020", "2021", "2022", "2023"]}
    _, charts, _ = filter_answer(block(pie_over_time), SOURCES)
    check(charts and charts[0]["type"] == "bar", "offline: pie over years -> bar", str(charts))
    donut_over_quarters = {**SPEC, "type": "donut"}
    _, charts, _ = filter_answer(block(donut_over_quarters), SOURCES)
    check(charts and charts[0]["type"] == "bar", "offline: donut over quarters -> bar", str(charts))
    shares = {**SPEC, "type": "pie", "labels": ["Amhara", "Tigray", "Afar", "Benishangul"]}
    _, charts, _ = filter_answer(block(shares), SOURCES)
    check(charts and charts[0]["type"] == "pie", "offline: pie of regional shares stays a pie", str(charts))
    negative = {**shares, "series": [{"name": "Net", "values": [42, -58, 51, 73]}]}
    _, charts, _ = filter_answer(block(negative), SOURCES)
    check(charts and charts[0]["type"] == "bar", "offline: pie with a negative value -> bar", str(charts))
    one_slice = {**shares, "labels": ["All"], "series": [{"name": "Net", "values": [42]}]}
    _, charts, _ = filter_answer(block(one_slice), SOURCES)
    check(charts and charts[0]["type"] == "bar", "offline: single-slice pie -> bar", str(charts))
    line_over_time = {**SPEC, "type": "line", "labels": ["2020", "2021", "2022", "2023"]}
    _, charts, _ = filter_answer(block(line_over_time), SOURCES)
    check(charts and charts[0]["type"] == "line", "offline: line over years untouched", str(charts))

    # Grounding: a value no cited source states is flagged, not removed.
    invented = {**SPEC, "series": [{"name": "Revenue", "values": [42, 58, 51, 80]}]}
    whole, charts, warnings = filter_answer(block(invented), SOURCES)
    check(charts and charts[0].get("unverified") == [80], "offline: invented value listed as unverified", str(charts))
    check(len(warnings) == 1 and "80" in warnings[0]["reason"] and warnings[0]["source_indices"] == [2],
          "offline: invented value warned", str(warnings))
    wrong_source = {**SPEC, "sources": [1]}
    _, charts, warnings = filter_answer(block(wrong_source), SOURCES)
    check(charts and charts[0].get("unverified") == [42, 58, 51, 73], "offline: wrong source -> all unverified", str(charts))
    no_source = {**SPEC, "sources": []}
    _, charts, warnings = filter_answer(block(no_source), SOURCES)
    check(charts and "unverified" not in charts[0] and not warnings, "offline: no sources -> checked against all", str(charts))
    formatted = {**SPEC, "labels": ["2021", "2022"], "series": [{"name": "Net revenue", "values": [6320815, 7.5]}], "sources": [1]}
    _, charts, _ = filter_answer(block(formatted), [{"content": "Net revenue was $6,320,815 (margin 7.5%)."}])
    check(charts and "unverified" not in charts[0], "offline: thousands separators and % match", str(charts))
    scaled = {**SPEC, "labels": ["2021"], "series": [{"name": "Net revenue ($M)", "values": [6.32]}], "sources": [1]}
    _, charts, _ = filter_answer(block(scaled), [{"content": "Net revenue was $6,320,815."}])
    check(charts and charts[0].get("unverified") == [6.32], "offline: a scaled value is unverified", str(charts))

    # The proposal prefilter: an answer with no chart and three or more
    # distinct numbers (citation markers aside) gets the proposal call.
    from app.charts import wants_proposal
    check(wants_proposal("Revenue was $42M in Q1 [1], $58M in Q2 [2] and $73M in Q4 [3]."), "offline: three figures -> proposal")
    check(not wants_proposal("Revenue was $73M in Q4 [1][2][3][4]."), "offline: one figure with four citations -> none")
    check(not wants_proposal("Q4 revenue was 73 and Q1 was 42 [1]."), "offline: two figures -> none")
    check(not wants_proposal("Here [1].\n" + block(SPEC)), "offline: answer that has a chart -> none")
    check(wants_proposal("Amhara 5,903.3, Tigray 55,012.1, Afar 26,047.95 people [4]."), "offline: thousands separators counted once each")

    from app.charts import build_proposal_prompt
    prompt = build_proposal_prompt("Trend?", "Q1 42, Q2 58, Q3 51 [2].", SOURCES)
    check('{"type": "line"' in prompt and "[2] | Quarter" in prompt and "Question: Trend?" in prompt,
          "offline: proposal prompt carries the example, the chunks and the question", prompt[-300:])

    # The citation segmenter keeps fence lines verbatim, without citations.
    from app.agent import _segment_citations
    answer = "Revenue rose [2].\n\n" + block(SPEC) + "\nQ4 was best [2]."
    segments = _segment_citations(answer)
    texts = [s["text"] for s in segments]
    check(json.dumps(SPEC) in texts, "offline: segmenter keeps the spec line verbatim", str(texts))
    check(all(not s["source_indices"] for s in segments if s["text"].startswith(("```", "{"))),
          "offline: fence lines carry no citations", str(segments))
    expected = "Revenue rose.\n\n" + block(SPEC) + "\nQ4 was best."
    check("\n".join(texts) == expected, "offline: rejoined segments rebuild the answer", "\n".join(texts))


async def live(base_url: str) -> None:
    import httpx

    conversations: list[str] = []
    async with httpx.AsyncClient(base_url=base_url, timeout=600.0, auth=KeycloakAuth()) as client:
        try:
            data = (FIXTURES / "sample.pptx").read_bytes()
            pptx = "application/vnd.openxmlformats-officedocument.presentationml.presentation"
            up = await client.post("/upload", files={"file": (UPLOAD_NAME, data, pptx)})
            check(up.status_code == 200 and up.json()["chunks_ingested"] > 0, "live: upload", up.text[:200])

            r = await ask(client, {"question": "Draw a bar chart of Kestrel's revenue by quarter."})
            check_done(r, "live: chart")
            if r["done"]:
                conversations.append(r["done"]["conversation_id"])
                charts = r["done"]["charts"]
                check(len(charts) >= 1, "live: chart: done has a chart", r["answer"][:300])
                check(r["answer"].count("```chart") == len(charts), "live: chart: one fence per chart", r["answer"])
                if charts:
                    c = charts[0]
                    print(f"     {json.dumps(c)}")
                    check(c["type"] == "bar", "live: chart: type is bar", c["type"])
                    values = [v for s in c["series"] for v in s["values"]]
                    check(sorted(values) == [42, 51, 58, 73], "live: chart: values are 42, 58, 51, 73", str(values))
                    check(len(c["labels"]) == 4, "live: chart: four labels", str(c["labels"]))
                    check("unverified" not in c, "live: chart: every value grounded", str(c.get("unverified")))
                    check(bool(c["sources"]), "live: chart: cites its sources", str(c))
                chart_warnings = [w for w in r["done"]["citation_warnings"] if w["reason"].startswith("chart")]
                check(not chart_warnings, "live: chart: no chart warnings", str(chart_warnings))
                prose = r["answer"].split("```chart")[0] + r["answer"].split("```")[-1]
                check(len(prose.strip()) > 40, "live: chart: prose around the chart", r["answer"][:300])
                check("json" not in prose.lower(), "live: chart: prose doesn't mention JSON", prose[:300])
                detail = (await client.get(f"/conversations/{r['done']['conversation_id']}")).json()
                stored = [m["content"] for m in detail["messages"] if m["role"] == "assistant"]
                check(stored and stored[-1] == r["answer"], "live: chart: stored answer matches the stream")

            f = await ask(client, {"question": "What was Kestrel's revenue in Q4?"})
            check_done(f, "live: figure")
            if f["done"]:
                conversations.append(f["done"]["conversation_id"])
                check("73" in f["answer"], "live: figure: answer has 73", f["answer"][:200])
                check(not f["done"]["charts"], "live: figure: no chart for a single figure", f["answer"][:300])
        finally:
            for cid in conversations:
                await client.delete(f"/conversations/{cid}")
            deleted = await client.delete(f"/documents/{UPLOAD_NAME}")
            check(deleted.status_code == 200, "live: delete the upload", deleted.text[:200])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true", help="also run against the API")
    parser.add_argument("--base-url", default="http://127.0.0.1:8001")
    args = parser.parse_args()
    offline()
    if args.live:
        asyncio.run(live(args.base_url))
    print(f"\n{'ALL PASSED' if not failures else f'{len(failures)} FAILED: ' + ', '.join(failures)}")
    sys.exit(1 if failures else 0)
