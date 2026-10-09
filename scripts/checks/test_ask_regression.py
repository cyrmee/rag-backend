"""End-to-end regression test of the HTTP API, run against the live server:
upload -> /ask (agent, follow-up, council, list_documents, web_search) ->
conversations -> cleanup. Checks answers to known questions as well as the
SSE event stream and the shape of the done event, so it catches both model
server regressions (tool calling, streaming, reasoning) and API regressions.

It uploads one fixture under a throwaway name and deletes it, and every
conversation it creates, afterwards. Needs the API server (default
http://127.0.0.1:8001) and, behind it, the chat, embedding and DB servers;
takes ~5-10 minutes.

    python scripts/checks/test_ask_regression.py
    python scripts/checks/test_ask_regression.py --base-url http://10.0.0.5:8001
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

import httpx

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"
# sample.docx: "Kestrel" quarterly revenue table (Q1-Q4: 42, 58, 51, 73) plus a chart image.
UPLOAD_NAME = "zz-regression-kestrel.docx"
COPY_NAME = "zz-regression-kestrel-copy.docx"
SOURCE_KEYS = {"content", "filename", "source_type", "source_format", "page_number", "document_url"}
DONE_KEYS = {"sources", "conversation_id", "citations", "user_message_id", "assistant_message_id", "title",
             "citation_warnings"}

failures: list[str] = []


def check(ok: bool, name: str, detail: str = "") -> bool:
    print(f"{'PASS' if ok else 'FAIL'} {name}{f' - {detail}' if detail and not ok else ''}", flush=True)
    if not ok:
        failures.append(name)
    return ok


async def ask(client: httpx.AsyncClient, body: dict) -> dict:
    """Reads one /ask SSE stream into {events, answer, tools, calls, done}."""
    events, answer, calls, done, error = [], "", [], None, None
    async with client.stream("POST", "/ask", json=body) as resp:
        resp.raise_for_status()
        event, data = None, []
        async for line in resp.aiter_lines():
            if line.startswith("event: "):
                event = line[len("event: "):]
            elif line.startswith("data: "):
                data.append(line[len("data: "):])
            elif line == "" and event:
                payload = "\n".join(data)
                events.append(event)
                if event == "answer":
                    answer += payload
                elif event == "tool_call":
                    calls.append(json.loads(payload))
                elif event == "done":
                    done = json.loads(payload)
                elif event == "error":
                    error = payload
                event, data = None, []
    tools = [c["name"] for c in calls]
    return {"events": events, "answer": answer, "tools": tools, "calls": calls, "done": done, "error": error}


def check_done(r: dict, name: str) -> None:
    """Shape checks every /ask turn must pass."""
    done = r["done"]
    if not check(done is not None and r["error"] is None, f"{name}: stream ends in done", str(r["error"])):
        return
    check(r["events"][-1] == "done", f"{name}: done is the last event", str(r["events"][-3:]))
    check(DONE_KEYS <= done.keys(), f"{name}: done has all fields", str(DONE_KEYS - done.keys()))
    check(bool(r["answer"].strip()), f"{name}: non-empty answer")
    sources = done["sources"]
    check(all(SOURCE_KEYS <= s.keys() for s in sources), f"{name}: every source has all fields")
    pairs = [(s["filename"], s["content"]) for s in sources]
    check(len(pairs) == len(set(pairs)), f"{name}: no duplicate sources")
    bad = [i for c in done["citations"] for i in c["source_indices"] if not 1 <= i <= len(sources)]
    check(not bad, f"{name}: citation indices point at sources", f"out of range: {bad}")
    check(isinstance(done["citation_warnings"], list), f"{name}: citation_warnings is a list")


async def main(base_url: str) -> None:
    conversations: list[str] = []
    async with httpx.AsyncClient(base_url=base_url, timeout=600.0) as client:
        try:
            r = await client.get("/health")
            check(r.status_code == 200 and r.json().get("status") == "ok", "health")

            # --- upload ---------------------------------------------------
            data = (FIXTURES / "sample.docx").read_bytes()
            docx = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            first = await client.post("/upload", files={"file": (UPLOAD_NAME, data, docx)})
            check(first.status_code == 200 and first.json()["chunks_ingested"] > 0, "upload", first.text[:200])
            again = await client.post("/upload", files={"file": (UPLOAD_NAME, data, docx)})
            check(again.status_code == 200 and again.json()["chunks_ingested"] == first.json()["chunks_ingested"],
                  "re-upload under the same name replaces, same chunk count", again.text[:200])
            copy = await client.post("/upload", files={"file": (COPY_NAME, data, docx)})
            check(copy.status_code == 409 and copy.json()["detail"].get("existing_filename") == UPLOAD_NAME,
                  "identical bytes under another name -> 409", f"{copy.status_code} {copy.text[:200]}")
            docs = (await client.get("/documents")).json()
            check(any(d["filename"] == UPLOAD_NAME for d in docs), "/documents lists the upload")

            # --- agent: fact from the upload, then a follow-up -------------
            r = await ask(client, {"question": "What was Kestrel's revenue in Q4, in $M?"})
            check_done(r, "agent")
            if r["done"]:
                conversations.append(r["done"]["conversation_id"])
                check("retrieve" in r["tools"], "agent: called retrieve", str(r["tools"]))
                check("73" in r["answer"], "agent: answer has Q4 = 73", r["answer"][:200])
                check(any(s["filename"] == UPLOAD_NAME for s in r["done"]["sources"]),
                      "agent: upload is among the sources")
                check(bool(r["done"]["title"]), "agent: new conversation gets a title")

                f = await ask(client, {"question": "And in Q2?", "conversation_id": r["done"]["conversation_id"]})
                check_done(f, "follow-up")
                if f["done"]:
                    check(f["done"]["conversation_id"] == r["done"]["conversation_id"],
                          "follow-up: same conversation")
                    check("58" in f["answer"], "follow-up: answer has Q2 = 58", f["answer"][:200])
                    check(f["done"]["title"] is None, "follow-up: title unchanged (null)")

            # --- council ---------------------------------------------------
            c = await ask(client, {
                "question": "Who is the lead engineer on Project Zephyr, and which transceiver does its relay use?",
                "council": True,
            })
            check_done(c, "council")
            if c["done"]:
                conversations.append(c["done"]["conversation_id"])
                low = c["answer"].lower()
                check("okafor" in low and "corvid-9" in low, "council: answer names Okafor and Corvid-9",
                      c["answer"][:200])
                # Council reports all its planned document searches as one retrieve call.
                angles = [a for call in c["calls"] if call["name"] == "retrieve"
                          for a in (call.get("args") or {}).get("angles", [])]
                check(len(angles) >= 2, "council: several planned search angles ran", str(c["calls"])[:300])

            # --- corpus question -> list_documents -------------------------
            d = await ask(client, {"question": "How many documents are in the knowledge base?"})
            check_done(d, "list_documents")
            if d["done"]:
                conversations.append(d["done"]["conversation_id"])
                check("list_documents" in d["tools"], "list_documents: tool called", str(d["tools"]))

            # --- web search -------------------------------------------------
            w = await ask(client, {"question": "What is the capital of Ethiopia?", "web_search": True})
            check_done(w, "web_search")
            if w["done"]:
                conversations.append(w["done"]["conversation_id"])
                check("web_search" in w["tools"], "web_search: tool called", str(w["tools"]))
                check("addis ababa" in w["answer"].lower(), "web_search: answer says Addis Ababa", w["answer"][:200])

            # --- validation -------------------------------------------------
            for n in (0, 11):
                v = await client.post("/ask", params={"max_iterations": n}, json={"question": "hi"})
                check(v.status_code == 422, f"max_iterations={n} -> 422", str(v.status_code))

            # --- conversations ----------------------------------------------
            if r["done"]:
                cid = r["done"]["conversation_id"]
                listed = (await client.get("/conversations")).json()
                check(any(x["id"] == cid for x in listed), "/conversations lists it")
                detail = (await client.get(f"/conversations/{cid}")).json()
                roles = [m["role"] for m in detail["messages"]]
                check(roles.count("user") == 2 and roles.count("assistant") == 2,
                      "/conversations/{id} has both turns", str(roles))
        finally:
            for cid in conversations:
                await client.delete(f"/conversations/{cid}")
            if conversations:
                gone = await client.get(f"/conversations/{conversations[0]}")
                check(gone.status_code == 404, "deleted conversation -> 404", str(gone.status_code))
            deleted = await client.delete(f"/documents/{UPLOAD_NAME}")
            check(deleted.status_code == 200, "delete the upload", deleted.text[:200])
            missing = await client.delete(f"/documents/{UPLOAD_NAME}")
            check(missing.status_code == 404, "deleting it again -> 404", str(missing.status_code))

    print(f"\n{'ALL PASSED' if not failures else f'{len(failures)} FAILED: ' + ', '.join(failures)}")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8001")
    asyncio.run(main(parser.parse_args().base_url))
