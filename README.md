# rag-backend

Local FastAPI + vLLM (chat) + Ollama (embeddings, vision) + pgvector RAG backend, with an agentic retrieval loop
and multi-format (PDF/DOCX/PPTX/XLSX) ingestion including vision captioning
of embedded charts/images.

## Setup

```bash
docker compose up -d   # postgres+pgvector and minio
docker exec -i $(docker compose ps -q db) psql -U raguser -d ragdb < sql/schema.sql
# upgrading an existing DB created before vision captioning was added?
docker exec -i $(docker compose ps -q db) psql -U raguser -d ragdb < sql/002_vision_columns.sql
# upgrading a DB created before duplicate-file detection? apply it, then hash existing files:
docker exec -i $(docker compose ps -q db) psql -U raguser -d ragdb < sql/009_document_files.sql
python scripts/maintenance/backfill_file_hashes.py   # add --delete to remove identical copies

ollama pull qwen3-embedding         # or your preferred embedding model tag
ollama pull qwen3-vl-caption        # or your preferred vision-captioning model tag

# chat model: separate vLLM (OpenAI-compatible) server, see "Notes on models"
vllm serve Qwen/Qwen3-30B-A3B \
  --served-model-name chat \
  --port 8101 \
  --max-model-len 32768 \
  --enable-auto-tool-choice --tool-call-parser hermes \
  --reasoning-parser qwen3

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
```

LibreOffice headless is also required on the machine running the app (not the
Ollama host) — used to render PPTX/XLSX charts to images when their
underlying data can't be extracted directly. Install it via your package
manager (e.g. `pacman -S libreoffice-still`, `apt install libreoffice`).

Extracted/rendered chart images are stored in MinIO (S3-compatible object
storage), not on local disk — `docker compose up -d` starts it alongside
Postgres. The app creates its bucket (`MINIO_BUCKET`, default `rag-images`)
automatically on startup. Browse stored images via the console at
`http://localhost:9001` (default credentials in `.env.example`;
**change `MINIO_ROOT_USER`/`MINIO_ROOT_PASSWORD` in `docker-compose.yml`
and the matching `MINIO_ACCESS_KEY`/`MINIO_SECRET_KEY` in `.env` before
running this anywhere but local dev**). `documents.source_image_path` holds
the object key (`{document_id}/{image_id}.png`), fetched via `app/storage.py`
by both the ingestion pipeline and the agent's `describe_image` tool.

`.env` is loaded from the process environment for scripts and `uvicorn`; either
`export $(cat .env)` in your shell or use a tool like `direnv`/`honcho` before
running commands below.

## Run with Docker

The backend, PostgreSQL, MinIO and SearXNG all run from `docker-compose.yml`;
the model servers (vLLM chat, Ollama embeddings/vision) run natively on the
host, because on macOS containers can't use the Apple GPU (Metal).

```bash
cp .env.example .env          # then edit; set MINIO_PUBLIC_ENDPOINT (below)
docker compose up -d --build  # backend on :8001, waits for db/minio health
docker compose logs -f backend
```

- **Model servers**: the backend reaches them on the host at
  `host.docker.internal:8101` (vLLM) and `:11434` (Ollama) by default;
  override with `DOCKER_CHAT_BASE_URL` / `DOCKER_OLLAMA_BASE_URL` in `.env`.
  The `DATABASE_URL`/`MINIO_ENDPOINT`/`OLLAMA_BASE_URL`/... values in `.env`
  are for running outside Docker - compose replaces them with service names.
- **`MINIO_PUBLIC_ENDPOINT`**: host:port users' browsers reach MinIO at
  (e.g. this machine's LAN or Tailscale IP + `:9000`). Links to original
  files in `/ask` sources are signed for it; unset, they point at
  `localhost:9000` and only open on this machine.
- **Database**: a fresh `pgdata` volume gets `sql/schema.sql` plus every
  numbered migration, in order (`docker/initdb.sh`). An existing volume is
  never touched - apply new migrations to it by hand:
  `docker compose exec -T db psql -U raguser -d ragdb < sql/0NN_....sql`.
- **Maintenance scripts** run inside the container, e.g.
  `docker compose exec backend python scripts/maintenance/backfill_file_hashes.py`.
- **Linux + NVIDIA GPU host**: `docker compose --profile gpu up -d` also starts
  `vllm` and `ollama` containers (set `DOCKER_CHAT_BASE_URL=http://vllm:8000/v1`
  and `DOCKER_OLLAMA_BASE_URL=http://ollama:11434`). **Untested** - there is no
  NVIDIA hardware here yet.
- After changing code: `docker compose up -d --build backend`.

## Run without Docker

```bash
source .venv/bin/activate
set -a && source .env && set +a
uvicorn app.main:app --reload --port 8001
```

- `POST /upload` — multipart file upload (`.pdf`, `.docx`, `.pptx`, `.xlsx`, `.txt`, `.md`).
  Extracts text (tables as markdown), embedded/rendered chart images
  (captioned via the vision model), and native chart data tables where
  available; chunks + embeds + stores everything. Re-uploading the same
  filename replaces its previous chunks (upsert, not append). A file whose
  bytes are identical to one already uploaded under a different filename
  is rejected with `409` and `{"detail": {"message", "existing_filename"}}`
  (matched by SHA-256, stored in `document_files`).
- `POST /ask` — `{"question": "...", "conversation_id": "..."}` (the latter
  optional), optional `?max_iterations=N` (1-10, default from
  `MAX_AGENT_ITERATIONS`). Always agentic and always streamed as
  [Server-Sent Events](https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events)
  — there's no single-pass or non-streaming variant. The chat model decides
  when and how many times to retrieve (multiple/refined queries for
  multi-part questions), can call `list_documents` for questions about the
  corpus itself (counts/filenames, not content), and can call
  `describe_image` on a specific figure from a retrieved chunk for a
  fresh, deeper vision-model look before answering. Each `retrieve` call is
  itself document-routed and multi-angle-decomposed (see
  `app/retrieval.py`): the question is split into a few distinct search
  angles, whole documents are ranked before diving into chunks, and each
  of the top few documents gets its own focused chunk search —
  considerably more thorough than a flat corpus-wide chunk search, at the
  cost of real added latency (multiple LLM/DB round trips per retrieve
  call, itself possibly called multiple times).
  Pass a prior response's `conversation_id` to continue that conversation
  (the model sees prior turns as history — only the user/assistant text of
  each turn is stored, not the tool-call choreography that produced it);
  omit it, or pass a stale/unknown one, to start a new one. Events:
  `thinking`/`answer` (tokens as they're generated — a turn that results
  in a tool call has no `answer` content, confirmed against the live
  model), `tool_call` (`{"name": "retrieve"|"list_documents"|
  "describe_image", "args": {...}}`) and `tool_result` (`{"name": ...,
  "preview": "..."}"`) around each tool invocation, then one final `done`
  event with `{"sources": [...], "conversation_id": "..."}` — `sources` is
  a list of objects, not plain strings: `{content, filename, source_type,
  source_format, page_number, document_url}`. `page_number` is a real PDF
  page for `.pdf`, a slide number for `.pptx`, a sheet order index for
  `.xlsx`, or a synthetic paragraph/table index for `.docx` (docx has no
  true page concept at the XML level); `null` for `.txt`/`.md`.
  `document_url` is a MinIO presigned link (1 hour expiry, regenerated
  fresh per request) to the original uploaded file, or `null` if it
  predates this feature and was never stored. An `error` event fires
  instead if Ollama is unreachable mid-stream.
- `GET /conversations` — list conversations (id, timestamps, first
  question as a preview), most recently updated first.
- `GET /conversations/{conversation_id}` — full turn history
  (`[{role, content}, ...]`) for one conversation; 404 if unknown.
- `GET /documents` — list ingested filenames and chunk counts.
- `DELETE /documents/{filename}` — remove all chunks for a file.

## Schema

The `documents` table tags every row with:
- `source_type` — `text` | `chart_data` (an extracted, exact chart/table data
  table) | `image_caption` (a vision-model caption of a chart/figure image).
- `source_format` — `pdf` | `docx` | `pptx` | `xlsx` | `txt` | `md`.
- `page_number` — populated for every row derived from a paginated/sectioned
  format (all of `text`/`chart_data`/`image_caption` for pdf/docx/pptx/xlsx);
  `null` for `.txt`/`.md`, which have no page concept.
- `source_image_path`, `bbox` — nullable, populated only for `image_caption`
  rows to trace back to the original figure (MinIO object key + bounding
  box, PDF-only for `bbox` in practice).

Original uploaded files are stored in MinIO under `documents/{filename}`
(overwritten on re-upload, matching the DB's per-filename upsert), separate
from the extracted chart images. `/ask` responses look this up per source
row and attach a presigned link — see `app/storage.py`.

Text chunking is now per-unit (per page/paragraph/slide/sheet) rather than
one big joined-then-rechunked blob, so that every output chunk keeps an
accurate `page_number`. One side effect: a short unit (e.g. a lone heading)
can surface as its own tiny chunk, and if two different documents happen to
share identical short text, retrieval can legitimately return that same
text from two different files — that's not duplication, since a document
worth deduping against is `(filename, content)`, not `content` alone.

## Standalone check scripts

`scripts/checks/` holds ad-hoc scripts that exercise real model-server/DB calls
(no mocking) to isolate infra issues from API-layer issues:

```bash
python scripts/checks/test_db.py               # insert + pgvector similarity query
python scripts/checks/test_ollama.py            # embedding (Ollama) + generation (vLLM) round trip
python scripts/checks/test_ingestion.py         # full parse -> chunk -> embed -> store pipeline
python scripts/checks/test_tool_calling.py      # confirms the chat model invokes the retrieve tool (streaming + non-streaming)
python scripts/checks/test_retrieve.py          # app/agent.py's retrieve() against the live DB
python scripts/checks/test_agent_loop.py        # multi-hop question triggers 2+ retrieve calls
python scripts/checks/test_vision_model.py      # captions real extracted chart images
python scripts/checks/test_describe_image.py    # vision captioning incl. retry/backoff
python scripts/checks/test_describe_image_tool.py  # agent's describe_image tool handler
python scripts/checks/test_caption_throughput.py   # concurrent captioning images/minute
python scripts/checks/test_retrieval_quality.py    # chart-specific queries hit the right row type
python scripts/checks/test_upload_idempotency.py   # re-uploading a file doesn't duplicate rows
python scripts/checks/compare_council.py        # agent vs council mode on known-answer questions (~25 min)
python scripts/checks/compare_council.py --set complex --runs 3   # same, multi-part stakeholder-style questions
python scripts/checks/compare_council.py --set complex --modes research   # research mode on the same questions (minutes each)
python scripts/checks/compare_council.py --set cross --modes auto   # questions whose answer spans several documents
python scripts/checks/test_citation_check.py    # council citation check: catch rate and false alarms on known claims
```

`scripts/maintenance/` holds one-off operational utilities:

```bash
python scripts/maintenance/dedupe_documents.py       # one-time cleanup of pre-upsert-fix duplicates
python scripts/maintenance/generate_test_fixtures.py # regenerates scripts/fixtures/sample.{pdf,docx,pptx,xlsx}
python scripts/maintenance/backfill_summaries.py     # summaries for documents that have none
python scripts/maintenance/backfill_summaries.py --redo   # rewrite every summary
python scripts/maintenance/backfill_file_hashes.py   # hash files ingested before 009, report identical copies (--delete removes them)
python scripts/maintenance/rechunk_documents.py --dry-run  # rebuild text chunks after a chunking change (keeps captions/chart data)
```

## Notes on models

- Model serving is split across two backends:
  - **Chat** (`/ask`'s agent loop, conversation titles, document summaries,
    query decomposition) runs on a vLLM OpenAI-compatible server at
    `CHAT_BASE_URL` (default `http://localhost:8101/v1`), model
    `CHAT_MODEL` (default `chat`, the `--served-model-name` above; the
    weights are `Qwen/Qwen3-30B-A3B`). `CHAT_API_KEY`, if set, is sent as a
    Bearer token (match vLLM's `--api-key`).
  - **Embeddings and vision captioning** stay on Ollama's native API at
    `OLLAMA_BASE_URL` (`EMBED_MODEL`, `VISION_MODEL`).
- `app/generation.py` is the only code that speaks the chat server's
  OpenAI format; it normalizes responses (tool-call arguments parsed to
  dicts, SSE tool-call fragments accumulated into whole calls, the reasoning
  parser's `reasoning_content`/`reasoning` mapped to `thinking`) so the
  agent loop doesn't care. `/ask` needs the server started with
  `--enable-auto-tool-choice --tool-call-parser hermes` (it's always
  agentic), and `--reasoning-parser qwen3` for `thinking` events to stream
  separately from the answer.
- With `web_search: true`, the agent's first model turn is forced (OpenAI
  `tool_choice`) to call `web_search`, and a `retrieve` with the same query
  runs alongside it so the documents are always checked too. Otherwise, and
  on every later turn, tool use is `"auto"` - the model decides. Titles,
  document summaries and query decomposition run with Qwen3's thinking
  switched off (`chat_template_kwargs.enable_thinking=false`).
- **Council mode** (`app/council.py`) changes how a turn starts: fast
  planner calls (thinking off, run in parallel) propose search angles -
  for document content, one search per who/what/when/where/why/how
  question that applies (up to `COUNCIL_ANGLES`), plus chart/figure
  captions, corpus listings, and web queries when `web_search` is on - and
  code runs all of those searches at once before
  the model's first turn. Document results from every angle are fused by
  rank and cut to `COUNCIL_MAX_CHUNKS`, so more angles broaden the search
  without growing the prompt. The model then answers with that evidence
  already in its history (and can still call tools for more). The searches
  show up as ordinary `tool_call`/`tool_result` events. Afterwards a
  thinking-off check runs per cited sentence, against only that
  sentence's own sources (so a claim can't pass on a source it didn't
  cite); the `done` event's `citation_warnings` lists any it flagged
  (always `[]` outside council mode).
- **Which mode answers:** `"council": true`/`false` on `/ask` forces it.
  Omitted, a thinking-off classifier routes complex questions (several
  facts combined, summaries, comparisons, obligations, histories) to
  council mode and simple ones (a fact or two, document counts/lists,
  small talk) to the regular agent - measured on the complex set: council
  13/15 vs agent 10/15 at the same latency. Turns with attachments stay on
  the regular agent unless forced; `AUTO_COUNCIL=false` turns routing off.
  The `done` event's `council` says which mode answered. In every mode, citation numbers with
  no matching source are dropped from `citations`. `"mode": "agent"|"council"`
  on `/ask` names the mode outright instead.
- **Research mode** (`"mode": "research"` on `/ask`, `app/research.py`) is
  the slow, thorough path for complex stakeholder questions - minutes per
  answer, never chosen automatically. The corpus (~4M tokens) is ~125x the
  chat window, and everything the other modes retrieve goes into one
  prompt; research mode reads outside it. A thinking-on planner turns the
  question into 6-12 sub-questions and, shown every document's summary
  (the whole corpus at document level fits in one prompt), picks up to
  `RESEARCH_MAX_DOCUMENTS` documents to read. Each round then searches
  every sub-question corpus-wide (8 documents x 6 chunks) and inside each
  picked document, fuses the results by rank, cuts them to
  `RESEARCH_MAX_CHUNKS`, reads picked documents under ~16k characters
  whole, and packs it all into batches of `RESEARCH_BATCH_TOKENS` that
  thinking-off calls read in parallel into one-line notes, each tied to
  the excerpt it came from. Every note is then checked against its own
  excerpt (one thinking-off call per excerpt, nothing else in the prompt)
  and dropped if the excerpt doesn't state it - asked to answer the
  questions, the reader will sometimes invent a plausible figure and pin
  it on an unrelated passage; asked whether the passage states it, the
  same model says no (measured: 6 of 23 notes dropped on a financial
  question, all six invented). A gap check then gives a verdict per
  sub-question and writes new searches for the unanswered ones; up to
  `RESEARCH_MAX_ROUNDS` rounds. The answer is written (thinking on,
  streamed, no tools) from the notes grouped by document, citing the
  excerpts behind them - so `sources` and the citation check work exactly
  as in council mode, on the real source text. Every internal call has an
  output cap, since a thinking-off reader that starts looping would
  otherwise run until the window is full (seen: one batch taking 265s).
  On the complex question set: 5/5 correct, 3.2 minutes average, 5.2
  longest, on this machine. Events: `research_plan`
  once (`sub_questions`, `documents`, `web_queries`), `research_progress`
  as each stage moves (`stage` plan/search/read/check/gaps/write, `round`,
  `message`, and `done`/`total` while reading), and the research itself as
  one `research` tool_call/tool_result pair so existing clients show
  something. The `done` event adds `mode` and `not_found` (sub-questions
  the documents didn't answer; the answer says so too). Web search joins
  in only with `web_search: true` (the planner writes the web queries).
  Attachments are folded into the question as in the other modes.
- vLLM fixes the context window at startup (`--max-model-len`); there's no
  per-request `num_ctx`. `CHAT_NUM_CTX` (default 32768) must match it -
  the app only uses it to budget attached-file text (`app/attachments.py`).
- The embedding model's native output is 4096-dim; `EMBED_DIM=1024` in `.env`
  uses Ollama's `dimensions` parameter to truncate it (Matryoshka-style, no
  meaningful quality loss at this ratio). The `documents.embedding` column is
  `vector(1024)` to match.
- `VISION_MODEL` (`qwen3-vl-caption` here) captions extracted/rendered chart
  images. There's no fallback model — if caption quality is poor, the fix is
  prompt iteration in `app/vision.py`, not a model swap.
- Extracted/rendered images persist in the MinIO `miniodata` volume, so they
  survive container restarts as long as that volume isn't removed.
