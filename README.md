# rag-backend

Local FastAPI + Ollama + pgvector RAG backend, with an agentic retrieval loop
and multi-format (PDF/DOCX/PPTX/XLSX) ingestion including vision captioning
of embedded charts/images.

## Setup

```bash
docker compose up -d   # postgres+pgvector and minio
docker exec -i $(docker compose ps -q db) psql -U raguser -d ragdb < sql/schema.sql
# upgrading an existing DB created before vision captioning was added?
docker exec -i $(docker compose ps -q db) psql -U raguser -d ragdb < sql/002_vision_columns.sql

ollama pull qwen3-embedding         # or your preferred embedding model tag
ollama pull gemma4:31b-mlx          # or your preferred chat model tag, must support tool-calling
ollama pull qwen3-vl-caption        # or your preferred vision-captioning model tag

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

## Run

```bash
source .venv/bin/activate
set -a && source .env && set +a
uvicorn app.main:app --reload --port 8001
```

- `POST /upload` — multipart file upload (`.pdf`, `.docx`, `.pptx`, `.xlsx`, `.txt`, `.md`).
  Extracts text (tables as markdown), embedded/rendered chart images
  (captioned via the vision model), and native chart data tables where
  available; chunks + embeds + stores everything. Re-uploading the same
  filename replaces its previous chunks (upsert, not append).
- `POST /ask` — `{"question": "..."}`, single-pass retrieve-then-generate:
  retrieves top-k relevant chunks and answers grounded in them.
- `POST /ask/agentic` — `{"question": "..."}`, optional `?max_iterations=N`
  (1-10, default from `MAX_AGENT_ITERATIONS`). The chat model decides when
  and how many times to retrieve (multiple/refined queries for multi-part
  questions), and can call `describe_image` on a specific figure from a
  retrieved chunk for a fresh, deeper vision-model look before answering.
- `GET /documents` — list ingested filenames and chunk counts.
- `DELETE /documents/{filename}` — remove all chunks for a file.

## Schema

The `documents` table tags every row with:
- `source_type` — `text` | `chart_data` (an extracted, exact chart/table data
  table) | `image_caption` (a vision-model caption of a chart/figure image).
- `source_format` — `pdf` | `docx` | `pptx` | `xlsx` | `txt` | `md`.
- `source_image_path`, `page_number`, `bbox` — nullable, populated for
  image-derived rows to trace back to the original figure.

## Standalone check scripts

`scripts/checks/` holds ad-hoc scripts that exercise real Ollama/DB calls
(no mocking) to isolate infra issues from API-layer issues:

```bash
python scripts/checks/test_db.py               # insert + pgvector similarity query
python scripts/checks/test_ollama.py            # embedding + generation round trip
python scripts/checks/test_ingestion.py         # full parse -> chunk -> embed -> store pipeline
python scripts/checks/test_tool_calling.py      # confirms the chat model invokes the retrieve tool
python scripts/checks/test_retrieve.py          # app/agent.py's retrieve() against the live DB
python scripts/checks/test_agent_loop.py        # multi-hop question triggers 2+ retrieve calls
python scripts/checks/compare_ask_routes.py     # /ask vs /ask/agentic, latency + tool-call counts
python scripts/checks/test_vision_model.py      # captions real extracted chart images
python scripts/checks/test_describe_image.py    # vision captioning incl. retry/backoff
python scripts/checks/test_describe_image_tool.py  # agent's describe_image tool handler
python scripts/checks/test_caption_throughput.py   # concurrent captioning images/minute
python scripts/checks/test_retrieval_quality.py    # chart-specific queries hit the right row type
python scripts/checks/test_upload_idempotency.py   # re-uploading a file doesn't duplicate rows
```

`scripts/maintenance/` holds one-off operational utilities:

```bash
python scripts/maintenance/dedupe_documents.py       # one-time cleanup of pre-upsert-fix duplicates
python scripts/maintenance/generate_test_fixtures.py # regenerates scripts/fixtures/sample.{pdf,docx,pptx,xlsx}
```

## Notes on models

- `gemma4:31b-mlx` is the current `CHAT_MODEL`. It's not a reasoning model, so
  `/api/generate` returns a clean `response` with no `thinking` field — the
  `<think>...</think>` regex strip in `app/generation.py` is a no-op for it,
  kept as a defensive no-cost fallback in case `CHAT_MODEL` is swapped back to
  a reasoning model (e.g. `deepseek-r1:70b`) via the same env var, no code
  changes required. It also needs to support Ollama's native tool-calling API
  for `/ask/agentic` to work.
- The embedding model's native output is 4096-dim; `EMBED_DIM=1024` in `.env`
  uses Ollama's `dimensions` parameter to truncate it (Matryoshka-style, no
  meaningful quality loss at this ratio). The `documents.embedding` column is
  `vector(1024)` to match.
- `VISION_MODEL` (`qwen3-vl-caption` here) captions extracted/rendered chart
  images. There's no fallback model — if caption quality is poor, the fix is
  prompt iteration in `app/vision.py`, not a model swap.
- Extracted/rendered images persist in the MinIO `miniodata` volume, so they
  survive container restarts as long as that volume isn't removed.
