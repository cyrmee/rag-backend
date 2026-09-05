# rag-backend

Local FastAPI + Ollama + pgvector RAG backend.

## Setup

```bash
docker compose up -d
docker exec -i $(docker compose ps -q db) psql -U raguser -d ragdb < sql/schema.sql

ollama pull qwen3-embedding      # or your preferred embedding model tag
ollama pull gemma4:31b-mlx       # or your preferred chat model tag

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
```

`.env` is loaded from the process environment for scripts and `uvicorn`; either
`export $(cat .env)` in your shell or use a tool like `direnv`/`honcho` before
running commands below.

## Run

```bash
source .venv/bin/activate
set -a && source .env && set +a
uvicorn app.main:app --reload --port 8001
```

- `POST /upload` — multipart file upload (`.pdf`, `.docx`, `.txt`, `.md`), chunks + embeds + stores it.
- `POST /ask` — `{"question": "..."}`, retrieves top-k relevant chunks and answers grounded in them.
- `GET /documents` — list ingested filenames and chunk counts.
- `DELETE /documents/{filename}` — remove all chunks for a file.

## Standalone test scripts

Run these to isolate DB/Ollama issues from API-layer issues:

```bash
python scripts/test_db.py         # insert + pgvector similarity query
python scripts/test_ollama.py     # embedding + generation round trip
python scripts/test_ingestion.py  # full parse -> chunk -> embed -> store pipeline
```

## Notes on models

- `gemma4:31b-mlx` is the current `CHAT_MODEL`. It's not a reasoning model, so
  `/api/generate` returns a clean `response` with no `thinking` field — the
  `<think>...</think>` regex strip in `app/generation.py` is a no-op for it,
  kept as a defensive no-cost fallback in case `CHAT_MODEL` is swapped back to
  a reasoning model (e.g. `deepseek-r1:70b`) via the same env var, no code
  changes required.
- The embedding model's native output is 4096-dim; `EMBED_DIM=1024` in `.env`
  uses Ollama's `dimensions` parameter to truncate it (Matryoshka-style, no
  meaningful quality loss at this ratio). The `documents.embedding` column is
  `vector(1024)` to match.
