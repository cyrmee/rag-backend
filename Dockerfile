# syntax=docker/dockerfile:1
FROM python:3.14-slim

# LibreOffice (Calc + Impress only) renders PPTX/XLSX charts to PDF when their
# data can't be read from the file (app/extractors). DejaVu covers Latin
# text, Abyssinica the Ethiopic script, so rendered Amharic isn't tofu.
# The apt and pip cache mounts keep downloaded packages across builds -
# including interrupted ones - so a slow mirror is only paid for once.
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    rm -f /etc/apt/apt.conf.d/docker-clean \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        libreoffice-calc libreoffice-impress \
        fonts-dejavu-core fonts-sil-abyssinica \
        curl

WORKDIR /app

COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt

COPY app ./app
COPY scripts ./scripts
COPY sql ./sql

# LibreOffice writes a user profile under $HOME on first run.
RUN useradd --create-home --uid 1000 rag
USER rag
ENV HOME=/home/rag PYTHONUNBUFFERED=1

EXPOSE 8001
HEALTHCHECK --interval=15s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS http://localhost:8001/health || exit 1

# One worker: per-message attachments live in process memory
# (app/attachments.py), so a second worker wouldn't see them.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8001", "--workers", "1"]
