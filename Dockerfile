# Agentic Graph RAG — slim demo (~1 GB). Structured queries + chat; no PDF ingest.
FROM python:3.11-slim-bookworm

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    PIP_NO_CACHE_DIR=1 \
    ENABLE_PDF_INGEST=false

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-slim.txt ./
RUN pip install --upgrade pip \
    && pip install -r requirements-slim.txt

COPY . .

RUN mkdir -p data/assets tmp_ingest output/ingestion \
    && chmod +x scripts/docker-entrypoint.sh

EXPOSE 8000
ENTRYPOINT ["/app/scripts/docker-entrypoint.sh"]
