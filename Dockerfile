# Agentic Graph RAG — lightweight image. Structured queries, chat, and PDF ingest.
#
# Two stages: compilers and headers are needed to BUILD wheels (hdbscan has no
# aarch64 wheel and compiles from source) but never to run them, so the final
# image copies just the finished virtualenv and drops the toolchain.
FROM python:3.11-slim-bookworm AS builder

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    ENABLE_PDF_INGEST=true

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

ENV VIRTUAL_ENV=/opt/venv PATH="/opt/venv/bin:$PATH"
RUN python -m venv "$VIRTUAL_ENV"

COPY requirements-slim.txt ./
# torch arrives transitively via sentence-transformers (cross-encoder
# reranking), and nothing constrained WHICH build, so pip resolved the CUDA
# one: cublas 543MB + cudnn 445MB + torch 427MB + cufft 214MB + triton 185MB
# — roughly 1.5GB of NVIDIA runtime downloaded on every build and shipped in
# every image, on machines that have no NVIDIA GPU to run it. Installing the
# CPU build first satisfies the dependency, so the transitive resolution
# never reaches for the CUDA wheels.
#
# The cache mount keeps pip's downloads outside the layer. BuildKit evicts
# this layer whenever its cache exceeds the daemon's GC ceiling, and it is by
# far the largest one, so it was being evicted and re-downloaded constantly —
# the mount means an evicted layer rebuilds from local wheels instead of the
# network.
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip \
    && pip install --index-url https://download.pytorch.org/whl/cpu torch \
    && pip install -r requirements-slim.txt


# ── runtime ────────────────────────────────────────────────────────────────
# Only what the code needs at run time: no compilers, no git, no pip cache.
FROM python:3.11-slim-bookworm

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    ENABLE_PDF_INGEST=true \
    VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:$PATH"

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libgl1 \
    libglib2.0-0 \
    tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/venv /opt/venv

COPY . .

RUN mkdir -p data/assets tmp_ingest output/ingestion \
    && chmod +x scripts/docker-entrypoint.sh

EXPOSE 8000
ENTRYPOINT ["/app/scripts/docker-entrypoint.sh"]
