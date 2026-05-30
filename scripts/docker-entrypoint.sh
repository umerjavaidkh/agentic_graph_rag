#!/usr/bin/env bash
set -euo pipefail

cd /app

if [[ -z "${OPENAI_API_KEY:-}" ]] || [[ "${OPENAI_API_KEY}" == "sk-your-key-here" ]]; then
  echo "ERROR: Set OPENAI_API_KEY in .env (copy from .env.example)." >&2
  exit 1
fi

echo "==> Waiting for Neo4j…"
python scripts/wait_for_neo4j.py

echo "==> Demo data (Northwind)…"
python scripts/init_demo_data.py

echo "==> Starting API on http://0.0.0.0:8000"
exec uvicorn src.api:app --host 0.0.0.0 --port 8000
