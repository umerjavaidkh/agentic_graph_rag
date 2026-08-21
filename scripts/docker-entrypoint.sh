#!/usr/bin/env bash
set -euo pipefail

cd /app

if [[ -z "${OPENAI_API_KEY:-}" ]] || [[ "${OPENAI_API_KEY}" == "sk-your-key-here" ]]; then
  echo "ERROR: Set OPENAI_API_KEY in .env (copy from .env.example)." >&2
  exit 1
fi

echo "==> Waiting for Neo4j…"
python scripts/wait_for_neo4j.py

# A command means this container is something other than the API -- an RQ
# worker, a one-off script, a shell. Run it and get out of the way.
#
# This used to end at `exec uvicorn` no matter what, so compose's
# `command:` was accepted and then silently discarded: every worker
# container started a second copy of the API instead of `rq worker`,
# nothing consumed the ingest queue, and scaling workers added API
# replicas. Ingestion still worked only because the API falls back to
# running jobs in-process when it cannot reach the queue.
if [[ $# -gt 0 ]]; then
  echo "==> Running: $*"
  exec "$@"
fi

# Only the API seeds demo data. Doing it here rather than before the
# dispatch above keeps N scaled workers from racing to load the same
# fixtures on startup.
echo "==> Demo data (Northwind)…"
python scripts/init_demo_data.py

echo "==> Starting API on http://0.0.0.0:8000"
exec uvicorn src.interface.api:app --host 0.0.0.0 --port 8000
