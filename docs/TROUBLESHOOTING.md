# Troubleshooting

**App not loading (port 8000 refused)**
```bash
docker ps --filter name=graphrag
docker logs graphrag-app --tail 50
# Common cause: missing OPENAI_API_KEY, or NEO4J_URI set to localhost in .env
docker compose up -d app
```

**Neo4j Browser: "Connection failed"** — use `neo4j://localhost:17687`, not `localhost:7687`.

**Worker not picking up jobs**
```bash
docker ps --filter name=worker
docker logs $(docker ps --filter name=worker -q | head -1) --tail 30
docker exec graphrag-redis redis-cli ping        # should return PONG
curl http://localhost:8000/ingest/queue/status   # check failed_jobs[]
```

**Job status lost after restart** — set `REDIS_URL` for durable storage; without it state lives only in the API process.

**Access denied on structured queries** — with Google sign-in, use `compliance_officer` or `admin` (default: `AUTH_DEFAULT_ROLE=compliance_officer`). In dev sidebar mode: structured needs `regular_001` / `regular_office`; documents need `public_001` / `public`; both graphs need `compliance_001` or `admin_001`.

**Ingestion returns 401/403** — sign in on `/upload` with an email listed in `AUTH_EMAIL_ROLE_MAP=…=admin`. Body `role`/`user_id` fields are **not** accepted on ingest routes.

**Google sign-in: Error 400 `origin_mismatch`** — add `http://localhost:8000` to Authorized JavaScript origins in Google Cloud Console (match the exact URL you open in the browser).

**Rebuild slow** — only rebuild the changed service:
```bash
docker compose up -d --build app
```

## Local development (without Docker)

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
# .env: set OPENAI_API_KEY, NEO4J_URI=bolt://localhost:7687, NEO4J_PASSWORD
uvicorn src.interface.api:app --reload --host 0.0.0.0 --port 8000
```
