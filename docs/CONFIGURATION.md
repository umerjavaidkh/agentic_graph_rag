# Configuration

Copy `.env.example` → `.env`. Only `OPENAI_API_KEY` is required to get started; everything else has a working default.

## Core

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | **required** | LLM, embeddings, routing |
| `NEO4J_USER` | `neo4j` | Neo4j username |
| `NEO4J_PASSWORD` | `password123` | Neo4j password |
| `CHAT_MODEL` | `gpt-4o-mini` | Document RAG synthesis; default for other stages |
| `STRUCTURED_MODEL` | *(CHAT_MODEL)* | Text-to-Cypher + structured answers |
| `ROUTING_MODEL` | *(CHAT_MODEL)* | MCP routing (`search_documents` vs `query_data`) |
| `AXIS2_MODEL` | *(CHAT_MODEL)* | Ingestion NER + relationship LLM pass |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | Retrieval + ingest embeddings |
| `VISION_MODEL` | `gpt-4o-mini` | Page vision (`ENABLE_PAGE_VISION=true`) |
| `APP_PORT` | `8000` | API port |

Active model resolution: `GET /config/models`

## Multi-tenancy

| Variable | Default | Description |
|----------|---------|-------------|
| `MULTI_TENANCY_ENABLED` | `false` | Property-based `tenant_id` isolation across Neo4j/Qdrant/MinIO |
| `DEFAULT_TENANT_ID` | `default` | Tenant id used when multi-tenancy is off or unspecified |
| `DOCUMENT_KNOWLEDGE_AREA_ID` | `esg` | KnowledgeArea id that gates document-RAG RBAC access — set this if you replace the seeded demo RBAC schema with your own |

## Audit logging

| Variable | Default | Description |
|----------|---------|-------------|
| `AUDIT_LOG_ENABLED` | `true` | Record who-did-what-when events (query, access denial, ingestion) |
| `AUDIT_LOG_STORE_QUESTION` | `true` | Include the question text in query audit events |
| `AUDIT_LOG_DIR` | `data/audit_log` | JSONL fallback directory (used when Redis is unavailable) |
| `AUDIT_LOG_JSONL_RETAIN_DAYS` | `90` | Days of JSONL audit files to retain |

Browse the trail at `GET /audit/events` (admin-only) or the **Audit log** panel on `/feedback`.

## Ingestion & scalability

| Variable | Default | Description |
|----------|---------|-------------|
| `REDIS_URL` | *(unset = in-process)* | Set to `redis://redis:6379/0` for workers |
| `INGEST_QUEUE_NAME` | `ingest` | RQ queue name |
| `AXIS2_NER_CONCURRENCY` | `8` | Parallel NER calls per doc |
| `AXIS2_LLM_PAIR_CONCURRENCY` | `6` | Parallel LLM relationship calls per doc |
| `AXIS2_MAX_LLM_PAIRS` | `300` | Cap on candidate pairs sent to LLM |
| `NEO4J_WRITE_BATCH` | `2000` | UNWIND chunk size for bulk writes |
| `DOC_SKIP_DUPLICATE_HASH` | `true` | Skip ingest when same file already active |
| `DOC_VERSION_RETAIN_METADATA` | `true` | Keep expired `DocRevision` nodes for audit |
| `ENABLE_PAGE_VISION` | `false` | Vision model descriptions for PDF pages |
| `ALLOW_CYPHER_INGEST` | `false` | Enable `.cypher` file upload endpoint |
| `ALLOW_DB_RESET` | `false` | Enable `/admin/reset-neo4j` |

### Enable Redis workers (parallel ingestion)

```env
REDIS_URL=redis://redis:6379/0
```

```bash
docker compose up --build              # starts Neo4j + Redis + API + 1 worker
docker compose up --scale worker=3     # scale to 3 parallel workers
```

Without `REDIS_URL`, jobs run inside FastAPI via `BackgroundTasks` — fine for local dev.

## Retrieval feedback loop

| Variable | Default | Description |
|----------|---------|-------------|
| `RETRIEVAL_FEEDBACK_ENABLED` | `false` | Record pipeline telemetry after queries (no behavior change) |
| `RETRIEVAL_FEEDBACK_ROUTING` | `false` | Apply labeled hints to routing/retrieval (`true` = self-improvement) |
| `RETRIEVAL_FEEDBACK_STORE_QUESTION` | `false` | Store first 120 chars of question in feedback events (privacy: keep `false` in prod) |
| `RETRIEVAL_FEEDBACK_MIN_SAMPLES` | `30` | Minimum labeled outcomes before a hint can apply |
| `RETRIEVAL_FEEDBACK_MIN_MARGIN` | `0.15` | Required pass-rate gap between best and second-best mode |
| `REDIS_URL` | *(unset)* | Recommended for production feedback aggregates (same Redis as ingestion) |

Typical workflow:

```env
RETRIEVAL_FEEDBACK_ENABLED=true
RETRIEVAL_FEEDBACK_ROUTING=false   # observe first; set true when labels are ready
RETRIEVAL_FEEDBACK_STORE_QUESTION=false
REDIS_URL=redis://redis:6379/0
```

1. Use **chat** (👍 / 👎 on each reply) or run eval with `--attach-feedback` to label pass/fail.
2. Open `/feedback` in a second browser tab while testing.
3. When patterns have enough labeled samples (`RETRIEVAL_FEEDBACK_MIN_SAMPLES`), enable `RETRIEVAL_FEEDBACK_ROUTING=true` and watch **Routing applied** in recent events.

```bash
# Label a prior query (e.g. from eval or thumbs)
curl -X POST http://localhost:8000/feedback/outcome \
  -H "Content-Type: application/json" \
  -d '{"request_id": "abc123", "passed": true}'

# Inspect pattern stats + suggested hint (read-only)
curl "http://localhost:8000/feedback/stats?question=Top%205%20products%20by%20revenue&agent=structured"

# Full ops dashboard JSON (powers /feedback UI)
curl http://localhost:8000/feedback/dashboard
```

Feedback shares the same Redis instance as ingestion (different key namespaces) — fine at current scale.

## Authentication

See [AUTHENTICATION.md](AUTHENTICATION.md) for the full auth/RBAC variable reference.

## NEO4J_URI — when to set it

| Setup | Value |
|-------|-------|
| Docker + bundled Neo4j | **Leave unset** |
| Docker + Neo4j on your Mac | `bolt://host.docker.internal:7687` |
| API on Mac + Neo4j in Docker | `bolt://localhost:17687` |
| API on Mac + local Neo4j | `bolt://localhost:7687` |
