# Agentic GraphRAG

> Neo4j-powered GraphRAG engine that unifies **structured analytics**, **document retrieval**, and **agentic orchestration** in a single knowledge layer.

Built with **Neo4j · FastAPI · LangGraph · OpenAI**.

---

## What it does

Most RAG systems store everything as flat chunks in a vector database. This project separates retrieval into two specialized paths:

| Path | Knowledge | Techniques |
|------|-----------|------------|
| **Structured** | Business entities, metrics, orders, products | Text-to-Cypher · graph traversal · aggregations |
| **Unstructured** | Policies, PDFs, manuals, reports | Hierarchical document graph · vector + full-text + graph expansion · TOC/visual retrieval |

An **MCP-style router** (`routing.py`) picks the right path — or both — for each query.

---

## What's new (June 2026)

| Change | Details |
|--------|---------|
| **Lightweight PDF parser** | PyMuPDF + pdfplumber. ~1 GB image, no Java. |
| **Image storage removed** | No binary JPEGs. Visual content stored as `Page.visual_content` text in Neo4j. |
| **Document versioning** | Immutable `DocRevision` per upload. Same file → skipped. Changed file → new revision, old one expired. |
| **Scalable ingestion** | Redis + RQ workers. Durable job state, auto-retry, per-doc lock, candidate-pair cap. Falls back to in-process `BackgroundTasks` when `REDIS_URL` is unset. |
| **Parallel Axis 2** | NER and LLM relationship passes run in parallel thread pools (8 NER / 6 LLM calls concurrently). |
| **Batched Neo4j writes** | `UNWIND` grouped by label/rel type. Default chunk 2 000 nodes. |
| **Bulk upload UI** | Drop multiple PDFs at once; each gets its own live status card on `/upload`. |
| **New API endpoints** | `GET /ingest/jobs` · `GET /ingest/queue/status`. No more 409 on concurrent uploads. |

---

## Tech stack

| Layer | Technology |
|-------|------------|
| Graph database | Neo4j 5.x |
| AI orchestration | LangGraph |
| API | FastAPI + Uvicorn |
| LLM / Embeddings | OpenAI (gpt-4o-mini, text-embedding-3-small) |
| PDF parsing | PyMuPDF + pdfplumber |
| Job queue | Redis + RQ *(optional — in-process fallback when unset)* |
| Containers | Docker / Docker Compose |

---

## Quick start

```bash
git clone https://github.com/umerjavaidkh/agentic_graph_rag.git
cd agentic_graph_rag
cp .env.example .env          # add OPENAI_API_KEY
docker compose up --build
```

Open:

| Page | URL |
|------|-----|
| **Chat** | http://localhost:8000/chat |
| **Upload** | http://localhost:8000/upload |
| **API docs** | http://localhost:8000/docs |
| **Health** | http://localhost:8000/health |

> Do **not** set `NEO4J_URI` in `.env` when using the bundled Docker Neo4j — it is wired automatically.

### Enable Redis workers (parallel ingestion)

Add to `.env`:

```env
REDIS_URL=redis://redis:6379/0
```

Then:

```bash
docker compose up --build              # starts Neo4j + Redis + API + 1 worker
docker compose up --scale worker=3    # scale to 3 parallel workers
```

Without `REDIS_URL`, jobs run inside FastAPI via `BackgroundTasks` — fine for local dev.

---

## Uploading documents

Go to **http://localhost:8000/upload**:

- **Drop multiple PDFs** onto the drop zone — they are submitted concurrently.
- Each file appears as a live job card with status, dispatch mode (`WORKER` / `IN-PROCESS`), version, and expandable logs.
- The **queue status bar** shows Redis connectivity, queue depth, and failed job count.

Or via `curl`:

```bash
# Single PDF
curl -X POST http://localhost:8000/ingest/unstructured \
  -F "file=@sample_data_to_test/unstructured/rag_document.pdf" \
  -F "doc_key=rag-document"

# Cypher data (requires ALLOW_CYPHER_INGEST=true)
curl -X POST http://localhost:8000/ingest/cypher \
  -F "file=@sample_data_to_test/structured/northwind-data.cypher" \
  -F "role=compliance_officer"
```

`doc_key` controls versioning: same key + same file → skipped; same key + changed file → new revision.

---

## Example questions

**Structured** (use `regular_001` or `compliance_001`):
```
Which customers ordered the most?
Top 5 products by sales?
```

**Unstructured** (use `public_001` or `compliance_001`):
```
What is the whistleblowing procedure?
Summarize section 3.
```

**Hybrid**: `Show compliance incidents and summarize the related policy guidance.`

---

## Architecture

```
User Query
    │
    ▼
MCP Router (routing.py)
    │
    ├─── Structured Agent  →  Text-to-Cypher  →  Neo4j
    │
    └─── Unstructured Agent  →  Hybrid retrieval  →  Neo4j
    │         (vector + full-text + graph expand + TOC/visual)
    │
    └─── Final Answer
```

**Ingestion write path:**

```
PDF → LightPdfParser
        │
        ├── Axis 1: Document → Chapter → Section → Page → Region
        ├── Page vision (optional, ENABLE_PAGE_VISION=true)
        └── Axis 2: Embeddings · NER · Clustering · LLM relationship pass
                      (parallel thread pools)
              │
              └── Neo4jExporter (UNWIND batched writes) → Neo4j
```

**With Redis workers:**

```
POST /ingest  →  API  →  Redis queue
                              │
                    ┌─────────┴─────────┐
                 Worker 1           Worker N
                    │                   │
             per-doc Redis lock (same doc serialised, different docs parallel)
             parse → Axis 2 → batched Neo4j writes → update job in Redis
                              │
GET /ingest/jobs/{id}  →  reads from Redis  →  200
```

---

## Configuration

Copy `.env.example` → `.env`. Key variables:

### Core

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | **required** | LLM, embeddings, routing |
| `NEO4J_USER` | `neo4j` | Neo4j username |
| `NEO4J_PASSWORD` | `password123` | Neo4j password |
| `CHAT_MODEL` | `gpt-4o-mini` | LLM for chat and Axis 2 |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | Embedding model |
| `APP_PORT` | `8000` | API port |

### Ingestion & scalability

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

### NEO4J_URI — when to set it

| Setup | Value |
|-------|-------|
| Docker + bundled Neo4j | **Leave unset** |
| Docker + Neo4j on your Mac | `bolt://host.docker.internal:7687` |
| API on Mac + Neo4j in Docker | `bolt://localhost:17687` |
| API on Mac + local Neo4j | `bolt://localhost:7687` |

---

## API reference

```bash
# Query
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "Which customers ordered the most?", "role": "regular_001"}'

# Upload PDF
curl -X POST http://localhost:8000/ingest/unstructured \
  -F "file=@doc.pdf" -F "doc_key=my-doc"

# Job status
curl http://localhost:8000/ingest/jobs/{job_id}
curl http://localhost:8000/ingest/jobs?limit=20
curl http://localhost:8000/ingest/queue/status

# Health
curl http://localhost:8000/health
```

Response fields from `/ingest/jobs/{id}`: `status`, `dispatch` (`worker` / `background_task`), `logical_doc_id`, `revision_id`, `version_number`, `skipped_duplicate`, `logs[]`, `error`.

---

## RBAC (demo users)

| User | Documents | Structured DB |
|------|-----------|---------------|
| `public_001` | ✅ | ❌ |
| `regular_001` | ❌ | ✅ |
| `compliance_001` | ✅ | ✅ |
| `admin_001` | ✅ | ✅ |

---

## Neo4j

| Purpose | Value |
|---------|-------|
| Browser | http://localhost:17474 |
| Bolt URL (in Browser login) | `neo4j://localhost:17687` |
| Username / Password | `neo4j` / `password123` |

Ports 17474 / 17687 avoid clashing with a local Neo4j on 7474 / 7687.

```bash
# Shell access
docker exec -it graphrag-neo4j cypher-shell -u neo4j -p password123
```

---

## Project structure

```
agentic_graph_rag/
├── sample_data_to_test/
│   ├── unstructured/          # rag_document.pdf, rag_document_2.pdf
│   └── structured/            # northwind-data.cypher
├── src/
│   ├── api.py                 # FastAPI routes, dispatch, job list, queue status
│   ├── config/settings.py     # All env-var settings
│   ├── ingestion/
│   │   ├── service.py         # IngestionManager (store-backed, per-doc lock)
│   │   ├── job_store.py       # RedisJobStore / InMemoryJobStore
│   │   ├── queue.py           # RQ queue helpers
│   │   ├── tasks.py           # run_ingest_job() — RQ worker callable
│   │   └── models.py          # IngestionStatus enum
│   ├── document/
│   │   ├── parser.py          # LightPdfParser (PyMuPDF + pdfplumber)
│   │   ├── page_vision.py     # Optional vision enrichment
│   │   └── versioning.py      # Logical doc ID, revision plans, hashing
│   ├── exporter/exporter.py   # Neo4jExporter — UNWIND batched writes
│   ├── semantic/axis2.py      # Axis 2 (parallel NER + LLM relationship pass)
│   ├── retrieval/
│   │   ├── unstructured/      # DocumentRAGRetriever, TOC helpers
│   │   └── structured/        # Text-to-Cypher + schema repair
│   ├── graph/                 # Neo4j constants, lifecycle helpers
│   ├── presentation/          # UI blocks (markdown, tables, charts)
│   ├── conversation/          # Thread memory + clarification
│   ├── auth/                  # RBAC (roles, knowledge areas)
│   └── prompts/               # LLM prompts
├── tests/
│   ├── test_scalable_pipeline_unit.py
│   ├── test_document_versioning_unit.py
│   └── test_toc_retrieval_unit.py
├── docker-compose.yml         # Neo4j + Redis + API + worker
├── Dockerfile
└── .env.example
```

---

## Troubleshooting

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

**Access denied on structured queries** — structured data needs `regular_001`, `compliance_001`, or `admin_001`. PDF questions need `public_001`, `compliance_001`, or `admin_001`.

**Rebuild slow** — only rebuild the changed service:
```bash
docker compose up -d --build app
```

---

## Local development (without Docker)

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
# .env: set OPENAI_API_KEY, NEO4J_URI=bolt://localhost:7687, NEO4J_PASSWORD
uvicorn src.api:app --reload --host 0.0.0.0 --port 8000
```

---

## Demo

| Ingestion | Unstructured | Structured |
| :---: | :---: | :---: |
| [![Ingestion](https://img.youtube.com/vi/2983DqSe0GM/0.jpg)](https://www.youtube.com/watch?v=2983DqSe0GM) | [![Unstructured](https://img.youtube.com/vi/s3Eceo20Eq4/0.jpg)](https://www.youtube.com/watch?v=s3Eceo20Eq4) | [![Structured](https://img.youtube.com/vi/XvigWQ5mB1g/0.jpg)](https://www.youtube.com/watch?v=XvigWQ5mB1g) |

**Medium article:** [Agentic Graph RAG — architecture and walkthrough](https://medium.com/p/0ee1f6baae26)

---

## Security

- Never commit `.env` — it is gitignored
- `ALLOW_CYPHER_INGEST` and `ALLOW_DB_RESET` are dangerous in production — keep them `false`
- Rotate your OpenAI key if it was ever exposed
