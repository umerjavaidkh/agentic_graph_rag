# Architecture

## Query path

```
User Query → MCP Router (routing.py + feedback_loop resolver)
                 ├─ Structured Agent → Text-to-Cypher / multistep → Neo4j → charts/tables
                 ├─ Unstructured Agent → hybrid retrieval → Neo4j → narrative + sources
                 └─ Hybrid (compliance role) → both paths → merged answer
                      │
                      ▼
              feedback_loop (observe → label → learn → optional apply)
                      │
                      ▼
              audit log (who / what / when / result — always on)
```

The router's tool descriptions and system prompt are schema-driven: `structured_entity_summary()` (`src/routing.py`) reads the live graph's node labels via `db.labels()` (filtered to exclude the ingested-document tree) and feeds that into the LLM prompt, instead of a hardcoded domain name. Bring your own Neo4j schema and routing adapts — it isn't tied to the bundled Northwind demo.

## Feedback loop

`src/feedback_loop/` — observe pipeline telemetry, label outcomes, optionally improve routing:

| Stage | What happens |
|-------|----------------|
| **Observe** | After `/query` and `/query/stream`, persist compact telemetry (Redis stream or JSONL) |
| **Label** | `POST /feedback/outcome` or eval `--attach-feedback` marks pass/fail per `request_id` |
| **Learn** | Aggregate pass rates by question **pattern** (intent flags) + retrieval mode or route tool |
| **Apply** | When `RETRIEVAL_FEEDBACK_ROUTING=true`, `FeedbackRoutingService` adjusts multistep vs text2cypher, document hybrid mode, or MCP route — only after enough labeled samples |

See [CONFIGURATION.md](CONFIGURATION.md#retrieval-feedback-loop) for the ops UI and workflow. Application code imports from `src/feedback_loop`; `src/telemetry/feedback/` is a deprecated shim.

**Unstructured retrieval modes** (selected per question): vector similarity · full-text · graph expand from NER · TOC structural fetch · page-by-number · phrase/fact lookup (URLs, licenses) · chapter-summary rollups for "what does this document/chapter discuss" overview questions.

## Retrieval strategies (registry pattern)

`src/retrieval/strategy_registry.py` is a flat, name-keyed registry — `register_structured`/`register_unstructured` at module load, `get_structured`/`get_unstructured` at call time — mirroring `src/document/parser_registry.py`'s extension-keyed parser dispatch. Each strategy is a standalone class with a single `retrieve(...)` method; nothing but the registry name couples a strategy to its caller.

| Side | Registered strategies | Dispatched by |
|------|------------------------|---------------|
| Structured | `text2cypher`, `multistep` | `StructuredRetriever.retrieve()` — schema + feedback-loop routing hint decide which runs first, with fallback |
| Unstructured | `structural_box_list`, `subsection_tree`, `structural_toc`, `structural_page`, `graph_rag_hybrid` | `HybridRetrieveMixin.hybrid_retrieve()` — question-shape checks (`is_box_list_request`, `is_toc_question`, …) pick a strategy, falling through to `graph_rag_hybrid` |

Unstructured strategies share six constructor-injected services (`src/retrieval/unstructured/services/`) — `RankingService`, `GraphSeedService`, `DocumentResolver`, `LexicalService`, `ChapterSummaryService`, `ResponseFormatter` — built once in `strategies/registration.py` and passed to every strategy that needs them, instead of the mixins-with-implicit-`self`-state pattern this replaced. Adding a new strategy means writing one class and registering it — no existing strategy or dispatch code changes.

## Audit log

`src/audit/` — dual-backend (Redis stream / daily JSONL) append-only event store, fire-and-forget so a write failure never blocks a real request. Records `QUERY`, `ACCESS_DENIED`, and `INGESTION_SUBMITTED` events with user, tenant, role, resource, and result. Query via `GET /audit/events` (admin-only) or the **Audit log** panel on `/feedback`. Defaults **on** — see [CONFIGURATION.md](CONFIGURATION.md).

## Ingestion write path

```
PDF → LightPdfParser
        │
        ├── Axis 1: Document → Chapter → Section → Page → Region
        ├── Page vision (optional, ENABLE_PAGE_VISION=true)
        ├── Axis 2: Embeddings · NER · Clustering · LLM relationship pass
        │             (parallel thread pools)
        └── Chapter summaries: 1 bounded LLM call per Chapter, from its
              own Sections' titles/excerpts (src/semantic/chapter_summary.py)
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

## Multi-tenancy

Property-based `tenant_id` isolation (not per-tenant databases) across Neo4j nodes/edges, Qdrant payloads, and MinIO key prefixes — gated behind `MULTI_TENANCY_ENABLED` (default off). Structured Cypher gets a deterministic (non-LLM) tenant-filter injection on every MATCH clause, with a fail-closed validator that refuses execution rather than trusting an LLM to remember the filter. See [CONFIGURATION.md](CONFIGURATION.md).

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
│   │   ├── parser_base.py     # DocumentParser Protocol
│   │   ├── parser_registry.py # Extension-keyed parser dispatch (.pdf:light, .pdf:table-aware)
│   │   ├── light/parser.py    # LightPdfParser — default (PyMuPDF + pdfplumber)
│   │   ├── table_aware/parser.py  # TableAwarePdfParser — fixes table over-segmentation
│   │   ├── page_vision.py     # Optional vision enrichment
│   │   └── versioning.py      # Logical doc ID, revision plans, hashing
│   ├── exporter/exporter.py   # Neo4jExporter — UNWIND batched writes
│   ├── semantic/
│   │   ├── axis2.py            # Axis 2 (parallel NER + LLM relationship pass)
│   │   └── chapter_summary.py  # Chapter-level rollup summaries (1 LLM call/chapter)
│   ├── retrieval/
│   │   ├── strategy_registry.py   # Name-keyed registry shared by both sides
│   │   ├── unstructured/          # DocumentRAGRetriever (thin facade)
│   │   │   ├── retriever.py       # Public API + backward-compat exports
│   │   │   ├── mixins/hybrid.py   # Dispatch only: question-shape → registered strategy
│   │   │   ├── strategies/        # box, subsection, toc, page, full_hybrid (+ base Protocol, registration)
│   │   │   ├── services/          # ranking, graph_seeds, document_resolver, lexical, chapter_summary, formatter
│   │   │   ├── query_intent.py    # Question-shape routing (TOC, page, synthesis, …)
│   │   │   ├── toc_retrieval.py, visual_retrieval.py, executor.py
│   │   └── structured/            # StructuredRetriever (facade)
│   │       ├── retriever.py
│   │       ├── strategies/        # text2cypher, multistep (+ base Protocol)
│   │       ├── cypher/            # generate, validate, repair, pipeline, tenant_injection
│   │       ├── multistep/         # planner, executor, context
│   │       ├── schema/ · policies/ · formatting/
│   ├── graph/                  # Neo4j constants, lifecycle helpers, tenancy
│   ├── audit/                  # Audit event store + recorder (Redis / JSONL)
│   ├── presentation/           # UI blocks (markdown, tables, charts)
│   ├── conversation/           # Thread memory + clarification
│   ├── feedback_loop/          # Observe → label → learn → optional routing apply
│   │   ├── pattern.py · profile.py · store.py · record.py · hints.py · dashboard.py
│   │   ├── resolver.py         # Shared MCP tool resolution (router + stream)
│   │   └── routing/             # Policy-based FeedbackRoutingService
│   ├── static/
│   │   ├── chat.html · upload.html · feedback.html   # Feedback + audit ops monitor
│   ├── auth/                   # RBAC + OIDC (Google JWT, JIT provision, deps)
│   ├── streaming/               # NDJSON /query/stream orchestrator
│   └── prompts/                 # LLM prompts
├── eval/                       # JSON smoke suites + validators
├── scripts/run_rag_eval.py     # Regression eval against /query
├── tests/                      # 400+ unit tests (pytest)
├── docker-compose.yml          # Neo4j + Redis + API + worker
├── Dockerfile
└── .env.example
```

## Neo4j access

| Purpose | Value |
|---------|-------|
| Browser | http://localhost:17474 |
| Bolt URL (in Browser login) | `neo4j://localhost:17687` |
| Username / Password | `neo4j` / `password123` |

Ports 17474 / 17687 avoid clashing with a local Neo4j on 7474 / 7687.

```bash
docker exec -it graphrag-neo4j cypher-shell -u neo4j -p password123
```
