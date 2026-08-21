# Agentic GraphRAG

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Status: Development in Progress](https://img.shields.io/badge/status-development%20in%20progress-orange.svg)](#status)

**Structured business data and unstructured documents in one Neo4j graph** — SQL-grade analytics *and* multi-hop reasoning over PDFs, Word, PowerPoint and Excel. You pick the source per query; nothing guesses.

**Scored 95/100 on a committed benchmark you can re-run for free.**

---

## Up and running

```bash
git clone https://github.com/umerjavaidkh/agentic_graph_rag.git
cd agentic_graph_rag
cp .env.example .env          # add OPENAI_API_KEY (embeddings)
docker compose up --build
```

That's it — **one URL, one page, everything behind tabs**: [localhost:8000](http://localhost:8000)

| | |
| --- | --- |
| **Chat** · **Ingestion** · **Graph Inspector** · **Feedback** | tabs across the top |
| API docs | `/docs` |

> Don't set `NEO4J_URI` when using the bundled Docker Neo4j — it's wired automatically.

**Load the samples:**

```bash
# structured — 4,000-order e-commerce sample, bundled, no download
python scripts/load_olist.py --source sample_data_to_test/structured/olist-sample

# documents — drag a folder onto /upload, or point at a directory
```

---

## Why this is different

| Typical flat RAG | Agentic GraphRAG |
| --- | --- |
| One chunk index for everything | Dedicated graphs for tables vs. documents |
| Similarity search only | Cypher for metrics + hybrid retrieval for documents |
| Weak on counts, joins, time series | Aggregations, rankings, charts from live Neo4j |
| Loses document structure | Document → Chapter → Section → Page → Region |
| Guesses when context is missing | Eval covers anti-hallucination and empty-result cases |

![Two axes over one corpus: pages linked in reading order around the ring (Axis 1 — PRECEDES / FOLLOWS), and semantic edges cutting across the middle to connect related pages that are nowhere near each other in the document (Axis 2)](docs/images/Gemini_Generated_Image_y27bnoy27bnoy27b.png)

<sub>The ring is **Axis 1** — pages in reading order. The chords are **Axis 2** — semantic links between pages far apart structurally. Both live in the same graph, which is what lets a question reach a page that keyword or page-order navigation never would.</sub>

**Bring your own schema.** The router reads the live graph at runtime instead of hardcoding a demo domain — a guard test fails the build if dataset-specific field names reappear.

---

## The benchmark is the claim

Text-to-Cypher over your own schema, measured against **[100 questions a business user would actually ask](eval/olist_business_suite.json)** — delivery performance, satisfaction drivers, seller concentration, retention, payment mix, and questions the data *cannot* answer.

| | |
| --- | --- |
| **95/100** | up from 79 at first baseline |
| **0** LLM-judge calls | every expected value computed from the graph by hand-written Cypher |
| **Free and deterministic** | same answer twice, costs nothing to run |

```bash
python scripts/eval_structured.py --suite eval/olist_business_suite.json
```

A pass means the system agrees with **the database**, not with another model. Every question was written before any were run, so the suite isn't selected for what already worked — and every result, including the generated Cypher for each failure, is committed:

- **[`olist_business_suite.json`](eval/olist_business_suite.json)** — the questions and their ground-truth queries
- **[`baseline_olist_business.json`](eval/baseline_olist_business.json)** — the last full run, every answer and query
- **[`OLIST_DATA_CONTEXT.md`](eval/OLIST_DATA_CONTEXT.md)** — the schema they were written from

The twelve-case suite this replaced read 11/12 while real business questions were failing. That gap is why the larger one exists.

---

## What's new

**Documents beyond PDF.** Word, PowerPoint and Excel are parsed natively — not converted. They *state* their structure where a PDF forces the parser to guess: a `.docx` declares heading levels in the paragraph style, a slide has one title, a sheet has a name. Section titles come out exactly as written.

**Pick a whole folder.** 📁 on the upload screen stages every supported file in a local directory and submits them concurrently. Unsupported files are filtered before upload, with the count shown.

**Connect a live database.** Any SQLAlchemy URL — Postgres, MySQL, SQL Server, SQLite. Reflection reads **declared foreign keys**, so relationships are known rather than inferred from column names. Dry run by default; the plan is reviewed before anything is written.

```bash
curl -X POST localhost:8000/ingest/tabular -H 'Content-Type: application/json' \
  -d '{"source":"postgresql://user:pass@host/db","user_id":"admin_001","role":"admin"}'
```

**Citations that name the page.** Every claim in an answer carries the page that supports it — printed page number *and* PDF index, since they differ — and a sentence nothing supports is marked, not quietly dropped.

**The right table of contents.** A filing with eight TOCs returns the one for the section you asked about, not the one at the front.

---

## See it in action

Every ingestion logged, every document's construction quality a measured report, every cited answer traceable to its source.

![Agentic GraphRAG chat — structured and unstructured retrieval in one session](docs/images/chat_demo.png)

**Click a citation, read the source.** The original document opens beside the answer, scrolled to the cited page.

![Source document viewer — the original PDF open in a side panel next to the cited answer](docs/images/document_viewer.png)

**Every action recorded** — user, tenant, role, result.

![Audit log — every ingestion and query recorded with user, tenant, role, and result](docs/images/audit_log.png)

**Ingestion quality is measured, not claimed** — coverage, edge counts, orphan checks per revision.

![Ingestion quality report — coverage, edge counts, and orphan-node check for a specific document revision](docs/images/ingestion_quality.png)

**Upload one file, a batch, or a folder.**

![Document Ingestion — upload single or batch documents](docs/images/document_ingestion.png)

**Watch the graph get built** — X1 structural, X2 semantic, and the final Neo4j state.

![Graph Inspector — interactive graph visualization across pipeline construction stages](docs/images/graph_inspector.png)

---

## Pluggable by design

Every layer is a seam, not a fixed pipeline — `DocumentParser`, `ModelProvider` / `BlobStore` / `VectorStore`, `StructuredStrategy` / `UnstructuredStrategy`. Register your own without forking.

Ships with **6 unstructured + 2 structured** retrieval strategies and **4 parsers** (geometry-first with RTL/bidi handling, PyMuPDF fallback, table-aware, plus Office), all resolved by name at runtime.

**Cost-effective by default.** Chat and synthesis run on a low-cost model (`gpt-4o-mini`, or `gemini-2.5-flash` on Gemini) — not a frontier tier — so the full pipeline doesn't need frontier spend. Swap with `CHAT_MODEL` any time. Providers: **OpenAI · Anthropic · Gemini**; embeddings always OpenAI.

Built with **Neo4j · FastAPI · LangGraph**.

---

## Status

| | |
| --- | --- |
| Dual-graph RAG — structured, documents, hybrid | ✅ |
| Documents: PDF, Word, PowerPoint, Excel | ✅ |
| Tabular: live databases, SQLite, CSV, Excel | ✅ |
| Structured benchmark — 100 business questions | ✅ **95/100**, deterministic |
| Multi-tenancy, RBAC, audit log, OIDC | ✅ |
| Scalable ingestion (Redis + RQ, versioning) | ✅ |
| Storage split — blob store + vector store | ✅ MinIO/Qdrant opt-in |
| Per-claim page citations, source viewer | ✅ |
| Document-side benchmark | 🚧 no equivalent to the structured 100 yet |
| 1000-document corpus validation → tagged release | 🚧 |
| CI on push/PR | 🚧 |

**Next:** validate across a 1000-document corpus, then cut a release. Then per-user memory across threads, multi-language answers, and a Kubernetes/Terraform reference.

---

## Docs

[Architecture](docs/ARCHITECTURE.md) · [API](docs/API.md) · [Configuration](docs/CONFIGURATION.md) · [Authentication](docs/AUTHENTICATION.md) · [Ingestion & retrieval](docs/INGESTION_RETRIEVAL_ARCHITECTURE.md) · [Troubleshooting](docs/TROUBLESHOOTING.md)

## Security

RBAC and tenant isolation are enforced in the graph, not just the UI. Cypher ingestion is disabled by default (`ALLOW_CYPHER_INGEST`), DB reset is disabled by default (`ALLOW_DB_RESET`), and database credentials are masked before they reach logs, responses or the graph.

## License

MIT — see [LICENSE](LICENSE). The bundled Olist sample is **CC BY-NC-SA**, not MIT; see its [ATTRIBUTION.md](sample_data_to_test/structured/olist-sample/ATTRIBUTION.md).
