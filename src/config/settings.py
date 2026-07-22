import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parents[1]  # src/
PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PROVIDER = os.environ.get("MODEL_PROVIDER", "openai").lower()
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small")
CHAT_MODEL = os.environ.get("CHAT_MODEL", "gpt-4o-mini")
# Per-pipeline overrides (each defaults to CHAT_MODEL when unset).
STRUCTURED_MODEL = os.environ.get("STRUCTURED_MODEL", CHAT_MODEL)  # Text-to-Cypher + structured synthesis
ROUTING_MODEL = os.environ.get("ROUTING_MODEL", CHAT_MODEL)  # MCP tool selection (search_documents vs query_data)
AXIS2_MODEL = os.environ.get("AXIS2_MODEL", CHAT_MODEL)  # Ingestion NER + optional relationship LLM pass

NEO4J_URI = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.environ.get("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "password123")
AUTO_LOAD_TO_NEO4J = os.environ.get("AUTO_LOAD_TO_NEO4J", "true").lower() in ("1", "true", "yes")

# Document versioning (logical doc + revision snapshots)
DOC_SKIP_DUPLICATE_HASH = os.environ.get("DOC_SKIP_DUPLICATE_HASH", "true").lower() in (
    "1",
    "true",
    "yes",
)
DOC_VERSION_RETAIN_METADATA = os.environ.get("DOC_VERSION_RETAIN_METADATA", "true").lower() in (
    "1",
    "true",
    "yes",
)

# Store ingestion artifacts (CSV/Cypher) on local disk under output/.
# Default OFF for scalable deployments; enable for debugging/auditing.
STORE_INGESTION_ARTIFACTS = os.environ.get("STORE_INGESTION_ARTIFACTS", "false").lower() in ("1", "true", "yes")

# SECURITY: Allows uploading and executing arbitrary Cypher against Neo4j.
# Keep disabled in production unless you also add strong authentication.
ALLOW_CYPHER_INGEST = os.environ.get("ALLOW_CYPHER_INGEST", "false").lower() in ("1", "true", "yes")

# If true, cypher ingestion will skip GenAI embedding statements like genai.vector.encode(...)
# Useful for loading schema/data when Neo4j GenAI credentials are not configured.
CYPHER_INGEST_SKIP_GENAI = os.environ.get("CYPHER_INGEST_SKIP_GENAI", "false").lower() in ("1", "true", "yes")

# If true, delete uploaded temp files in tmp_ingest/ after jobs finish.
# Disable to keep raw inputs for debugging.
CLEANUP_TMP_INGEST = os.environ.get("CLEANUP_TMP_INGEST", "true").lower() in ("1", "true", "yes")

# Bulk/corpus ingestion safety rails (directory-scan or manifest ingestion).
CORPUS_MAX_FILES = int(os.environ.get("CORPUS_MAX_FILES", "100000"))
CORPUS_MAX_PDF_PAGES = int(os.environ.get("CORPUS_MAX_PDF_PAGES", "2000"))
CORPUS_SCAN_TIMEOUT = os.environ.get("CORPUS_SCAN_TIMEOUT", "6h")

# SECURITY: Allows wiping the Neo4j database (DROP indexes/constraints + delete all nodes).
# Keep disabled unless you're in a trusted dev environment.
ALLOW_DB_RESET = os.environ.get("ALLOW_DB_RESET", "false").lower() in ("1", "true", "yes")

# Unstructured retrieval: broad fetch then filter before LLM
RETRIEVAL_CANDIDATE_POOL = int(os.environ.get("RETRIEVAL_CANDIDATE_POOL", "30"))
RETRIEVAL_FINAL_LIMIT = int(os.environ.get("RETRIEVAL_FINAL_LIMIT", "8"))
RETRIEVAL_MIN_RERANK_SCORE = float(os.environ.get("RETRIEVAL_MIN_RERANK_SCORE", "0.12"))

# Page vision fallback (cheap model, selective pages) — tables/charts/diagrams → visual_content
ENABLE_PAGE_VISION = os.environ.get("ENABLE_PAGE_VISION", "false").lower() in ("1", "true", "yes")
VISION_MODEL = os.environ.get("VISION_MODEL", "gpt-4o-mini")


def get_model_config() -> dict[str, str]:
    """Active model IDs per pipeline stage (for /config/models and ops dashboards)."""
    return {
        "provider": MODEL_PROVIDER,
        "chat": CHAT_MODEL,
        "structured": STRUCTURED_MODEL,
        "routing": ROUTING_MODEL,
        "embedding": EMBEDDING_MODEL,
        "axis2": AXIS2_MODEL,
        "vision": VISION_MODEL,
    }


VISION_DPI = int(os.environ.get("VISION_DPI", "120"))
VISION_IMAGE_DETAIL = os.environ.get("VISION_IMAGE_DETAIL", "low")  # low | high (cost)
VISION_MAX_PAGES_PER_DOC = int(os.environ.get("VISION_MAX_PAGES_PER_DOC", "25"))
VISION_SELECTIVE = os.environ.get("VISION_SELECTIVE", "true").lower() in ("1", "true", "yes")
VISION_MIN_TEXT_CHARS = int(os.environ.get("VISION_MIN_TEXT_CHARS", "350"))

# Lightweight PDF parser
PDF_PARSER_BACKEND = os.environ.get("PDF_PARSER_BACKEND", "light").lower()
PDF_ENABLE_PDFPLUMBER = os.environ.get("PDF_ENABLE_PDFPLUMBER", "true").lower() in ("1", "true", "yes")
PDF_LOW_TEXT_CHARS = int(os.environ.get("PDF_LOW_TEXT_CHARS", "120"))
# Per-page cap for pdfplumber fallback (find_tables/layout can hang on some PDFs).
PDF_PLUMBER_PAGE_TIMEOUT_SEC = int(os.environ.get("PDF_PLUMBER_PAGE_TIMEOUT_SEC", "25"))
PDF_ENABLE_OCR = os.environ.get("PDF_ENABLE_OCR", "false").lower() in ("1", "true", "yes")
PDF_OCR_BACKEND = os.environ.get("PDF_OCR_BACKEND", "none").lower()
PDF_OCR_DPI = int(os.environ.get("PDF_OCR_DPI", "200"))
PDF_OCR_LANG = os.environ.get("PDF_OCR_LANG", "eng")

# Structured queries: skip LLM synthesis when Cypher rows are self-explanatory (table/chart UI).
STRUCTURED_FAST_ANSWER = os.environ.get("STRUCTURED_FAST_ANSWER", "false").lower() in ("1", "true", "yes")
# If true, always run the multistep LLM planner before Text-to-Cypher (slower; default uses regex gate).
STRUCTURED_ALWAYS_MULTISTEP_PLAN = os.environ.get(
    "STRUCTURED_ALWAYS_MULTISTEP_PLAN", "false"
).lower() in ("1", "true", "yes")
# When single-shot Text-to-Cypher returns no rows, try the multistep planner once.
STRUCTURED_EMPTY_MULTISTEP_FALLBACK = os.environ.get(
    "STRUCTURED_EMPTY_MULTISTEP_FALLBACK", "true"
).lower() in ("1", "true", "yes")
# Skip routing LLM when question clearly targets documents vs graph data.
FAST_ROUTE_QUERIES = os.environ.get("FAST_ROUTE_QUERIES", "true").lower() in ("1", "true", "yes")


def llm_max_tokens(env_key: str, default: int, *, minimum: int = 1, maximum: int = 128000) -> int:
    """Read and clamp an LLM max_tokens value from the environment."""
    raw = os.environ.get(env_key)
    if raw is None:
        return max(minimum, min(default, maximum))
    try:
        val = int(str(raw).strip())
    except ValueError:
        return max(minimum, min(default, maximum))
    return max(minimum, min(val, maximum))


# ── LLM max_tokens budgets (per call site) ───────────────────────────────
STRUCTURED_SYNTHESIS_MAX_TOKENS = llm_max_tokens("STRUCTURED_SYNTHESIS_MAX_TOKENS", 600, minimum=100)
STRUCTURED_SYNTHESIS_LONG_MAX_TOKENS = llm_max_tokens(
    "STRUCTURED_SYNTHESIS_LONG_MAX_TOKENS", 4096, minimum=500
)
STRUCTURED_TEXT2CYPHER_MAX_TOKENS = llm_max_tokens("STRUCTURED_TEXT2CYPHER_MAX_TOKENS", 500)
STRUCTURED_TEXT2CYPHER_LONG_MAX_TOKENS = llm_max_tokens("STRUCTURED_TEXT2CYPHER_LONG_MAX_TOKENS", 900)
STRUCTURED_TEXT2CYPHER_LONG_QUERY_CHARS = int(
    os.environ.get("STRUCTURED_TEXT2CYPHER_LONG_QUERY_CHARS", "180")
)
# Answer verification: free rule-based checks always run; this only gates the
# extra small LLM cross-check (cost-conscious — off by default).
STRUCTURED_VERIFY_ENABLED = os.environ.get("STRUCTURED_VERIFY_ENABLED", "false").lower() in (
    "1",
    "true",
    "yes",
)
STRUCTURED_VERIFY_MAX_TOKENS = llm_max_tokens("STRUCTURED_VERIFY_MAX_TOKENS", 120, minimum=40)
# Same pattern for the document (unstructured) answer path.
DOCUMENT_VERIFY_ENABLED = os.environ.get("DOCUMENT_VERIFY_ENABLED", "false").lower() in (
    "1",
    "true",
    "yes",
)
DOCUMENT_VERIFY_MAX_TOKENS = llm_max_tokens("DOCUMENT_VERIFY_MAX_TOKENS", 120, minimum=40)
# Cypher execution / repair budgets (lower = faster; repair runs before LLM regen).
STRUCTURED_CYPHER_MAX_ATTEMPTS = int(os.environ.get("STRUCTURED_CYPHER_MAX_ATTEMPTS", "2"))
STRUCTURED_CYPHER_SQL_LLM_RETRIES = int(os.environ.get("STRUCTURED_CYPHER_SQL_LLM_RETRIES", "1"))
STRUCTURED_EMPTY_RESULT_LLM_RETRIES = int(os.environ.get("STRUCTURED_EMPTY_RESULT_LLM_RETRIES", "1"))
STRUCTURED_MULTISTEP_STEP_ATTEMPTS = int(os.environ.get("STRUCTURED_MULTISTEP_STEP_ATTEMPTS", "2"))
# Fixed override for multistep planner; empty = use heuristic tiers below.
STRUCTURED_PLAN_MAX_TOKENS = (os.environ.get("STRUCTURED_PLAN_MAX_TOKENS") or "").strip()
STRUCTURED_PLAN_TOKENS_SMALL = llm_max_tokens("STRUCTURED_PLAN_TOKENS_SMALL", 900, minimum=300)
STRUCTURED_PLAN_TOKENS_MEDIUM = llm_max_tokens("STRUCTURED_PLAN_TOKENS_MEDIUM", 1600, minimum=300)
STRUCTURED_PLAN_TOKENS_LARGE = llm_max_tokens("STRUCTURED_PLAN_TOKENS_LARGE", 2200, minimum=300)
STRUCTURED_PLAN_QUERY_MEDIUM_CHARS = int(os.environ.get("STRUCTURED_PLAN_QUERY_MEDIUM_CHARS", "160"))
STRUCTURED_PLAN_QUERY_LARGE_CHARS = int(os.environ.get("STRUCTURED_PLAN_QUERY_LARGE_CHARS", "260"))
STRUCTURED_PLAN_SCHEMA_MEDIUM_CHARS = int(os.environ.get("STRUCTURED_PLAN_SCHEMA_MEDIUM_CHARS", "3500"))
STRUCTURED_PLAN_SCHEMA_LARGE_CHARS = int(os.environ.get("STRUCTURED_PLAN_SCHEMA_LARGE_CHARS", "6000"))

DOCUMENT_SYNTHESIS_MAX_TOKENS = llm_max_tokens("DOCUMENT_SYNTHESIS_MAX_TOKENS", 600, minimum=100)
DOCUMENT_SYNTHESIS_LONG_MAX_TOKENS = llm_max_tokens("DOCUMENT_SYNTHESIS_LONG_MAX_TOKENS", 1400, minimum=100)

VISION_LLM_MAX_TOKENS = llm_max_tokens("VISION_LLM_MAX_TOKENS", 2000, minimum=256)

AXIS2_NER_MAX_TOKENS = llm_max_tokens("AXIS2_NER_MAX_TOKENS", 200)
AXIS2_RELATION_MAX_TOKENS = llm_max_tokens("AXIS2_RELATION_MAX_TOKENS", 150)

# ── Scalable ingestion pipeline ────────────────────────────────────────────
# Redis broker URL. When unset, the pipeline falls back to in-process
# BackgroundTasks (single-process, dev-friendly, no Redis required).
REDIS_URL = os.environ.get("REDIS_URL", "")

# RQ queue name consumed by `rq worker` containers.
INGEST_QUEUE_NAME = os.environ.get("INGEST_QUEUE_NAME", "ingest")

# Number of RQ worker threads per worker process (passed to `rq worker --burst`
# or used by the worker entrypoint). Override per deployment.
INGEST_WORKER_CONCURRENCY = int(os.environ.get("INGEST_WORKER_CONCURRENCY", "2"))

# Axis 2 — parallel NER: max simultaneous LLM calls for entity extraction.
AXIS2_NER_CONCURRENCY = int(os.environ.get("AXIS2_NER_CONCURRENCY", "8"))

# API-process thread pools (src/api.py) that run blocking work (LLM calls,
# Neo4j reads/writes) off the asyncio event loop. Defaults match what was
# previously hardcoded — override per deployment to match expected
# concurrent request volume and available CPU/connection headroom.
API_QUERY_EXECUTOR_WORKERS = int(os.environ.get("API_QUERY_EXECUTOR_WORKERS", "4"))
# Only used when REDIS_URL is unset (dev / single-process mode) — the RQ
# path (INGEST_WORKER_CONCURRENCY above) handles ingestion otherwise.
API_INGEST_EXECUTOR_WORKERS = int(os.environ.get("API_INGEST_EXECUTOR_WORKERS", "2"))

# Axis 2 — parallel LLM relationship pass: max simultaneous calls.
AXIS2_LLM_PAIR_CONCURRENCY = int(os.environ.get("AXIS2_LLM_PAIR_CONCURRENCY", "6"))

# Axis 2 — cap on candidate pairs fed to the expensive LLM relationship pass.
# Pairs are ranked by embedding similarity; only the top-k are sent to the LLM.
AXIS2_MAX_LLM_PAIRS = int(os.environ.get("AXIS2_MAX_LLM_PAIRS", "300"))

# Neo4j: UNWIND batch size for node/edge bulk writes.
NEO4J_WRITE_BATCH = int(os.environ.get("NEO4J_WRITE_BATCH", "2000"))

# ── Blob storage (raw text / visual_content, kept out of Neo4j properties) ──
BLOB_STORE_BACKEND = os.environ.get("BLOB_STORE_BACKEND", "local").lower()  # local | minio
LOCAL_BLOB_STORE_DIR = os.environ.get("LOCAL_BLOB_STORE_DIR", str(PROJECT_ROOT / "data" / "blobs"))
MINIO_ENDPOINT = os.environ.get("MINIO_ENDPOINT", "localhost:9000")
MINIO_ACCESS_KEY = os.environ.get("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.environ.get("MINIO_SECRET_KEY", "minioadmin")
MINIO_BUCKET = os.environ.get("MINIO_BUCKET", "graphrag-content")
MINIO_SECURE = os.environ.get("MINIO_SECURE", "false").lower() in ("1", "true", "yes")

# ── Vector storage (embeddings, kept out of Neo4j properties) ──────────────
VECTOR_STORE_BACKEND = os.environ.get("VECTOR_STORE_BACKEND", "memory").lower()  # memory | qdrant
VECTOR_DIM = int(os.environ.get("VECTOR_DIM", "1536"))  # matches EMBEDDING_MODEL default
QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
QDRANT_COLLECTION = os.environ.get("QDRANT_COLLECTION", "sections")
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY", "")

# MCP routing: one tool call; args echo the user question verbatim.
ROUTE_MAX_TOKENS_MIN = llm_max_tokens("ROUTE_MAX_TOKENS_MIN", 64, minimum=32)
ROUTE_MAX_TOKENS_BASE = llm_max_tokens("ROUTE_MAX_TOKENS_BASE", 128, minimum=64)
ROUTE_MAX_TOKENS_CAP = llm_max_tokens("ROUTE_MAX_TOKENS_CAP", 1024, minimum=128)
# Fixed override; when set (digits only), skips length-based estimate.
ROUTE_MAX_TOKENS = (os.environ.get("ROUTE_MAX_TOKENS") or "").strip()

# ── Retrieval feedback (observe-only by default) ───────────────────────────
# Persists existing pipeline telemetry after /query — no retrieval behavior change.
RETRIEVAL_FEEDBACK_ENABLED = os.environ.get("RETRIEVAL_FEEDBACK_ENABLED", "false").lower() in (
    "1",
    "true",
    "yes",
)
# When true, read-side hints may suggest a mode (caller must opt in to act on them).
RETRIEVAL_FEEDBACK_ROUTING = os.environ.get("RETRIEVAL_FEEDBACK_ROUTING", "false").lower() in (
    "1",
    "true",
    "yes",
)
RETRIEVAL_FEEDBACK_STORE_QUESTION = os.environ.get(
    "RETRIEVAL_FEEDBACK_STORE_QUESTION", "false"
).lower() in ("1", "true", "yes")
RETRIEVAL_FEEDBACK_DIR = os.environ.get(
    "RETRIEVAL_FEEDBACK_DIR",
    str(PROJECT_ROOT / "data" / "feedback"),
)
RETRIEVAL_FEEDBACK_JSONL_RETAIN_DAYS = int(
    os.environ.get("RETRIEVAL_FEEDBACK_JSONL_RETAIN_DAYS", "30")
)
RETRIEVAL_FEEDBACK_REDIS_STREAM = os.environ.get(
    "RETRIEVAL_FEEDBACK_REDIS_STREAM", "rag:feedback:events"
)
RETRIEVAL_FEEDBACK_STREAM_MAXLEN = int(
    os.environ.get("RETRIEVAL_FEEDBACK_STREAM_MAXLEN", "100000")
)
RETRIEVAL_FEEDBACK_REQ_TTL_SEC = int(
    os.environ.get("RETRIEVAL_FEEDBACK_REQ_TTL_SEC", str(7 * 24 * 3600))
)
RETRIEVAL_FEEDBACK_AGG_TTL_DAYS = int(os.environ.get("RETRIEVAL_FEEDBACK_AGG_TTL_DAYS", "90"))
RETRIEVAL_FEEDBACK_MIN_SAMPLES = int(os.environ.get("RETRIEVAL_FEEDBACK_MIN_SAMPLES", "30"))
RETRIEVAL_FEEDBACK_MIN_MARGIN = float(os.environ.get("RETRIEVAL_FEEDBACK_MIN_MARGIN", "0.15"))
RETRIEVAL_FEEDBACK_HINT_CACHE_SEC = int(
    os.environ.get("RETRIEVAL_FEEDBACK_HINT_CACHE_SEC", "60")
)


# When true, POST /query/stream returns phased NDJSON (tokens + early charts).
QUERY_STREAM_ENABLED = os.environ.get("QUERY_STREAM_ENABLED", "true").lower() in (
    "1",
    "true",
    "yes",
)

# Multi-tenancy: property-based tenant_id isolation across Neo4j/Qdrant/MinIO.
MULTI_TENANCY_ENABLED = os.environ.get("MULTI_TENANCY_ENABLED", "false").lower() in (
    "1",
    "true",
    "yes",
)
DEFAULT_TENANT_ID = os.environ.get("DEFAULT_TENANT_ID", "default")

# KnowledgeArea id that gates document-RAG access in the seeded RBAC schema
# (src/auth/rbac_schema.cypher). Defaults to "esg" to match this repo's demo
# seed data — deployments with their own KnowledgeArea taxonomy should set
# this instead of editing src/retrieval/unstructured/mixins/policies.py.
DOCUMENT_KNOWLEDGE_AREA_ID = os.environ.get("DOCUMENT_KNOWLEDGE_AREA_ID", "esg")

# ── Audit logging (compliance/security trail — defaults ON) ───────────────
# Unlike other flags in this file, writing an audit event never changes
# what the user sees, so this defaults on: real compliance value from the
# moment of upgrade, no surprise behavior change.
AUDIT_LOG_ENABLED = os.environ.get("AUDIT_LOG_ENABLED", "true").lower() in (
    "1",
    "true",
    "yes",
)
AUDIT_LOG_STORE_QUESTION = os.environ.get("AUDIT_LOG_STORE_QUESTION", "true").lower() in (
    "1",
    "true",
    "yes",
)
AUDIT_LOG_DIR = os.environ.get(
    "AUDIT_LOG_DIR",
    str(PROJECT_ROOT / "data" / "audit_log"),
)
AUDIT_LOG_JSONL_RETAIN_DAYS = int(os.environ.get("AUDIT_LOG_JSONL_RETAIN_DAYS", "90"))
AUDIT_LOG_REDIS_STREAM = os.environ.get("AUDIT_LOG_REDIS_STREAM", "rag:audit:stream")
AUDIT_LOG_STREAM_MAXLEN = int(os.environ.get("AUDIT_LOG_STREAM_MAXLEN", "1000000"))
AUDIT_LOG_REQ_TTL_SEC = int(os.environ.get("AUDIT_LOG_REQ_TTL_SEC", str(90 * 24 * 3600)))


def estimate_route_max_tokens(question: str) -> int:
    """Budget for MCP tool routing: base + room to echo question in tool arguments."""
    if ROUTE_MAX_TOKENS.isdigit():
        return max(ROUTE_MAX_TOKENS_MIN, min(int(ROUTE_MAX_TOKENS), ROUTE_MAX_TOKENS_CAP))
    q_len = len((question or "").strip())
    # ~3 chars/token for JSON args + fixed overhead for tool name/metadata.
    estimated = ROUTE_MAX_TOKENS_BASE + (q_len // 3) + 96
    return max(ROUTE_MAX_TOKENS_MIN, min(estimated, ROUTE_MAX_TOKENS_CAP))
