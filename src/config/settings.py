import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parents[1]  # src/
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# openai | anthropic (alias: claude) | gemini (alias: google) — see
# model_providers/factory.py's get_chat_provider(). Only affects chat/
# completion; embeddings always use OpenAI regardless (Anthropic has no
# embeddings API, and Neo4j's vector index has a hardcoded dimension).
MODEL_PROVIDER = os.environ.get("MODEL_PROVIDER", "openai").lower()
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "")
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small")  # always OpenAI, see above

# CHAT_MODEL's default depends on MODEL_PROVIDER so switching providers
# alone (without also remembering to override every *_MODEL var) doesn't
# silently try to call e.g. Anthropic with an OpenAI model name. An
# explicit CHAT_MODEL env var always wins over this default either way.
_DEFAULT_CHAT_MODEL_BY_PROVIDER = {
    "openai": "gpt-4o-mini",
    "anthropic": "claude-sonnet-5",
    "gemini": "gemini-2.5-flash",
}
CHAT_MODEL = os.environ.get("CHAT_MODEL") or _DEFAULT_CHAT_MODEL_BY_PROVIDER.get(
    MODEL_PROVIDER, "gpt-4o-mini"
)

# Resolved API key for whichever provider MODEL_PROVIDER actually names —
# for cheap early-exit gates ("is chat enrichment even configured?") that
# need a plain boolean/string check before constructing any provider
# object. Kept in sync with model_providers.factory.get_chat_provider()'s
# own resolution, duplicated here (not imported from factory) to avoid a
# settings <-> model_providers import cycle.
_CANONICAL_PROVIDER = {
    "openai": "openai",
    "anthropic": "anthropic",
    "claude": "anthropic",
    "gemini": "gemini",
    "google": "gemini",
}.get(MODEL_PROVIDER, "openai")
CHAT_PROVIDER_API_KEY = {
    "openai": OPENAI_API_KEY,
    "anthropic": ANTHROPIC_API_KEY,
    "gemini": GOOGLE_API_KEY,
}[_CANONICAL_PROVIDER]
# Per-pipeline overrides (each defaults to CHAT_MODEL when unset).
STRUCTURED_MODEL = os.environ.get("STRUCTURED_MODEL", CHAT_MODEL)  # Text-to-Cypher + structured synthesis

# Escalation model, used only when the first attempt produced Cypher that
# failed or was rejected. Measured on the eleven cases the small model got
# wrong: seven were answered correctly on the larger one, including a join
# that walked back out of Payment into a different Order and an avg() taken
# over rows already collapsed per seller. Both are reasoning errors rather
# than missing schema, so a retry with the same model reproduces them.
# Escalating only on failure keeps the common path on the cheap model.
STRUCTURED_FALLBACK_MODEL = os.environ.get("STRUCTURED_FALLBACK_MODEL", "gpt-4.1")
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

# Cross-encoder reranking: the merged vector/fulltext/graph/lexical candidate
# pool is scored by heuristic weights that don't judge query-relevance
# directly, so unrelated chunks sharing generic terms/entities (dates, "u.s.")
# can tie with the chunk that actually answers the question. A cross-encoder
# re-scores each (query, chunk) pair directly and reorders on that, same
# retrieve-then-rerank pattern most production RAG stacks use.
RERANK_ENABLED = os.environ.get("RERANK_ENABLED", "true").lower() in ("1", "true", "yes")
# Backend key resolved via src/retrieval/reranker_registry.py — swap/A-B by
# registering a new backend under a new key and pointing this at it.
# Default is "rrf" (reciprocal rank fusion, no dependency, no model load);
# "cross_encoder" is more precise but requires torch/sentence-transformers.
RERANK_BACKEND = os.environ.get("RERANK_BACKEND", "rrf")
RERANK_MODEL = os.environ.get("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")

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

# PDF parser backend: "rtldoc" (default — geometry-first, correct RTL/bidi
# handling, vector-rule table detection, no OCR/model — see
# https://github.com/umerjavaidkh/rtldoc), "light" (plain PyMuPDF/pdfplumber
# heuristics), or "table-aware" (light + extra vetoes against table
# fragments/running headers misread as headings).
PDF_PARSER_BACKEND = os.environ.get("PDF_PARSER_BACKEND", "rtldoc").lower()
PDF_ENABLE_PDFPLUMBER = os.environ.get("PDF_ENABLE_PDFPLUMBER", "true").lower() in ("1", "true", "yes")
PDF_LOW_TEXT_CHARS = int(os.environ.get("PDF_LOW_TEXT_CHARS", "120"))
# Per-page cap for pdfplumber fallback (find_tables/layout can hang on some PDFs).
PDF_PLUMBER_PAGE_TIMEOUT_SEC = int(os.environ.get("PDF_PLUMBER_PAGE_TIMEOUT_SEC", "25"))

# Wall-clock cap on a single chat/completion call. The vendor SDKs default to
# 600s with retries on top, so one stalled request blocks a caller for up to
# half an hour with nothing in the logs -- which is exactly what stalled a
# 100-case eval run at case 33 and looked like a hang with no cause.
LLM_REQUEST_TIMEOUT_SEC = float(os.environ.get("LLM_REQUEST_TIMEOUT_SEC", "90"))
LLM_MAX_RETRIES = int(os.environ.get("LLM_MAX_RETRIES", "2"))
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
# Input-side budget for the synthesis prompt's retrieved-chunk context, in
# characters (cheap token proxy, same convention as the STRUCTURED_PLAN_*_CHARS
# tiers above). A whole-chapter node's own .text can legitimately span dozens
# of pages once chapter detection is accurate (see the TOC-based parsing fix)
# -- with no cap, a handful of such chunks blew past both the model's context
# window and the org's tokens-per-minute rate limit in one call (verified
# live: a 46-page chapter's chunks alone requested ~1.35M tokens against a
# 200k TPM limit). 160k chars ≈ 40k tokens at a ~4-chars/token estimate --
# comfortably under gpt-4o-mini's 128k context with room for the system
# prompt/output and concurrent traffic against the shared TPM limit, while
# large enough to fit even the biggest single real chapter whole (the
# largest chapter in the physics textbook that surfaced this is ~149k
# chars) rather than truncating its tail, which is exactly where a
# "list every X in this chapter" question needs full coverage.
DOCUMENT_SYNTHESIS_CONTEXT_MAX_CHARS = int(
    os.environ.get("DOCUMENT_SYNTHESIS_CONTEXT_MAX_CHARS", "160000")
)

VISION_LLM_MAX_TOKENS = llm_max_tokens("VISION_LLM_MAX_TOKENS", 2000, minimum=256)

# 200 was calibrated for the pre-chunking flat-string entity format. Two
# compounding causes made it (and 350) too tight once chunking + typed
# entities landed: (1) the typed {"text": ..., "type": ...} objects are
# more verbose than flat strings, and (2) without an explicit instruction
# the model defaults to PRETTY-PRINTED JSON (newlines + indentation), which
# measured 3-4x the tokens of the equivalent compact JSON for the same 10
# entities -- verified live on a real 10-K page: a single compact-JSON
# excerpt needs ~100 tokens for 10 typed entities, the same content
# pretty-printed needed 350+ and still got cut off mid-object, silently
# degrading to zero entities for the whole call. Fixed at the prompt level
# (compact-JSON instruction in axis2.py's NER prompt) so this budget is a
# comfortable multiple of the real ~100-token measured need, not a tight
# fit -- axis2.py also self-heals via batch-split-and-retry on any
# remaining parse failure, so this doesn't need to be a worst-case bound.
AXIS2_NER_MAX_TOKENS = llm_max_tokens("AXIS2_NER_MAX_TOKENS", 300)
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

# Axis 2 — excerpts per NER LLM call (nodes since chunking, since a long
# node's chunks are each their own excerpt). One call per excerpt doesn't
# scale: a single 7,165-node document (Section+Page nodes needing NER)
# burned an entire 10,000-request daily OpenAI quota by itself, reproduced
# twice live. Batching multiple excerpts into one call cuts request count
# by roughly this factor — same "batch instead of one-call-per-item"
# pattern chapter-summary enrichment already uses.
#
# Was 15, lowered after finding that a *large* multi-excerpt batch has a
# second, distinct failure mode beyond the token-overflow one the
# self-healing split-retry already handles: the model can silently return
# an empty array for one excerpt's index in a crowded batch without any
# parse error at all (a real response, just wrong for that one index) --
# not something a retry can detect or fix, since there's no exception to
# catch. Measured live across all 142 long-text pages of a real 10-K:
#   batch=15 (old default): 17.6% zero-entity, 26.1% <=3 entities
#   batch=5:                 1.4% zero-entity,  3.5% <=3 entities
#   batch=3:                 0.0% zero-entity,  0.7% <=3 entities
# 3 costs ~5x the request count of 15 (still far below one-call-per-node),
# and ~12s slower across 142 pages in that measurement -- worth it to
# clear the project's 95%+ coverage target without leaving a known,
# unfixable-by-retry gap.
AXIS2_NER_BATCH_SIZE = int(os.environ.get("AXIS2_NER_BATCH_SIZE", "3"))

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

# Axis 2 — independently re-verify each CONTRADICTS/ELABORATES/PREREQUISITE_OF
# edge with a second, separate LLM call before keeping it, instead of trusting
# the same generation's self-reported `confidence` as the only signal (that
# call already "wants" to find a relationship -- it was prompted to name one).
# A second call given only the two texts and the claimed relationship type,
# with no visibility into the first call's stated reason, is a cheap
# independent check without adding a new model dependency. Doubles LLM calls
# for this edge type specifically -- already the expensive, opt-in
# (run_llm_pass=True) path, so the added cost is proportional to a cost
# that was already being paid on purpose. Off switch provided since it's a
# real added cost, not because grounding is ever wrong to want.
AXIS2_GROUND_LLM_EDGES = os.environ.get("AXIS2_GROUND_LLM_EDGES", "true").lower() in ("1", "true", "yes")

# Axis 2 — max SEMANTICALLY_SIMILAR / SAME_CATEGORY edges per node (a kNN cap,
# not a flat similarity threshold alone). Without this, both edge builders
# scale with corpus size in a way that blows up fast: SEMANTICALLY_SIMILAR
# creates an edge for every pair above threshold with no per-node bound, and
# SAME_CATEGORY's cluster count is capped at 10 regardless of node count, so
# cluster (and thus intra-cluster pair) size grows with the corpus instead of
# staying flat. Verified live: a 7,165-node textbook produced 2.17M edges
# (~303/node) — almost entirely from SAME_CATEGORY's ~716-member clusters,
# each fully interconnected (C(716,2) * 10 clusters ≈ 2.56M). Capped to each
# node's top-k most-similar neighbors, this bounds total edges to O(n*k)
# instead of O(n^2) regardless of corpus size.
AXIS2_MAX_SIMILARITY_EDGES_PER_NODE = int(os.environ.get("AXIS2_MAX_SIMILARITY_EDGES_PER_NODE", "20"))

# Chapter-summary enrichment: one LLM call per Chapter, fed section titles +
# excerpts (not full body text) to bound both prompt size and cost — see
# src/semantic/chapter_summary.py.
CHAPTER_SUMMARY_MODEL = os.environ.get("CHAPTER_SUMMARY_MODEL", CHAT_MODEL)
CHAPTER_SUMMARY_MAX_TOKENS = llm_max_tokens("CHAPTER_SUMMARY_MAX_TOKENS", 200)
CHAPTER_SUMMARY_CONCURRENCY = int(os.environ.get("CHAPTER_SUMMARY_CONCURRENCY", "4"))
# Per-section excerpt length and total chapter context cap (characters, not
# tokens — a cheap, conservative bound) fed into the summarization prompt.
CHAPTER_SUMMARY_SECTION_EXCERPT_CHARS = int(
    os.environ.get("CHAPTER_SUMMARY_SECTION_EXCERPT_CHARS", "400")
)
CHAPTER_SUMMARY_MAX_CONTEXT_CHARS = int(
    os.environ.get("CHAPTER_SUMMARY_MAX_CONTEXT_CHARS", "6000")
)

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

# ── Hydrator (blob_key -> full text, kept out of Neo4j properties) ─────────
HYDRATOR_CACHE = os.environ.get("HYDRATOR_CACHE", "true").lower() == "true"

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
