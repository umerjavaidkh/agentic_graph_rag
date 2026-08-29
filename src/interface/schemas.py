"""Request and response models for the HTTP API.

Kept apart from the routes so a router reads without scrolling past eight
Pydantic classes, and so two routers can share a model without importing
each other.
"""
import asyncio
import functools
import json
import logging
import re
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Optional
from fastapi import BackgroundTasks, FastAPI, File, Form, Header, HTTPException, Query, UploadFile
from fastapi.responses import RedirectResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from ..unstructured.graph.constants import DOC_REVISION_LABEL, DOCUMENT_LOGICAL_LABEL
from ..shared.neo4j.driver import close_neo4j_driver, get_neo4j_driver
from ..shared.unicode_text import fold
from ..shared.neo4j.tenancy import tenant_filter
from ..unstructured.document.graph_snapshot import (
    X1_STAGE,
    X2_STAGE,
    query_final_snapshot_sync,
    query_page_scoped_snapshot_sync,
    read_snapshot as read_graph_snapshot,
)
from ..unstructured.document.purge import delete_document
from ..unstructured.document.versioning import source_file_blob_key
from ..shared.storage.blob.factory import get_blob_store
from pydantic import BaseModel, Field, field_validator
from ..shared.audit import AuditEventType, get_audit_store, record_audit_event
from .bridge import ask
from ..shared.conversation import clear_turn
from ..shared.logging_config import setup_logging
from ..shared.auth.rbac_setup import GraphRBAC
from ..shared.auth.oidc import auth_public_config, resolve_admin_session, resolve_scoped_thread_id, resolve_user_context
from ..shared.config.settings import (
    ALLOW_CYPHER_INGEST,
    ALLOW_DB_RESET,
    API_INGEST_EXECUTOR_WORKERS,
    API_QUERY_EXECUTOR_WORKERS,
    CORPUS_SCAN_TIMEOUT,
    NEO4J_PASSWORD,
    NEO4J_URI,
    NEO4J_USER,
    OPENAI_API_KEY,
    PROJECT_ROOT,
    REDIS_URL,
    RETRIEVAL_FEEDBACK_ENABLED,
    RETRIEVAL_FEEDBACK_HINT_CACHE_SEC,
    RETRIEVAL_FEEDBACK_MIN_MARGIN,
    RETRIEVAL_FEEDBACK_MIN_SAMPLES,
    QUERY_STREAM_ENABLED,
    get_model_config,
)
from .streaming.query_stream import iter_query_stream
from ..unstructured.ingestion.service import IngestionManager
from ..pipeline.ingestion.job_store import get_job_store
from ..pipeline.ingestion.queue import enqueue_ingest, list_failed_jobs, queue_depth
from ..unstructured.ingestion.validation import build_ingestion_quality_report, list_ingested_documents
from ..shared.feedback import (
    best_mode_for_question,
    build_dashboard_overview,
    get_feedback_store,
    maybe_attach_feedback_outcome,
    maybe_record_retrieval_feedback,
    pattern_hash,
    retrieval_pattern,
)


class IngestionResponse(BaseModel):
    job_id: str
    status: str
    message: str
    output_dir: str
    dispatch: Optional[str] = None  # "worker" | "background_task"


class IngestionStatusResponse(BaseModel):
    job_id: str
    job_type: str
    status: str
    created_at: str
    started_at: Optional[str]
    finished_at: Optional[str]
    output_dir: str
    neo4j_load_status: Optional[str]
    neo4j_load_message: Optional[str]
    logs: List[str]
    error: Optional[str]
    logical_doc_id: Optional[str] = None
    revision_id: Optional[str] = None
    content_hash: Optional[str] = None
    version_number: Optional[int] = None
    skipped_duplicate: bool = False
    child_job_ids: List[str] = []


class CorpusIngestRequest(BaseModel):
    source: str = Field(
        ...,
        description=(
            "Absolute path to a directory to scan (recursively), or a manifest "
            "file of absolute paths (one per line; '#' comments allowed)."
        ),
    )
    job_name: Optional[str] = None
    doc_key_prefix: Optional[str] = Field(
        default=None,
        description="Combined with each file's name to form a per-file logical id.",
    )
    user_id: Optional[str] = None
    role: Optional[str] = None
    tenant_id: Optional[str] = None


class IngestionJobSummary(BaseModel):
    job_id: str
    job_type: str
    status: str
    created_at: str
    name: Optional[str]
    logical_doc_id: Optional[str]
    revision_id: Optional[str] = None
    version_number: Optional[int] = None
    skipped_duplicate: bool = False
    error: Optional[str] = None


class QueryRequest(BaseModel):
    question:    str           = Field(..., description="User's question")
    role:        Optional[str] = Field(default=None, description="Dev only when AUTH_ALLOW_BODY_FALLBACK")
    user_id:     Optional[str] = Field(default=None, description="Dev only when AUTH_ALLOW_BODY_FALLBACK")
    tenant_id:   Optional[str] = Field(default=None, description="Dev only when AUTH_ALLOW_BODY_FALLBACK")
    department:  Optional[str] = Field(default=None, description="User department")
    thread_id:   Optional[str] = Field(default="default")
    retrieval_mode: Optional[str] = Field(
        default="unstructured",
        description=(
            "Which retrieval path handles the query (no auto-routing): "
            "'unstructured' (documents only, default), 'structured' "
            "(graph/data only), or 'hybrid' (both). Unknown/omitted values "
            "fall back to 'unstructured'."
        ),
    )

    @field_validator("question")
    @classmethod
    def _normalize_question(cls, value: str) -> str:
        """NFKC-normalize the question before anything reads it.

        The same word can arrive in several encodings -- Arabic presentation
        forms, full-width Latin from a CJK keyboard, a `ﬁ` ligature pasted
        out of a PDF -- and each one tokenizes to something the corpus does
        not contain. Normalizing once here means no downstream matcher has
        to know that, and both `/query` and `/query/stream` are covered by
        the one rule because both take this model.

        Queries only, deliberately. Running the same normalization over
        STORED text would rewrite English content, so it needs a re-ingest
        and a re-measurement -- Phase 2, when the corpus is rebuilt anyway.
        Until then a query is normalized and a document is not, which
        matches strictly more than the reverse would.
        """
        return fold(value)


class ClearThreadRequest(BaseModel):
    thread_id: Optional[str] = Field(default="default")
    user_id: Optional[str] = Field(default=None, description="Dev only when AUTH_ALLOW_BODY_FALLBACK")
    role: Optional[str] = Field(default=None, description="Dev only when AUTH_ALLOW_BODY_FALLBACK")
    tenant_id: Optional[str] = Field(default=None, description="Dev only when AUTH_ALLOW_BODY_FALLBACK")


class QueryResponse(BaseModel):
    answer:       str
    sources:      list
    keywords:     list
    total_chunks: int
    agent:        str   # "unstructured" | "structured" | "hybrid"
    strategy:     str   # retrieval / query mode
    access_level: str
    route_tool:   Optional[str] = None   # MCP tool chosen by LLM router
    route_method: Optional[str] = None   # e.g. llm_mcp
    presentation: Optional[dict] = None  # { kind, blocks[] } for rich UI
    query_type:   Optional[str] = None
    follow_up:    Optional[str] = None  # set when last-turn context was used
    telemetry:    Optional[dict] = None  # {_telemetry} from router (tokens/tries)
    request_id:   Optional[str] = None   # correlates with feedback / logs
    low_confidence:  bool = False        # structured-path answer verification signal
    confidence_note: Optional[str] = None  # reason when low_confidence is True
    underspecified: bool = False  # question named no document and implied none
    document_id:    Optional[str] = None  # logical doc id the answer was grounded in (document paths only)
    # Plausible documents when the question named none clearly. Present only
    # when the resolver declined to pick: on a 50-document corpus an unscoped
    # question lands on the wrong document often enough that offering the
    # choice beats guessing, and beats answering "not covered" from a
    # document the user never meant.
    document_candidates: Optional[List[dict]] = None
    document_title: Optional[str] = None  # UI transparency: "answering from <document>"
    # Per-claim citations: [{text, source_id, page, title, overlap}], with
    # source_id None where a sentence has no confident support. Declared here
    # because response_model strips anything this model does not name -- the
    # router set "claims" all along and every /query response silently
    # dropped it, so the UI could only ever show a document-level citation.
    claims: list = []


class FeedbackOutcomeRequest(BaseModel):
    request_id: str
    passed: bool
    case_id: Optional[str] = None


class TabularIngestRequest(BaseModel):
    source: str = Field(
        ...,
        description=(
            "A CSV directory, .xlsx workbook, .sqlite file, or a database "
            "connection URL (postgresql://…, mysql://…, sqlite:///…)."
        ),
    )
    load: bool = Field(
        default=False,
        description=(
            "Dry run unless true. Labels are inferred from table names, so two "
            "unrelated sources collide constantly -- a small fixture and a "
            "production dataset both infer :Product. Reviewing the plan before "
            "writing is the point, not a formality."
        ),
    )
    user_id: Optional[str] = None
    role: Optional[str] = None
    tenant_id: Optional[str] = None


class TabularPlanTable(BaseModel):
    name: str
    label: str
    columns: List[str]
    primary_key: Optional[str] = None
    # column -> "table.column" it points at
    foreign_keys: dict = {}
    row_count: Optional[int] = None


class TabularIngestResponse(BaseModel):
    # Always the sanitised form: a connection URL carries a password and this
    # response is logged, cached and shown in a browser.
    source: str
    dry_run: bool
    tables: List[TabularPlanTable]
    warnings: List[str] = []
    # Present only after an actual load; per-table rows written.
    counts: Optional[dict] = None
