"""Static UI entry points and the public auth config."""
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
from ...unstructured.graph.constants import DOC_REVISION_LABEL, DOCUMENT_LOGICAL_LABEL
from ...shared.neo4j.driver import close_neo4j_driver, get_neo4j_driver
from ...shared.neo4j.tenancy import tenant_filter
from ...unstructured.document.graph_snapshot import (
    X1_STAGE,
    X2_STAGE,
    query_final_snapshot_sync,
    query_page_scoped_snapshot_sync,
    read_snapshot as read_graph_snapshot,
)
from ...unstructured.document.purge import delete_document
from ...unstructured.document.versioning import source_file_blob_key
from ...shared.storage.blob.factory import get_blob_store
from pydantic import BaseModel, Field
from ...shared.audit import AuditEventType, get_audit_store, record_audit_event
from ..bridge import ask
from ...shared.conversation import clear_turn
from ...shared.logging_config import setup_logging
from ...shared.auth.rbac_setup import GraphRBAC
from ...shared.auth.oidc import auth_public_config, resolve_admin_session, resolve_scoped_thread_id, resolve_user_context
from ...shared.config.settings import (
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
from ..streaming.query_stream import iter_query_stream
from ...unstructured.ingestion.service import IngestionManager
from ...pipeline.ingestion.job_store import get_job_store
from ...pipeline.ingestion.queue import enqueue_ingest, list_failed_jobs, queue_depth
from ...unstructured.ingestion.validation import build_ingestion_quality_report, list_ingested_documents
from ...shared.feedback import (
    best_mode_for_question,
    build_dashboard_overview,
    get_feedback_store,
    maybe_attach_feedback_outcome,
    maybe_record_retrieval_feedback,
    pattern_hash,
    retrieval_pattern,
)
from fastapi import APIRouter

from ..deps import (_dispatch_ingest_job, _ingest_executor, _query_executor,
                    _run_ingest_job_local, ingestion_manager)
from ..schemas import (ClearThreadRequest, CorpusIngestRequest, FeedbackOutcomeRequest,
                       IngestionJobSummary, IngestionResponse, IngestionStatusResponse,
                       QueryRequest, QueryResponse)

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/")
async def root():
    return RedirectResponse(url="/static/chat.html")


@router.get("/upload")
async def upload_page():
    # Redirect into the mounted StaticFiles app (same pattern as root())
    # instead of reading the file off disk on every request — the old
    # per-request read_text() blocked the event loop for real disk I/O
    # that StaticFiles already serves with caching/conditional-GET support.
    return RedirectResponse(url="/static/upload.html")


@router.get("/chat")
async def chat_page():
    return RedirectResponse(url="/static/chat.html")


@router.get("/graph-inspector")
async def graph_inspector_page():
    return RedirectResponse(url="/static/graph_inspector.html")


@router.get("/feedback")
async def feedback_dashboard_page():
    """Ops dashboard: feedback store, hints, and routing apply status."""
    return RedirectResponse(url="/static/feedback.html")


@router.get("/auth/config")
async def auth_config():
    """Public OIDC settings for the chat UI (no secrets)."""
    out = auth_public_config()
    out["feedback_enabled"] = RETRIEVAL_FEEDBACK_ENABLED
    return out


@router.get("/auth/me")
async def auth_me(authorization: Optional[str] = Header(default=None)):
    """Return the resolved principal from a Bearer token (or dev body fallback)."""
    session = resolve_user_context(authorization=authorization)
    out = {
        "user_id": session.user.user_id,
        "role": session.user.role.value,
        "department": session.user.department,
        "auth_mode": session.auth_mode,
    }
    if session.claims:
        out["email"] = session.claims.email
        out["name"] = session.claims.name
    return out
