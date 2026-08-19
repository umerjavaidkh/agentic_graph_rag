"""Recording answer outcomes and reading the feedback dashboard."""
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


@router.post("/feedback/outcome")
async def feedback_outcome(body: FeedbackOutcomeRequest):
    """
    Attach pass/fail to a prior query (eval runner, user thumbs).

    Requires RETRIEVAL_FEEDBACK_ENABLED=true on the server.
    """
    if not RETRIEVAL_FEEDBACK_ENABLED:
        raise HTTPException(status_code=404, detail="Retrieval feedback is disabled.")
    maybe_attach_feedback_outcome(
        body.request_id,
        passed=body.passed,
        case_id=body.case_id,
    )
    return {"status": "accepted", "request_id": body.request_id}


@router.get("/feedback/stats")
async def feedback_stats(question: str, agent: Optional[str] = None):
    """
    Read aggregated mode stats for a question pattern (ops / dashboards).

    Does not change retrieval behavior.
    """
    if not RETRIEVAL_FEEDBACK_ENABLED:
        raise HTTPException(status_code=404, detail="Retrieval feedback is disabled.")
    pattern = retrieval_pattern(question, agent=agent or "")
    p_hash = pattern_hash(pattern)
    stats = get_feedback_store().aggregate_stats(p_hash)
    hint = best_mode_for_question(
        question,
        agent=agent or "",
        min_samples=RETRIEVAL_FEEDBACK_MIN_SAMPLES,
        min_margin=RETRIEVAL_FEEDBACK_MIN_MARGIN,
        cache_sec=RETRIEVAL_FEEDBACK_HINT_CACHE_SEC,
    )
    return {
        "pattern": pattern,
        "pattern_hash": p_hash,
        "by_mode": stats,
        "hint": (
            {
                "mode": hint.mode,
                "pass_rate": hint.pass_rate,
                "samples": hint.samples,
                "confidence": hint.confidence,
            }
            if hint
            else None
        ),
    }


@router.get("/feedback/dashboard")
async def feedback_dashboard(
    recent_limit: int = 50,
    pattern_limit: int = 25,
):
    """
    Aggregated feedback analytics for the ops dashboard UI.

    Read-only; does not change retrieval.
    """
    if not RETRIEVAL_FEEDBACK_ENABLED:
        raise HTTPException(status_code=404, detail="Retrieval feedback is disabled.")
    return build_dashboard_overview(
        recent_limit=min(max(recent_limit, 1), 200),
        pattern_limit=min(max(pattern_limit, 1), 100),
    )
