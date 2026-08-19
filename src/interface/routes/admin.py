"""Operational endpoints: audit trail, health, model config, DB reset."""
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


def _reset_neo4j_sync() -> tuple[int, int]:
    driver = get_neo4j_driver()
    dropped_indexes = 0
    dropped_constraints = 0
    with driver.session() as session:
        try:
            rows = session.run("SHOW INDEXES YIELD name RETURN name").data()
            for r in rows:
                name = r.get("name")
                if not name:
                    continue
                try:
                    session.run(f"DROP INDEX `{name}` IF EXISTS").consume()
                    dropped_indexes += 1
                except Exception:
                    pass
        except Exception:
            pass

        try:
            rows = session.run("SHOW CONSTRAINTS YIELD name RETURN name").data()
            for r in rows:
                name = r.get("name")
                if not name:
                    continue
                try:
                    session.run(f"DROP CONSTRAINT `{name}` IF EXISTS").consume()
                    dropped_constraints += 1
                except Exception:
                    pass
        except Exception:
            pass

        session.run("MATCH (n) DETACH DELETE n").consume()

    return dropped_indexes, dropped_constraints


@router.get("/audit/events", response_model=List[dict])
async def list_audit_events(
    limit: int = 100,
    user_id: Optional[str] = Query(None),
    tenant_id: Optional[str] = Query(None),
    event_type: Optional[str] = Query(None),
    since: Optional[str] = Query(None),
    until: Optional[str] = Query(None),
    authorization: Optional[str] = Header(default=None),
    role: Optional[str] = Query(None),
):
    """
    Query the audit trail (who did what, when, to what data).

    Admin-only. Filters are ANDed; omit any to widen the result. `user_id`
    filters the results — it is not the identity used for the admin check
    (dev-fallback only gates on role=admin), so filtering by another user's
    id doesn't affect whether the request is allowed.
    """
    resolve_admin_session(
        authorization=authorization,
        body_role=role,
    )
    events = get_audit_store().query(
        user_id=user_id,
        tenant_id=tenant_id,
        event_type=event_type,
        since=since,
        until=until,
        limit=min(max(limit, 1), 1000),
    )
    return [e.to_dict() for e in events]


@router.post("/admin/reset-neo4j")
async def reset_neo4j(
    authorization: Optional[str] = Header(default=None),
    user_id: Optional[str] = Query(None),
    role: Optional[str] = Query(None),
):
    """
    DANGEROUS: Wipes Neo4j database contents.

    - Disabled by default (ALLOW_DB_RESET=true to enable)
    - Admin role required (JWT when AUTH_ENABLED, else body role=admin)
    """
    if not ALLOW_DB_RESET:
        raise HTTPException(
            status_code=403,
            detail="DB reset is disabled. Set ALLOW_DB_RESET=true to enable.",
        )

    resolve_admin_session(
        authorization=authorization,
        body_user_id=user_id,
        body_role=role,
    )

    loop = asyncio.get_running_loop()
    dropped_indexes, dropped_constraints = await loop.run_in_executor(_query_executor, _reset_neo4j_sync)

    return {
        "status": "ok",
        "dropped_indexes": dropped_indexes,
        "dropped_constraints": dropped_constraints,
        "message": "Neo4j wiped (best-effort). RBAC will be re-initialized on next API startup.",
    }


@router.get("/health")
async def health():
    return {"status": "ok"}


@router.get("/config/models")
async def config_models():
    """
    Active LLM/embedding model per pipeline stage (from .env).
    Change models in .env and restart the app (or workers) to apply.
    """
    models = get_model_config()
    return {
        "models": models,
        "env_keys": {
            "chat": "CHAT_MODEL",
            "structured": "STRUCTURED_MODEL",
            "routing": "ROUTING_MODEL",
            "embedding": "EMBEDDING_MODEL",
            "axis2": "AXIS2_MODEL",
            "vision": "VISION_MODEL",
        },
        "defaults_when_unset": "ROUTING_MODEL, STRUCTURED_MODEL, and AXIS2_MODEL fall back to CHAT_MODEL",
    }
