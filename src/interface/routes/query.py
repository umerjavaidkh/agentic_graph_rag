"""Asking questions: the blocking endpoint, the stream, and thread reset."""
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



def _document_candidates(result: dict, retrieved_context: dict, request) -> Optional[list]:
    """Documents this question might have meant, when none was resolved.

    Computed here rather than inside the retrieval graph because a question
    can reach an answer by two routes -- the LangGraph pipeline and a direct
    tool dispatch (`route_method: llm_mcp`) -- and only one of them runs
    that graph. Threading the value through both is how it ends up present
    on some answers and missing on others for no reason a user could see.

    Returns None, not [], when there is nothing to offer: a picker that
    renders empty is worse than no picker.
    """
    existing = result.get("document_candidates") or retrieved_context.get("document_candidates")
    if existing:
        return existing

    # Only when nothing was resolved. A confident answer needs no picker.
    if result.get("document_id") or retrieved_context.get("document_id"):
        return None

    try:
        from ...unstructured.retrieval.retriever import DocumentRAGRetriever

        cands = DocumentRAGRetriever().document_candidates(
            request.question, user_context=getattr(request, "user_context", None)
        )
        return cands or None
    except Exception:  # a suggestion must never fail a working answer
        logger.debug("document candidates unavailable", exc_info=True)
        return None


@router.post("/query", response_model=QueryResponse)
async def query(
    request: QueryRequest,
    authorization: Optional[str] = Header(default=None),
):
    request_id = uuid.uuid4().hex[:12]
    session = resolve_user_context(
        authorization=authorization,
        body_user_id=request.user_id,
        body_role=request.role,
        body_department=request.department,
        body_tenant_id=request.tenant_id,
    )
    thread_id = resolve_scoped_thread_id(session, request.thread_id)
    context = session.user
    user_id = context.user_id
    question_preview = (request.question or "")[:160]
    logger.info(
        "query start request_id=%s user=%s auth=%s thread=%s q=%r",
        request_id,
        user_id,
        session.auth_mode,
        thread_id,
        question_preview,
    )
    try:
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            _query_executor,
            lambda: ask(
                request.question,
                user_context=context,
                thread_id=thread_id,
                request_id=request_id,
                retrieval_mode=request.retrieval_mode,
            ),
        )

        telemetry = result.get("_telemetry") or {}
        totals = telemetry.get("totals") or {}
        route = telemetry.get("route") or {}
        logger.info(
            "query ok request_id=%s agent=%s route=%s tokens=%s failed_step=%s",
            request_id,
            result.get("agent"),
            route.get("tool"),
            totals.get("total_tokens"),
            telemetry.get("failed_step"),
        )

        maybe_record_retrieval_feedback(
            request_id=request_id,
            question=request.question,
            result=result,
        )

        retrieved_context = result.get("retrieved_context") or {}

        return QueryResponse(
            answer       = result.get("answer", "No answer generated."),
            sources      = result.get("sources", []),
            keywords     = result.get("keywords", []),
            total_chunks = len(result.get("sources", [])),
            agent        = result.get("agent", "unstructured"),
            strategy     = result.get("strategy", "semantic"),
            access_level = result.get("_access_level", context.role.value),
            route_tool   = result.get("_route_tool"),
            route_method = result.get("_route_method"),
            presentation = result.get("presentation"),
            query_type   = result.get("query_type"),
            follow_up    = result.get("_follow_up"),
            telemetry    = telemetry,
            claims       = result.get("claims", []),
            request_id   = request_id,
            low_confidence  = bool(result.get("low_confidence")),
            confidence_note = result.get("confidence_note"),
            # Set when retrieval declined to guess because the question named
            # no document and implied none. Passed explicitly like every field
            # here: the envelope is built field-by-field, so a key the answer
            # carries but this call does not name is silently dropped.
            underspecified  = bool(
                result.get("underspecified")
                or retrieved_context.get("underspecified")
            ),
            document_id     = result.get("document_id") or retrieved_context.get("document_id"),
            # Offered only when the resolver declined because two documents
            # matched the query about equally. Guessing there lands on the
            # runner-up about half the time and answers confidently from the
            # wrong document.
            document_candidates = _document_candidates(result, retrieved_context, request),
            document_title  = result.get("document_title") or retrieved_context.get("document_title"),
        )
    except ValueError as ve:
        logger.warning(
            "query validation error request_id=%s: %s",
            request_id,
            ve,
            exc_info=True,
        )
        raise HTTPException(status_code=400, detail=str(ve))
    except HTTPException:
        raise
    except Exception:
        logger.exception("query failed request_id=%s", request_id)
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error (request_id={request_id}). Check server logs.",
        )


@router.post("/query/stream")
async def query_stream(
    request: QueryRequest,
    authorization: Optional[str] = Header(default=None),
):
    """
    Stream query results as NDJSON lines.

    Event types: status, presentation (partial charts/tables), token, done, error.
    Retrieval matches /query; synthesis tokens stream when LLM is used.
    """
    if not QUERY_STREAM_ENABLED:
        raise HTTPException(status_code=404, detail="Query streaming is disabled.")

    request_id = uuid.uuid4().hex[:12]
    session = resolve_user_context(
        authorization=authorization,
        body_user_id=request.user_id,
        body_role=request.role,
        body_department=request.department,
        body_tenant_id=request.tenant_id,
    )
    thread_id = resolve_scoped_thread_id(session, request.thread_id)
    context = session.user
    logger.info(
        "query stream start request_id=%s user=%s auth=%s thread=%s",
        request_id,
        context.user_id,
        session.auth_mode,
        thread_id,
    )

    def _stream():
        try:
            yield from iter_query_stream(
                request.question,
                user_context=context,
                thread_id=thread_id,
                request_id=request_id,
                retrieval_mode=request.retrieval_mode,
            )
        except Exception:
            logger.exception("query stream failed request_id=%s", request_id)
            yield json.dumps(
                {
                    "type": "error",
                    "message": f"Stream failed (request_id={request_id}).",
                    "request_id": request_id,
                }
            ) + "\n"

    return StreamingResponse(_stream(), media_type="application/x-ndjson")


@router.post("/chat/clear")
async def chat_clear(
    request: ClearThreadRequest,
    authorization: Optional[str] = Header(default=None),
):
    """Forget the last critical document turn for this user's thread (e.g. New chat)."""
    session = resolve_user_context(
        authorization=authorization,
        body_user_id=request.user_id,
        body_role=request.role,
        body_tenant_id=request.tenant_id,
    )
    thread_id = resolve_scoped_thread_id(session, request.thread_id)
    clear_turn(thread_id)
    return {"ok": True, "thread_id": thread_id}
