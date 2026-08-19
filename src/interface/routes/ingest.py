"""Submitting work and following it: uploads, corpus scans, jobs, quality."""
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


def _list_ingestion_quality_documents_sync(
    tenant_id: Optional[str], limit: int, search: Optional[str]
) -> list[dict]:
    driver = get_neo4j_driver()
    with driver.session() as session:
        return list_ingested_documents(
            session, tenant_id=tenant_id, limit=min(max(limit, 1), 500), search=search
        )


@router.post("/ingest/unstructured", response_model=IngestionResponse)
async def ingest_unstructured(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    job_name: Optional[str] = Form(None),
    doc_key: Optional[str] = Form(
        None,
        description=(
            "Stable logical document key (e.g. annual-report-2021). "
            "Re-ingests with the same key create a new revision; identical file hash is skipped."
        ),
    ),
    authorization: Optional[str] = Header(default=None),
    user_id: Optional[str] = Form(None),
    role: Optional[str] = Form(None),
    tenant_id: Optional[str] = Form(None),
):
    session = resolve_admin_session(
        authorization=authorization,
        body_user_id=user_id,
        body_role=role,
        body_tenant_id=tenant_id,
    )
    # No 409 gate: multiple concurrent uploads are fine. The per-doc Redis lock
    # (inside IngestionManager._doc_lock) serialises revision installs for the
    # same logical document while allowing different documents to run in parallel.
    job = ingestion_manager.submit_unstructured(
        file, session.user.tenant_id, job_name=job_name, doc_key=doc_key
    )
    record_audit_event(
        event_type=AuditEventType.INGESTION_SUBMITTED,
        user_id=session.user.user_id,
        tenant_id=session.user.tenant_id,
        role=session.user.role.value,
        resource=job.id,
        action=job_name or file.filename,
        metadata={"job_type": "unstructured"},
    )
    dispatch = _dispatch_ingest_job(job.id, background_tasks)
    return IngestionResponse(
        job_id=job.id,
        status=job.status.value,
        message="Unstructured ingestion job submitted.",
        output_dir=str(job.output_dir) if job.output_dir else "",
        dispatch=dispatch,
    )


@router.post("/ingest/corpus", response_model=IngestionResponse)
async def ingest_corpus(
    request: CorpusIngestRequest,
    background_tasks: BackgroundTasks,
    authorization: Optional[str] = Header(default=None),
):
    """
    Scan a server-accessible directory (recursively) or manifest file and
    fan out one unstructured-ingestion job per accepted file, after a cheap
    dedup + structural-sanity triage pass (no LLM calls in triage).

    Note: this job ("corpus" type) can reach `completed` once every child
    job has been created/enqueued — that does not mean every document has
    finished ingesting. Poll each id in the response's `child_job_ids`
    (via GET /ingest/jobs/{job_id}) individually for their own status.
    """
    session = resolve_admin_session(
        authorization=authorization,
        body_user_id=request.user_id,
        body_role=request.role,
        body_tenant_id=request.tenant_id,
    )
    try:
        job = ingestion_manager.submit_corpus(
            request.source,
            session.user.tenant_id,
            job_name=request.job_name,
            doc_key_prefix=request.doc_key_prefix,
        )
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    record_audit_event(
        event_type=AuditEventType.INGESTION_SUBMITTED,
        user_id=session.user.user_id,
        tenant_id=session.user.tenant_id,
        role=session.user.role.value,
        resource=job.id,
        action=request.job_name or request.source,
        metadata={"job_type": "corpus"},
    )
    dispatch = _dispatch_ingest_job(job.id, background_tasks, job_timeout=CORPUS_SCAN_TIMEOUT)
    return IngestionResponse(
        job_id=job.id,
        status=job.status.value,
        message="Corpus ingestion job submitted.",
        output_dir=str(job.output_dir) if job.output_dir else "",
        dispatch=dispatch,
    )


@router.post("/ingest/cypher", response_model=IngestionResponse)
async def ingest_cypher(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    job_name: Optional[str] = Form(None),
    openai_key: Optional[str] = Form(default=None),
    authorization: Optional[str] = Header(default=None),
    user_id: Optional[str] = Form(None),
    role: Optional[str] = Form(None),
    tenant_id: Optional[str] = Form(None),
):
    """
    Upload and execute arbitrary Cypher against Neo4j.

    Security:
    - Disabled by default (set ALLOW_CYPHER_INGEST=true to enable)
    - Admin role required (JWT when AUTH_ENABLED, else body role=admin)
    - Not covered by automatic tenant-stamping (see IngestionManager._process_cypher);
      restrict to trusted admin uploads in a genuinely multi-tenant deployment.
    """
    if not ALLOW_CYPHER_INGEST:
        raise HTTPException(status_code=403, detail="Cypher ingestion is disabled. Set ALLOW_CYPHER_INGEST=true to enable.")

    session = resolve_admin_session(
        authorization=authorization,
        body_user_id=user_id,
        body_role=role,
        body_tenant_id=tenant_id,
    )

    cypher_params = {}
    effective_openai_key = openai_key or OPENAI_API_KEY
    if effective_openai_key:
        cypher_params["openAIKey"] = effective_openai_key

    job = ingestion_manager.submit_cypher(
        file, session.user.tenant_id, job_name=job_name, cypher_params=cypher_params or None
    )
    record_audit_event(
        event_type=AuditEventType.INGESTION_SUBMITTED,
        user_id=session.user.user_id,
        tenant_id=session.user.tenant_id,
        role=session.user.role.value,
        resource=job.id,
        action=job_name or file.filename,
        metadata={"job_type": "cypher"},
    )
    dispatch = _dispatch_ingest_job(job.id, background_tasks)
    return IngestionResponse(
        job_id=job.id,
        status=job.status.value,
        message="Cypher ingestion job submitted.",
        output_dir=str(job.output_dir) if job.output_dir else "",
        dispatch=dispatch,
    )


@router.get("/ingest/jobs", response_model=List[IngestionJobSummary])
async def list_ingestion_jobs(
    limit: int = 50,
    authorization: Optional[str] = Header(default=None),
    user_id: Optional[str] = Query(None),
    role: Optional[str] = Query(None),
):
    """
    List recent ingestion jobs (newest first).

    Works in both in-process (InMemoryJobStore) and Redis-backed modes.
    """
    resolve_admin_session(
        authorization=authorization,
        body_user_id=user_id,
        body_role=role,
    )
    store = get_job_store()
    ids = store.list_ids(limit=limit)
    summaries = []
    for jid in reversed(ids):  # newest first
        job = store.get(jid)
        if job is None:
            continue
        summaries.append(
            IngestionJobSummary(
                job_id=job.id,
                job_type=job.type,
                status=job.status.value,
                created_at=job.created_at.isoformat() + "Z",
                name=job.name,
                logical_doc_id=job.logical_doc_id,
                revision_id=job.revision_id,
                version_number=job.version_number,
                skipped_duplicate=bool(job.skipped_duplicate),
                error=job.error,
            )
        )
    return summaries


@router.get("/ingest/jobs/{job_id}", response_model=IngestionStatusResponse)
async def get_ingestion_job(
    job_id: str,
    authorization: Optional[str] = Header(default=None),
    user_id: Optional[str] = Query(None),
    role: Optional[str] = Query(None),
):
    resolve_admin_session(
        authorization=authorization,
        body_user_id=user_id,
        body_role=role,
    )
    job = ingestion_manager.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")

    return IngestionStatusResponse(
        job_id=job.id,
        job_type=job.type,
        status=job.status.value,
        created_at=job.created_at.isoformat() + "Z",
        started_at=job.started_at.isoformat() + "Z" if job.started_at else None,
        finished_at=job.finished_at.isoformat() + "Z" if job.finished_at else None,
        output_dir=str(job.output_dir) if job.output_dir else "",
        neo4j_load_status=job.neo4j_load_status,
        neo4j_load_message=job.neo4j_load_message,
        logs=job.logs,
        error=job.error,
        logical_doc_id=job.logical_doc_id,
        revision_id=job.revision_id,
        content_hash=job.content_hash,
        version_number=job.version_number,
        skipped_duplicate=job.skipped_duplicate,
        child_job_ids=job.child_job_ids,
    )


@router.get("/ingest/quality")
async def list_ingestion_quality_documents(
    tenant_id: Optional[str] = Query(None),
    limit: int = 200,
    search: Optional[str] = Query(None, description="Filter by id/title/filename substring (case-insensitive)"),
    authorization: Optional[str] = Header(default=None),
    user_id: Optional[str] = Query(None),
    role: Optional[str] = Query(None),
):
    """List ingested documents (ACTIVE revisions) for the quality-report picker.
    Admin-only. Supports `search` so a corpus of hundreds of documents doesn't
    have to be listed in full and filtered client-side (e.g. in a dropdown)."""
    resolve_admin_session(
        authorization=authorization,
        body_user_id=user_id,
        body_role=role,
    )
    loop = asyncio.get_running_loop()
    # Neo4j reads are synchronous — off the event loop, same as /query, so
    # this endpoint doesn't stall every other in-flight request while it runs.
    docs = await loop.run_in_executor(
        _query_executor, _list_ingestion_quality_documents_sync, tenant_id, limit, search
    )
    return docs


@router.get("/ingest/quality/{logical_doc_id}")
async def get_ingestion_quality_report(
    logical_doc_id: str,
    authorization: Optional[str] = Header(default=None),
    user_id: Optional[str] = Query(None),
    role: Optional[str] = Query(None),
):
    """
    Cheap, LLM-free ingestion-quality report for one document (node/edge
    counts, text/NER/embedding coverage, page continuity, orphan nodes).
    Admin-only. Pure Cypher aggregation — no OpenAI calls, safe to call often.
    """
    resolve_admin_session(
        authorization=authorization,
        body_user_id=user_id,
        body_role=role,
    )
    driver = get_neo4j_driver()
    loop = asyncio.get_running_loop()
    report = await loop.run_in_executor(_query_executor, build_ingestion_quality_report, driver, logical_doc_id)
    if not report.get("found"):
        raise HTTPException(status_code=404, detail=f"No ingested document found for {logical_doc_id!r}")
    return report


@router.get("/ingest/quality/{logical_doc_id}/pages")
async def get_page_level_quality_report(
    logical_doc_id: str,
    authorization: Optional[str] = Header(default=None),
    user_id: Optional[str] = Query(None),
    role: Optional[str] = Query(None),
    tenant_id: Optional[str] = Query(None),
):
    """
    Page-level coverage report: for each page, how much of its raw source
    text actually made it into the graph (word-count ratio), plus its
    entity and semantic-edge counts. Admin-only. Re-parses the stored
    source PDF on demand (same "scored on demand, never automatic" posture
    as /documents/{id}/ontology-score) — costs CPU per call, no LLM calls.
    """
    session = resolve_admin_session(
        authorization=authorization, body_user_id=user_id, body_role=role, body_tenant_id=tenant_id,
    )
    context = session.user

    from ..unstructured.document.page_report import run_for_doc as run_page_report

    driver = get_neo4j_driver()
    loop = asyncio.get_running_loop()
    report = await loop.run_in_executor(
        _query_executor,
        lambda: run_page_report(driver, logical_doc_id, tenant_id=context.tenant_id),
    )
    if not report.get("found"):
        raise HTTPException(status_code=404, detail=f"No ingested document found for {logical_doc_id!r}")
    return report


@router.get("/ingest/queue/status")
async def ingest_queue_status(
    authorization: Optional[str] = Header(default=None),
    user_id: Optional[str] = Query(None),
    role: Optional[str] = Query(None),
):
    """
    Queue depth and dead-letter (failed) job visibility.

    Returns queue depth and recent failed jobs from the RQ FailedJobRegistry.
    When Redis is not configured all counts are None.
    """
    resolve_admin_session(
        authorization=authorization,
        body_user_id=user_id,
        body_role=role,
    )
    depth = queue_depth()
    failed = list_failed_jobs(limit=20)
    return {
        "redis_configured": bool(REDIS_URL),
        "queue_depth": depth,
        "failed_jobs": failed,
    }
