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
from .graph.constants import DOC_REVISION_LABEL, DOCUMENT_LOGICAL_LABEL
from .graph.driver import close_neo4j_driver, get_neo4j_driver
from .graph.tenancy import tenant_filter
from .document.graph_snapshot import (
    X1_STAGE,
    X2_STAGE,
    query_final_snapshot_sync,
    query_page_scoped_snapshot_sync,
    read_snapshot as read_graph_snapshot,
)
from .document.purge import delete_document
from .document.versioning import source_file_blob_key
from .storage.blob.factory import get_blob_store
from pydantic import BaseModel, Field

from .audit import AuditEventType, get_audit_store, record_audit_event
from .bridge import ask
from .conversation import clear_turn
from .logging_config import setup_logging
from .auth.rbac_setup import GraphRBAC
from .auth.oidc import auth_public_config, resolve_admin_session, resolve_scoped_thread_id, resolve_user_context
from .config.settings import (
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
from .streaming import iter_query_stream
from .ingestion.service import IngestionManager
from .ingestion.job_store import get_job_store
from .ingestion.queue import enqueue_ingest, list_failed_jobs, queue_depth
from .ingestion.validation import build_ingestion_quality_report, list_ingested_documents
from .feedback_loop import (
    best_mode_for_question,
    build_dashboard_overview,
    get_feedback_store,
    maybe_attach_feedback_outcome,
    maybe_record_retrieval_feedback,
    pattern_hash,
    retrieval_pattern,
)

logger = logging.getLogger(__name__)

app = FastAPI(title="Agentic Graph RAG API")

# Shared ingestion manager (store-backed — works in both in-process and worker modes).
ingestion_manager = IngestionManager()

# Fallback executor: used only when REDIS_URL is not set (dev / single-process mode).
_ingest_executor = ThreadPoolExecutor(max_workers=API_INGEST_EXECUTOR_WORKERS, thread_name_prefix="ingest")
# Run sync RAG pipeline (LLM + Neo4j) off the asyncio event loop.
_query_executor = ThreadPoolExecutor(max_workers=API_QUERY_EXECUTOR_WORKERS, thread_name_prefix="query")


async def _run_ingest_job_local(job_id: str) -> None:
    """In-process fallback: run the job in a thread when Redis is not configured."""
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(_ingest_executor, ingestion_manager.run_job, job_id)


def _dispatch_ingest_job(
    job_id: str, background_tasks: BackgroundTasks, *, job_timeout: str = "30m"
) -> str:
    """
    Dispatch a job to RQ workers when Redis is configured, or run it
    locally via BackgroundTasks when it is not.  Returns the dispatch mode.

    job_timeout only affects the RQ path — the BackgroundTasks/ThreadPoolExecutor
    fallback has no timeout concept.
    """
    rq_job = enqueue_ingest(job_id, job_timeout=job_timeout)  # None when REDIS_URL not set
    if rq_job is not None:
        return "worker"
    background_tasks.add_task(_run_ingest_job_local, job_id)
    return "background_task"


@app.on_event("startup")
async def _ensure_rbac_schema_initialized():
    """
    Auto-initialize RBAC seed schema/data in Neo4j if missing.

    This is idempotent (Cypher uses MERGE/IF NOT EXISTS) and safe to run on each boot.
    """
    setup_logging()
    logger.info("Agentic Graph RAG API starting")
    rbac = GraphRBAC(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
    try:
        if not rbac.is_initialized():
            rbac.setup_schema(str(PROJECT_ROOT / "src" / "auth" / "rbac_schema.cypher"))
    finally:
        rbac.close()


@app.on_event("shutdown")
async def _close_neo4j_driver():
    close_neo4j_driver()


app.mount(
    "/static",
    StaticFiles(directory=Path(__file__).resolve().parent / "static"),
    name="static",
)

@app.get("/")
async def root():
    return RedirectResponse(url="/static/chat.html")


@app.get("/upload")
async def upload_page():
    # Redirect into the mounted StaticFiles app (same pattern as root())
    # instead of reading the file off disk on every request — the old
    # per-request read_text() blocked the event loop for real disk I/O
    # that StaticFiles already serves with caching/conditional-GET support.
    return RedirectResponse(url="/static/upload.html")


@app.get("/chat")
async def chat_page():
    return RedirectResponse(url="/static/chat.html")


@app.get("/graph-inspector")
async def graph_inspector_page():
    return RedirectResponse(url="/static/graph_inspector.html")


@app.get("/feedback")
async def feedback_dashboard_page():
    """Ops dashboard: feedback store, hints, and routing apply status."""
    return RedirectResponse(url="/static/feedback.html")


@app.get("/auth/config")
async def auth_config():
    """Public OIDC settings for the chat UI (no secrets)."""
    out = auth_public_config()
    out["feedback_enabled"] = RETRIEVAL_FEEDBACK_ENABLED
    return out


@app.get("/auth/me")
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
    document_id:    Optional[str] = None  # logical doc id the answer was grounded in (document paths only)
    document_title: Optional[str] = None  # UI transparency: "answering from <document>"


@app.post("/query", response_model=QueryResponse)
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
            request_id   = request_id,
            low_confidence  = bool(result.get("low_confidence")),
            confidence_note = result.get("confidence_note"),
            document_id     = result.get("document_id") or retrieved_context.get("document_id"),
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


@app.post("/query/stream")
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


@app.post("/chat/clear")
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


class FeedbackOutcomeRequest(BaseModel):
    request_id: str
    passed: bool
    case_id: Optional[str] = None


@app.post("/feedback/outcome")
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


@app.get("/feedback/stats")
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


@app.get("/feedback/dashboard")
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


@app.post("/ingest/unstructured", response_model=IngestionResponse)
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


@app.post("/ingest/corpus", response_model=IngestionResponse)
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


@app.post("/ingest/cypher", response_model=IngestionResponse)
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


@app.get("/ingest/jobs", response_model=List[IngestionJobSummary])
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


@app.get("/audit/events", response_model=List[dict])
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


@app.get("/ingest/jobs/{job_id}", response_model=IngestionStatusResponse)
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


def _list_ingestion_quality_documents_sync(
    tenant_id: Optional[str], limit: int, search: Optional[str]
) -> list[dict]:
    driver = get_neo4j_driver()
    with driver.session() as session:
        return list_ingested_documents(
            session, tenant_id=tenant_id, limit=min(max(limit, 1), 500), search=search
        )


@app.get("/ingest/quality")
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


@app.get("/ingest/quality/{logical_doc_id}")
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


@app.get("/ingest/quality/{logical_doc_id}/pages")
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

    from .document.page_report import run_for_doc as run_page_report

    driver = get_neo4j_driver()
    loop = asyncio.get_running_loop()
    report = await loop.run_in_executor(
        _query_executor,
        lambda: run_page_report(driver, logical_doc_id, tenant_id=context.tenant_id),
    )
    if not report.get("found"):
        raise HTTPException(status_code=404, detail=f"No ingested document found for {logical_doc_id!r}")
    return report


def _active_revision_source_meta_sync(driver, logical_doc_id: str, tenant_id: str) -> Optional[dict]:
    """Look up the ACTIVE revision's tenant/revision/source-filename/title —
    everything needed to derive the same blob key ingestion wrote to
    (source_file_blob_key) without re-deriving it from a fresh ingestion plan.
    """
    with driver.session() as session:
        row = session.run(
            f"""
            MATCH (dl:{DOCUMENT_LOGICAL_LABEL} {{logical_id: $lid}})-[:ACTIVE_REVISION]->(rev:{DOC_REVISION_LABEL})
            WHERE {tenant_filter("rev")}
            RETURN rev.tenant_id AS tenant_id, rev.revision_id AS revision_id,
                   rev.source_filename AS source_filename, coalesce(dl.title, dl.logical_id) AS title
            LIMIT 1
            """,
            lid=logical_doc_id,
            tenant_id=tenant_id,
        ).single()
        return dict(row) if row else None


@app.get("/documents/{logical_doc_id}/file")
async def get_document_source_file(
    logical_doc_id: str,
    authorization: Optional[str] = Header(default=None),
    user_id: Optional[str] = Query(None),
    role: Optional[str] = Query(None),
    tenant_id: Optional[str] = Query(None),
):
    """
    Stream the original uploaded file for a document's ACTIVE revision, for
    the chat UI's "view source" side panel. Same auth as /query (any
    authenticated user, tenant-scoped) — not admin-only, since this exposes
    nothing beyond what the document agent already returns in chat answers.
    Only available for documents ingested after the source-file-persistence
    fix (or backfilled); older revisions 404 with a clear reason.
    """
    session = resolve_user_context(
        authorization=authorization,
        body_user_id=user_id,
        body_role=role,
        body_tenant_id=tenant_id,
    )
    context = session.user
    driver = get_neo4j_driver()
    loop = asyncio.get_running_loop()
    meta = await loop.run_in_executor(
        _query_executor,
        _active_revision_source_meta_sync,
        driver,
        logical_doc_id,
        context.tenant_id,
    )
    if not meta:
        raise HTTPException(status_code=404, detail=f"No document found for {logical_doc_id!r}")

    key = source_file_blob_key(
        tenant_id=meta["tenant_id"] or context.tenant_id,
        logical_id=logical_doc_id,
        revision_id=meta["revision_id"],
        source_filename=meta["source_filename"] or "",
    )
    blob_store = get_blob_store()
    data = await loop.run_in_executor(_query_executor, blob_store.get_bytes, key)
    if data is None:
        raise HTTPException(
            status_code=404,
            detail="Original source file not available for this document (ingested before the "
            "document-viewer feature, or not yet backfilled).",
        )

    suffix = Path(meta["source_filename"] or "").suffix.lower() or ".pdf"
    content_type = {".pdf": "application/pdf"}.get(suffix, "application/octet-stream")
    # Title is document metadata, not directly attacker-controlled at request
    # time, but still strip header-unsafe characters before it lands in
    # Content-Disposition (defense in depth against injection/malformed headers).
    safe_title = re.sub(r'[\r\n"]+', "_", meta["title"] or logical_doc_id)
    download_name = f"{safe_title}{suffix}"
    return Response(
        content=data,
        media_type=content_type,
        headers={"Content-Disposition": f'inline; filename="{download_name}"'},
    )


@app.delete("/documents/{logical_doc_id}")
async def delete_document_everywhere(
    logical_doc_id: str,
    keep_source: bool = Query(
        False,
        description="Keep the original uploaded file in the blob store. Everything "
        "else is derived and can be rebuilt by re-ingesting that file; the file "
        "itself cannot be rebuilt from anything.",
    ),
    authorization: Optional[str] = Header(default=None),
    user_id: Optional[str] = Query(None),
    role: Optional[str] = Query(None),
    tenant_id: Optional[str] = Query(None),
):
    """
    Delete a document from every store it lives in, keyed on its logical id.

    A document is spread across three: structure in Neo4j, text and reports
    and the uploaded file in the blob store, embeddings in the vector store.
    Deleting only the Neo4j half is what left 50,642 orphaned blobs and 6,195
    orphaned vectors behind, so this removes every revision from all three and
    then the document node itself.

    Admin-only and irreversible. `keep_source=true` spares the original upload
    so the document can be re-ingested later.
    """
    session = resolve_admin_session(
        authorization=authorization,
        body_user_id=user_id,
        body_role=role,
        body_tenant_id=tenant_id,
    )
    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(
        _query_executor,
        functools.partial(
            delete_document,
            get_neo4j_driver(),
            logical_id=logical_doc_id,
            tenant_id=session.user.tenant_id,
            keep_source=keep_source,
        ),
    )
    if result is None:
        raise HTTPException(
            status_code=404, detail=f"No document found for {logical_doc_id!r}"
        )
    return {
        "status": "ok",
        "logical_doc_id": logical_doc_id,
        "revisions_deleted": result["revisions"],
        "nodes_deleted": result["nodes"],
        "blobs_deleted": result["blobs"],
        "source_file_kept": keep_source,
        # Non-empty means part of the delete did not land: the document is gone
        # from Neo4j (so it cannot be queried) but some storage was left behind.
        # scripts/gc_orphaned_storage.py reclaims it.
        "errors": result["errors"],
    }


@app.get("/documents/{logical_doc_id}/graph-snapshot/{stage}")
async def get_document_graph_snapshot(
    logical_doc_id: str,
    stage: str,
    authorization: Optional[str] = Header(default=None),
    user_id: Optional[str] = Query(None),
    role: Optional[str] = Query(None),
    tenant_id: Optional[str] = Query(None),
    revision_id: Optional[str] = Query(None, description="Defaults to the ACTIVE revision"),
):
    """
    X1 (structural) or X2 (structural + semantic) graph-construction
    snapshot, for the graph-inspector UI. Admin-only: this exposes
    internal graph-construction detail (raw entity lists, edge weights)
    beyond what /query answers ever surface.

    Only available for documents ingested after this feature existed (or
    re-ingested) -- older revisions 404 with a clear reason, same pattern
    as /documents/{id}/file for pre-source-persistence revisions.
    """
    if stage not in (X1_STAGE, X2_STAGE):
        raise HTTPException(status_code=400, detail=f"stage must be one of: {X1_STAGE}, {X2_STAGE}")

    session = resolve_admin_session(
        authorization=authorization, body_user_id=user_id, body_role=role, body_tenant_id=tenant_id,
    )
    context = session.user
    driver = get_neo4j_driver()
    loop = asyncio.get_running_loop()

    resolved_revision_id = revision_id
    if not resolved_revision_id:
        meta = await loop.run_in_executor(
            _query_executor, _active_revision_source_meta_sync, driver, logical_doc_id, context.tenant_id,
        )
        if not meta:
            raise HTTPException(status_code=404, detail=f"No document found for {logical_doc_id!r}")
        resolved_revision_id = meta["revision_id"]

    blob_store = get_blob_store()
    snapshot = await loop.run_in_executor(
        _query_executor,
        lambda: read_graph_snapshot(
            blob_store,
            tenant_id=context.tenant_id,
            logical_doc_id=logical_doc_id,
            revision_id=resolved_revision_id,
            stage=stage,
        ),
    )
    if snapshot is None:
        raise HTTPException(
            status_code=404,
            detail=f"No {stage} snapshot for this document (ingested before the graph-inspector "
            "feature existed, semantic enrichment was skipped, or the snapshot write failed).",
        )
    return snapshot


def _query_final_graph_snapshot_sync(driver, logical_doc_id: str, revision_id: str) -> dict:
    with driver.session() as session:
        return query_final_snapshot_sync(session, logical_doc_id, revision_id)


@app.get("/documents/{logical_doc_id}/graph-final")
async def get_document_graph_final(
    logical_doc_id: str,
    authorization: Optional[str] = Header(default=None),
    user_id: Optional[str] = Query(None),
    role: Optional[str] = Query(None),
    tenant_id: Optional[str] = Query(None),
    revision_id: Optional[str] = Query(None, description="Defaults to the ACTIVE revision"),
):
    """
    The "final" stage: what's actually persisted in Neo4j right now for
    this document, queried live rather than read from a file -- Neo4j
    itself is the always-current source of truth, so there's nothing to
    write during ingestion for this stage.
    """
    session = resolve_admin_session(
        authorization=authorization, body_user_id=user_id, body_role=role, body_tenant_id=tenant_id,
    )
    context = session.user
    driver = get_neo4j_driver()
    loop = asyncio.get_running_loop()

    resolved_revision_id = revision_id
    if not resolved_revision_id:
        meta = await loop.run_in_executor(
            _query_executor, _active_revision_source_meta_sync, driver, logical_doc_id, context.tenant_id,
        )
        if not meta:
            raise HTTPException(status_code=404, detail=f"No document found for {logical_doc_id!r}")
        resolved_revision_id = meta["revision_id"]

    return await loop.run_in_executor(
        _query_executor, _query_final_graph_snapshot_sync, driver, logical_doc_id, resolved_revision_id,
    )


def _query_page_scoped_graph_snapshot_sync(
    driver, logical_doc_id: str, revision_id: str, page_number: int
) -> dict:
    with driver.session() as session:
        return query_page_scoped_snapshot_sync(session, logical_doc_id, revision_id, page_number)


@app.get("/documents/{logical_doc_id}/pages/{page_number}/graph")
async def get_document_page_scoped_graph(
    logical_doc_id: str,
    page_number: int,
    authorization: Optional[str] = Header(default=None),
    user_id: Optional[str] = Query(None),
    role: Optional[str] = Query(None),
    tenant_id: Optional[str] = Query(None),
    revision_id: Optional[str] = Query(None, description="Defaults to the ACTIVE revision"),
):
    """
    The "inner edges" view for one page: every node whose page range
    covers this page (a multi-page Chapter/Section included, not just
    nodes that collapse to exactly this one page) other than the Document
    root, and edges between them. The document-wide graph
    (/documents/{id}/graph-final, or the X1/X2 snapshots) is the "outer
    edges" view — the full cross-page/cross-section structural and
    semantic graph — and is unaffected by this endpoint. Live Neo4j query,
    same admin-only pattern as graph-final.
    """
    session = resolve_admin_session(
        authorization=authorization, body_user_id=user_id, body_role=role, body_tenant_id=tenant_id,
    )
    context = session.user
    driver = get_neo4j_driver()
    loop = asyncio.get_running_loop()

    resolved_revision_id = revision_id
    if not resolved_revision_id:
        meta = await loop.run_in_executor(
            _query_executor, _active_revision_source_meta_sync, driver, logical_doc_id, context.tenant_id,
        )
        if not meta:
            raise HTTPException(status_code=404, detail=f"No document found for {logical_doc_id!r}")
        resolved_revision_id = meta["revision_id"]

    return await loop.run_in_executor(
        _query_executor,
        _query_page_scoped_graph_snapshot_sync,
        driver, logical_doc_id, resolved_revision_id, page_number,
    )


@app.get("/documents/{logical_doc_id}/ontology-score")
async def get_document_ontology_score(
    logical_doc_id: str,
    authorization: Optional[str] = Header(default=None),
    user_id: Optional[str] = Query(None),
    role: Optional[str] = Query(None),
    tenant_id: Optional[str] = Query(None),
    axis1_only: bool = Query(False, description="Skip axis2 (no LLM calls, free/fast)"),
    axis2_only: bool = Query(False, description="Skip axis1"),
    sample_size: int = Query(15, ge=1, le=50, description="Axis-2 sample size (edges + entities)"),
):
    """
    Measured ontology-accuracy score for this document, per
    docs/DESIGN_unstructured_graph_v2.md's >=90% gate -- axis1
    (structural, free) scored against the PDF's own embedded outline or
    structural invariants; axis2 (idea-linking) via sampled LLM-judge
    precision (real cost: one small call per sampled edge/entity, hence
    both the opt-in axis1_only/axis2_only flags and the capped
    sample_size -- this is meant to be triggered explicitly from the
    Graph Inspector's Quality panel, not run automatically).
    """
    resolve_admin_session(
        authorization=authorization, body_user_id=user_id, body_role=role, body_tenant_id=tenant_id,
    )
    if axis1_only and axis2_only:
        raise HTTPException(status_code=400, detail="axis1_only and axis2_only are mutually exclusive")

    from .document.ontology_report import run_for_doc

    driver = get_neo4j_driver()
    loop = asyncio.get_running_loop()
    report = await loop.run_in_executor(
        _query_executor,
        lambda: run_for_doc(
            driver, logical_doc_id, sample_size,
            skip_axis1=axis2_only, skip_axis2=axis1_only,
        ),
    )
    if not report.get("found"):
        raise HTTPException(status_code=404, detail=f"No document found for {logical_doc_id!r}")
    return report


@app.get("/ingest/queue/status")
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


@app.post("/admin/reset-neo4j")
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


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/config/models")
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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
