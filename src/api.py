import asyncio
import json
import logging
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Optional

from fastapi import BackgroundTasks, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from .graph.driver import close_neo4j_driver, get_neo4j_driver
from pydantic import BaseModel, Field

from .bridge import ask
from .conversation import clear_turn
from .logging_config import setup_logging
from .auth.rbac_setup import GraphRBAC
from .auth.oidc import auth_public_config, require_admin_session, resolve_user_context
from .config.settings import (
    ALLOW_CYPHER_INGEST,
    ALLOW_DB_RESET,
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
from .telemetry.feedback import (
    best_mode_for_question,
    maybe_attach_feedback_outcome,
    maybe_record_retrieval_feedback,
    pattern_hash,
    retrieval_pattern,
)
from .telemetry.feedback.store import get_feedback_store

logger = logging.getLogger(__name__)

app = FastAPI(title="Agentic Graph RAG API")

# Shared ingestion manager (store-backed — works in both in-process and worker modes).
ingestion_manager = IngestionManager()

# Fallback executor: used only when REDIS_URL is not set (dev / single-process mode).
_ingest_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="ingest")
# Run sync RAG pipeline (LLM + Neo4j) off the asyncio event loop.
_query_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="query")


async def _run_ingest_job_local(job_id: str) -> None:
    """In-process fallback: run the job in a thread when Redis is not configured."""
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(_ingest_executor, ingestion_manager.run_job, job_id)


def _dispatch_ingest_job(job_id: str, background_tasks: BackgroundTasks) -> str:
    """
    Dispatch a job to RQ workers when Redis is configured, or run it
    locally via BackgroundTasks when it is not.  Returns the dispatch mode.
    """
    rq_job = enqueue_ingest(job_id)  # returns None when REDIS_URL not set
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


@app.get("/upload", response_class=HTMLResponse)
async def upload_page():
    html_path = Path(__file__).resolve().parent / "static" / "upload.html"
    return HTMLResponse(html_path.read_text(encoding="utf-8"))


@app.get("/chat", response_class=HTMLResponse)
async def chat_page():
    html_path = Path(__file__).resolve().parent / "static" / "chat.html"
    return HTMLResponse(html_path.read_text(encoding="utf-8"))


@app.get("/auth/config")
async def auth_config():
    """Public OIDC settings for the chat UI (no secrets)."""
    return auth_public_config()


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
    department:  Optional[str] = Field(default=None, description="User department")
    thread_id:   Optional[str] = Field(default="default")


class ClearThreadRequest(BaseModel):
    thread_id: Optional[str] = Field(default="default")


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


@app.post("/query", response_model=QueryResponse)
async def query(
    request: QueryRequest,
    authorization: Optional[str] = Header(default=None),
):
    request_id = uuid.uuid4().hex[:12]
    thread_id = request.thread_id or "default"
    session = resolve_user_context(
        authorization=authorization,
        body_user_id=request.user_id,
        body_role=request.role,
        body_department=request.department,
    )
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
    thread_id = request.thread_id or "default"
    session = resolve_user_context(
        authorization=authorization,
        body_user_id=request.user_id,
        body_role=request.role,
        body_department=request.department,
    )
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
async def chat_clear(request: ClearThreadRequest):
    """Forget the last critical document turn for this thread (e.g. New chat)."""
    clear_turn(request.thread_id or "default")
    return {"ok": True, "thread_id": request.thread_id or "default"}


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
):
    require_admin_session(authorization=authorization)
    # No 409 gate: multiple concurrent uploads are fine. The per-doc Redis lock
    # (inside IngestionManager._doc_lock) serialises revision installs for the
    # same logical document while allowing different documents to run in parallel.
    job = ingestion_manager.submit_unstructured(file, job_name=job_name, doc_key=doc_key)
    dispatch = _dispatch_ingest_job(job.id, background_tasks)
    return IngestionResponse(
        job_id=job.id,
        status=job.status.value,
        message="Unstructured ingestion job submitted.",
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
):
    """
    Upload and execute arbitrary Cypher against Neo4j.

    Security:
    - Disabled by default (set ALLOW_CYPHER_INGEST=true to enable)
    - Requires Google/OIDC sign-in and admin role
    """
    if not ALLOW_CYPHER_INGEST:
        raise HTTPException(status_code=403, detail="Cypher ingestion is disabled. Set ALLOW_CYPHER_INGEST=true to enable.")

    require_admin_session(authorization=authorization)

    cypher_params = {}
    effective_openai_key = openai_key or OPENAI_API_KEY
    if effective_openai_key:
        cypher_params["openAIKey"] = effective_openai_key

    job = ingestion_manager.submit_cypher(file, job_name=job_name, cypher_params=cypher_params or None)
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
):
    """
    List recent ingestion jobs (newest first).

    Works in both in-process (InMemoryJobStore) and Redis-backed modes.
    """
    require_admin_session(authorization=authorization)
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


@app.get("/ingest/jobs/{job_id}", response_model=IngestionStatusResponse)
async def get_ingestion_job(
    job_id: str,
    authorization: Optional[str] = Header(default=None),
):
    require_admin_session(authorization=authorization)
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
    )


@app.get("/ingest/queue/status")
async def ingest_queue_status(authorization: Optional[str] = Header(default=None)):
    """
    Queue depth and dead-letter (failed) job visibility.

    Returns queue depth and recent failed jobs from the RQ FailedJobRegistry.
    When Redis is not configured all counts are None.
    """
    require_admin_session(authorization=authorization)
    depth = queue_depth()
    failed = list_failed_jobs(limit=20)
    return {
        "redis_configured": bool(REDIS_URL),
        "queue_depth": depth,
        "failed_jobs": failed,
    }


@app.post("/admin/reset-neo4j")
async def reset_neo4j(authorization: Optional[str] = Header(default=None)):
    """
    DANGEROUS: Wipes Neo4j database contents.

    - Disabled by default (ALLOW_DB_RESET=true to enable)
    - Requires Google/OIDC sign-in and admin role
    """
    if not ALLOW_DB_RESET:
        raise HTTPException(
            status_code=403,
            detail="DB reset is disabled. Set ALLOW_DB_RESET=true to enable.",
        )

    require_admin_session(authorization=authorization)

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
