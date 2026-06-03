import asyncio
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Optional

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from neo4j import GraphDatabase
from pydantic import BaseModel, Field

from .bridge import ask
from .conversation import clear_turn
from .auth.roles import Role, UserContext, validate_role
from .auth.rbac_setup import GraphRBAC
from .assets.cleanup import cleanup_all_document_assets
from .assets.factory import get_asset_store
from .config.settings import (
    ALLOW_CYPHER_INGEST,
    ALLOW_DB_RESET,
    ASSETS_DIR,
    CLEANUP_ASSETS_ON_DB_RESET,
    NEO4J_PASSWORD,
    NEO4J_URI,
    NEO4J_USER,
    OPENAI_API_KEY,
    PROJECT_ROOT,
)
from .ingestion.service import IngestionManager

app = FastAPI(title="Agentic Graph RAG API")

# ingestion manager state
ingestion_manager = IngestionManager()
# One PDF at a time keeps ingestion CPU/IO predictable and /health responsive.
_ingest_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ingest")


async def _run_ingest_job(job_id: str) -> None:
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(_ingest_executor, ingestion_manager.run_job, job_id)

@app.on_event("startup")
async def _ensure_rbac_schema_initialized():
    """
    Auto-initialize RBAC seed schema/data in Neo4j if missing.

    This is idempotent (Cypher uses MERGE/IF NOT EXISTS) and safe to run on each boot.
    """
    rbac = GraphRBAC(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
    try:
        if not rbac.is_initialized():
            rbac.setup_schema(str(PROJECT_ROOT / "src" / "auth" / "rbac_schema.cypher"))
    finally:
        rbac.close()

app.mount(
    "/static",
    StaticFiles(directory=Path(__file__).resolve().parent / "static"),
    name="static",
)

_assets_path = Path(ASSETS_DIR)
_assets_path.mkdir(parents=True, exist_ok=True)


@app.get("/assets/{asset_path:path}")
async def serve_asset(asset_path: str):
    """Serve page images from local storage or MinIO (via bytes proxy)."""
    store = get_asset_store()
    data = store.get_bytes(asset_path)
    if not data:
        raise HTTPException(status_code=404, detail="Asset not found")
    media = "image/jpeg" if asset_path.lower().endswith(".jpg") else "application/octet-stream"
    return Response(content=data, media_type=media)


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


class IngestionResponse(BaseModel):
    job_id: str
    status: str
    message: str
    output_dir: str


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


class QueryRequest(BaseModel):
    question:    str           = Field(..., description="User's question")
    role:        Optional[str] = Field(default="public", description="User role for access control")
    user_id:     Optional[str] = Field(default="public_001", description="User identifier")
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


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    try:
        role = validate_role(request.role or "public")
        context = UserContext(
            user_id=request.user_id or "public_001",
            role=role,
            department=request.department,
        )

        result = ask(
            request.question,
            user_context=context,
            thread_id=request.thread_id or "default",
        )

        return QueryResponse(
            answer       = result.get("answer", "No answer generated."),
            sources      = result.get("sources", []),
            keywords     = result.get("keywords", []),
            total_chunks = len(result.get("sources", [])),
            agent        = result.get("agent", "unstructured"),
            strategy     = result.get("strategy", "semantic"),
            access_level = result.get("_access_level", role.value),
            route_tool   = result.get("_route_tool"),
            route_method = result.get("_route_method"),
            presentation = result.get("presentation"),
            query_type   = result.get("query_type"),
            follow_up    = result.get("_follow_up"),
            telemetry    = result.get("_telemetry"),
        )
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat/clear")
async def chat_clear(request: ClearThreadRequest):
    """Forget the last critical document turn for this thread (e.g. New chat)."""
    clear_turn(request.thread_id or "default")
    return {"ok": True, "thread_id": request.thread_id or "default"}


@app.post("/ingest/unstructured", response_model=IngestionResponse)
async def ingest_unstructured(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    job_name: Optional[str] = Form(None),
    doc_key: Optional[str] = Form(
        None,
        description=(
            "Stable logical document key (e.g. godata-manual). "
            "Re-ingests with the same key create a new revision; identical file hash is skipped."
        ),
    ),
):
    if ingestion_manager.has_active_job():
        raise HTTPException(
            status_code=409,
            detail="Another ingestion job is already running. Wait for it to finish before uploading again.",
        )
    job = ingestion_manager.submit_unstructured(file, job_name=job_name, doc_key=doc_key)
    background_tasks.add_task(_run_ingest_job, job.id)
    return IngestionResponse(
        job_id=job.id,
        status=job.status.value,
        message="Unstructured ingestion job submitted.",
        output_dir=str(job.output_dir),
    )


@app.post("/ingest/cypher", response_model=IngestionResponse)
async def ingest_cypher(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    job_name: Optional[str] = Form(None),
    role: Optional[str] = Form(default="public"),
    user_id: Optional[str] = Form(default="public_001"),
    department: Optional[str] = Form(default=None),
    openai_key: Optional[str] = Form(default=None),
):
    """
    Upload and execute arbitrary Cypher against Neo4j.

    Security:
    - Disabled by default (set ALLOW_CYPHER_INGEST=true to enable)
    - When enabled, requires role >= COMPLIANCE_OFFICER
    """
    if not ALLOW_CYPHER_INGEST:
        raise HTTPException(status_code=403, detail="Cypher ingestion is disabled. Set ALLOW_CYPHER_INGEST=true to enable.")

    try:
        validated_role = validate_role(role or "public")
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))

    ctx = UserContext(
        user_id=user_id or "public_001",
        role=validated_role,
        department=department,
    )
    if not ctx.has_role(Role.COMPLIANCE_OFFICER):
        raise HTTPException(status_code=403, detail="Insufficient role to execute Cypher ingestion (requires compliance_officer or admin).")

    cypher_params = {}
    effective_openai_key = openai_key or OPENAI_API_KEY
    if effective_openai_key:
        # Supports scripts that expect $openAIKey (common in Neo4j Browser examples).
        cypher_params["openAIKey"] = effective_openai_key

    job = ingestion_manager.submit_cypher(file, job_name=job_name, cypher_params=cypher_params or None)
    background_tasks.add_task(_run_ingest_job, job.id)
    return IngestionResponse(
        job_id=job.id,
        status=job.status.value,
        message="Cypher ingestion job submitted.",
        output_dir=str(job.output_dir),
    )


@app.post("/admin/reset-neo4j")
async def reset_neo4j(
    role: Optional[str] = Form(default="public"),
    user_id: Optional[str] = Form(default="public_001"),
    department: Optional[str] = Form(default=None),
):
    """
    DANGEROUS: Wipes Neo4j database contents.

    - Disabled by default (ALLOW_DB_RESET=true to enable)
    - Requires admin role
    """
    if not ALLOW_DB_RESET:
        raise HTTPException(
            status_code=403,
            detail="DB reset is disabled. Set ALLOW_DB_RESET=true to enable.",
        )

    try:
        validated_role = validate_role(role or "public")
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))

    ctx = UserContext(
        user_id=user_id or "public_001",
        role=validated_role,
        department=department,
    )
    if not ctx.has_role(Role.ADMIN):
        raise HTTPException(status_code=403, detail="Insufficient role (requires admin).")

    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    dropped_indexes = 0
    dropped_constraints = 0
    try:
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
    finally:
        driver.close()

    assets_removed = 0
    if CLEANUP_ASSETS_ON_DB_RESET:
        try:
            assets_removed = cleanup_all_document_assets()
        except Exception:
            pass

    return {
        "status": "ok",
        "dropped_indexes": dropped_indexes,
        "dropped_constraints": dropped_constraints,
        "assets_removed": assets_removed,
        "message": (
            "Neo4j wiped (best-effort). RBAC will be re-initialized on next API startup."
            + (
                f" Removed {assets_removed} asset file(s) from storage."
                if assets_removed
                else ""
            )
        ),
    }


@app.get("/ingest/jobs/{job_id}", response_model=IngestionStatusResponse)
async def get_ingestion_job(job_id: str):
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


@app.get("/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)