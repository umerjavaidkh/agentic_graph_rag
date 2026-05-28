import json
from pathlib import Path
from typing import List, Optional

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .bridge import ask
from .auth.roles import Role, UserContext, validate_role
from .auth.rbac_setup import GraphRBAC
from .config.settings import ALLOW_CYPHER_INGEST, NEO4J_PASSWORD, NEO4J_URI, NEO4J_USER, OPENAI_API_KEY
from .ingestion.service import IngestionManager

app = FastAPI(title="ESG Compliance Agent API")

# ingestion manager state
ingestion_manager = IngestionManager()

@app.on_event("startup")
async def _ensure_rbac_schema_initialized():
    """
    Auto-initialize RBAC seed schema/data in Neo4j if missing.

    This is idempotent (Cypher uses MERGE/IF NOT EXISTS) and safe to run on each boot.
    """
    rbac = GraphRBAC(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
    try:
        if not rbac.is_initialized():
            rbac.setup_schema("src/auth/rbac_schema.cypher")
    finally:
        rbac.close()

app.mount(
    "/static",
    StaticFiles(directory=Path(__file__).resolve().parent / "static"),
    name="static",
)


@app.get("/upload", response_class=HTMLResponse)
async def upload_page():
    html_path = Path(__file__).resolve().parent / "static" / "upload.html"
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


class QueryRequest(BaseModel):
    question:    str           = Field(..., description="User's question")
    role:        Optional[str] = Field(default="public", description="User role for access control")
    user_id:     Optional[str] = Field(default="public_001", description="User identifier")
    department:  Optional[str] = Field(default=None, description="User department")
    thread_id:   Optional[str] = Field(default="default")


class QueryResponse(BaseModel):
    answer:       str
    sources:      list
    keywords:     list
    total_chunks: int
    agent:        str   # "unstructured" | "structured" | "hybrid"
    strategy:     str   # "semantic" | "text2cypher" | "multi_hop" | "vector"
    access_level: str


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    try:
        role = validate_role(request.role or "public")
        context = UserContext(
            user_id=request.user_id or "public_001",
            role=role,
            department=request.department,
        )

        result = ask(request.question, user_context=context)

        return QueryResponse(
            answer       = result.get("answer", "No answer generated."),
            sources      = result.get("sources", []),
            keywords     = result.get("keywords", []),
            total_chunks = len(result.get("sources", [])),
            agent        = result.get("agent", "unstructured"),
            strategy     = result.get("strategy", "semantic"),
            access_level = result.get("_access_level", role.value),
        )
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ingest/unstructured", response_model=IngestionResponse)
async def ingest_unstructured(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    job_name: Optional[str] = Form(None),
):
    job = ingestion_manager.submit_unstructured(file, job_name=job_name)
    background_tasks.add_task(ingestion_manager.run_job, job.id)
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
    background_tasks.add_task(ingestion_manager.run_job, job.id)
    return IngestionResponse(
        job_id=job.id,
        status=job.status.value,
        message="Cypher ingestion job submitted.",
        output_dir=str(job.output_dir),
    )


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
    )


@app.get("/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)