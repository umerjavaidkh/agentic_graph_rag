import json
from pathlib import Path
from typing import List, Optional

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .bridge import ask
from .auth.roles import UserContext, validate_role
from .auth.rbac_setup import GraphRBAC
from .config.settings import NEO4J_PASSWORD, NEO4J_URI, NEO4J_USER
from .ingestion.models import StructuredCSVMapping
from .ingestion.service import IngestionManager
from .unstructured.retriever import ESGComplianceRetriever
from .structured.retriever import StructuredRetriever

app = FastAPI(title="ESG Compliance Agent API")

# for shutdown cleanup
_unstructured_retriever = ESGComplianceRetriever()
_structured_retriever   = StructuredRetriever()

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
            user_id=request.user_id or "anonymous",
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


@app.post("/ingest/structured", response_model=IngestionResponse)
async def ingest_structured(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    mapping: str = Form(...),
    job_name: Optional[str] = Form(None),
):
    try:
        mapping_payload = json.loads(mapping)
        csv_mapping = StructuredCSVMapping.parse_obj(mapping_payload)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid mapping payload: {exc}")

    job = ingestion_manager.submit_structured(file, csv_mapping, job_name=job_name)
    background_tasks.add_task(ingestion_manager.run_job, job.id)
    return IngestionResponse(
        job_id=job.id,
        status=job.status.value,
        message="Structured ingestion job submitted.",
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


@app.on_event("shutdown")
async def shutdown():
    _unstructured_retriever.close()
    _structured_retriever.close()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)