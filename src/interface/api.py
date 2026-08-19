"""The FastAPI application: lifecycle, static mount, and the routers.

The endpoints live in routes/, grouped by purpose -- pages, query, feedback,
ingest, documents, admin. They were all in this file, so ingest, retrieval,
feedback, audit and RBAC were interleaved across 1369 lines and neither half
of the system could be read on its own.

Nothing about the HTTP surface changed: same paths, same methods, same
response models, checked by diffing the generated OpenAPI schema before and
after. The routers take the thread pools and the ingestion manager from deps
rather than building their own -- each is sized from config, so a per-router
copy would multiply the thread count.

Startup/shutdown handlers and the static mount stay here: they belong to the
application, not to any one group of endpoints.
"""
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
from ..unstructured.graph.constants import DOC_REVISION_LABEL, DOCUMENT_LOGICAL_LABEL
from ..shared.neo4j.driver import close_neo4j_driver, get_neo4j_driver
from ..shared.neo4j.tenancy import tenant_filter
from ..unstructured.document.graph_snapshot import (
    X1_STAGE,
    X2_STAGE,
    query_final_snapshot_sync,
    query_page_scoped_snapshot_sync,
    read_snapshot as read_graph_snapshot,
)
from ..unstructured.document.purge import delete_document
from ..unstructured.document.versioning import source_file_blob_key
from ..shared.storage.blob.factory import get_blob_store
from pydantic import BaseModel, Field
from ..shared.audit import AuditEventType, get_audit_store, record_audit_event
from .bridge import ask
from ..shared.conversation import clear_turn
from ..shared.logging_config import setup_logging
from ..shared.auth.rbac_setup import GraphRBAC
from ..shared.auth.oidc import auth_public_config, resolve_admin_session, resolve_scoped_thread_id, resolve_user_context
from ..shared.config.settings import (
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
from .streaming.query_stream import iter_query_stream
from ..unstructured.ingestion.service import IngestionManager
from ..pipeline.ingestion.job_store import get_job_store
from ..pipeline.ingestion.queue import enqueue_ingest, list_failed_jobs, queue_depth
from ..unstructured.ingestion.validation import build_ingestion_quality_report, list_ingested_documents
from ..shared.feedback import (
    best_mode_for_question,
    build_dashboard_overview,
    get_feedback_store,
    maybe_attach_feedback_outcome,
    maybe_record_retrieval_feedback,
    pattern_hash,
    retrieval_pattern,
)

from .routes import admin, documents, feedback, ingest, pages, query

logger = logging.getLogger(__name__)


app = FastAPI(title="Agentic Graph RAG API")


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
            rbac.setup_schema()
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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


# Order is cosmetic -- FastAPI matches on path, not registration order -- but
# it reads the way a request flows: pages, then asking, then the pipeline.
app.include_router(pages.router)
app.include_router(query.router)
app.include_router(feedback.router)
app.include_router(ingest.router)
app.include_router(documents.router)
app.include_router(admin.router)
