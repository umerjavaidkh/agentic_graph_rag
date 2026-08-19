"""Process-wide singletons the routers share.

The two thread pools especially MUST be created once. Each is sized from
config, so a per-router copy would silently multiply the thread count by the
number of routers -- a behaviour change no test would notice.
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


ingestion_manager = IngestionManager()


_ingest_executor = ThreadPoolExecutor(max_workers=API_INGEST_EXECUTOR_WORKERS, thread_name_prefix="ingest")


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
