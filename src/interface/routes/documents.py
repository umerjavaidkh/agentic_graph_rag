"""Reading one ingested document -- file, graphs, score -- and deleting it."""
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


def _query_page_scoped_graph_snapshot_sync(
    driver, logical_doc_id: str, revision_id: str, page_number: int
) -> dict:
    with driver.session() as session:
        return query_page_scoped_snapshot_sync(session, logical_doc_id, revision_id, page_number)


def _query_final_graph_snapshot_sync(driver, logical_doc_id: str, revision_id: str) -> dict:
    with driver.session() as session:
        return query_final_snapshot_sync(session, logical_doc_id, revision_id)


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


@router.get("/documents/{logical_doc_id}/file")
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


@router.delete("/documents/{logical_doc_id}")
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


@router.get("/documents/{logical_doc_id}/graph-snapshot/{stage}")
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


@router.get("/documents/{logical_doc_id}/graph-final")
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


@router.get("/documents/{logical_doc_id}/pages/{page_number}/graph")
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


@router.get("/documents/{logical_doc_id}/ontology-score")
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

    from ...unstructured.document.ontology_report import run_for_doc

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
