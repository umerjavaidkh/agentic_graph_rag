"""Logical document identity + revision snapshots for scalable re-ingest."""
from __future__ import annotations

import hashlib
import re
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from ..models import DKGEdge, DKGNode, NodeType, RelType


def file_content_sha256(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            block = fh.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


# SEC EDGAR sample corpus filenames follow TICKER_FORM_YYYY-MM-DD.ext, where
# the date comes from EDGAR's own `filingDate` field (see
# scripts/fetch_sec_edgar_corpus.py) — the actual submission date, distinct
# from the period-end date the filing covers. That real filing date is
# usually NOT printed anywhere in the PDF body (EDGAR's "Filed:" stamp lives
# in the filing's HTML/index wrapper, not the document itself), so no amount
# of retrieval tuning can make synthesis find it in the text — the document
# genuinely doesn't say it. Extracting it from the filename (already stored
# as DocRevision.source_filename) sidesteps that entirely.
_FILING_DATE_RE = re.compile(r"(\d{4}-\d{2}-\d{2})(?:\.[A-Za-z0-9]+)?$")


def extract_filing_date_from_filename(filename: str) -> str | None:
    """Pull a trailing YYYY-MM-DD from a filename, or None if it doesn't match."""
    if not filename:
        return None
    m = _FILING_DATE_RE.search(Path(filename).name)
    return m.group(1) if m else None


def slug_logical_key(stem: str) -> str:
    safe = re.sub(r"[^a-zA-Z0-9]+", "_", (stem or "document").lower()).strip("_")
    return safe[:120] or "document"


def upload_filename_stem(file_path: Path, job_id: str | None = None) -> str:
    """
    Stem used for default logical ids.

    Temp uploads are stored as ``{job_id}_{original_name}``; strip the job prefix
    so re-ingests of the same PDF share one logical document.
    """
    stem = file_path.stem
    if job_id:
        prefix = f"{job_id}_"
        if stem.startswith(prefix):
            return stem[len(prefix) :]
    return stem


def resolve_logical_id(
    file_path: Path,
    *,
    doc_key: str | None = None,
    job_id: str | None = None,
) -> str:
    if doc_key and doc_key.strip():
        safe = re.sub(r"[^a-zA-Z0-9._-]+", "_", doc_key.strip().lower()).strip("_")
        return safe[:160] or slug_logical_key(upload_filename_stem(file_path, job_id))
    stem = upload_filename_stem(file_path, job_id)
    return f"doc_{slug_logical_key(stem)}"


@dataclass
class DocumentRevisionPlan:
    logical_id: str
    revision_id: str
    version_number: int
    content_hash: str
    content_root_id: str
    title: str
    source_filename: str
    tenant_id: str


def source_file_blob_key(
    *, tenant_id: str, logical_id: str, revision_id: str, source_filename: str
) -> str:
    """Deterministic BlobStore key for a revision's original uploaded file.

    Takes plain fields (not a DocumentRevisionPlan) so both callers can use
    it: the ingestion write path (ingestion/service.py, which has a live
    plan) and the document-viewer read path (api.py's GET
    /documents/{id}/file, which only has values re-queried from Neo4j
    metadata written by revision_metadata_nodes) always agree on where a
    revision's source bytes live.
    """
    suffix = Path(source_filename or "").suffix.lower() or ".bin"
    return f"documents/{tenant_id}/{logical_id}/{revision_id}/source{suffix}"


def build_revision_plan(
    file_path: Path,
    *,
    tenant_id: str,
    doc_key: str | None = None,
    job_id: str | None = None,
    version_number: int = 1,
    content_root_id: str | None = None,
    logical_id: str | None = None,
) -> DocumentRevisionPlan:
    # An explicit logical_id wins over one derived from doc_key/filename: the
    # caller may have found that this exact content already belongs to a
    # different logical document, and the revision must land there so the
    # older copy is superseded rather than joined by a duplicate.
    logical_id = logical_id or resolve_logical_id(
        file_path, doc_key=doc_key, job_id=job_id
    )
    revision_id = f"{logical_id}:r{version_number}"
    clean_stem = upload_filename_stem(file_path, job_id)
    root = content_root_id or f"doc_{slug_logical_key(clean_stem)}"
    return DocumentRevisionPlan(
        logical_id=logical_id,
        revision_id=revision_id,
        version_number=version_number,
        content_hash=file_content_sha256(file_path),
        content_root_id=f"{revision_id}::{root}",
        title=clean_stem,
        source_filename=file_path.name,
        tenant_id=tenant_id,
    )


def apply_revision_to_graph(
    nodes: list[DKGNode],
    edges: list[DKGEdge],
    plan: DocumentRevisionPlan,
) -> tuple[list[DKGNode], list[DKGEdge]]:
    """Prefix node ids and stamp lineage fields on every content node/edge."""
    id_map: dict[str, str] = {}

    def remap(nid: str) -> str:
        if nid in id_map:
            return id_map[nid]
        if nid.startswith(f"{plan.revision_id}::"):
            id_map[nid] = nid
            return nid
        new_id = f"{plan.revision_id}::{nid}"
        id_map[nid] = new_id
        return new_id

    out_nodes: list[DKGNode] = []
    for node in nodes:
        old_id = node.id
        node.id = remap(old_id)
        if node.type in (NodeType.DOCUMENT, NodeType.DOCUMENT.value, NodeType.BOOK, NodeType.BOOK.value):
            node.id = plan.content_root_id
            node.title = plan.title or node.title
        node.logical_doc_id = plan.logical_id
        node.revision_id = plan.revision_id
        node.lifecycle_status = "ACTIVE"
        node.content_hash = plan.content_hash
        node.tenant_id = plan.tenant_id
        out_nodes.append(node)

    out_edges: list[DKGEdge] = []
    for edge in edges:
        edge.source_id = remap(edge.source_id)
        edge.target_id = remap(edge.target_id)
        edge.properties = {
            **(edge.properties or {}),
            "revision_id": plan.revision_id,
            "logical_doc_id": plan.logical_id,
        }
        edge.tenant_id = plan.tenant_id
        out_edges.append(edge)

    return out_nodes, out_edges


def revision_metadata_nodes(plan: DocumentRevisionPlan) -> tuple[list[DKGNode], list[DKGEdge]]:
    """DocumentLogical + DocRevision nodes wired to the content root."""
    now = datetime.now(timezone.utc).isoformat()
    logical = DKGNode(
        id=plan.logical_id,
        type="DocumentLogical",
        title=plan.title,
        text=plan.title,
        search_text=plan.title,
        order=0,
        logical_doc_id=plan.logical_id,
        revision_id=None,
        lifecycle_status="ACTIVE",
        content_hash=plan.content_hash,
        tenant_id=plan.tenant_id,
    )
    revision = DKGNode(
        id=plan.revision_id,
        type="DocRevision",
        title=f"{plan.title} v{plan.version_number}",
        text=plan.source_filename,
        search_text=plan.source_filename,
        order=plan.version_number,
        logical_doc_id=plan.logical_id,
        revision_id=plan.revision_id,
        lifecycle_status="ACTIVE",
        content_hash=plan.content_hash,
        version_number=plan.version_number,
        ingested_at=now,
        source_filename=plan.source_filename,
        tenant_id=plan.tenant_id,
    )
    edges = [
        DKGEdge(plan.logical_id, plan.revision_id, RelType.HAS_REVISION, axis=1),
        DKGEdge(plan.logical_id, plan.revision_id, RelType.ACTIVE_REVISION, axis=1),
        DKGEdge(plan.revision_id, plan.content_root_id, RelType.ROOT, axis=1),
    ]
    return [logical, revision], edges
