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


def build_revision_plan(
    file_path: Path,
    *,
    doc_key: str | None = None,
    job_id: str | None = None,
    version_number: int = 1,
    content_root_id: str | None = None,
) -> DocumentRevisionPlan:
    logical_id = resolve_logical_id(file_path, doc_key=doc_key, job_id=job_id)
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
        order=0,
        logical_doc_id=plan.logical_id,
        revision_id=None,
        lifecycle_status="ACTIVE",
        content_hash=plan.content_hash,
    )
    revision = DKGNode(
        id=plan.revision_id,
        type="DocRevision",
        title=f"{plan.title} v{plan.version_number}",
        text=plan.source_filename,
        order=plan.version_number,
        logical_doc_id=plan.logical_id,
        revision_id=plan.revision_id,
        lifecycle_status="ACTIVE",
        content_hash=plan.content_hash,
        version_number=plan.version_number,
        ingested_at=now,
        source_filename=plan.source_filename,
    )
    edges = [
        DKGEdge(plan.logical_id, plan.revision_id, RelType.HAS_REVISION, axis=1),
        DKGEdge(plan.logical_id, plan.revision_id, RelType.ACTIVE_REVISION, axis=1),
        DKGEdge(plan.revision_id, plan.content_root_id, RelType.ROOT, axis=1),
    ]
    return [logical, revision], edges
