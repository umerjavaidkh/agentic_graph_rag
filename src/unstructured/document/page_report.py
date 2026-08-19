"""page_report.py — wires page_validation.py's pure scoring up against a
live document.

Two paths to a page's raw text:
  - fast (the common case): read the report `write_page_report` persisted
    during ingestion (src/ingestion/service.py, right after
    Axis1StructuralBuilder.build() -- see check_construction_coverage in
    page_validation.py), which already compared this exact ingestion run's
    `ir`/`nodes` against each other. No re-parse needed.
  - fallback: for a revision ingested before this feature existed (no
    persisted report in blob storage), re-parse the stored source file
    through the same parser_registry dispatch ingestion itself used.

Either way, entity_count/edge_count per page are always read live from
Neo4j here, never persisted -- they can change independently of the text
coverage question (re-enrichment, edge pruning) and cost nothing to query.

Split the same way ontology_report.py sits in front of
ontology_validation.py: pure scoring stays Neo4j/parser-free there, this
file is the impure wiring the Graph Inspector's Pages panel calls into
(GET /ingest/quality/{id}/pages in src/api.py).
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any, Optional

from .parser_registry import get_parser
from ..ingestion.validation import resolve_active_revision
from ...shared.storage.blob.factory import get_blob_store
from ...shared.storage.hydrator import get_hydrator
from .page_validation import score_page, summarize_pages
from .versioning import source_file_blob_key

_SEMANTIC_REL_TYPES = (
    "SEMANTICALLY_SIMILAR",
    "SHARES_ENTITY",
    "SAME_CATEGORY",
    "CONTRADICTS",
    "ELABORATES",
    "PREREQUISITE_OF",
)


def _page_report_blob_key(tenant_id: str, logical_doc_id: str, revision_id: str) -> str:
    return f"page_reports/{tenant_id}/{logical_doc_id}/{revision_id}.json"


def write_page_report(
    blob_store, *, tenant_id: str, logical_doc_id: str, revision_id: str, report: dict[str, Any]
) -> str:
    """Persists the construction-time coverage report (page_validation.
    check_construction_coverage's output) so later reads of this revision
    don't need to re-parse the source file -- best-effort, same as the X1/
    X2 graph snapshots: propagates any blob-store error to the caller
    rather than swallowing it, so ingestion/service.py's own try/except
    (which already tolerates snapshot-write failures) decides how to log
    it, instead of this function silently hiding a real storage problem."""
    key = _page_report_blob_key(tenant_id, logical_doc_id, revision_id)
    blob_store.put(key, json.dumps(report), content_type="application/json")
    return key


def read_page_report(
    blob_store, *, tenant_id: str, logical_doc_id: str, revision_id: str
) -> Optional[dict[str, Any]]:
    key = _page_report_blob_key(tenant_id, logical_doc_id, revision_id)
    raw = blob_store.get(key)
    if raw is None:
        return None
    return json.loads(raw)


def _fetch_pdf_bytes(logical_doc_id: str, revision_meta: dict, tenant_id: str) -> Optional[bytes]:
    key = source_file_blob_key(
        tenant_id=revision_meta.get("tenant_id") or tenant_id,
        logical_id=logical_doc_id,
        revision_id=revision_meta["revision_id"],
        source_filename=revision_meta.get("source_filename") or "",
    )
    try:
        return get_blob_store().get_bytes(key)
    except Exception:
        return None


def _raw_page_text_by_number(pdf_bytes: bytes, source_filename: str) -> dict[int, str]:
    """parse_ir() takes a path, not bytes, so the stored source is spooled
    to a temp file with the original suffix first -- parser_registry's
    dispatch is extension-keyed and needs a real suffix to pick the same
    backend ingestion used."""
    suffix = Path(source_filename or "").suffix or ".pdf"
    with tempfile.NamedTemporaryFile(suffix=suffix) as tmp:
        tmp.write(pdf_bytes)
        tmp.flush()
        parser = get_parser(tmp.name)
        ir = parser.parse_ir(tmp.name)
    return {page.page: page.text for page in ir.pages}


def _fetch_graph_pages(session, logical_doc_id: str, revision_id: str) -> list[dict]:
    """One row per Page node: its text (hydrated from full-text blob
    storage when available -- same "read the exact same text pre- and
    post lean-storage-migration" reasoning as ontology_report._sample_edges
    -- falling back to the chunk-bounded search_text otherwise), plus a
    count of semantic edges touching it so the Pages panel can show how
    connected a page is without a second round trip."""
    rows = session.run(
        f"""
        MATCH (p:Page) WHERE p.logical_doc_id = $logical_doc_id AND p.revision_id = $revision_id
        OPTIONAL MATCH (p)-[r]-()
        WHERE type(r) IN $rel_types
        WITH p, count(r) AS edge_count
        RETURN p.order AS page_number, coalesce(p.search_text, '') AS search_text,
               p.blob_key_text AS blob_key_text,
               size(coalesce(p.entities, [])) AS entity_count, edge_count
        ORDER BY p.order
        """,
        logical_doc_id=logical_doc_id,
        revision_id=revision_id,
        rel_types=list(_SEMANTIC_REL_TYPES),
    )
    return [dict(r) for r in rows]


def _merge_live_counts(pages: list[dict[str, Any]], graph_by_number: dict[int, dict]) -> list[dict[str, Any]]:
    """entity_count/edge_count don't feed into a page's coverage/status
    (score_page computes those from raw_text/graph_text alone) -- safe to
    overlay live Neo4j counts onto an already-scored page dict without
    re-scoring anything."""
    merged = []
    for p in pages:
        g = graph_by_number.get(p["page_number"], {"entity_count": 0, "edge_count": 0})
        merged.append({**p, "entity_count": g["entity_count"], "edge_count": g["edge_count"]})
    return merged


def run_for_doc(driver, logical_doc_id: str, *, tenant_id: str = "default") -> dict:
    with driver.session() as session:
        active = resolve_active_revision(session, logical_doc_id)
        if active is None:
            return {"logical_doc_id": logical_doc_id, "found": False}
        revision_id = active["revision_id"]
        graph_pages = _fetch_graph_pages(session, logical_doc_id, revision_id)

    hydrator = get_hydrator()
    graph_by_number: dict[int, dict] = {}
    for row in graph_pages:
        text = hydrator.hydrate(row["blob_key_text"], fallback=row["search_text"] or "")
        graph_by_number[row["page_number"]] = {
            "graph_text": text,
            "entity_count": row["entity_count"] or 0,
            "edge_count": row["edge_count"] or 0,
        }

    blob_store = get_blob_store()
    persisted = read_page_report(
        blob_store,
        tenant_id=active.get("tenant_id") or tenant_id,
        logical_doc_id=logical_doc_id,
        revision_id=revision_id,
    )
    if persisted is not None:
        return {
            "logical_doc_id": logical_doc_id,
            "revision_id": revision_id,
            "found": True,
            "source": "persisted",
            "pages": _merge_live_counts(persisted["pages"], graph_by_number),
            "summary": persisted["summary"],
        }

    pdf_bytes = _fetch_pdf_bytes(logical_doc_id, {**active, "revision_id": revision_id}, tenant_id)
    if pdf_bytes is None:
        return {
            "logical_doc_id": logical_doc_id,
            "revision_id": revision_id,
            "found": True,
            "error": "Original source file not available for this revision (ingested before "
            "source-file persistence, or not backfilled) -- page-level coverage needs to "
            "re-parse it.",
        }

    raw_pages = _raw_page_text_by_number(pdf_bytes, active.get("source_filename") or "")

    all_page_numbers = sorted(set(raw_pages) | set(graph_by_number))
    pages = [
        score_page(
            page_number=n,
            raw_text=raw_pages.get(n, ""),
            graph_text=(graph_by_number.get(n) or {}).get("graph_text", ""),
            entity_count=(graph_by_number.get(n) or {}).get("entity_count", 0),
            edge_count=(graph_by_number.get(n) or {}).get("edge_count", 0),
        )
        for n in all_page_numbers
    ]

    return {
        "logical_doc_id": logical_doc_id,
        "revision_id": revision_id,
        "found": True,
        "source": "reparsed",
        "pages": [p.as_dict() for p in pages],
        "summary": summarize_pages(pages),
    }
