"""graph_snapshot.py — capture + read graph-construction snapshots for the
graph-inspector UI.

X1 (structural) and X2 (structural + semantic) are transient in-memory
states during ingestion (src/ingestion/service.py's `nodes, edges` lists
at two specific points) -- unlike the final Neo4j state, nothing persists
them once the ingestion job finishes, so they must be captured DURING
ingestion, not re-derived afterward. Written to the blob store (not local
disk) so this works the same whether ingestion runs in-process or on a
scaled-out worker.

Stages:
  x1_structural — right after apply_revision_to_graph (lineage stamped,
                  content_hash/logical_doc_id/revision_id set), before
                  Axis-2 runs. CONTAINS/PART_OF/PRECEDES/FOLLOWS edges only.
  x2_semantic    — right after Axis2Builder.build() adds its edges, before
                  Neo4j load. Structural edges + SHARES_ENTITY/
                  SAME_CATEGORY/SEMANTICALLY_SIMILAR/etc.

The "final" stage (what's actually persisted in Neo4j) is intentionally
NOT captured here — Neo4j itself is that snapshot, always current, so the
graph-inspector API queries it live instead of reading a third file that
could drift from reality.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Optional

from ..graph.constants import DOC_REVISION_LABEL, DOCUMENT_ROOT_CYPHER
from ..models import DKGEdge, DKGNode

X1_STAGE = "x1_structural"
X2_STAGE = "x2_semantic"


def _blob_key(tenant_id: str, logical_doc_id: str, revision_id: str, stage: str) -> str:
    return f"graph_snapshots/{tenant_id}/{logical_doc_id}/{revision_id}/{stage}.json"


def _node_to_dict(node: DKGNode) -> dict[str, Any]:
    node_type = node.type
    return {
        "id": node.id,
        "type": node_type.value if hasattr(node_type, "value") else node_type,
        "title": node.title,
        "order": node.order,
        "depth": node.depth,
        "page_start": node.page_start,
        "page_end": node.page_end,
        "text_len": len(node.text or ""),
        "n_entities": len(node.entities or []),
        "has_embedding": node.embedding is not None,
        "region_kind": node.region_kind,
    }


def _edge_to_dict(edge: DKGEdge) -> dict[str, Any]:
    rel_type = edge.rel_type
    tier = edge.confidence_tier
    return {
        "source_id": edge.source_id,
        "target_id": edge.target_id,
        "rel_type": rel_type.value if hasattr(rel_type, "value") else rel_type,
        "axis": edge.axis,
        "weight": edge.weight,
        "confidence": edge.confidence,
        "confidence_tier": tier.value if hasattr(tier, "value") else str(tier),
        "properties": edge.properties or {},
    }


def build_snapshot(
    stage: str,
    *,
    logical_doc_id: str,
    revision_id: str,
    nodes: list[DKGNode],
    edges: list[DKGEdge],
) -> dict[str, Any]:
    """Pure serialization -- no I/O, so this is directly unit-testable."""
    return {
        "stage": stage,
        "logical_doc_id": logical_doc_id,
        "revision_id": revision_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "node_count": len(nodes),
        "edge_count": len(edges),
        "nodes": [_node_to_dict(n) for n in nodes],
        "edges": [_edge_to_dict(e) for e in edges],
    }


def write_snapshot(
    blob_store,
    stage: str,
    *,
    tenant_id: str,
    logical_doc_id: str,
    revision_id: str,
    nodes: list[DKGNode],
    edges: list[DKGEdge],
) -> str:
    """Builds and writes a snapshot, returns the blob key it was written
    under. Propagates any blob-store error to the caller rather than
    swallowing it here -- a snapshot is a debugging aid, not required for
    ingestion correctness, so the caller (ingestion/service.py) wraps this
    call in its own try/except, the same way it already tolerates vision/
    semantic-enrichment step failures, instead of this function silently
    hiding a real storage problem from anyone who might want to know."""
    snapshot = build_snapshot(
        stage, logical_doc_id=logical_doc_id, revision_id=revision_id, nodes=nodes, edges=edges
    )
    key = _blob_key(tenant_id, logical_doc_id, revision_id, stage)
    blob_store.put(key, json.dumps(snapshot), content_type="application/json")
    return key


def read_snapshot(
    blob_store, *, tenant_id: str, logical_doc_id: str, revision_id: str, stage: str
) -> Optional[dict[str, Any]]:
    key = _blob_key(tenant_id, logical_doc_id, revision_id, stage)
    raw = blob_store.get(key)
    if raw is None:
        return None
    return json.loads(raw)


def _live_node_rows_to_dicts(node_rows) -> list[dict[str, Any]]:
    return [
        {
            "id": r["id"],
            "type": next((l for l in r["labels"] if l != "DocRevision"), r["labels"][0] if r["labels"] else "Unknown"),
            "title": r["title"],
            "order": r["order"],
            "depth": r["depth"],
            "page_start": r["page_start"],
            "page_end": r["page_end"],
            "text_len": r["text_len"],
            "n_entities": r["n_entities"],
            "has_embedding": r["has_embedding"],
            "region_kind": r["region_kind"],
        }
        for r in node_rows
    ]


def _live_edge_rows_to_dicts(edge_rows) -> list[dict[str, Any]]:
    edges = []
    for r in edge_rows:
        props = r["properties"]
        if isinstance(props, str) and props:
            try:
                props = json.loads(props)
            except Exception:
                props = {"_raw": props}
        elif not props:
            props = {}
        edges.append({
            "source_id": r["source_id"],
            "target_id": r["target_id"],
            "rel_type": r["rel_type"],
            "axis": r["axis"],
            "weight": r["weight"],
            "confidence": r["confidence"],
            "confidence_tier": r["confidence_tier"],
            "properties": props,
        })
    return edges


_LIVE_NODE_RETURN = """
    RETURN n.id AS id, labels(n) AS labels, n.title AS title, n.order AS order,
           n.depth AS depth, n.page_start AS page_start, n.page_end AS page_end,
           size(coalesce(n.search_text, '')) AS text_len, size(coalesce(n.entities, [])) AS n_entities,
           n.embedding IS NOT NULL AS has_embedding, n.region_kind AS region_kind
"""

_LIVE_EDGE_RETURN = """
    RETURN a.id AS source_id, b.id AS target_id, type(r) AS rel_type,
           coalesce(r.axis, 0) AS axis, coalesce(r.weight, 1.0) AS weight,
           coalesce(r.confidence, 1.0) AS confidence,
           coalesce(r.confidence_tier, '') AS confidence_tier,
           coalesce(r.properties, '') AS properties
"""


def query_final_snapshot_sync(session, logical_doc_id: str, revision_id: str) -> dict[str, Any]:
    """The "final" stage isn't captured as a file -- Neo4j itself is that
    snapshot, always current, so this queries it live instead of a third
    on-disk copy that could drift from what's actually persisted. Same
    JSON shape as build_snapshot() so the graph-inspector UI renders all
    three stages identically."""
    node_rows = session.run(
        f"""
        MATCH (n) WHERE n.logical_doc_id = $logical_doc_id AND n.revision_id = $revision_id
        {_LIVE_NODE_RETURN}
        """,
        logical_doc_id=logical_doc_id, revision_id=revision_id,
    )
    nodes = _live_node_rows_to_dicts(node_rows)

    edge_rows = session.run(
        f"""
        MATCH (a)-[r]->(b)
        WHERE a.logical_doc_id = $logical_doc_id AND a.revision_id = $revision_id
        {_LIVE_EDGE_RETURN}
        """,
        logical_doc_id=logical_doc_id, revision_id=revision_id,
    )
    edges = _live_edge_rows_to_dicts(edge_rows)

    return {
        "stage": "final_neo4j",
        "logical_doc_id": logical_doc_id,
        "revision_id": revision_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "node_count": len(nodes),
        "edge_count": len(edges),
        "nodes": nodes,
        "edges": edges,
    }


def query_page_scoped_snapshot_sync(
    session, logical_doc_id: str, revision_id: str, page_number: int
) -> dict[str, Any]:
    """The graph-inspector's page-level ("inner edges") view: every node
    whose page range covers this page (page_start <= N <= page_end) --
    NOT only nodes that collapse to exactly this one page. An earlier
    version required exact single-page containment, which excludes every
    multi-page Chapter/Section from every page's view entirely (verified
    live: a document where most sections happen to be single-page looked
    fine, but any document with a real multi-page section -- the common
    case for SEC filings, textbooks, anything with substantial chapters --
    would show nothing for most pages at all).

    The Document root (and its DocRevision sibling) is excluded even
    though its range technically covers every page -- including it would
    make every page's "focused" view balloon to the size of the whole
    document, which defeats the point of a page-scoped view. In practice
    only the handful of Chapter/Section/Page/Region nodes whose range
    actually reaches this page remain, which is usually a small, useful
    set (one chapter, the section(s) inside it that reach this page, the
    page itself, any regions on it) -- not "everything"."""
    node_rows = session.run(
        f"""
        MATCH (n) WHERE n.logical_doc_id = $logical_doc_id AND n.revision_id = $revision_id
          AND n.page_start <= $page_number AND n.page_end >= $page_number
          AND NOT n:{DOCUMENT_ROOT_CYPHER} AND NOT n:{DOC_REVISION_LABEL}
        {_LIVE_NODE_RETURN}
        """,
        logical_doc_id=logical_doc_id, revision_id=revision_id, page_number=page_number,
    )
    nodes = _live_node_rows_to_dicts(node_rows)

    edge_rows = session.run(
        f"""
        MATCH (a)-[r]->(b)
        WHERE a.logical_doc_id = $logical_doc_id AND a.revision_id = $revision_id
          AND a.page_start <= $page_number AND a.page_end >= $page_number
          AND b.page_start <= $page_number AND b.page_end >= $page_number
          AND NOT a:{DOCUMENT_ROOT_CYPHER} AND NOT a:{DOC_REVISION_LABEL}
          AND NOT b:{DOCUMENT_ROOT_CYPHER} AND NOT b:{DOC_REVISION_LABEL}
        {_LIVE_EDGE_RETURN}
        """,
        logical_doc_id=logical_doc_id, revision_id=revision_id, page_number=page_number,
    )
    edges = _live_edge_rows_to_dicts(edge_rows)

    return {
        "stage": "page_scoped",
        "logical_doc_id": logical_doc_id,
        "revision_id": revision_id,
        "page_number": page_number,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "node_count": len(nodes),
        "edge_count": len(edges),
        "nodes": nodes,
        "edges": edges,
    }
