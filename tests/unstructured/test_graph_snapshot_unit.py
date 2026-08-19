"""
tests/test_graph_snapshot_unit.py — graph-construction snapshot
serialization for the graph-inspector UI (src/document/graph_snapshot.py).

Covers build_snapshot (pure serialization), write_snapshot/read_snapshot
against a fake in-memory blob store, and query_page_scoped_snapshot_sync's
internal/external merge logic against a fake queued-response Neo4j session
(query_final_snapshot_sync is a simple single-query pass-through and stays
integration-level, not covered here -- query_page_scoped_snapshot_sync earns
its own coverage since it runs three queries and merges/dedups their results,
real logic that can silently break the graph-inspector's page view).

Run with:
    python -m pytest tests/test_graph_snapshot_unit.py -v
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


from src.unstructured.document.graph_snapshot import (
    X1_STAGE,
    X2_STAGE,
    build_snapshot,
    query_page_scoped_snapshot_sync,
    read_snapshot,
    write_snapshot,
)
from src.unstructured.models import DKGEdge, DKGNode, EdgeConfidenceTier, NodeType, RelType


class _FakeBlobStore:
    def __init__(self):
        self.data: dict[str, str] = {}

    def put(self, key, content, *, content_type="text/plain"):
        self.data[key] = content
        return key

    def get(self, key):
        return self.data.get(key)


def _node(node_id: str, **kw) -> DKGNode:
    defaults = dict(type=NodeType.SECTION, title=node_id, text="body text", order=0)
    defaults.update(kw)
    return DKGNode(id=node_id, **defaults)


def _edge(source_id: str, target_id: str, **kw) -> DKGEdge:
    defaults = dict(rel_type=RelType.CONTAINS, axis=1)
    defaults.update(kw)
    return DKGEdge(source_id=source_id, target_id=target_id, **defaults)


# ── build_snapshot ────────────────────────────────────────────────────────


def test_build_snapshot_shape():
    nodes = [_node("n1", depth=1, page_start=1, page_end=5)]
    edges = [_edge("n1", "n2")]
    snap = build_snapshot(X1_STAGE, logical_doc_id="doc1", revision_id="doc1:r1", nodes=nodes, edges=edges)

    assert snap["stage"] == X1_STAGE
    assert snap["logical_doc_id"] == "doc1"
    assert snap["revision_id"] == "doc1:r1"
    assert snap["node_count"] == 1
    assert snap["edge_count"] == 1
    assert "generated_at" in snap


def test_build_snapshot_node_fields_and_enum_serialization():
    node = _node("n1", type=NodeType.CHAPTER, depth=1, page_start=1, page_end=10)
    node.entities = ["a", "b"]
    node.embedding = [0.1, 0.2]
    snap = build_snapshot(X1_STAGE, logical_doc_id="d", revision_id="d:r1", nodes=[node], edges=[])

    n = snap["nodes"][0]
    assert n["id"] == "n1"
    assert n["type"] == "Chapter"  # enum -> plain string, not NodeType.CHAPTER
    assert n["depth"] == 1
    assert n["page_start"] == 1
    assert n["page_end"] == 10
    assert n["n_entities"] == 2
    assert n["has_embedding"] is True


def test_build_snapshot_node_without_embedding_or_entities():
    node = _node("n1")
    snap = build_snapshot(X1_STAGE, logical_doc_id="d", revision_id="d:r1", nodes=[node], edges=[])
    n = snap["nodes"][0]
    assert n["has_embedding"] is False
    assert n["n_entities"] == 0


def test_build_snapshot_edge_fields_and_enum_serialization():
    edge = _edge(
        "a", "b", rel_type=RelType.SHARES_ENTITY, axis=2, weight=3.0,
        confidence=0.8, confidence_tier=EdgeConfidenceTier.INFERRED,
        properties={"shared_entities": ["x"]},
    )
    snap = build_snapshot(X2_STAGE, logical_doc_id="d", revision_id="d:r1", nodes=[], edges=[edge])
    e = snap["edges"][0]
    assert e["source_id"] == "a"
    assert e["target_id"] == "b"
    assert e["rel_type"] == "SHARES_ENTITY"
    assert e["axis"] == 2
    assert e["weight"] == 3.0
    assert e["properties"] == {"shared_entities": ["x"]}


def test_build_snapshot_is_json_serializable():
    """The whole point of this module -- must round-trip through json.dumps
    without a custom encoder (no raw enums/dataclasses left unconverted)."""
    node = _node("n1", type=NodeType.PAGE)
    edge = _edge("n1", "n2", rel_type=RelType.PART_OF, confidence_tier=EdgeConfidenceTier.EXTRACTED)
    snap = build_snapshot(X1_STAGE, logical_doc_id="d", revision_id="d:r1", nodes=[node], edges=[edge])
    dumped = json.dumps(snap)
    reloaded = json.loads(dumped)
    assert reloaded["nodes"][0]["type"] == "Page"
    assert reloaded["edges"][0]["rel_type"] == "PART_OF"


def test_build_snapshot_empty_nodes_and_edges():
    snap = build_snapshot(X1_STAGE, logical_doc_id="d", revision_id="d:r1", nodes=[], edges=[])
    assert snap["node_count"] == 0
    assert snap["edge_count"] == 0
    assert snap["nodes"] == []
    assert snap["edges"] == []


# ── write_snapshot / read_snapshot round trip ────────────────────────────


def test_write_then_read_round_trips():
    store = _FakeBlobStore()
    nodes = [_node("n1", depth=1)]
    edges = [_edge("n1", "n2")]

    key = write_snapshot(
        store, X1_STAGE, tenant_id="default", logical_doc_id="doc1", revision_id="doc1:r1",
        nodes=nodes, edges=edges,
    )
    assert key in store.data

    result = read_snapshot(store, tenant_id="default", logical_doc_id="doc1", revision_id="doc1:r1", stage=X1_STAGE)
    assert result is not None
    assert result["node_count"] == 1
    assert result["nodes"][0]["id"] == "n1"


def test_read_snapshot_missing_returns_none():
    store = _FakeBlobStore()
    result = read_snapshot(
        store, tenant_id="default", logical_doc_id="nonexistent", revision_id="nonexistent:r1", stage=X1_STAGE,
    )
    assert result is None


def test_write_snapshot_keys_scoped_by_tenant_doc_revision_stage():
    """Two different (tenant, doc, revision, stage) combos must not collide."""
    store = _FakeBlobStore()
    write_snapshot(store, X1_STAGE, tenant_id="t1", logical_doc_id="d1", revision_id="d1:r1", nodes=[], edges=[])
    write_snapshot(store, X2_STAGE, tenant_id="t1", logical_doc_id="d1", revision_id="d1:r1", nodes=[], edges=[])
    write_snapshot(store, X1_STAGE, tenant_id="t2", logical_doc_id="d1", revision_id="d1:r1", nodes=[], edges=[])
    assert len(store.data) == 3


def test_write_snapshot_propagates_blob_store_failure_to_caller():
    """write_snapshot itself does NOT swallow errors -- the caller
    (ingestion/service.py) is responsible for the try/except that keeps a
    snapshot write failure from failing the whole ingestion job. This test
    documents that boundary so it isn't accidentally reversed later."""
    class RaisingStore:
        def put(self, *a, **kw):
            raise RuntimeError("blob store down")

    import pytest

    with pytest.raises(RuntimeError):
        write_snapshot(
            RaisingStore(), X1_STAGE, tenant_id="t", logical_doc_id="d", revision_id="d:r1", nodes=[], edges=[],
        )


# ── query_page_scoped_snapshot_sync (internal + external merge) ─────────────


class _FakeSession:
    """Returns queued responses in call order -- query_page_scoped_snapshot_sync
    runs exactly 3 queries (nodes, internal edges, external edges), in that
    order, each expecting a different row shape."""

    def __init__(self, responses: list[list[dict]]):
        self._responses = list(responses)

    def run(self, cypher, **kwargs):
        return self._responses.pop(0)


def _node_row(node_id: str, *, page: int = 1) -> dict:
    return {
        "id": node_id, "labels": ["Section"], "title": node_id, "order": 0,
        "depth": 1, "page_start": page, "page_end": page, "text_len": 10,
        "n_entities": 0, "has_embedding": False, "region_kind": None,
    }


def _internal_edge_row(source_id: str, target_id: str) -> dict:
    return {
        "source_id": source_id, "target_id": target_id, "rel_type": "CONTAINS",
        "axis": 1, "weight": 1.0, "confidence": 1.0, "confidence_tier": "EXTRACTED",
        "properties": "{}",
    }


def _external_edge_row(source_id: str, target_id: str, *, b_id: str, weight: float = 1.0) -> dict:
    return {
        "source_id": source_id, "target_id": target_id, "rel_type": "SHARES_ENTITY",
        "axis": 2, "weight": weight, "confidence": 0.8, "confidence_tier": "INFERRED",
        "properties": "{}",
        "b_id": b_id, "b_labels": ["Page"], "b_title": f"Page for {b_id}", "b_order": 5,
        "b_depth": 99, "b_page_start": 9, "b_page_end": 9, "b_text_len": 20,
        "b_n_entities": 3, "b_has_embedding": False, "b_region_kind": None,
    }


def test_page_scoped_snapshot_internal_only_when_no_external_edges():
    session = _FakeSession([
        [_node_row("n1"), _node_row("n2")],
        [_internal_edge_row("n1", "n2")],
        [],
    ])
    snap = query_page_scoped_snapshot_sync(session, "doc1", "doc1:r1", 1)

    assert snap["stage"] == "page_scoped"
    assert snap["node_count"] == 2
    assert snap["edge_count"] == 1
    assert snap["internal_node_count"] == 2
    assert snap["external_node_count"] == 0
    assert all(n["is_external"] is False for n in snap["nodes"])
    assert snap["edges"][0]["scope"] == "internal"


def test_page_scoped_snapshot_includes_external_edges_and_neighbor_nodes():
    session = _FakeSession([
        [_node_row("n1")],
        [],
        [_external_edge_row("n1", "ext1", b_id="ext1")],
    ])
    snap = query_page_scoped_snapshot_sync(session, "doc1", "doc1:r1", 1)

    assert snap["external_edge_count"] == 1
    assert snap["external_node_count"] == 1
    node_ids = {n["id"] for n in snap["nodes"]}
    assert "ext1" in node_ids  # external neighbor node included, not just referenced
    ext_node = next(n for n in snap["nodes"] if n["id"] == "ext1")
    assert ext_node["is_external"] is True
    assert snap["edges"][0]["scope"] == "external"
    assert snap["edges"][0]["target_id"] == "ext1"


def test_page_scoped_snapshot_external_edges_do_not_dangle_reference_missing_nodes():
    """An edge referencing a node outside the internal node set must bring
    that node along -- otherwise a renderer that drops dangling-reference
    edges (as graph_inspector.html's does) would silently show nothing for
    every external edge despite them being present in the response."""
    session = _FakeSession([
        [_node_row("n1")],
        [],
        [_external_edge_row("n1", "ext1", b_id="ext1")],
    ])
    snap = query_page_scoped_snapshot_sync(session, "doc1", "doc1:r1", 1)

    node_ids = {n["id"] for n in snap["nodes"]}
    for e in snap["edges"]:
        assert e["source_id"] in node_ids
        assert e["target_id"] in node_ids


def test_page_scoped_snapshot_dedupes_external_node_shared_by_multiple_edges():
    session = _FakeSession([
        [_node_row("n1"), _node_row("n2")],
        [],
        [
            _external_edge_row("n1", "ext1", b_id="ext1"),
            _external_edge_row("n2", "ext1", b_id="ext1"),
        ],
    ])
    snap = query_page_scoped_snapshot_sync(session, "doc1", "doc1:r1", 1)

    ext_nodes = [n for n in snap["nodes"] if n["id"] == "ext1"]
    assert len(ext_nodes) == 1  # not duplicated despite 2 edges referencing it
    assert snap["external_edge_count"] == 2
    assert snap["external_node_count"] == 1


def test_page_scoped_snapshot_counts_are_consistent():
    session = _FakeSession([
        [_node_row("n1"), _node_row("n2")],
        [_internal_edge_row("n1", "n2")],
        [_external_edge_row("n1", "ext1", b_id="ext1"), _external_edge_row("n1", "ext2", b_id="ext2")],
    ])
    snap = query_page_scoped_snapshot_sync(session, "doc1", "doc1:r1", 1)

    assert snap["node_count"] == snap["internal_node_count"] + snap["external_node_count"]
    assert snap["edge_count"] == snap["internal_edge_count"] + snap["external_edge_count"]
    assert snap["node_count"] == 4  # n1, n2, ext1, ext2
    assert snap["edge_count"] == 3  # 1 internal + 2 external
