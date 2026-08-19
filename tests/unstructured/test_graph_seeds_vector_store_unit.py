"""
tests/test_graph_seeds_vector_store_unit.py — GraphSeedService VectorStore read path.

Covers the read-side wiring: when VECTOR_STORE_BACKEND == "qdrant",
vector_seed dispatches to the external VectorStore instead of Neo4j's
native vector index, and hydrates title/node_label from Neo4j plus full
text via BlobHydrator (docs/DESIGN_unstructured_graph_v2.md phase 3 --
text moved off `n.text` to blob storage, hydrated via `n.blob_key_text`)
for the matched ids.

Originally written against mixins/graph_seeds.py's GraphSeedsMixin, before
the loosely-coupled retrieval refactor extracted this into a standalone
GraphSeedService (src/retrieval/unstructured/services/graph_seeds.py). The
service has no dependency on the mixins/__init__.py chain (which pulls in
neo4j/auth.rbac_setup via hybrid.py), so unlike the old mixin-based version
of this file, no sys.modules stubbing gymnastics are needed here — a plain
import just works.

Run with:
    python -m pytest tests/test_graph_seeds_vector_store_unit.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest


from src.unstructured.retrieval.services import graph_seeds as graph_seeds_mod
from src.unstructured.retrieval.services.graph_seeds import GraphSeedService
from src.unstructured.retrieval.services.ranking import RankingService


class _FakeSession:
    def __init__(self, rows):
        self.rows = rows
        self.last_cypher = None

    def run(self, cypher, **kwargs):
        self.last_cypher = cypher
        return self.rows


@pytest.fixture()
def service() -> GraphSeedService:
    return GraphSeedService(RankingService())


class _FakeHydrator:
    """Stands in for BlobHydrator -- maps a blob_key straight to a
    deterministic string instead of hitting a real blob store, so these
    tests can assert on hydrated `text` without a BlobStore dependency."""

    def hydrate(self, blob_key, fallback=""):
        if not blob_key:
            return fallback
        return f"hydrated:{blob_key}"


def test_vector_seed_via_vector_store_hydrates_from_neo4j(service, monkeypatch):
    fake_store = type(
        "FakeVectorStore",
        (),
        {"query": lambda self, embedding, top_k=10, filters=None: [("id1", 0.9), ("id2", 0.5)]},
    )()
    monkeypatch.setattr(graph_seeds_mod, "get_vector_store", lambda: fake_store)
    monkeypatch.setattr(graph_seeds_mod, "get_hydrator", _FakeHydrator)

    session = _FakeSession(
        [
            {"id": "id1", "title": "Title 1", "blob_key_text": "blob/id1", "node_label": "Section"},
            {"id": "id2", "title": "Title 2", "blob_key_text": "blob/id2", "node_label": "Section"},
        ]
    )

    results = service.vector_seed_via_vector_store(session, [0.1, 0.2], limit=5)

    assert [r["id"] for r in results] == ["id1", "id2"]
    assert results[0]["score"] == 0.9
    # text comes from BlobHydrator.hydrate(blob_key_text), not a raw Cypher
    # column -- this is the actual regression test for the phase-3 rewire.
    assert results[0]["text"] == "hydrated:blob/id1"
    # Two consecutive WHERE clauses on the same MATCH is a Cypher syntax error
    # (`_FakeSession` never validates the query, so this regressed silently
    # until exercised against a real Neo4j) -- must be a single WHERE with AND.
    assert session.last_cypher.count("WHERE") == 1


def test_vector_seed_via_vector_store_empty_hits_returns_empty(service, monkeypatch):
    fake_store = type("FakeVectorStore", (), {"query": lambda self, *a, **k: []})()
    monkeypatch.setattr(graph_seeds_mod, "get_vector_store", lambda: fake_store)

    session = _FakeSession([])
    results = service.vector_seed_via_vector_store(session, [0.1, 0.2], limit=5)

    assert results == []


def test_vector_seed_via_vector_store_drops_ids_missing_from_neo4j(service, monkeypatch):
    """Vector hits for ids no longer present in Neo4j (e.g. expired revision) are dropped."""
    fake_store = type(
        "FakeVectorStore",
        (),
        {"query": lambda self, *a, **k: [("id1", 0.9), ("stale_id", 0.8)]},
    )()
    monkeypatch.setattr(graph_seeds_mod, "get_vector_store", lambda: fake_store)
    monkeypatch.setattr(graph_seeds_mod, "get_hydrator", _FakeHydrator)

    # Neo4j only returns id1 — stale_id has no matching node (e.g. purged on supersede).
    session = _FakeSession(
        [{"id": "id1", "title": "T", "blob_key_text": "blob/id1", "node_label": "Section"}]
    )
    results = service.vector_seed_via_vector_store(session, [0.1, 0.2], limit=5)

    assert [r["id"] for r in results] == ["id1"]


def test_vector_seed_dispatches_to_vector_store_when_backend_is_qdrant(service, monkeypatch):
    monkeypatch.setattr(graph_seeds_mod, "VECTOR_STORE_BACKEND", "qdrant")
    called = {}

    def fake_dispatch(self, session, embedding, limit, tenant_id="", document_id=""):
        called["invoked"] = True
        return ["via_vector_store"]

    monkeypatch.setattr(GraphSeedService, "vector_seed_via_vector_store", fake_dispatch)

    session = _FakeSession([])
    result = service.vector_seed(session, [0.1], limit=5)

    assert called.get("invoked") is True
    assert result == ["via_vector_store"]


def test_vector_seed_uses_neo4j_native_index_when_backend_is_memory(service, monkeypatch):
    monkeypatch.setattr(graph_seeds_mod, "VECTOR_STORE_BACKEND", "memory")
    session = _FakeSession([])

    service.vector_seed(session, [0.1], limit=5)

    assert "db.index.vector.queryNodes" in session.last_cypher
