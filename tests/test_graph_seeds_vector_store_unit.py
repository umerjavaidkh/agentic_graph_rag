"""
tests/test_graph_seeds_vector_store_unit.py — GraphSeedsMixin VectorStore read path.

Covers the Phase 4 read-side wiring: when VECTOR_STORE_BACKEND == "qdrant",
_vector_seed dispatches to the external VectorStore instead of Neo4j's
native vector index, and hydrates title/text/node_label from Neo4j for the
matched ids.

Run with:
    python -m pytest tests/test_graph_seeds_vector_store_unit.py -v
"""
from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path

import pytest

_root = Path(__file__).resolve().parents[1]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

_SIBLING_MIXINS = {
    "box_strategy": "BoxStrategyMixin",
    "document_resolver": "DocumentResolverMixin",
    "hybrid": "HybridRetrieveMixin",
    "lexical": "LexicalRetrievalMixin",
    "page_strategy": "PageStrategyMixin",
    "policies": "PoliciesMixin",
    "ranking": "RankingMixin",
    "subsection": "SubsectionMixin",
    "toc_strategy": "TocStrategyMixin",
}


@pytest.fixture()
def graph_seeds_mod():
    """
    Fresh, un-stubbed src.retrieval.unstructured.mixins.graph_seeds.

    Other test modules in this suite stub src.graph*/src.document*/etc. with
    plain placeholder modules at *import* time, and pytest imports every test
    module during collection before running any test — so a stale stub (e.g.
    an empty "src.graph" with no __path__) can leak in here depending on
    collection order. Re-importing fresh inside a fixture (i.e. at
    test-execution time, after all collection has finished) sidesteps that.

    mixins/__init__.py also imports every sibling mixin, and hybrid.py pulls
    in auth.rbac_setup -> real `neo4j` (not installed in this env). We only
    need the real graph_seeds.py, so the siblings are pre-stubbed in
    sys.modules; Python's import machinery finds them already cached and
    never touches the real (heavy-dependency) files.

    Only clear src.graph* and src.retrieval.unstructured* — NOT the whole
    src.retrieval.* tree. src.retrieval.structured.verification is a
    completely unrelated sibling subpackage that other test files import
    directly and hold live references into; blanket-deleting it here would
    force a second, distinct module instance into existence the next time
    any test does `import src.retrieval.structured.verification`, silently
    breaking monkeypatch targeting in those files.
    """
    for name in list(sys.modules):
        if name == "src.graph" or name.startswith("src.graph."):
            del sys.modules[name]
        if name == "src.retrieval.unstructured" or name.startswith("src.retrieval.unstructured."):
            del sys.modules[name]

    for submod, cls_name in _SIBLING_MIXINS.items():
        name = f"src.retrieval.unstructured.mixins.{submod}"
        mod = types.ModuleType(name)
        setattr(mod, cls_name, type(cls_name, (), {}))
        sys.modules[name] = mod

    return importlib.import_module("src.retrieval.unstructured.mixins.graph_seeds")


class _FakeSession:
    def __init__(self, rows):
        self.rows = rows
        self.last_cypher = None

    def run(self, cypher, **kwargs):
        self.last_cypher = cypher
        return self.rows


def _seeder(mod):
    return type("_Seeder", (mod.GraphSeedsMixin,), {})()


def test_vector_seed_via_vector_store_hydrates_from_neo4j(graph_seeds_mod, monkeypatch):
    fake_store = type(
        "FakeVectorStore",
        (),
        {"query": lambda self, embedding, top_k=10, filters=None: [("id1", 0.9), ("id2", 0.5)]},
    )()
    monkeypatch.setattr(graph_seeds_mod, "get_vector_store", lambda: fake_store)

    session = _FakeSession(
        [
            {"id": "id1", "title": "Title 1", "text": "Text 1", "node_label": "Section"},
            {"id": "id2", "title": "Title 2", "text": "Text 2", "node_label": "Section"},
        ]
    )

    results = _seeder(graph_seeds_mod)._vector_seed_via_vector_store(session, [0.1, 0.2], limit=5)

    assert [r["id"] for r in results] == ["id1", "id2"]
    assert results[0]["score"] == 0.9
    assert results[0]["text"] == "Text 1"


def test_vector_seed_via_vector_store_empty_hits_returns_empty(graph_seeds_mod, monkeypatch):
    fake_store = type("FakeVectorStore", (), {"query": lambda self, *a, **k: []})()
    monkeypatch.setattr(graph_seeds_mod, "get_vector_store", lambda: fake_store)

    session = _FakeSession([])
    results = _seeder(graph_seeds_mod)._vector_seed_via_vector_store(session, [0.1, 0.2], limit=5)

    assert results == []


def test_vector_seed_via_vector_store_drops_ids_missing_from_neo4j(graph_seeds_mod, monkeypatch):
    """Vector hits for ids no longer present in Neo4j (e.g. expired revision) are dropped."""
    fake_store = type(
        "FakeVectorStore",
        (),
        {"query": lambda self, *a, **k: [("id1", 0.9), ("stale_id", 0.8)]},
    )()
    monkeypatch.setattr(graph_seeds_mod, "get_vector_store", lambda: fake_store)

    # Neo4j only returns id1 — stale_id has no matching node (e.g. purged on supersede).
    session = _FakeSession([{"id": "id1", "title": "T", "text": "Text", "node_label": "Section"}])
    results = _seeder(graph_seeds_mod)._vector_seed_via_vector_store(session, [0.1, 0.2], limit=5)

    assert [r["id"] for r in results] == ["id1"]


def test_vector_seed_dispatches_to_vector_store_when_backend_is_qdrant(graph_seeds_mod, monkeypatch):
    monkeypatch.setattr(graph_seeds_mod, "VECTOR_STORE_BACKEND", "qdrant")
    called = {}

    def fake_dispatch(self, session, embedding, limit, tenant_id=""):
        called["invoked"] = True
        return ["via_vector_store"]

    monkeypatch.setattr(graph_seeds_mod.GraphSeedsMixin, "_vector_seed_via_vector_store", fake_dispatch)

    session = _FakeSession([])
    result = _seeder(graph_seeds_mod)._vector_seed(session, [0.1], limit=5)

    assert called.get("invoked") is True
    assert result == ["via_vector_store"]


def test_vector_seed_uses_neo4j_native_index_when_backend_is_memory(graph_seeds_mod, monkeypatch):
    monkeypatch.setattr(graph_seeds_mod, "VECTOR_STORE_BACKEND", "memory")
    session = _FakeSession([])

    _seeder(graph_seeds_mod)._vector_seed(session, [0.1], limit=5)

    assert "db.index.vector.queryNodes" in session.last_cypher
