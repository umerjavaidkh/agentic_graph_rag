"""tests/test_document_scoping_unit.py — document-scoped retrieval in
GraphSeedService (vector_seed / fulltext_seed / graph_expand) and its
_document_filter helper.

Covers the fix for the terminal hybrid retrieval path (FullHybridStrategy)
never filtering by document — only by tenant — so a generic question with
no distinguishing vocabulary ("what is discussed in Part II of this
filing?") would silently search the whole tenant corpus once more than one
document was ingested.

Run with:
    python -m pytest tests/test_document_scoping_unit.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_root = Path(__file__).resolve().parents[1]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from src.retrieval.unstructured.services.graph_seeds import GraphSeedService, _document_filter
from src.retrieval.unstructured.services.ranking import RankingService


class _FakeSession:
    def __init__(self, rows):
        self.rows = rows
        self.last_cypher = None
        self.last_kwargs = None

    def run(self, cypher, **kwargs):
        self.last_cypher = cypher
        self.last_kwargs = kwargs
        return self.rows


@pytest.fixture()
def service() -> GraphSeedService:
    return GraphSeedService(RankingService())


# ── _document_filter helper ──────────────────────────────────────────────


def test_document_filter_degrades_to_true_when_no_document_resolved():
    assert _document_filter("n", "") == "true"


def test_document_filter_scopes_alias_when_document_resolved():
    assert _document_filter("n", "aapl-10k-2024") == "n.logical_doc_id = $document_id"


def test_document_filter_uses_the_given_alias():
    assert _document_filter("related", "amzn-10q") == "related.logical_doc_id = $document_id"


# ── vector_seed ───────────────────────────────────────────────────────────


def test_vector_seed_passes_document_id_param(service, monkeypatch):
    from src.retrieval.unstructured.services import graph_seeds as graph_seeds_mod

    monkeypatch.setattr(graph_seeds_mod, "VECTOR_STORE_BACKEND", "memory")
    session = _FakeSession([])

    service.vector_seed(session, [0.1], limit=5, tenant_id="default", document_id="aapl-10k-2024")

    assert session.last_kwargs["document_id"] == "aapl-10k-2024"
    assert "n.logical_doc_id = $document_id" in session.last_cypher


def test_vector_seed_unscoped_when_no_document_id(service, monkeypatch):
    from src.retrieval.unstructured.services import graph_seeds as graph_seeds_mod

    monkeypatch.setattr(graph_seeds_mod, "VECTOR_STORE_BACKEND", "memory")
    session = _FakeSession([])

    service.vector_seed(session, [0.1], limit=5, tenant_id="default")

    assert session.last_kwargs["document_id"] == ""


def test_vector_seed_overfetches_from_index_when_scoping(service, monkeypatch):
    """The vector index's own top-K is computed BEFORE the document filter
    runs, so scoping must request a larger candidate pool from the index
    itself (not just cap the final output), or a document's real matches
    could be crowded out of the unfiltered global top-K entirely."""
    from src.retrieval.unstructured.services import graph_seeds as graph_seeds_mod

    monkeypatch.setattr(graph_seeds_mod, "VECTOR_STORE_BACKEND", "memory")

    session_scoped = _FakeSession([])
    service.vector_seed(session_scoped, [0.1], limit=5, document_id="aapl-10k-2024")
    assert session_scoped.last_kwargs["index_limit"] > 5
    assert session_scoped.last_kwargs["limit"] == 5

    session_unscoped = _FakeSession([])
    service.vector_seed(session_unscoped, [0.1], limit=5, document_id="")
    assert session_unscoped.last_kwargs["index_limit"] == 5


# ── fulltext_seed ─────────────────────────────────────────────────────────


def test_fulltext_seed_passes_document_id_and_overfetches(service):
    session = _FakeSession([])

    service.fulltext_seed(session, "what is discussed in part II", limit=5, document_id="amzn-10q")

    assert session.last_kwargs["document_id"] == "amzn-10q"
    assert session.last_kwargs["index_limit"] > 5
    assert "n.logical_doc_id = $document_id" in session.last_cypher


def test_fulltext_seed_unscoped_when_no_document_id(service):
    session = _FakeSession([])

    service.fulltext_seed(session, "what is discussed in part II", limit=5)

    assert session.last_kwargs["document_id"] == ""
    assert session.last_kwargs["index_limit"] == 5


# ── graph_expand ──────────────────────────────────────────────────────────


@pytest.mark.parametrize("hops", [1, 2])
def test_graph_expand_scopes_related_nodes_to_document(service, hops):
    session = _FakeSession([])

    service.graph_expand(session, ["seed1"], hops=hops, limit=10, document_id="aapl-10k-2024")

    assert session.last_kwargs["document_id"] == "aapl-10k-2024"
    assert "related.logical_doc_id = $document_id" in session.last_cypher


def test_graph_expand_unscoped_when_no_document_id(service):
    session = _FakeSession([])

    service.graph_expand(session, ["seed1"], hops=1, limit=10)

    assert session.last_kwargs["document_id"] == ""
