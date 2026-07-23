"""tests/test_document_resolver_hint_unit.py — resolve_document_for_query's
document_id_hint priority ordering.

Covers conversation continuity: "what's on page 6 of this document" (no
document name, no distinguishing vocabulary) should stay on the document
the thread was already discussing, instead of falling through to
vector-majority resolution's "biggest document wins" behavior. Priority:
explicit name (strict) > conversation hint > vector-majority > generic-term
> largest-document fallback — an explicitly named *different* document
must always override the hint (that's the real topic-switch signal).

These test the priority logic directly (stubbing the resolver's own
sub-methods) rather than faking the full Neo4j session chain — the
individual sub-methods' Cypher behavior is covered elsewhere
(test_document_scoping_unit.py) and live verification.

Run with:
    python -m pytest tests/test_document_resolver_hint_unit.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

_root = Path(__file__).resolve().parents[1]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from src.retrieval.unstructured.services.document_resolver import DocumentResolver
from src.retrieval.unstructured.services.graph_seeds import GraphSeedService
from src.retrieval.unstructured.services.ranking import RankingService


@pytest.fixture()
def resolver() -> DocumentResolver:
    return DocumentResolver(GraphSeedService(RankingService()))


def test_hint_used_when_strict_and_nothing_else_stronger(resolver, monkeypatch):
    monkeypatch.setattr(
        resolver, "resolve_document_for_query_strict", lambda *a, **kw: (None, None)
    )
    monkeypatch.setattr(
        resolver, "_validate_document_id", lambda session, doc_id, tenant_id="": "STRATEC Policy"
    )
    resolver.resolve_document_by_vector = MagicMock(side_effect=AssertionError("must not reach vector"))

    doc_id, title = resolver.resolve_document_for_query(
        session=None, query="What is discussed on page 6 of this document?",
        tenant_id="default", document_id_hint="stratec-compliance-policy-2025",
    )
    assert doc_id == "stratec-compliance-policy-2025"
    assert title == "STRATEC Policy"


def test_explicit_name_overrides_hint():
    resolver = DocumentResolver(GraphSeedService(RankingService()))
    resolver.resolve_document_for_query_strict = MagicMock(return_value=("jpm-10k-2017-02-28", "JPM 10-K"))
    resolver._validate_document_id = MagicMock(side_effect=AssertionError("must not consult hint"))
    resolver.resolve_document_by_vector = MagicMock(side_effect=AssertionError("must not reach vector"))

    doc_id, title = resolver.resolve_document_for_query(
        session=None, query="What does the JPM filing say about risk?",
        tenant_id="default", document_id_hint="stratec-compliance-policy-2025",
    )
    assert doc_id == "jpm-10k-2017-02-28"
    assert title == "JPM 10-K"


def test_invalid_hint_falls_through_to_vector():
    """The hinted document may no longer exist (deleted/expired since the
    prior turn) — must not be trusted blindly."""
    resolver = DocumentResolver(GraphSeedService(RankingService()))
    resolver.resolve_document_for_query_strict = MagicMock(return_value=(None, None))
    resolver._validate_document_id = MagicMock(return_value=None)  # hint doesn't exist
    resolver.document_match_terms = MagicMock(return_value=[])  # no term signal -> pure vector fallback
    resolver.resolve_document_by_vector = MagicMock(return_value=("aapl-10k-2024", "AAPL 10-K"))

    doc_id, title = resolver.resolve_document_for_query(
        session=None, query="some ambiguous question",
        tenant_id="default", document_id_hint="deleted-doc-id",
    )
    assert doc_id == "aapl-10k-2024"


def test_no_hint_behaves_as_before():
    resolver = DocumentResolver(GraphSeedService(RankingService()))
    resolver.resolve_document_for_query_strict = MagicMock(return_value=(None, None))
    resolver._validate_document_id = MagicMock(side_effect=AssertionError("no hint given, must not be called"))
    resolver.document_match_terms = MagicMock(return_value=[])  # no term signal -> pure vector fallback
    resolver.resolve_document_by_vector = MagicMock(return_value=("amzn-10q-2016-07-29", "AMZN 10-Q"))

    doc_id, title = resolver.resolve_document_for_query(
        session=None, query="some ambiguous question", tenant_id="default",
    )
    assert doc_id == "amzn-10q-2016-07-29"
