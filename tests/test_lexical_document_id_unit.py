"""tests/test_lexical_document_id_unit.py — LexicalService's document_id
skip-re-resolution contract.

Covers the fix where structural_phrase_retrieve/structural_keyword_retrieve
each independently re-resolved the query's document via DocumentResolver,
even though FullHybridStrategy (their only real caller) had already
resolved it once — two fully redundant Neo4j round trips per query. Passing
document_id now skips that internal resolution; leaving it unset (None)
preserves the old resolve-internally behavior for any other caller.

Run with:
    python -m pytest tests/test_lexical_document_id_unit.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_root = Path(__file__).resolve().parents[1]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

# A previously-collected test file (e.g. test_ingestion_manager_di_unit.py)
# may have stubbed src.graph.constants/src.graph.driver/src.auth* with bare
# fakes to avoid needing a real Neo4j/RBAC setup for its own tests — this
# file needs the REAL src.graph.constants (lexical.py imports
# DOCUMENT_ROOT_CYPHER from it). Only clear entries that are actually fake
# stubs (a hand-built types.ModuleType has neither __file__ nor __path__);
# mirrors test_rbac_setup_unit.py's identical guard.
for _mod_name in list(sys.modules):
    if _mod_name.startswith("src.auth") or _mod_name.startswith("src.graph"):
        _mod = sys.modules[_mod_name]
        if getattr(_mod, "__file__", None) is None and getattr(_mod, "__path__", None) is None:
            del sys.modules[_mod_name]

from src.retrieval.unstructured.services.lexical import LexicalService
from src.retrieval.unstructured.services.ranking import RankingService


class _FakeSession:
    def __init__(self):
        self.calls: list[tuple[str, dict]] = []

    def run(self, cypher, **kwargs):
        self.calls.append((cypher, kwargs))
        return []


class _FakeDocumentResolver:
    def __init__(self):
        self.resolve_calls = 0

    def resolve_document_for_query(self, session, query, tenant_id=""):
        self.resolve_calls += 1
        return "resolved-doc-id", "Resolved Doc Title"


@pytest.fixture()
def resolver() -> _FakeDocumentResolver:
    return _FakeDocumentResolver()


@pytest.fixture()
def lexical(resolver) -> LexicalService:
    return LexicalService(RankingService(), resolver)


_QUERY = "What dividends per share did Apple declare in fiscal year 2024?"


# ── structural_keyword_retrieve ──────────────────────────────────────────


def test_keyword_retrieve_resolves_internally_when_document_id_unset(lexical, resolver):
    session = _FakeSession()
    lexical.structural_keyword_retrieve(session, _QUERY, tenant_id="default")
    assert resolver.resolve_calls == 1


def test_keyword_retrieve_skips_resolution_when_document_id_given(lexical, resolver):
    session = _FakeSession()
    lexical.structural_keyword_retrieve(
        session, _QUERY, tenant_id="default", document_id="aapl-10k-2024"
    )
    assert resolver.resolve_calls == 0
    # doc_id flows through to the Cypher params of at least one query.
    assert any(kwargs.get("doc_id") == "aapl-10k-2024" for _cypher, kwargs in session.calls)


def test_keyword_retrieve_normalizes_empty_document_id_to_none(lexical, resolver):
    """document_id="" means the caller already resolved and found nothing —
    must become None, not the literal empty string, or _doc_scope_cypher's
    `$doc_id IS NULL` unscoped branch never fires and every document is
    excluded instead of every document being eligible."""
    session = _FakeSession()
    lexical.structural_keyword_retrieve(session, _QUERY, tenant_id="default", document_id="")
    assert resolver.resolve_calls == 0
    assert any(kwargs.get("doc_id") is None for _cypher, kwargs in session.calls)


# ── structural_phrase_retrieve ───────────────────────────────────────────


def test_phrase_retrieve_resolves_internally_when_document_id_unset(lexical, resolver):
    session = _FakeSession()
    lexical.structural_phrase_retrieve(session, _QUERY, tenant_id="default")
    assert resolver.resolve_calls == 1


def test_phrase_retrieve_skips_resolution_when_document_id_given(lexical, resolver):
    session = _FakeSession()
    lexical.structural_phrase_retrieve(
        session, _QUERY, tenant_id="default", document_id="aapl-10k-2024"
    )
    assert resolver.resolve_calls == 0
    assert any(kwargs.get("doc_id") == "aapl-10k-2024" for _cypher, kwargs in session.calls)


def test_phrase_retrieve_normalizes_empty_document_id_to_none(lexical, resolver):
    session = _FakeSession()
    lexical.structural_phrase_retrieve(session, _QUERY, tenant_id="default", document_id="")
    assert resolver.resolve_calls == 0
    assert any(kwargs.get("doc_id") is None for _cypher, kwargs in session.calls)
