"""
A named retrieval mode must decide which source answers -- with no crossover.

Structured and document retrieval each fall back to the other in several
places: a weak answer, zero rows, an RBAC denial. Each has a reasonable local
justification and none is visible from outside, so a question asked of the
business data came back as "This document does not cover it" -- naming the
wrong corpus, and giving no sign the selected source had been abandoned.
Guarding the fallback sites one at a time kept missing one, so the boundary
is enforced in a single place and pinned here.
"""
from src.interface.routing import enforce_mode


def _result(agent: str) -> dict:
    return {"agent": agent, "answer": "This document does not cover it.", "sources": [{"id": "s"}]}


def test_structured_request_never_returns_a_document_answer():
    out = enforce_mode(_result("unstructured"), "query_data")
    assert out["agent"] == "structured"
    assert "document" not in out["answer"].lower() or "switch to" in out["answer"].lower()
    assert out["sources"] == []


def test_document_request_never_returns_a_structured_answer():
    out = enforce_mode(_result("structured"), "search_documents")
    assert out["agent"] == "unstructured"
    assert out["sources"] == []


def test_matching_agent_passes_through_untouched():
    original = _result("structured")
    assert enforce_mode(original, "query_data") is original


def test_no_mode_named_leaves_routing_alone():
    """Defensive only: ask() cannot reach this.

    resolve_mode_override always returns a concrete tool -- there is no auto
    mode -- so in practice the boundary is enforced on every query. This
    pins the guard's behaviour if a caller ever passes None directly, rather
    than describing a routing path that exists.
    """
    original = _result("unstructured")
    assert enforce_mode(original, None) is original


def test_replacement_says_which_source_was_searched():
    """The reply has to explain itself, or it is just a different wrong answer."""
    out = enforce_mode(_result("unstructured"), "query_data")
    assert "structured" in out["answer"].lower()
    assert out["low_confidence"] is True
