"""
tests/test_retrieval_mode_override_unit.py — explicit retrieval_mode override.

Guards the API/UI "Retrieval" selector that lets a caller force the
structured or unstructured agent instead of the LLM router. This closes a
real cross-document bug: a bare follow-up ("book value per common share")
that flipped from the structured to the unstructured agent mid-conversation
re-resolved the document from scratch and answered from the wrong company's
10-K. An explicit mode makes the two paths fully separable.

`resolve_mode_override` maps the mode string to a forced MCP tool (or None
for auto). `resolve_query_tool(..., forced_tool=...)` must then honor that
tool outright while STILL running follow-up resolution (so document
continuity — document_id, focus_section_id — carries forward).

Run with:
    python -m pytest tests/test_retrieval_mode_override_unit.py -v
"""
from __future__ import annotations

import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest


def _stub_module(name: str) -> types.ModuleType:
    mod = types.ModuleType(name)
    sys.modules[name] = mod
    return mod


if "neo4j" not in sys.modules:
    _stub_module("neo4j")
sys.modules["neo4j"].GraphDatabase = MagicMock()

for _n in ["src.auth", "src.auth.roles"]:
    if _n not in sys.modules:
        _stub_module(_n)
sys.modules["src.auth.roles"].UserContext = MagicMock

if "src.routing" in sys.modules:
    del sys.modules["src.routing"]

from src.routing import DEFAULT_RETRIEVAL_MODE, MODE_TO_TOOL, resolve_mode_override

_DEFAULT_TOOL = MODE_TO_TOOL[DEFAULT_RETRIEVAL_MODE]


@pytest.mark.parametrize(
    "mode,expected_tool",
    [
        ("structured", "query_data"),
        ("unstructured", "search_documents"),
        ("hybrid", "query_hybrid"),
        # Case / whitespace tolerant (UI values arrive verbatim).
        ("  Structured  ", "query_data"),
        ("UNSTRUCTURED", "search_documents"),
        # Auto is gone: legacy/unknown/empty values fall back to the default
        # mode (documents only) rather than invoking the LLM router.
        ("auto", _DEFAULT_TOOL),
        ("", _DEFAULT_TOOL),
        (None, _DEFAULT_TOOL),
        ("nonsense", _DEFAULT_TOOL),
    ],
)
def test_resolve_mode_override(mode, expected_tool):
    assert resolve_mode_override(mode) == expected_tool


def test_default_mode_is_documents_only():
    assert DEFAULT_RETRIEVAL_MODE == "unstructured"
    assert resolve_mode_override("auto") == "search_documents"


def test_mode_to_tool_covers_all_real_agents():
    assert set(MODE_TO_TOOL.values()) == {
        "query_data",
        "search_documents",
        "query_hybrid",
    }


def test_forced_tool_wins_over_baseline_and_follow_up(monkeypatch):
    """resolve_query_tool must honor forced_tool and skip baseline routing."""
    import src.feedback_loop.resolver as resolver

    # A prior turn that, on its own, would resolve as a document follow-up.
    monkeypatch.setattr(resolver, "get_turn", lambda _tid: {"document_id": "doc_gs"})
    monkeypatch.setattr(
        resolver,
        "resolve_follow_up",
        lambda _q, _p: {
            "question": _q,
            "use_prior": True,
            "follow_up_kind": "subsection_detail",  # would force search_documents
            "document_id": "doc_gs",
        },
    )

    # If baseline routing is consulted, the test fails loudly.
    def _boom(*_a, **_k):
        raise AssertionError("baseline routing must be skipped when forced")

    monkeypatch.setattr(resolver, "select_mcp_tool", _boom)

    tool, resolved = resolver.resolve_query_tool(
        "book value per common share", "t1", forced_tool="query_data"
    )
    assert tool == "query_data"
    # Continuity still flows through: the document id is preserved.
    assert resolved.get("document_id") == "doc_gs"


def test_auto_still_uses_baseline(monkeypatch):
    """Without forced_tool, baseline routing drives the decision."""
    import src.feedback_loop.resolver as resolver

    monkeypatch.setattr(resolver, "get_turn", lambda _tid: None)
    monkeypatch.setattr(resolver, "select_mcp_tool", lambda _q: "search_documents")
    monkeypatch.setattr(
        resolver,
        "get_feedback_routing",
        lambda: types.SimpleNamespace(route_tool=lambda _q, baseline: baseline),
    )

    tool, _resolved = resolver.resolve_query_tool("anything", "t2")
    assert tool == "search_documents"
