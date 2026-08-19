"""
tests/test_router_search_documents_confidence_unit.py — router.search_documents()
forwards low_confidence/confidence_note from the esg_agent's LangGraph output.

Mirrors test_router_query_data_confidence_unit.py: search_documents() previously
built its `out` dict with no low_confidence/confidence_note keys at all (nothing
to forward, since the document path never computed them either).

Run with:
    python -m pytest tests/test_router_search_documents_confidence_unit.py -v
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
sys.modules["neo4j"].Driver = MagicMock

for _n in ["src.shared.auth", "src.shared.auth.rbac_setup", "src.shared.auth.roles"]:
    if _n not in sys.modules:
        _stub_module(_n)
sys.modules["src.shared.auth.rbac_setup"].GraphRBAC = MagicMock()
sys.modules["src.shared.auth.rbac_setup"].initialize_rbac_schema = MagicMock()
sys.modules["src.shared.auth.roles"].UserContext = MagicMock
sys.modules["src.shared.auth.roles"].DEFAULT_PUBLIC_CONTEXT = MagicMock(role=MagicMock(value="public"))

# router.py imports `structured_agent`/`esg_agent` directly from the real
# graph modules at module level — stub both wholesale rather than importing
# the real (heavy) LangGraph pipelines (same reasoning as the query_data test).
for _n in ["src.retrieval.structured.graph", "src.retrieval.unstructured.graph"]:
    if _n in sys.modules:
        del sys.modules[_n]
_structured_graph_stub = _stub_module("src.retrieval.structured.graph")
_structured_graph_stub.structured_agent = MagicMock()
_unstructured_graph_stub = _stub_module("src.retrieval.unstructured.graph")
_unstructured_graph_stub.esg_agent = MagicMock()

if "src.router" in sys.modules:
    del sys.modules["src.router"]

import src.router as router_mod


def test_search_documents_forwards_low_confidence_and_note(monkeypatch):
    fake_result = {
        "answer": "I could not find relevant information in the ingested documents.",
        "sources": [],
        "keywords": [],
        "query_type": "graph_rag",
        "low_confidence": True,
        "confidence_note": "No relevant passages were found in the ingested documents for this question.",
    }
    router_mod.esg_agent.invoke = MagicMock(return_value=fake_result)
    monkeypatch.setattr(router_mod, "build_presentation", lambda **kwargs: None)
    monkeypatch.setattr(router_mod, "get_turn", lambda thread_id: None)
    monkeypatch.setattr(router_mod, "save_turn", lambda *a, **k: None)
    monkeypatch.setattr(router_mod, "resolve_follow_up", lambda question, prior: {"question": question})
    monkeypatch.setattr(router_mod, "get_telemetry", lambda: None)

    out = router_mod.search_documents("What does the manual say about a topic it never covers?")

    assert out["low_confidence"] is True
    assert out["confidence_note"] == fake_result["confidence_note"]


def test_search_documents_defaults_when_result_has_no_confidence_keys(monkeypatch):
    fake_result = {"answer": "ok", "sources": [], "keywords": [], "query_type": "graph_rag"}
    router_mod.esg_agent.invoke = MagicMock(return_value=fake_result)
    monkeypatch.setattr(router_mod, "build_presentation", lambda **kwargs: None)
    monkeypatch.setattr(router_mod, "get_turn", lambda thread_id: None)
    monkeypatch.setattr(router_mod, "save_turn", lambda *a, **k: None)
    monkeypatch.setattr(router_mod, "resolve_follow_up", lambda question, prior: {"question": question})
    monkeypatch.setattr(router_mod, "get_telemetry", lambda: None)

    out = router_mod.search_documents("anything")

    assert out["low_confidence"] is False
    assert out["confidence_note"] is None


# ── structured-misroute autofix attribution ──────────────────────────────────
#
# Regression: a structured-shaped question that reached the document agent
# (default retrieval mode, or a document-RBAC denial) and got silently
# re-answered by the structured agent via document_agent_structured_guard/
# run_structured_autofix used to still be reported as "agent": "unstructured"
# / route_tool "search_documents" here -- this function never checked the
# esg_agent result for the "_autofix_agent" marker that flags the redirect,
# so callers (and eval suites asserting route_tool=="query_data") had no way
# to tell a real, RBAC-correct structured answer from a genuine document
# search. Verified live: user regular_001 (has structured access, lacks
# document access) asking "Which supplier provides Chai?" got the correct
# answer's content but agent/route_tool still said unstructured/search_documents.


def test_search_documents_reports_structured_agent_when_autofixed(monkeypatch):
    fake_result = {
        "answer": "The supplier that provides Chai is Exotic Liquids.",
        "sources": [{"id": "row_0", "title": "Exotic Liquids"}],
        "keywords": [],
        "query_type": "graph_rag",
        "low_confidence": False,
        "strategy": "structured",
        "_autofix_agent": "structured",
    }
    router_mod.esg_agent.invoke = MagicMock(return_value=fake_result)
    captured_presentation_kwargs = {}

    def _fake_build_presentation(**kwargs):
        captured_presentation_kwargs.update(kwargs)
        return None

    monkeypatch.setattr(router_mod, "build_presentation", _fake_build_presentation)
    monkeypatch.setattr(router_mod, "get_turn", lambda thread_id: None)
    monkeypatch.setattr(router_mod, "save_turn", lambda *a, **k: None)
    monkeypatch.setattr(router_mod, "resolve_follow_up", lambda question, prior: {"question": question})
    monkeypatch.setattr(router_mod, "get_telemetry", lambda: None)

    out = router_mod.search_documents("Which supplier provides Chai?")

    assert out["agent"] == "structured"
    assert out["_autofix_agent"] == "structured"
    assert out["strategy"] == "structured"
    assert captured_presentation_kwargs["agent"] == "structured"


def test_search_documents_stays_unstructured_when_not_autofixed(monkeypatch):
    fake_result = {
        "answer": "The compliance policy requires reporting concerns.",
        "sources": [{"id": "c1", "title": "Policy"}],
        "keywords": [],
        "query_type": "graph_rag",
        "low_confidence": False,
    }
    router_mod.esg_agent.invoke = MagicMock(return_value=fake_result)
    monkeypatch.setattr(router_mod, "build_presentation", lambda **kwargs: None)
    monkeypatch.setattr(router_mod, "get_turn", lambda thread_id: None)
    monkeypatch.setattr(router_mod, "save_turn", lambda *a, **k: None)
    monkeypatch.setattr(router_mod, "resolve_follow_up", lambda question, prior: {"question": question})
    monkeypatch.setattr(router_mod, "get_telemetry", lambda: None)

    out = router_mod.search_documents("What does the compliance policy say?")

    assert out["agent"] == "unstructured"
    assert "_autofix_agent" not in out
    assert out["strategy"] == "graph_rag"
