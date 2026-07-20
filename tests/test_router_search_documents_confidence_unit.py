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

_root = Path(__file__).resolve().parents[1]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))


def _stub_module(name: str) -> types.ModuleType:
    mod = types.ModuleType(name)
    sys.modules[name] = mod
    return mod


if "neo4j" not in sys.modules:
    _stub_module("neo4j")
sys.modules["neo4j"].GraphDatabase = MagicMock()
sys.modules["neo4j"].Driver = MagicMock

for _n in ["src.auth", "src.auth.rbac_setup", "src.auth.roles"]:
    if _n not in sys.modules:
        _stub_module(_n)
sys.modules["src.auth.rbac_setup"].GraphRBAC = MagicMock()
sys.modules["src.auth.rbac_setup"].initialize_rbac_schema = MagicMock()
sys.modules["src.auth.roles"].UserContext = MagicMock
sys.modules["src.auth.roles"].DEFAULT_PUBLIC_CONTEXT = MagicMock(role=MagicMock(value="public"))

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
