"""
tests/test_router_query_data_confidence_unit.py — router.query_data() forwards
low_confidence/confidence_note from the structured_agent's LangGraph output.

Regression guard for a real bug: query_data() built its `out` dict without
ever copying `low_confidence`/`confidence_note` from `structured_agent.invoke()`'s
result — computed, then silently dropped, every time.

Run with:
    python -m pytest tests/test_router_query_data_confidence_unit.py -v
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
# graph modules at module level — those need a live Neo4j-backed retriever
# to construct. query_data() only calls structured_agent.invoke(state), so
# stub both graph modules wholesale rather than importing the real (heavy)
# LangGraph pipelines.
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


def test_query_data_forwards_low_confidence_and_note(monkeypatch):
    fake_result = {
        "answer": "42 customers.",
        "sources": [{"id": "row_0"}],
        "strategy": "text2cypher",
        "low_confidence": True,
        "confidence_note": "Question asks for a count, but the query has no COUNT(...)/size(...).",
    }
    router_mod.structured_agent.invoke = MagicMock(return_value=fake_result)
    monkeypatch.setattr(router_mod, "build_presentation", lambda **kwargs: None)
    monkeypatch.setattr(router_mod, "get_turn", lambda thread_id: None)
    monkeypatch.setattr(router_mod, "save_turn", lambda *a, **k: None)
    monkeypatch.setattr(router_mod, "resolve_follow_up", lambda question, prior: {"question": question})
    monkeypatch.setattr(router_mod, "get_telemetry", lambda: None)

    out = router_mod.query_data("how many customers do we have")

    assert out["low_confidence"] is True
    assert out["confidence_note"] == fake_result["confidence_note"]


def test_query_data_defaults_when_result_has_no_confidence_keys(monkeypatch):
    fake_result = {"answer": "ok", "sources": [], "strategy": "text2cypher"}
    router_mod.structured_agent.invoke = MagicMock(return_value=fake_result)
    monkeypatch.setattr(router_mod, "build_presentation", lambda **kwargs: None)
    monkeypatch.setattr(router_mod, "get_turn", lambda thread_id: None)
    monkeypatch.setattr(router_mod, "save_turn", lambda *a, **k: None)
    monkeypatch.setattr(router_mod, "resolve_follow_up", lambda question, prior: {"question": question})
    monkeypatch.setattr(router_mod, "get_telemetry", lambda: None)

    out = router_mod.query_data("anything")

    assert out["low_confidence"] is False
    assert out["confidence_note"] is None
