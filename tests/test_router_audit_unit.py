"""
tests/test_router_audit_unit.py — router.ask() records audit events.

Covers both branches: a successful query records a QUERY event with
route_tool/agent/low_confidence metadata, and an RBAC-denied structured
question records an ACCESS_DENIED event instead of silently returning.

Run with:
    python -m pytest tests/test_router_audit_unit.py -v
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
sys.modules["src.shared.auth.roles"].DEFAULT_PUBLIC_CONTEXT = MagicMock(
    user_id="public_001", tenant_id="default", role=MagicMock(value="public")
)

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


def _fake_ctx():
    ctx = MagicMock()
    ctx.user_id = "user_a"
    ctx.tenant_id = "tenant_a"
    ctx.role.value = "regular_office"
    return ctx


def test_ask_records_query_event_on_success(monkeypatch):
    recorded = {}
    monkeypatch.setattr(
        router_mod, "record_audit_event", lambda **kw: recorded.update(kw)
    )
    monkeypatch.setattr(router_mod, "resolve_query_tool", lambda q, t: ("search_documents", {}))
    monkeypatch.setattr(router_mod, "get_telemetry", lambda: None)
    monkeypatch.setattr(router_mod, "start_telemetry", lambda: None)
    monkeypatch.setattr(router_mod, "clear_telemetry", lambda: None)
    monkeypatch.setattr(router_mod, "save_turn", lambda *a, **k: None)

    fake_result = {
        "answer": "the answer",
        "agent": "unstructured",
        "strategy": "graph_rag",
        "low_confidence": True,
        "_route_tool": "search_documents",
    }
    monkeypatch.setattr(router_mod, "run_via_mcp_tool", lambda *a, **k: fake_result)

    ctx = _fake_ctx()
    out = router_mod.ask("What is the policy?", user_context=ctx, request_id="req1")

    assert out is fake_result
    assert recorded["event_type"] == router_mod.AuditEventType.QUERY
    assert recorded["user_id"] == "user_a"
    assert recorded["tenant_id"] == "tenant_a"
    assert recorded["role"] == "regular_office"
    assert recorded["request_id"] == "req1"
    assert recorded["result"] == "success"
    assert recorded["metadata"]["route_tool"] == "search_documents"
    assert recorded["metadata"]["low_confidence"] is True


def test_ask_records_access_denied_event_on_rbac_denial(monkeypatch):
    recorded = {}
    monkeypatch.setattr(
        router_mod, "record_audit_event", lambda **kw: recorded.update(kw)
    )
    monkeypatch.setattr(router_mod, "resolve_query_tool", lambda q, t: ("query_data", {}))
    monkeypatch.setattr(router_mod, "is_structured_data_question", lambda q: True)
    monkeypatch.setattr(router_mod, "get_telemetry", lambda: None)
    monkeypatch.setattr(router_mod, "start_telemetry", lambda: None)
    monkeypatch.setattr(router_mod, "clear_telemetry", lambda: None)
    monkeypatch.setattr(router_mod, "save_turn", lambda *a, **k: None)
    monkeypatch.setattr(
        router_mod, "make_structured_access_denied_result", lambda q, ctx: {"answer": "denied"}
    )

    fake_rbac = MagicMock()
    fake_rbac.can_query_knowledge_area.return_value = False
    monkeypatch.setattr(router_mod, "_rbac_check", lambda: fake_rbac)

    ctx = _fake_ctx()
    out = router_mod.ask("How many products do we have?", user_context=ctx, request_id="req2")

    assert out == {"answer": "denied"}
    assert recorded["event_type"] == router_mod.AuditEventType.ACCESS_DENIED
    assert recorded["user_id"] == "user_a"
    assert recorded["tenant_id"] == "tenant_a"
    assert recorded["result"] == "denied"
    assert recorded["reason"] == "rbac_denied"
    assert recorded["resource"] == "structured"
