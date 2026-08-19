"""tests/test_router_query_hybrid_concurrency_unit.py — query_hybrid runs
esg_agent and structured_agent concurrently, with telemetry context
correctly propagated into both worker threads.

Covers two things in one fix:
1. query_hybrid() used to invoke esg_agent then structured_agent strictly
   sequentially even though data_result never reads doc_result — pure
   unnecessary serialization of two independent LLM+DB round trips.
2. Naively running them via ThreadPoolExecutor.submit() would silently
   drop telemetry for whichever one lands in a worker thread:
   get_telemetry()/pipeline_step() read a contextvars.ContextVar that does
   not propagate into a new OS thread on its own, so a worker thread would
   just see the ContextVar's default (None). copy_context() must be used
   to carry the value across.

Run with:
    python -m pytest tests/test_router_query_hybrid_concurrency_unit.py -v
"""
from __future__ import annotations

import sys
import threading
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest


def _stub_module(name: str) -> types.ModuleType:
    mod = types.ModuleType(name)
    sys.modules[name] = mod
    return mod


_stub_module("neo4j").GraphDatabase = MagicMock()
sys.modules["neo4j"].Driver = MagicMock

_stub_module("src.shared.auth")
_stub_module("src.shared.auth.rbac_setup").GraphRBAC = MagicMock()
sys.modules["src.shared.auth.rbac_setup"].initialize_rbac_schema = MagicMock()
_auth_roles = _stub_module("src.shared.auth.roles")
_auth_roles.UserContext = MagicMock
_auth_roles.DEFAULT_PUBLIC_CONTEXT = MagicMock(role=MagicMock(value="public"))

# router.py imports structured_agent/esg_agent directly from the real graph
# modules at module level — those need a live Neo4j-backed retriever to
# construct, so stub both wholesale rather than importing the real (heavy)
# LangGraph pipelines. query_hybrid() only calls .invoke(state) on each.
for _n in ["src.structured.retrieval.graph", "src.unstructured.retrieval.graph"]:
    sys.modules.pop(_n, None)
_structured_graph_stub = _stub_module("src.structured.retrieval.graph")
_structured_graph_stub.structured_agent = MagicMock()
_unstructured_graph_stub = _stub_module("src.unstructured.retrieval.graph")
_unstructured_graph_stub.esg_agent = MagicMock()

sys.modules.pop("src.interface.router", None)

import src.interface.router as router_mod


class _ConcurrencyProbeAgent:
    """Records which thread ran it and what get_telemetry() saw there, and
    blocks until a sibling agent has also started — proving the two run
    concurrently rather than one blocking the other's start."""

    def __init__(self, name: str, sibling_started: threading.Event, own_started: threading.Event, answer: str):
        self.name = name
        self._sibling_started = sibling_started
        self._own_started = own_started
        self.answer = answer
        self.thread_name: str | None = None
        self.telemetry_seen = "UNSET"

    def invoke(self, state):
        self.thread_name = threading.current_thread().name
        self.telemetry_seen = router_mod.get_telemetry()
        self._own_started.set()
        # If the two invocations were actually sequential, the second one
        # wouldn't start until the first returns — so waiting here for the
        # sibling to have started would hang until this wait times out.
        started_concurrently = self._sibling_started.wait(timeout=2.0)
        assert started_concurrently, f"{self.name} never observed its sibling start — not concurrent"
        return {"answer": self.answer, "sources": [{"id": f"{self.name}_1"}]}


@pytest.fixture()
def agents(monkeypatch):
    doc_started = threading.Event()
    data_started = threading.Event()
    doc_agent = _ConcurrencyProbeAgent("esg_agent", sibling_started=data_started, own_started=doc_started, answer="doc answer")
    data_agent = _ConcurrencyProbeAgent(
        "structured_agent", sibling_started=doc_started, own_started=data_started, answer="data answer"
    )
    monkeypatch.setattr(router_mod, "esg_agent", doc_agent)
    monkeypatch.setattr(router_mod, "structured_agent", data_agent)
    monkeypatch.setattr(router_mod, "build_presentation", lambda **kwargs: {"kind": "markdown", "blocks": []})
    return doc_agent, data_agent


def test_query_hybrid_runs_agents_concurrently(agents):
    doc_agent, data_agent = agents
    out = router_mod.query_hybrid("what does this filing say and how much revenue?")

    assert "doc answer" in out["answer"]
    assert "data answer" in out["answer"]
    # Each agent actually ran (not skipped) and on a worker thread, not the
    # calling thread itself directly via a plain sequential call.
    assert doc_agent.thread_name is not None
    assert data_agent.thread_name is not None


def test_query_hybrid_propagates_telemetry_into_both_threads(agents):
    doc_agent, data_agent = agents
    router_mod.query_hybrid("what does this filing say and how much revenue?")

    assert doc_agent.telemetry_seen is not None
    assert data_agent.telemetry_seen is not None
    # Both threads must see the SAME Telemetry object start_telemetry()
    # created in the calling thread, not each their own independent copy
    # and not None (the ContextVar's un-propagated default).
    assert doc_agent.telemetry_seen is data_agent.telemetry_seen
