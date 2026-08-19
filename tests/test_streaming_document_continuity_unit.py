"""
tests/test_streaming_document_continuity_unit.py — streaming path carries
document_id/document_title into the saved turn snapshot.

Regression: conversation continuity (document_id_hint, see
test_document_resolver_hint_unit.py / test_thread_memory_document_continuity_unit.py)
was built and verified against router.py's non-streaming /query path, but the
real chat UI uses /query/stream (src/streaming/query_stream.py), a separate
orchestrator. Its iter_document_stream() "done" events never carried
document_id/document_title, and _enrich_and_persist()'s `out` dict (what
actually gets saved via save_turn) never included retrieved_context at all --
so extract_critical_from_result()'s `result.get("retrieved_context")` was
always `{}` for any turn that went through streaming, silently breaking
continuity end-to-end despite the strategy-level fix being correct. Verified
live: a 4-turn browser conversation on Go.Data's annual report lost the
document on turn 4, resolving to an unrelated JPM filing instead.

Run with:
    python -m pytest tests/test_streaming_document_continuity_unit.py -v
"""
from __future__ import annotations

import json
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

for _n in ["src.retrieval.structured.graph", "src.unstructured.retrieval.graph"]:
    if _n in sys.modules:
        del sys.modules[_n]
_structured_graph_stub = _stub_module("src.retrieval.structured.graph")
_structured_graph_stub.structured_agent = MagicMock()
_structured_graph_stub._build_fast_structured_answer = MagicMock()
_structured_graph_stub._should_fast_structured_answer = MagicMock()
_structured_graph_stub.retrieve_node = MagicMock()
_unstructured_graph_stub = _stub_module("src.unstructured.retrieval.graph")
_unstructured_graph_stub.esg_agent = MagicMock()
_unstructured_graph_stub._STRUCTURAL_FAST_MODES = frozenset({"structural_toc"})
_unstructured_graph_stub._build_fast_unstructured_answer = MagicMock(return_value="Fast answer.")
_unstructured_graph_stub._fix_misrouted_structured_answer = lambda answer, question: answer
_unstructured_graph_stub.retrieve_node = MagicMock()

# An earlier-collected test file (test_scalable_pipeline_unit.py) stubs
# src.unstructured.document (and its submodules, including page_vision) wholesale as
# empty modules and never undoes it -- that stub lingers in sys.modules for
# the rest of the pytest process. Force a fresh, real import of the whole
# src.unstructured.document tree here rather than picking up that stale stub (same fix
# pattern as the src.shared.auth/src.unstructured.document blocks in that file itself: never
# trust another test's mutation of a shared module).
for _mod_name in list(sys.modules):
    if _mod_name == "src.unstructured.document" or _mod_name.startswith("src.unstructured.document."):
        del sys.modules[_mod_name]

import src.unstructured.streaming as document_stream_mod
import src.streaming.query_stream as query_stream_mod


def _last_done_event(lines: list[str]) -> dict:
    events = [json.loads(line) for line in lines if json.loads(line).get("type") == "done"]
    assert events, "stream produced no 'done' event"
    return events[-1]


def test_structural_fast_mode_done_event_carries_document_id(monkeypatch):
    retrieved = {
        "chunks": [{"id": "c1", "title": "TOC", "text": "..."}],
        "mode": "structural_toc",
        "document_id": "godata-annual-report-2023",
        "document_title": "Go.Data Annual Report",
    }
    monkeypatch.setattr(
        document_stream_mod, "retrieve_node", lambda state: {"retrieved_context": retrieved}
    )
    monkeypatch.setattr(document_stream_mod, "build_presentation", lambda **kwargs: None)
    monkeypatch.setattr(document_stream_mod, "document_agent_structured_guard", lambda *a, **k: None)

    lines = list(
        document_stream_mod.iter_document_stream(
            "what is the table of contents", user_context=None, resolved_question="what is the table of contents"
        )
    )
    done = _last_done_event(lines)

    assert done["document_id"] == "godata-annual-report-2023"
    assert done["document_title"] == "Go.Data Annual Report"


def test_llm_synthesis_done_event_carries_document_id(monkeypatch):
    retrieved = {
        "chunks": [{"id": "c1", "title": "Intro", "text": "Some content."}],
        "mode": "graph_rag",
        "document_id": "godata-annual-report-2023",
        "document_title": "Go.Data Annual Report",
    }
    monkeypatch.setattr(
        document_stream_mod, "retrieve_node", lambda state: {"retrieved_context": retrieved}
    )
    monkeypatch.setattr(document_stream_mod, "build_presentation", lambda **kwargs: None)
    monkeypatch.setattr(document_stream_mod, "document_agent_structured_guard", lambda *a, **k: None)
    monkeypatch.setattr(
        document_stream_mod, "compute_confidence", lambda *a, **k: (False, None)
    )
    fake_provider = MagicMock()
    fake_provider.chat_completion_stream = lambda **kwargs: iter(["Here ", "you go."])
    monkeypatch.setattr(document_stream_mod, "get_chat_provider", lambda: fake_provider)

    lines = list(
        document_stream_mod.iter_document_stream(
            "what does the introduction discuss",
            user_context=None,
            resolved_question="what does the introduction discuss",
        )
    )
    done = _last_done_event(lines)

    assert done["document_id"] == "godata-annual-report-2023"
    assert done["document_title"] == "Go.Data Annual Report"


def test_enrich_and_persist_saves_document_id_under_retrieved_context(monkeypatch):
    saved: dict = {}
    monkeypatch.setattr(query_stream_mod, "save_turn", lambda thread_id, question, out: saved.update(out))
    monkeypatch.setattr(query_stream_mod, "get_telemetry", lambda: None)
    monkeypatch.setattr(query_stream_mod, "clear_telemetry", lambda: None)

    final_payload = {
        "answer": "Here you go.",
        "sources": [],
        "agent": "unstructured",
        "strategy": "graph_rag",
        "query_type": "graph_rag",
        "document_id": "godata-annual-report-2023",
        "document_title": "Go.Data Annual Report",
    }

    query_stream_mod._enrich_and_persist(
        tool_name="search_documents",
        question="what does the introduction discuss",
        thread_id="thread-1",
        ctx=MagicMock(role=MagicMock(value="admin")),
        resolved={"use_prior": False},
        final=final_payload,
        request_id=None,
    )

    assert saved["retrieved_context"]["document_id"] == "godata-annual-report-2023"
    assert saved["retrieved_context"]["document_title"] == "Go.Data Annual Report"


def test_enrich_and_persist_handles_missing_document_id(monkeypatch):
    saved: dict = {}
    monkeypatch.setattr(query_stream_mod, "save_turn", lambda thread_id, question, out: saved.update(out))
    monkeypatch.setattr(query_stream_mod, "get_telemetry", lambda: None)
    monkeypatch.setattr(query_stream_mod, "clear_telemetry", lambda: None)

    final_payload = {"answer": "ok", "sources": [], "agent": "structured", "strategy": "text2cypher"}

    query_stream_mod._enrich_and_persist(
        tool_name="query_data",
        question="how many customers",
        thread_id="thread-1",
        ctx=MagicMock(role=MagicMock(value="admin")),
        resolved={"use_prior": False},
        final=final_payload,
        request_id=None,
    )

    assert saved["retrieved_context"]["document_id"] is None
    assert saved["retrieved_context"]["document_title"] is None
