"""
tests/test_streaming_structured_confidence_unit.py — iter_structured_stream
verification wiring (the streaming duplicate of graph.py's logic).

Proves the final "done" NDJSON event carries low_confidence/confidence_note
for the error, fast-path, and LLM-synthesis branches — none of which emitted
either field before this change.

Run with:
    python -m pytest tests/test_streaming_structured_confidence_unit.py -v
"""
from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# Repo root located by searching upward for src/, not by counting parents:
# a fixed index silently points at the wrong directory the moment this
# file changes nesting depth.
_root = next(p for p in Path(__file__).resolve().parents if (p / "src").is_dir())


def _stub_module(name: str) -> types.ModuleType:
    mod = types.ModuleType(name)
    sys.modules[name] = mod
    return mod


for _n in ["langgraph", "langgraph.graph"]:
    if _n not in sys.modules:
        _stub_module(_n)
sys.modules["langgraph.graph"].StateGraph = MagicMock()
sys.modules["langgraph.graph"].END = MagicMock()

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

for _mod_name in (
    "src.structured.retrieval.graph",
    "src.structured.retrieval.retriever",
    "src.structured.streaming",
):
    if _mod_name in sys.modules:
        del sys.modules[_mod_name]
_retriever_stub = _stub_module("src.structured.retrieval.retriever")
_retriever_stub.StructuredRetriever = lambda *a, **k: MagicMock()

# src.interface.streaming.__init__ imports query_stream.py, which pulls in router.py,
# routing.py, feedback_loop, conversation — none of which iter_structured_stream
# needs. Pre-populate a stub package (real __path__, so submodule lookup on
# disk still works) instead of letting Python run the real __init__.py.
if "src.interface.streaming" not in sys.modules:
    _streaming_pkg = types.ModuleType("src.interface.streaming")
    _streaming_pkg.__path__ = [str(_root / "src" / "interface" / "streaming")]
    _streaming_pkg.__package__ = "src.interface.streaming"
    sys.modules["src.interface.streaming"] = _streaming_pkg

from src.structured.streaming import iter_structured_stream
import src.structured.streaming as structured_stream_mod


def _chunks(cypher: str, rows: list[dict]) -> list[dict]:
    return [
        {
            "id": f"row_{i}",
            "title": f"Row {i}",
            "text": "\n".join(f"{k}: {v}" for k, v in row.items()),
            "raw": row,
            "score": 1.0,
            "cypher": cypher,
            "related": [],
        }
        for i, row in enumerate(rows)
    ]


def _last_done_event(lines: list[str]) -> dict:
    done_events = [json.loads(line) for line in lines if json.loads(line).get("type") == "done"]
    assert done_events, "stream produced no 'done' event"
    return done_events[-1]


def test_error_branch_emits_confidence_fields(monkeypatch):
    chunks = [
        {
            "id": "error",
            "title": "Query Error",
            "text": "Generated Cypher failed: syntax error",
            "score": 0.0,
            "related": [],
            "cypher": "MATCH (n RETURN n",
        }
    ]
    monkeypatch.setattr(
        structured_stream_mod,
        "retrieve_node",
        lambda state: {"retrieved_context": {"chunks": chunks, "strategy": "text2cypher"}, "strategy": "text2cypher"},
    )

    lines = list(iter_structured_stream("anything", user_context=None, resolved_question="anything"))
    done = _last_done_event(lines)

    assert done["low_confidence"] is True
    assert done["confidence_note"]


def test_fast_path_flags_low_confidence_on_rule_mismatch(monkeypatch):
    chunks = _chunks("MATCH (o:Order) RETURN o.total AS total", [{"total": 42.0}])
    monkeypatch.setattr(
        structured_stream_mod,
        "retrieve_node",
        lambda state: {"retrieved_context": {"chunks": chunks, "strategy": "text2cypher"}, "strategy": "text2cypher"},
    )
    monkeypatch.setattr(structured_stream_mod, "_should_fast_structured_answer", lambda chunks, strategy: True)
    monkeypatch.setattr(structured_stream_mod, "get_chat_provider", lambda: MagicMock())
    # This test is about the confidence flag itself, not the document
    # fallback it now triggers — stub it out so low_confidence=True doesn't
    # pull in the real unstructured retrieval stack.
    monkeypatch.setattr(structured_stream_mod, "_try_document_fallback_stream", lambda *a, **k: None)

    lines = list(
        iter_structured_stream(
            "what is the average order value", user_context=None, resolved_question="what is the average order value"
        )
    )
    done = _last_done_event(lines)

    assert done["low_confidence"] is True
    assert "AVG" in done["confidence_note"]


def test_fast_path_clean_query_is_high_confidence(monkeypatch):
    chunks = _chunks("MATCH (c:Customer) RETURN count(c) AS n", [{"n": 5}])
    monkeypatch.setattr(
        structured_stream_mod,
        "retrieve_node",
        lambda state: {"retrieved_context": {"chunks": chunks, "strategy": "text2cypher"}, "strategy": "text2cypher"},
    )
    monkeypatch.setattr(structured_stream_mod, "_should_fast_structured_answer", lambda chunks, strategy: True)
    monkeypatch.setattr(structured_stream_mod, "get_chat_provider", lambda: MagicMock())

    lines = list(
        iter_structured_stream(
            "how many customers do we have", user_context=None, resolved_question="how many customers do we have"
        )
    )
    done = _last_done_event(lines)

    assert done["low_confidence"] is False
    assert done["confidence_note"] is None


def test_llm_synthesis_path_carries_confidence(monkeypatch):
    chunks = _chunks("MATCH (c:Customer) RETURN c.name AS name", [{"name": "Acme"}])
    monkeypatch.setattr(
        structured_stream_mod,
        "retrieve_node",
        lambda state: {"retrieved_context": {"chunks": chunks, "strategy": "text2cypher"}, "strategy": "text2cypher"},
    )
    monkeypatch.setattr(structured_stream_mod, "_should_fast_structured_answer", lambda chunks, strategy: False)

    fake_provider = MagicMock()
    fake_provider.chat_completion_stream = lambda **kwargs: iter(["Here ", "you go."])
    monkeypatch.setattr(structured_stream_mod, "get_chat_provider", lambda: fake_provider)

    lines = list(
        iter_structured_stream("list customers", user_context=None, resolved_question="list customers")
    )
    done = _last_done_event(lines)

    assert done["answer"] == "Here you go."
    assert done["low_confidence"] is False
    assert done["confidence_note"] is None
