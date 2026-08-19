"""
tests/test_document_synthesis_context_budget_unit.py — synthesis prompt
context is bounded regardless of how large individual retrieved chunks are.

Regression: found live after the embedded-PDF-outline chapter-detection fix
(light/parser.py's _usable_toc) made a whole Chapter node's .text span
dozens of real pages instead of a fragment. _generate_document_answer
concatenated every retrieved chunk's full text into one prompt with no
input-side budget -- only the OUTPUT side was capped
(DOCUMENT_SYNTHESIS_MAX_TOKENS). A handful of chunks from a 46-page chapter
requested ~1.35M tokens in one call against gpt-4o-mini's 128k context
window and the org's 200k tokens-per-minute rate limit, crashing the query
with a 500 instead of answering. Verified live: "List every Check Your
Understanding question in Chapter 15." on the re-ingested physics textbook
triggered openai.RateLimitError before this fix.

Fixed via DOCUMENT_SYNTHESIS_CONTEXT_MAX_CHARS: a running character budget
across all chunks, truncating the tail rather than sending everything.

Run with:
    python -m pytest tests/test_document_synthesis_context_budget_unit.py -v
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


# graph.py does `from langgraph.graph import END, StateGraph` and builds a
# real StateGraph/compile() at module level -- stub both, permissively, so
# importing the real graph.py doesn't need the actual langgraph package
# (same pattern as test_structured_graph_confidence_unit.py).
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
sys.modules["src.shared.auth.roles"].DEFAULT_PUBLIC_CONTEXT = MagicMock(role=MagicMock(value="public"))

# graph.py does `retriever = DocumentRAGRetriever()` at module level, which
# would otherwise need a live Neo4j driver/RBAC/schema chain to construct.
# Not exercised by _generate_document_answer (only retrieve_node uses
# `retriever`) -- stub the class to a no-op constructor.
for _mod_name in ("src.unstructured.retrieval.graph", "src.unstructured.retrieval.retriever"):
    if _mod_name in sys.modules:
        del sys.modules[_mod_name]
_retriever_stub = _stub_module("src.unstructured.retrieval.retriever")
_retriever_stub.DocumentRAGRetriever = lambda *a, **k: MagicMock()
_retriever_stub.is_page_question = lambda q: False
_retriever_stub.is_synthesis_question = lambda q: False
_retriever_stub.is_toc_question = lambda q: False
_retriever_stub.is_visual_page_question = lambda q: False

import src.unstructured.retrieval.graph as graph_module


def _fake_response(text: str = "answer"):
    return MagicMock(choices=[MagicMock(message=MagicMock(content=text))])


def test_context_text_is_truncated_to_the_configured_char_budget(monkeypatch):
    huge_chunk = {
        "id": "c1",
        "title": "Chapter 15. Oscillations",
        "text": "x" * (graph_module.DOCUMENT_SYNTHESIS_CONTEXT_MAX_CHARS * 3),
    }
    captured = {}

    def _fake_chat_completion(model, messages, **kwargs):
        captured["system_prompt"] = messages[0]["content"]
        return _fake_response()

    monkeypatch.setattr(graph_module, "provider", MagicMock(chat_completion=_fake_chat_completion))
    monkeypatch.setattr(graph_module, "load_prompt", lambda name, **kw: kw.get("context", ""))
    monkeypatch.setattr(graph_module, "compute_confidence", lambda *a, **kw: (False, None))
    monkeypatch.setattr(graph_module, "document_agent_structured_guard", lambda *a, **kw: None)

    graph_module._generate_document_answer(
        "List every Check Your Understanding question in Chapter 15.",
        {"mode": "graph_rag"},
        [huge_chunk],
    )

    assert len(captured["system_prompt"]) <= graph_module.DOCUMENT_SYNTHESIS_CONTEXT_MAX_CHARS + 200
    assert "truncated" in captured["system_prompt"]


def test_later_chunks_are_dropped_once_budget_is_exhausted(monkeypatch):
    chunks = [
        {"id": f"c{i}", "title": f"Section {i}", "text": "y" * (graph_module.DOCUMENT_SYNTHESIS_CONTEXT_MAX_CHARS // 2)}
        for i in range(5)
    ]
    captured = {}

    def _fake_chat_completion(model, messages, **kwargs):
        captured["system_prompt"] = messages[0]["content"]
        return _fake_response()

    monkeypatch.setattr(graph_module, "provider", MagicMock(chat_completion=_fake_chat_completion))
    monkeypatch.setattr(graph_module, "load_prompt", lambda name, **kw: kw.get("context", ""))
    monkeypatch.setattr(graph_module, "compute_confidence", lambda *a, **kw: (False, None))
    monkeypatch.setattr(graph_module, "document_agent_structured_guard", lambda *a, **kw: None)

    graph_module._generate_document_answer(
        "What does this document cover overall?",
        {"mode": "graph_rag"},
        chunks,
    )

    # Only the first two ~half-budget chunks fit; the rest are dropped.
    assert "Section 0" in captured["system_prompt"]
    assert "Section 4" not in captured["system_prompt"]


def test_small_chunks_are_unaffected_no_truncation_marker(monkeypatch):
    chunks = [{"id": "c1", "title": "Small Section", "text": "normal short content"}]
    captured = {}

    def _fake_chat_completion(model, messages, **kwargs):
        captured["system_prompt"] = messages[0]["content"]
        return _fake_response()

    monkeypatch.setattr(graph_module, "provider", MagicMock(chat_completion=_fake_chat_completion))
    monkeypatch.setattr(graph_module, "load_prompt", lambda name, **kw: kw.get("context", ""))
    monkeypatch.setattr(graph_module, "compute_confidence", lambda *a, **kw: (False, None))
    monkeypatch.setattr(graph_module, "document_agent_structured_guard", lambda *a, **kw: None)

    graph_module._generate_document_answer(
        "What is this section about?",
        {"mode": "graph_rag"},
        chunks,
    )

    assert "normal short content" in captured["system_prompt"]
    assert "truncated" not in captured["system_prompt"]
