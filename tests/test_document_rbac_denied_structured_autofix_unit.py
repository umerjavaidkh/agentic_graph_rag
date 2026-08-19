"""
tests/test_document_rbac_denied_structured_autofix_unit.py — the
document-RBAC-denial + structured-misroute-autofix interaction in
_generate_document_answer.

Regression: access_denied_response() (services/formatter.py) returns a
NON-EMPTY chunks list — one synthetic "access_denied" marker chunk — when
the user lacks document access. _generate_document_answer's misroute guard
(document_agent_structured_guard, which redirects a structured-shaped
question to the structured agent when the user actually has structured
access) was gated on `if not chunks:`, so it never ran for this case:
execution fell straight to the flat "access denied" message without ever
trying the redirect, even for a user who has real structured access and
asked an obviously structured/analytics question. Verified live: user
regular_001 (structured access yes, document/"esg" access no) asking
"Which supplier provides Chai?" got a flat access-denied message instead
of the real Northwind answer, across the whole structured/advanced eval
suites (which default to no explicit retrieval_mode, so every case lands
on the document agent first).

Run with:
    python -m pytest tests/test_document_rbac_denied_structured_autofix_unit.py -v
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


for _n in ["langgraph", "langgraph.graph"]:
    if _n not in sys.modules:
        _stub_module(_n)
sys.modules["langgraph.graph"].StateGraph = MagicMock()
sys.modules["langgraph.graph"].END = MagicMock()

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

for _mod_name in ("src.retrieval.unstructured.graph", "src.retrieval.unstructured.retriever"):
    if _mod_name in sys.modules:
        del sys.modules[_mod_name]
_retriever_stub = _stub_module("src.retrieval.unstructured.retriever")
_retriever_stub.DocumentRAGRetriever = lambda *a, **k: MagicMock()
_retriever_stub.is_page_question = lambda q: False
_retriever_stub.is_synthesis_question = lambda q: False
_retriever_stub.is_toc_question = lambda q: False
_retriever_stub.is_visual_page_question = lambda q: False

import src.retrieval.unstructured.graph as graph_module


def _access_denied_chunk(user_id: str = "regular_001") -> dict:
    return {
        "id": "access_denied",
        "title": "Access Denied",
        "text": f"User {user_id} does not have permission to query Agentic Graph RAG data.",
        "score": 0.0,
        "related": [],
    }


def test_denied_chunk_redirects_to_structured_when_guard_fixes_it(monkeypatch):
    """User lacks document access but has structured access + a structured-shaped
    question: the guard's redirect must win over the flat denial message."""
    guard_result = {
        "answer": "The supplier that provides Chai is Exotic Liquids.",
        "low_confidence": False,
        "confidence_note": None,
        "sources": [{"id": "row_0", "title": "Exotic Liquids"}],
        "strategy": "structured",
        "_autofix_agent": "structured",
    }
    monkeypatch.setattr(graph_module, "document_agent_structured_guard", lambda *a, **kw: guard_result)

    out = graph_module._generate_document_answer(
        "Which supplier provides Chai?",
        {},
        [_access_denied_chunk()],
    )

    assert out == guard_result


def test_denied_chunk_falls_back_to_flat_message_when_guard_declines(monkeypatch):
    """A real document question (not structured-shaped) with no document
    access must still get the plain access-denied message, unchanged."""
    monkeypatch.setattr(graph_module, "document_agent_structured_guard", lambda *a, **kw: None)

    out = graph_module._generate_document_answer(
        "What does the compliance policy say about whistleblowing?",
        {},
        [_access_denied_chunk("public_001")],
    )

    assert out == {
        "answer": "User public_001 does not have permission to query Agentic Graph RAG data.",
        "low_confidence": False,
    }


def test_denied_chunk_skips_guard_entirely_when_skip_structured_guard_set(monkeypatch):
    """The structured path's own low-confidence document fallback
    (skip_structured_guard=True) must never bounce back to structured --
    that would recreate the very answer it's trying to improve on."""
    guard = MagicMock(return_value={"_autofix_agent": "structured"})
    monkeypatch.setattr(graph_module, "document_agent_structured_guard", guard)

    out = graph_module._generate_document_answer(
        "Which supplier provides Chai?",
        {},
        [_access_denied_chunk()],
        skip_structured_guard=True,
    )

    guard.assert_not_called()
    assert out == {
        "answer": "User regular_001 does not have permission to query Agentic Graph RAG data.",
        "low_confidence": False,
    }


def test_empty_chunks_without_denial_still_uses_guard_as_before(monkeypatch):
    """Pre-existing behavior (genuinely empty retrieval, no denial marker)
    must be unaffected by folding the denied-chunk case into the same branch."""
    monkeypatch.setattr(graph_module, "document_agent_structured_guard", lambda *a, **kw: None)
    monkeypatch.setattr(graph_module, "compute_confidence", lambda *a, **kw: (True, "no chunks"))

    out = graph_module._generate_document_answer(
        "What does this document cover overall?",
        {},
        [],
    )

    assert out["answer"] == "I could not find relevant information in the ingested documents."
    assert out["low_confidence"] is True
