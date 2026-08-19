"""
tests/test_routing_misroute_autofix_attribution_unit.py — run_via_mcp_tool()
correctly attributes a document-agent-side structured autofix.

Regression: a structured-shaped question that reaches the document agent
(default retrieval mode, or a document-RBAC denial with unrelated structured
access) can get silently re-answered by the structured agent via
document_agent_structured_guard/run_structured_autofix (see
retrieval/unstructured/graph.py's _generate_document_answer and
router.py's search_documents()). Before this fix, run_via_mcp_tool() left
`_route_tool`/`_route_method` as "search_documents"/"fast" in that case,
so callers (and eval suites asserting route_tool == "query_data") had no
way to see that the structured agent actually served the answer. Mirrors
the existing, opposite-direction branch a few lines above (query_data
access denied -> falls back to search_documents), which already updates
tool_name/route_method the same way.

Run with:
    python -m pytest tests/test_routing_misroute_autofix_attribution_unit.py -v
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

for _n in ["src.shared.auth", "src.shared.auth.roles"]:
    if _n not in sys.modules:
        _stub_module(_n)
sys.modules["src.shared.auth.roles"].UserContext = MagicMock

if "src.routing" in sys.modules:
    del sys.modules["src.routing"]

import src.routing as routing_mod


def test_run_via_mcp_tool_reattributes_document_agent_structured_autofix(monkeypatch):
    monkeypatch.setattr(routing_mod, "_fast_route_tool", lambda q: "search_documents")

    fake_result = {
        "answer": "The supplier that provides Chai is Exotic Liquids.",
        "agent": "structured",
        "_autofix_agent": "structured",
    }
    handlers = {"search_documents": lambda question, user_context=None, thread_id="default": fake_result}

    out = routing_mod.run_via_mcp_tool(
        "Which supplier provides Chai?", "search_documents", handlers
    )

    assert out["_route_tool"] == "query_data"
    assert out["_route_method"] == "misroute_autofix"


def test_run_via_mcp_tool_leaves_plain_document_answer_unattributed(monkeypatch):
    monkeypatch.setattr(routing_mod, "_fast_route_tool", lambda q: "search_documents")

    fake_result = {
        "answer": "The compliance policy requires reporting concerns.",
        "agent": "unstructured",
    }
    handlers = {"search_documents": lambda question, user_context=None, thread_id="default": fake_result}

    out = routing_mod.run_via_mcp_tool(
        "What does the compliance policy say?", "search_documents", handlers
    )

    assert out["_route_tool"] == "search_documents"
    assert out["_route_method"] == "fast"
