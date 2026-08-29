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

from unittest.mock import MagicMock

import pytest


@pytest.fixture
def routing_mod(stubbed_import):
    """The routing module, imported with its heavy dependencies stood in for.

    Scoped to a test rather than installed at module scope: pytest imports
    every test module during collection, so module-level stubs are still in
    place when a later module is imported and quietly decide what it sees.
    """
    return stubbed_import(
        "src.interface.routing",
        stubs={
            "neo4j": {"GraphDatabase": MagicMock()},
            "src.shared.auth": {},
            "src.shared.auth.roles": {"UserContext": MagicMock},
        },
    )


def test_run_via_mcp_tool_reattributes_document_agent_structured_autofix(routing_mod, monkeypatch):
    monkeypatch.setattr(routing_mod, "_fast_route_tool", lambda q: "search_documents")

    fake_result = {
        "answer": "The supplier that provides Chai is Exotic Liquids.",
        "agent": "structured",
        "_autofix_agent": "structured",
    }
    # Takes `language` because every document handler now does: run_via_mcp_tool
    # passes it to the language-aware tools, and a double that cannot accept it
    # is not standing in for the real contract.
    handlers = {
        "search_documents": lambda question, user_context=None, thread_id="default",
        language="en": fake_result
    }

    out = routing_mod.run_via_mcp_tool(
        "Which supplier provides Chai?", "search_documents", handlers
    )

    assert out["_route_tool"] == "query_data"
    assert out["_route_method"] == "misroute_autofix"


def test_run_via_mcp_tool_leaves_plain_document_answer_unattributed(routing_mod, monkeypatch):
    monkeypatch.setattr(routing_mod, "_fast_route_tool", lambda q: "search_documents")

    fake_result = {
        "answer": "The compliance policy requires reporting concerns.",
        "agent": "unstructured",
    }
    # Takes `language` because every document handler now does: run_via_mcp_tool
    # passes it to the language-aware tools, and a double that cannot accept it
    # is not standing in for the real contract.
    handlers = {
        "search_documents": lambda question, user_context=None, thread_id="default",
        language="en": fake_result
    }

    out = routing_mod.run_via_mcp_tool(
        "What does the compliance policy say?", "search_documents", handlers
    )

    assert out["_route_tool"] == "search_documents"
    assert out["_route_method"] == "fast"
