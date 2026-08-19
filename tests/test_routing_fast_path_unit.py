"""
tests/test_routing_fast_path_unit.py — keyword-based fast-path routing.

Regression guard for a real bug found via end-to-end testing: a document
question referencing "Table 1" (e.g. "revenue breakdown ... according to
Table 1") was confidently fast-routed to the structured Northwind path
(because "revenue" matched _DATA_ROUTE) since _DOC_ROUTE had no pattern
for a bare table/figure reference — only "table on"/"table of contents".
The fix makes "table" a bare-word document cue (mirroring "figure", which
was already bare), so a genuinely ambiguous question now defers to the
LLM router via the existing "both matched -> None" safety valve, instead
of confidently answering from the wrong domain.

Run with:
    python -m pytest tests/test_routing_fast_path_unit.py -v
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

from src.routing import _DATA_ROUTE, _DOC_ROUTE, _fast_route_tool


@pytest.mark.parametrize(
    "question,expected_tool",
    [
        # The exact bug: a document question mentioning "Table 1" and "revenue"
        # must NOT confidently fast-route to the structured (wrong-domain) path.
        ("What was the revenue breakdown by region according to Table 1?", None),
        ("What does Figure 2 show about sales trends?", None),
        # Clearly structured-only — still fast-routes as before.
        ("How many products do we have?", "query_data"),
        ("What are the top 5 best selling products?", "query_data"),
        # Clearly document-only — still fast-routes as before.
        ("What does the compliance policy manual say about whistleblowing?", "search_documents"),
        ("What sections are in the annual report?", "search_documents"),
        ("What does section 2 say about compliance procedure?", "search_documents"),
        # Genuinely ambiguous business/document overlap — safety valve still works.
        # ("suppliers" is a _DATA_ROUTE term too, so this must defer, not guess.)
        ("As shown in Table 3, list all suppliers mentioned.", None),
        ("Show me the products table", None),
    ],
)
def test_fast_route_tool(question, expected_tool):
    assert _fast_route_tool(question) == expected_tool


def test_doc_route_matches_bare_table_and_figure_references():
    assert _DOC_ROUTE.search("see Table 1 for details")
    assert _DOC_ROUTE.search("as shown in Figure 2")
    assert _DOC_ROUTE.search("refer to table 3.1")


def test_data_route_still_matches_revenue_and_products():
    assert _DATA_ROUTE.search("What was our total revenue?")
    assert _DATA_ROUTE.search("How many products are there?")
