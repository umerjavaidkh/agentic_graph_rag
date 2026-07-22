"""
tests/test_parser_reference_edges_unit.py — table/figure cross-reference resolution.

Regression guard for a real bug: "see Table 3" in prose never resolved to
the actual Table 3 region node, because the alias map only indexed
Chapter/Section numeric shorthand, never Region nodes.

Run with:
    python -m pytest tests/test_parser_reference_edges_unit.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_root = Path(__file__).resolve().parents[1]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from src.document.light.parser import LightPdfParser
from src.models import DKGEdge, DKGNode, NodeType, RelType


def _section(node_id: str, text: str, order: int = 0) -> DKGNode:
    return DKGNode(id=node_id, type=NodeType.SECTION, title=f"Section {order}", text=text, order=order)


def _region(node_id: str, title: str, kind: str, order: int = 0) -> DKGNode:
    node = DKGNode(id=node_id, type=NodeType.REGION, title=title, text=title, order=order)
    node.region_kind = kind
    return node


def test_table_reference_resolves_to_region_node():
    section = _section("sec1", "See Table 3 for details on revenue.")
    table = _region("region_table3", "Table 3: Revenue by Segment", "table")
    nodes = [section, table]
    edges: list[DKGEdge] = []

    parser = LightPdfParser()
    parser._detect_reference_edges(nodes, edges)

    refs = [e for e in edges if e.rel_type == RelType.REFERENCES]
    assert len(refs) == 1
    assert refs[0].source_id == "sec1"
    assert refs[0].target_id == "region_table3"


def test_figure_reference_resolves_to_region_node():
    section = _section("sec1", "As shown in Figure 2, sales increased sharply.")
    figure = _region("region_fig2", "Figure 2: Sales Trend", "figure")
    nodes = [section, figure]
    edges: list[DKGEdge] = []

    parser = LightPdfParser()
    parser._detect_reference_edges(nodes, edges)

    refs = [e for e in edges if e.rel_type == RelType.REFERENCES]
    assert len(refs) == 1
    assert refs[0].source_id == "sec1"
    assert refs[0].target_id == "region_fig2"


def test_no_matching_table_number_produces_no_edge():
    section = _section("sec1", "See Table 9 for details.")
    table = _region("region_table3", "Table 3: Revenue by Segment", "table")
    nodes = [section, table]
    edges: list[DKGEdge] = []

    parser = LightPdfParser()
    parser._detect_reference_edges(nodes, edges)

    assert edges == []


def test_chapter_section_aliasing_still_works():
    """Existing chapter/section numeric-shorthand aliasing is unaffected."""
    section_a = DKGNode(id="a", type=NodeType.SECTION, title="Intro", text="See Section 2 for more.", order=1)
    section_b = DKGNode(id="b", type=NodeType.SECTION, title="Details", text="Body text.", order=2)
    nodes = [section_a, section_b]
    edges: list[DKGEdge] = []

    parser = LightPdfParser()
    parser._detect_reference_edges(nodes, edges)

    refs = [e for e in edges if e.rel_type == RelType.REFERENCES]
    assert len(refs) == 1
    assert refs[0].source_id == "a"
    assert refs[0].target_id == "b"


def test_decimal_table_number_resolves():
    section = _section("sec1", "Refer to Table 3.1 for the breakdown.")
    table = _region("region_table3_1", "Table 3.1: Quarterly Breakdown", "table")
    nodes = [section, table]
    edges: list[DKGEdge] = []

    parser = LightPdfParser()
    parser._detect_reference_edges(nodes, edges)

    refs = [e for e in edges if e.rel_type == RelType.REFERENCES]
    assert len(refs) == 1
    assert refs[0].target_id == "region_table3_1"
