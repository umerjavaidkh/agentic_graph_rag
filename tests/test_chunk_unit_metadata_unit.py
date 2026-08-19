"""
tests/test_chunk_unit_metadata_unit.py — chunks record what they are part of.

Both defects here were the same shape: a chunk knew its own text and nothing
about its place, so retrieval could not tell a fragment from a whole.

  * A table continued across pages was several unrelated chunks. Asked how
    many institutions Table A3 listed -- a table running over pages 47-49 --
    retrieval answered from one page and gave a count with no sign the rest
    existed. Now one `unit_id` spans the run.

  * A chunk did not know which section it sat in, so two segments repeating
    the same row labels were indistinguishable: asked for International
    Upstream's liquids production, retrieval returned a sibling segment's
    figure. Now `section_path` carries the trail of ancestor titles.

Both are computed once at ingestion rather than re-derived per query, so
every retrieval strategy sees the same relationship.

Run with:
    python -m pytest tests/test_chunk_unit_metadata_unit.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path


from src.graph.axis1_structural import _link_continuations, _stamp_section_paths
from src.models import DKGEdge, DKGNode, NodeType, RelType


def _node(node_id: str, title: str, node_type=NodeType.SECTION) -> DKGNode:
    return DKGNode(id=node_id, type=node_type, title=title, text="x", order=0)


# ── continuation units ──────────────────────────────────────────────────────


def test_continued_table_becomes_one_unit_in_reading_order():
    nodes = [
        _node("a", "Table A3. Use of Go.Data in WHO European Region."),
        _node("b", "Table A3. Use of Go.Data in WHO European Region. (Suite)"),
        _node("c", "Table A3. Use of Go.Data in WHO European Region. (continued)"),
    ]
    _link_continuations(nodes)
    assert [n.unit_id for n in nodes] == ["a", "a", "a"]
    assert [n.unit_part for n in nodes] == [1, 2, 3]


def test_unrelated_table_is_not_joined():
    nodes = [_node("a", "Table A3. Regions."), _node("b", "Table A4. Something Else.")]
    _link_continuations(nodes)
    assert all(n.unit_id is None for n in nodes)


def test_continuation_with_no_preceding_head_is_ignored():
    """A marker alone must never create a unit — the document has to have
    introduced the title first."""
    nodes = [_node("a", "Orphan thing (continued)")]
    _link_continuations(nodes)
    assert nodes[0].unit_id is None


def test_every_title_ending_in_a_period_is_not_a_continuation():
    """Regression: trimming punctuation before checking for a marker made
    each such title a continuation of itself."""
    nodes = [_node("a", "Introduction."), _node("b", "Summary.")]
    _link_continuations(nodes)
    assert all(n.unit_id is None for n in nodes)


# ── section breadcrumb ──────────────────────────────────────────────────────


def _contains(parent: str, child: str) -> DKGEdge:
    return DKGEdge(parent, child, RelType.CONTAINS, axis=1)


def test_breadcrumb_is_the_trail_of_ancestor_titles():
    nodes = [
        _node("d", "cvx.pdf", NodeType.DOCUMENT),
        _node("c", "Item 7", NodeType.CHAPTER),
        _node("s", "International Upstream"),
        _node("p", "Page 42", NodeType.PAGE),
    ]
    _stamp_section_paths(nodes, [_contains("d", "c"), _contains("c", "s"), _contains("s", "p")])
    assert nodes[3].section_path == "Item 7 > International Upstream"


def test_document_root_is_left_out_of_the_breadcrumb():
    """Every chunk shares the root, so it locates nothing."""
    nodes = [_node("d", "cvx.pdf", NodeType.DOCUMENT), _node("c", "Item 7", NodeType.CHAPTER)]
    _stamp_section_paths(nodes, [_contains("d", "c")])
    assert nodes[1].section_path == ""


def test_top_level_node_has_an_empty_breadcrumb():
    nodes = [_node("a", "Standalone")]
    _stamp_section_paths(nodes, [])
    assert nodes[0].section_path == ""


def test_cyclic_containment_terminates():
    """Malformed structure must not hang the walk."""
    nodes = [_node("a", "A"), _node("b", "B")]
    _stamp_section_paths(nodes, [_contains("a", "b"), _contains("b", "a")])
    assert all(isinstance(n.section_path, str) for n in nodes)


def test_page_node_continuation_is_found_in_body_text():
    """Page nodes are titled positionally ("Page 41 (PDF 48)"), so a table
    continued across pages carried its marker only in the body — the
    motivating case, which title-only matching missed entirely while Region
    nodes (Box 9) linked fine."""
    def page(node_id: str, title: str, text: str) -> DKGNode:
        return DKGNode(id=node_id, type=NodeType.PAGE, title=title, text=text, order=0)

    nodes = [
        page("p47", "Page 40 (PDF 47)", "Table A3. Use of Go.Data in WHO European Region.\nrows"),
        page("p48", "Page 41 (PDF 48)", "Table A3. Use of Go.Data in WHO European Region. (Suite)\nrows"),
        page("p49", "Page 42 (PDF 49)", "Table A4. Use of Go.Data in PAHO regions.\nrows"),
    ]
    _link_continuations(nodes)
    assert [n.unit_id for n in nodes] == ["p47", "p47", None]
    assert [n.unit_part for n in nodes] == [1, 2, 0]
