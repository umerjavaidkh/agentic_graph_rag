"""
tests/test_parser_item_heading_hierarchy_unit.py — SEC "Item N[Letter]"
headings nest correctly through the full structure-building pipeline
(Axis1StructuralBuilder._build_from_extracts + _link_number_hierarchy),
not just at the patterns.py helper-function level.

RtldocPdfParser and TableAwarePdfParser both subclass LightPdfParser and
only override page-extraction; Axis1StructuralBuilder (moved out of
LightPdfParser in docs/DESIGN_unstructured_graph_v2.md phase 2) is the
one shared construction implementation all three backends' IR ultimately
runs through -- this test exercises that shared code path.

Run with:
    python -m pytest tests/test_parser_item_heading_hierarchy_unit.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path


from src.document.light.parser import LightPdfParser, _PageExtract, _PdfBlock
from src.graph.axis1_structural import Axis1StructuralBuilder
from src.graph.chunker import StructuralChunker
from src.models import NodeType, RelType


def _block(text: str, page: int, font_size: float) -> _PdfBlock:
    return _PdfBlock(
        text=text, page=page, bbox=[10.0, 10.0, 200.0, 30.0],
        avg_font_size=font_size, max_font_size=font_size,
    )


def _contains_pairs(edges) -> set[tuple[str, str]]:
    return {(e.source_id, e.target_id) for e in edges if e.rel_type == RelType.CONTAINS}


def test_item_1a_nests_under_item_1():
    # Headings at font 20 clear the median-derived threshold; body text at
    # font 10 stays well under it, matching how _heading_font_threshold
    # actually separates the two in a real document.
    blocks = [
        _block("Item 1. Business", 1, 20.0),
        _block("We design, manufacture, and sell widgets.", 1, 10.0),
        _block("Item 1A. Risk Factors", 2, 20.0),
        _block("Our business is subject to numerous risks.", 2, 10.0),
        _block("Item 2. Properties", 3, 20.0),
        _block("We lease office space in several countries.", 3, 10.0),
    ]
    extracts = [
        _PageExtract(page=1, text="", blocks=[blocks[0], blocks[1]]),
        _PageExtract(page=2, text="", blocks=[blocks[2], blocks[3]]),
        _PageExtract(page=3, text="", blocks=[blocks[4], blocks[5]]),
    ]

    parser = LightPdfParser()
    ir = parser._to_document_ir(extracts, toc=None, source_name="sample-10k", page_count=3)
    chunks = StructuralChunker().chunk(ir)
    nodes, edges = Axis1StructuralBuilder()._build_from_extracts(ir, chunks)

    # Chapter titles are the content-derived title only ("Business"), not
    # prefixed with the section number the way section titles are -- the
    # number is still tracked internally via number_map for hierarchy
    # linking even though it isn't rendered into the chapter's title.
    chapters = {n.title: n for n in nodes if n.type == NodeType.CHAPTER}
    sections = {n.title: n for n in nodes if n.type == NodeType.SECTION}

    item1 = chapters["Business"]
    item1a = next(n for title, n in sections.items() if title.startswith("Item 1A "))
    item2 = chapters["Properties"]

    pairs = _contains_pairs(edges)
    assert (item1.id, item1a.id) in pairs, "Item 1A must be CONTAINS-linked under Item 1, not a sibling chapter"
    assert item1a.id not in {n.id for n in chapters.values()}, "Item 1A must be a Section, not a Chapter"
    # Item 2 is a new top-level chapter, not nested under Item 1/1A.
    assert (item1.id, item2.id) not in pairs
    assert (item1a.id, item2.id) not in pairs
