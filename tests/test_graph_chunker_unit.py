"""
tests/test_graph_chunker_unit.py — src/graph/chunker.py's StructuralChunker
default impl (docs/DESIGN_unstructured_graph_v2.md phase 2, step 2).

Run with:
    python -m pytest tests/test_graph_chunker_unit.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

_root = Path(__file__).resolve().parents[1]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from src.document.ir import DocumentIR, PageBlock
from src.graph.chunker import Chunk, StructuralChunker


def test_structural_chunker_one_chunk_per_page():
    ir = DocumentIR(
        source_name="doc", page_count=3,
        pages=[
            PageBlock(page=1, text="alpha"),
            PageBlock(page=2, text="beta"),
            PageBlock(page=3, text="gamma"),
        ],
    )

    chunks = StructuralChunker().chunk(ir)

    assert len(chunks) == 3
    assert [c.text for c in chunks] == ["alpha", "beta", "gamma"]
    assert [c.page_start for c in chunks] == [1, 2, 3]
    assert [c.page_end for c in chunks] == [1, 2, 3]
    assert [c.source_pages for c in chunks] == [[1], [2], [3]]
    assert all(isinstance(c, Chunk) for c in chunks)


def test_structural_chunker_ids_are_stable_and_unique():
    ir = DocumentIR(
        source_name="doc", page_count=2,
        pages=[PageBlock(page=1, text="a"), PageBlock(page=2, text="b")],
    )

    chunks = StructuralChunker().chunk(ir)

    assert chunks[0].id == "page_1"
    assert chunks[1].id == "page_2"
    assert chunks[0].id != chunks[1].id


def test_structural_chunker_empty_document():
    ir = DocumentIR(source_name="doc", page_count=0, pages=[])
    assert StructuralChunker().chunk(ir) == []
