"""
tests/test_document_ir_unit.py — Document IR data model (src/document/ir.py).

Pure dataclass behavior: defaults, content_hash determinism, no Neo4j/
blob/vector imports leaking into the module.

Run with:
    python -m pytest tests/test_document_ir_unit.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path


from src.document.ir import Block, DocumentIR, PageBlock


def test_block_defaults():
    b = Block(text="hello", page=1)
    assert b.bbox is None
    assert b.source == "pymupdf"
    assert b.kind == "text"
    assert b.low_confidence is False
    assert b.extra == {}


def test_block_extra_bag_is_independent_per_instance():
    a = Block(text="a", page=1)
    b = Block(text="b", page=1)
    a.extra["role"] = "heading"
    assert b.extra == {}


def test_page_block_defaults():
    p = PageBlock(page=1, text="content")
    assert p.blocks == []
    assert p.regions == []
    assert p.confidence == 0.0


def test_document_ir_defaults_no_toc():
    ir = DocumentIR(source_name="doc", page_count=10)
    assert ir.toc is None
    assert ir.pages == []
    assert ir.content_hash == ""


def test_finalize_sets_content_hash():
    ir = DocumentIR(
        source_name="doc",
        page_count=1,
        pages=[PageBlock(page=1, text="hello world")],
    )
    ir.finalize()
    assert ir.content_hash != ""
    assert len(ir.content_hash) == 64  # sha256 hex digest


def test_finalize_is_deterministic():
    ir1 = DocumentIR(source_name="doc", page_count=1, pages=[PageBlock(page=1, text="hello")])
    ir2 = DocumentIR(source_name="doc", page_count=1, pages=[PageBlock(page=1, text="hello")])
    assert ir1.finalize().content_hash == ir2.finalize().content_hash


def test_finalize_differs_on_different_text():
    ir1 = DocumentIR(source_name="doc", page_count=1, pages=[PageBlock(page=1, text="hello")])
    ir2 = DocumentIR(source_name="doc", page_count=1, pages=[PageBlock(page=1, text="goodbye")])
    assert ir1.finalize().content_hash != ir2.finalize().content_hash


def test_finalize_distinguishes_page_boundary_from_concatenation():
    # "ab" + "c" must hash differently from "a" + "bc" -- a null-byte
    # separator between pages (not naive string concatenation) is what
    # keeps content_hash sensitive to how text was split across pages,
    # not just its concatenated content.
    ir1 = DocumentIR(
        source_name="doc", page_count=2,
        pages=[PageBlock(page=1, text="ab"), PageBlock(page=2, text="c")],
    )
    ir2 = DocumentIR(
        source_name="doc", page_count=2,
        pages=[PageBlock(page=1, text="a"), PageBlock(page=2, text="bc")],
    )
    assert ir1.finalize().content_hash != ir2.finalize().content_hash


def test_finalize_returns_self_for_chaining():
    ir = DocumentIR(source_name="doc", page_count=1, pages=[PageBlock(page=1, text="x")])
    assert ir.finalize() is ir


def test_ir_module_has_no_storage_layer_imports():
    import src.document.ir as ir_module

    src_text = Path(ir_module.__file__).read_text()
    for forbidden in ("neo4j", "qdrant", "minio", "boto3", "Neo4jDriver"):
        assert forbidden not in src_text, f"ir.py must stay storage-agnostic; found {forbidden!r}"
