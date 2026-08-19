"""
tests/test_axis1_structural_golden_unit.py — golden-output regression for
the Axis1StructuralBuilder extraction (docs/DESIGN_unstructured_graph_v2.md
phase 2, step 3).

Axis1StructuralBuilder moves (not rewrites) LightPdfParser's
_build_from_toc/_build_from_extracts + every helper they call, retyped
from _PdfBlock/_PageExtract to Block/PageBlock/DocumentIR. "Moved, not
rewritten" is a claim, not a given -- this test proves it by diffing the
new path's output against a snapshot captured from the pre-refactor
`parse()` on two real fixtures (one TOC-driven, one heuristic-driven),
not just checking "doesn't crash".

This already caught one real bug during development: a missing `break`
after the first matching reference-edge key, silently copied out of
_detect_reference_edges, which would have produced ~3x too many
REFERENCES edges (every title_lookup key that matched, instead of just
the first) on any document with ambiguous/overlapping reference phrases.

Fixture snapshots (tests/fixtures/axis1_golden/*.json) intentionally
omit node/edge `text`/`properties` to keep them small -- id/type/title/
page range/depth/order is enough to catch a structural regression, and
a separate test below diffs full node `text` directly (not via a
fixture) since re-parsing is cheap enough to do live.

Run with:
    python -m pytest tests/test_axis1_structural_golden_unit.py -v
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

# Repo root located by searching upward for src/, not by counting parents:
# a fixed index silently points at the wrong directory the moment this
# file changes nesting depth.
_root = next(p for p in Path(__file__).resolve().parents if (p / "src").is_dir())

from src.unstructured.document.light.parser import LightPdfParser
from src.unstructured.graph.axis1_structural import Axis1StructuralBuilder
from src.unstructured.graph.chunker import StructuralChunker

_ALL_FIXTURES = [
    ("rag_document", "sample_data_to_test/unstructured/rag_document.pdf"),
    # Not committed to git (18MB, sample_data_to_test/unstructured/textbooks/
    # is untracked) -- this is the only TOC-driven real-PDF fixture available
    # (the small tracked PDFs all lack an embedded outline), so it's kept as
    # an opportunistic local check rather than skipped outright: present ->
    # exercised, absent (a fresh clone, CI) -> skipped rather than failing.
    ("university_physics", "sample_data_to_test/unstructured/textbooks/UniversityPhysicsVolume1.pdf"),
]
_FIXTURES = [
    pytest.param(name, path, marks=pytest.mark.skip(reason=f"fixture not present: {path}"))
    if not (_root / path).exists()
    else (name, path)
    for name, path in _ALL_FIXTURES
]


def _snapshot(nodes, edges) -> dict:
    return {
        "nodes": sorted(
            [
                {
                    "id": n.id,
                    "type": n.type.value if hasattr(n.type, "value") else n.type,
                    "title": n.title,
                    "page_start": n.page_start,
                    "page_end": n.page_end,
                    "depth": n.depth,
                    "order": n.order,
                    "region_kind": getattr(n, "region_kind", None),
                }
                for n in nodes
            ],
            key=lambda x: x["id"],
        ),
        "edges": sorted(
            [
                {
                    "source_id": e.source_id,
                    "target_id": e.target_id,
                    "rel_type": e.rel_type.value if hasattr(e.rel_type, "value") else e.rel_type,
                    "axis": e.axis,
                }
                for e in edges
            ],
            key=lambda x: (x["source_id"], x["target_id"], x["rel_type"]),
        ),
    }


def _fixture_path(name: str) -> Path:
    return _root / "tests" / "fixtures" / "axis1_golden" / f"{name}_golden.json"


@pytest.mark.parametrize("name,pdf_path", _FIXTURES)
def test_axis1_structural_builder_matches_golden_snapshot(name, pdf_path):
    golden = json.loads(_fixture_path(name).read_text())

    parser = LightPdfParser()
    ir = parser.parse_ir(pdf_path)
    chunks = StructuralChunker().chunk(ir)
    nodes, edges = Axis1StructuralBuilder().build(ir, chunks)

    snapshot = _snapshot(nodes, edges)
    assert snapshot["nodes"] == golden["nodes"]
    assert snapshot["edges"] == golden["edges"]


@pytest.mark.parametrize("name,pdf_path", _FIXTURES)
def test_axis1_structural_builder_matches_legacy_parse_exactly(name, pdf_path):
    """Stronger than the golden-file check: re-runs the OLD parse() path
    live (rather than a pre-captured snapshot) and diffs node .text too,
    so this can't drift out of sync with parse() the way a static
    fixture theoretically could."""
    parser = LightPdfParser()
    legacy_nodes, legacy_edges = parser.parse(pdf_path)

    ir = parser.parse_ir(pdf_path)
    chunks = StructuralChunker().chunk(ir)
    new_nodes, new_edges = Axis1StructuralBuilder().build(ir, chunks)

    legacy_by_id = {n.id: n for n in legacy_nodes}
    new_by_id = {n.id: n for n in new_nodes}
    assert set(legacy_by_id) == set(new_by_id)
    for nid, legacy_node in legacy_by_id.items():
        new_node = new_by_id[nid]
        assert legacy_node.title == new_node.title, nid
        assert legacy_node.text == new_node.text, nid
        assert legacy_node.page_start == new_node.page_start, nid
        assert legacy_node.page_end == new_node.page_end, nid

    def edge_key(e):
        return (e.source_id, e.target_id, e.rel_type.value if hasattr(e.rel_type, "value") else e.rel_type)

    assert {edge_key(e) for e in legacy_edges} == {edge_key(e) for e in new_edges}
    assert len(legacy_edges) == len(new_edges)
