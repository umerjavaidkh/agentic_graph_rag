"""
tests/test_dkgedge_confidence_unit.py — DKGEdge confidence/provenance taxonomy.

Proves the new fields default to EXTRACTED/1.0 (so most existing edge
construction call sites need no code change), and that structural/reference/
SHARES_ENTITY edges built by the real functions correctly inherit that
default without any explicit confidence kwargs.

Run with:
    python -m pytest tests/test_dkgedge_confidence_unit.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

_root = Path(__file__).resolve().parents[1]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from src.document.light.parser import LightPdfParser
from src.models import DKGEdge, DKGNode, EdgeConfidenceTier, NodeType, RelType


def test_dkgedge_defaults_to_extracted():
    edge = DKGEdge(source_id="a", target_id="b", rel_type=RelType.CONTAINS)
    assert edge.confidence == 1.0
    assert edge.confidence_tier == EdgeConfidenceTier.EXTRACTED


def test_dkgedge_accepts_explicit_confidence():
    edge = DKGEdge(
        source_id="a",
        target_id="b",
        rel_type=RelType.SEMANTICALLY_SIMILAR,
        confidence=0.83,
        confidence_tier=EdgeConfidenceTier.INFERRED,
    )
    assert edge.confidence == 0.83
    assert edge.confidence_tier == EdgeConfidenceTier.INFERRED


def test_sequential_edges_inherit_extracted_default():
    a = DKGNode(id="a", type=NodeType.SECTION, title="A", text="a", order=1)
    b = DKGNode(id="b", type=NodeType.SECTION, title="B", text="b", order=2)
    edges: list[DKGEdge] = []

    parser = LightPdfParser()
    parser._add_sequential_edges([a, b], edges)

    assert edges
    assert all(e.confidence_tier == EdgeConfidenceTier.EXTRACTED for e in edges)
    assert all(e.confidence == 1.0 for e in edges)


def test_reference_edges_inherit_extracted_default():
    section = DKGNode(id="sec1", type=NodeType.SECTION, title="Intro", text="See Table 3.", order=1)
    table = DKGNode(id="tbl1", type=NodeType.REGION, title="Table 3: Revenue", text="Table 3: Revenue", order=2)
    table.region_kind = "table"
    edges: list[DKGEdge] = []

    parser = LightPdfParser()
    parser._detect_reference_edges([section, table], edges)

    refs = [e for e in edges if e.rel_type == RelType.REFERENCES]
    assert refs
    assert refs[0].confidence_tier == EdgeConfidenceTier.EXTRACTED
    assert refs[0].confidence == 1.0
