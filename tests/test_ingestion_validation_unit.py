"""tests/test_ingestion_validation_unit.py — pure-logic pieces of the cheap,
LLM-free ingestion quality report (no Neo4j needed for these).

Run with:
    python -m pytest tests/test_ingestion_validation_unit.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path


from src.ingestion.validation import _coverage, _flag_anomalies, _page_continuity


def test_coverage_percentage():
    c = _coverage(total=200, good_count=180)
    assert c == {"total": 200, "good": 180, "pct": 90.0}


def test_coverage_zero_total_no_division_error():
    c = _coverage(total=0, good_count=0)
    assert c == {"total": 0, "good": 0, "pct": 0.0}


def test_coverage_handles_none_inputs():
    c = _coverage(total=None, good_count=None)
    assert c == {"total": 0, "good": 0, "pct": 0.0}


def test_page_continuity_no_gaps():
    pc = _page_continuity([1, 2, 3, 4, 5])
    assert pc == {"count": 5, "min": 1, "max": 5, "gaps": []}


def test_page_continuity_detects_gaps():
    pc = _page_continuity([1, 2, 4, 5, 8])
    assert pc["gaps"] == [3, 6, 7]
    assert pc["count"] == 5
    assert pc["min"] == 1
    assert pc["max"] == 8


def test_page_continuity_empty_input():
    pc = _page_continuity([])
    assert pc == {"count": 0, "min": None, "max": None, "gaps": []}


def _base_kwargs(**overrides):
    base = dict(
        node_counts={"Document": 1, "Chapter": 10, "Section": 20, "Page": 30},
        semantic_edge_count=50,
        text_coverage={"total": 50, "good": 49, "pct": 98.0, "empty_or_near_empty": 1},
        ner_coverage={"total": 50, "good": 45, "pct": 90.0},
        embedding_coverage={"total": 30, "good": 30, "pct": 100.0},
        page_continuity={"count": 30, "min": 1, "max": 30, "gaps": []},
        orphan_nodes=0,
    )
    base.update(overrides)
    return base


def test_flag_anomalies_clean_document_has_no_flags():
    flags = _flag_anomalies(**_base_kwargs())
    assert flags == []


def test_flag_anomalies_no_nodes_short_circuits():
    flags = _flag_anomalies(**_base_kwargs(node_counts={}))
    assert len(flags) == 1
    assert "No nodes found" in flags[0]


def test_flag_anomalies_low_text_coverage():
    kwargs = _base_kwargs(
        text_coverage={"total": 50, "good": 30, "pct": 60.0, "empty_or_near_empty": 20}
    )
    flags = _flag_anomalies(**kwargs)
    assert any("chars of text" in f for f in flags)


def test_flag_anomalies_zero_ner_coverage():
    kwargs = _base_kwargs(ner_coverage={"total": 50, "good": 0, "pct": 0.0})
    flags = _flag_anomalies(**kwargs)
    assert any("NER coverage" in f for f in flags)


def test_flag_anomalies_zero_embedding_coverage():
    kwargs = _base_kwargs(embedding_coverage={"total": 30, "good": 0, "pct": 0.0})
    flags = _flag_anomalies(**kwargs)
    assert any("embedding coverage" in f for f in flags)


def test_flag_anomalies_no_semantic_edges():
    kwargs = _base_kwargs(semantic_edge_count=0)
    flags = _flag_anomalies(**kwargs)
    assert any("No semantic edges" in f for f in flags)


def test_flag_anomalies_no_semantic_edges_not_flagged_for_tiny_docs():
    # A single-chapter, single-section doc has too few nodes for pairwise
    # semantic edges to be expected — shouldn't false-positive.
    kwargs = _base_kwargs(
        node_counts={"Document": 1, "Chapter": 1, "Section": 0, "Page": 1},
        semantic_edge_count=0,
    )
    flags = _flag_anomalies(**kwargs)
    assert not any("No semantic edges" in f for f in flags)


def test_flag_anomalies_page_gaps():
    kwargs = _base_kwargs(page_continuity={"count": 29, "min": 1, "max": 30, "gaps": [15]})
    flags = _flag_anomalies(**kwargs)
    assert any("missing page number" in f for f in flags)


def test_flag_anomalies_orphan_nodes():
    kwargs = _base_kwargs(orphan_nodes=3)
    flags = _flag_anomalies(**kwargs)
    assert any("no parent CONTAINS edge" in f for f in flags)
