"""
tests/test_axis2_generic_entity_filter_unit.py — SHARES_ENTITY edges don't
anchor on generic, low-information entities.

Regression: found via sampled LLM-judge audit of already-ingested SEC
filings (scripts/validate_ontology_accuracy.py) -- SHARES_ENTITY edges
were routinely built on terms like "company", "2023", the filer's own
name, appearing in most sections of a document. Technically true (both
passages do contain the word) but not a meaningful connection: every
section of a 10-K mentions "the Company", so an edge anchored on it
doesn't distinguish two related sections from two unrelated ones.
Measured baseline: axis2 idea-linking scored 56.45% average across 15
ingested documents (0/15 passing the >=90% ontology-accuracy target),
with exactly this pattern in every judge-flagged invalid example.

Fixed via _informative_entities: a document-frequency ratio within the
document's own entity-bearing nodes (mirrors the IDF philosophy already
used for lexical ranking elsewhere in this repo -- document_resolver.py,
LexicalService -- applied here to which entities may anchor an edge).

Second regression, opposite direction: found via live verification of a
short single-topic tutorial (~20 entity-bearing nodes) -- the flat 40%
cutoff above also excluded that document's own core subject ("sample
mean", 70% document frequency) from anchoring any edge, so its summary
section failed to connect to the sections it was actually summarizing.
Fixed by making the ratio adaptive (_adaptive_genericity_ratio): permissive
for a small corpus, decaying toward the original validated 40% floor as
corpus size grows toward real SEC-filing scale. Tests below at n=200
preserve the original regression's protection at the scale it was
measured on; tests at n~20 cover the new small-corpus case.

Run with:
    python -m pytest tests/test_axis2_generic_entity_filter_unit.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

_root = Path(__file__).resolve().parents[1]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from src.models import DKGNode, NodeType
from src.semantic.axis2 import (
    Axis2Builder,
    _ENTITY_GENERICITY_DF_RATIO,
    _ENTITY_GENERICITY_DF_RATIO_CEILING,
    _ENTITY_GENERICITY_MIN_NODES,
    _adaptive_genericity_ratio,
    _informative_entities,
)


def _builder() -> Axis2Builder:
    builder = Axis2Builder.__new__(Axis2Builder)
    builder.client = MagicMock()
    return builder


def _node(node_id: str, entities: list[str]) -> DKGNode:
    n = DKGNode(id=node_id, type=NodeType.SECTION, title=node_id, text="x", order=0)
    n.entities = entities
    return n


# ── _informative_entities ────────────────────────────────────────────────────


def test_below_min_nodes_no_filtering_applied():
    # 4 nodes < _ENTITY_GENERICITY_MIN_NODES (5) -- ratio math would be
    # unreliable at this sample size, so every entity stays eligible.
    entity_to_nodes = {"generic": [0, 1, 2, 3]}
    result = _informative_entities(entity_to_nodes, total_entity_nodes=4)
    assert result == {"generic"}


def test_entity_in_most_nodes_excluded_at_large_corpus_scale():
    # Real SEC-filing scale (hundreds of entity-bearing nodes) -- the
    # adaptive ratio has decayed close to the original validated 40%
    # floor by this size, so this preserves the original regression's
    # protection at the scale it was actually measured on. A small-corpus
    # equivalent of this same setup is intentionally NOT expected to
    # exclude "generic" anymore -- see the adaptive-ratio tests below.
    total = 200
    max_df = int(total * _adaptive_genericity_ratio(total))
    entity_to_nodes = {
        "generic": list(range(max_df + 1)),  # just over the adaptive threshold
        "specific": [0, 1],
    }
    result = _informative_entities(entity_to_nodes, total_entity_nodes=total)
    assert "generic" not in result
    assert "specific" in result


def test_entity_exactly_at_ratio_threshold_included():
    total = 200
    max_df = int(total * _adaptive_genericity_ratio(total))
    entity_to_nodes = {"borderline": list(range(max_df))}
    result = _informative_entities(entity_to_nodes, total_entity_nodes=total)
    assert "borderline" in result


# ── _adaptive_genericity_ratio ───────────────────────────────────────────────


def test_adaptive_ratio_permissive_for_small_corpus():
    # The motivating case: a ~20-node single-topic tutorial's own core
    # subject sits at 70% document frequency -- the adaptive ratio at this
    # scale must clear that comfortably, not just barely.
    assert _adaptive_genericity_ratio(20) >= 0.70


def test_adaptive_ratio_converges_toward_floor_for_large_corpus():
    # By real SEC-filing scale, behavior should be close to the original
    # validated 40% floor, not meaningfully more permissive.
    ratio = _adaptive_genericity_ratio(200)
    assert _ENTITY_GENERICITY_DF_RATIO <= ratio <= _ENTITY_GENERICITY_DF_RATIO + 0.05


def test_adaptive_ratio_monotonically_decreases_with_corpus_size():
    assert _adaptive_genericity_ratio(10) > _adaptive_genericity_ratio(50) > _adaptive_genericity_ratio(200)


def test_adaptive_ratio_bounded_between_floor_and_ceiling():
    for n in (5, 20, 80, 200, 5000):
        ratio = _adaptive_genericity_ratio(n)
        assert _ENTITY_GENERICITY_DF_RATIO <= ratio <= _ENTITY_GENERICITY_DF_RATIO_CEILING


def test_small_single_topic_document_core_subject_now_informative():
    # Direct reproduction of the live bug: a 20-node document's own core
    # subject appears in 14/20 (70%) of entity-bearing nodes -- under the
    # old flat 40% cutoff this was excluded, breaking the summary
    # section's connections to the sections it actually summarizes. Must
    # now be informative.
    total = 20
    entity_to_nodes = {"sample mean": list(range(14)), "rare aside": [0, 1]}
    result = _informative_entities(entity_to_nodes, total_entity_nodes=total)
    assert "sample mean" in result
    assert "rare aside" in result


# ── _build_entity_edges wiring ────────────────────────────────────────────────


def test_generic_entity_does_not_anchor_edge_in_large_document():
    # 6 nodes (>= min threshold): "company" appears in all 6 (generic),
    # "kenvue divestiture" appears in only 2 (specific, informative).
    nodes = [
        _node("n0", ["company", "kenvue divestiture"]),
        _node("n1", ["company", "kenvue divestiture"]),
        _node("n2", ["company"]),
        _node("n3", ["company"]),
        _node("n4", ["company"]),
        _node("n5", ["company"]),
    ]
    builder = _builder()
    edges = builder._build_entity_edges(nodes)

    # Only n0-n1 share a genuinely informative entity; every other pair's
    # only common ground is the generic "company", which must not anchor
    # an edge on its own.
    assert len(edges) == 1
    assert {edges[0].source_id, edges[0].target_id} == {"n0", "n1"}
    assert "company" not in edges[0].properties["shared_entities"]
    assert "kenvue divestiture" in edges[0].properties["shared_entities"]


def test_only_generic_entity_shared_produces_no_edge():
    nodes = [
        _node("n0", ["company"]),
        _node("n1", ["company"]),
        _node("n2", ["company"]),
        _node("n3", ["company"]),
        _node("n4", ["company"]),
        _node("n5", ["company"]),
    ]
    builder = _builder()
    edges = builder._build_entity_edges(nodes)
    assert edges == []


def test_small_document_below_min_nodes_unaffected_by_genericity_filter():
    # Only 2 entity-bearing nodes -- below _ENTITY_GENERICITY_MIN_NODES, so
    # even a term shared by both nodes must still form an edge (matches
    # pre-existing possessive-variant regression test's expectation).
    a = _node("a", ["company", "force"])
    b = _node("b", ["company", "energy"])
    builder = _builder()
    edges = builder._build_entity_edges([a, b])
    assert len(edges) == 1
    assert "company" in edges[0].properties["shared_entities"]
