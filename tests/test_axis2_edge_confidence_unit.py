"""
tests/test_axis2_edge_confidence_unit.py — axis2.py edge confidence tagging.

Covers the taxonomy assignment for each edge type axis2 builds:
SEMANTICALLY_SIMILAR/ELABORATES-etc get INFERRED with the real signal
preserved; SAME_CATEGORY gets AMBIGUOUS with a flat score; SHARES_ENTITY
needs no change (inherits EXTRACTED/1.0 defaults).

Run with:
    python -m pytest tests/test_axis2_edge_confidence_unit.py -v
"""
from __future__ import annotations

import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest

_root = Path(__file__).resolve().parents[1]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

# hdbscan isn't installed in this dev env -- stub it so
# _build_category_edges can be exercised without a real install. Mirrors
# the existing stub style already used for this module elsewhere.
if "hdbscan" not in sys.modules:
    hdbscan_mod = types.ModuleType("hdbscan")
    sys.modules["hdbscan"] = hdbscan_mod


class _FakeHDBSCAN:
    def __init__(self, min_cluster_size=5, metric="euclidean"):
        self.min_cluster_size = min_cluster_size

    def fit_predict(self, vecs):
        # Deterministic: put everything in one cluster (no -1 noise) so a
        # SAME_CATEGORY edge is guaranteed between every pair, regardless
        # of vector count.
        return [0] * len(vecs)


sys.modules["hdbscan"].HDBSCAN = _FakeHDBSCAN

from src.models import DKGNode, EdgeConfidenceTier, NodeType, RelType
from src.semantic.axis2 import (
    SAME_CATEGORY_CONFIDENCE,
    Axis2Builder,
)


def _embedded_node(node_id: str, embedding: list[float], entities: list[str] | None = None) -> DKGNode:
    node = DKGNode(id=node_id, type=NodeType.SECTION, title=node_id, text="text", order=0)
    node.embedding = embedding
    node.entities = entities or []
    return node


def _builder() -> Axis2Builder:
    builder = Axis2Builder.__new__(Axis2Builder)
    builder.client = MagicMock()
    return builder


def test_semantically_similar_edges_get_inferred_with_real_score():
    a = _embedded_node("a", [1.0, 0.0, 0.0])
    b = _embedded_node("b", [0.99, 0.01, 0.0])  # nearly identical -> high cosine sim

    builder = _builder()
    edges = builder._build_similarity_edges([a, b])

    assert len(edges) == 1
    edge = edges[0]
    assert edge.rel_type == RelType.SEMANTICALLY_SIMILAR
    assert edge.confidence_tier == EdgeConfidenceTier.INFERRED
    assert 0.0 < edge.confidence <= 1.0
    assert edge.confidence == edge.weight  # same underlying cosine score


def test_shares_entity_edges_need_no_change_inherit_extracted():
    a = _embedded_node("a", [1.0, 0.0], entities=["acme corp"])
    b = _embedded_node("b", [0.0, 1.0], entities=["ACME Corp", "widgets"])

    builder = _builder()
    edges = builder._build_entity_edges([a, b])

    assert len(edges) == 1
    edge = edges[0]
    assert edge.rel_type == RelType.SHARES_ENTITY
    assert edge.confidence_tier == EdgeConfidenceTier.EXTRACTED
    assert edge.confidence == 1.0


def test_same_category_edges_get_ambiguous_with_flat_score():
    a = _embedded_node("a", [1.0, 0.0, 0.0])
    b = _embedded_node("b", [0.0, 1.0, 0.0])
    c = _embedded_node("c", [0.0, 0.0, 1.0])

    builder = _builder()
    _, edges = builder._build_category_edges([a, b, c])

    assert edges
    for edge in edges:
        assert edge.rel_type == RelType.SAME_CATEGORY
        assert edge.confidence_tier == EdgeConfidenceTier.AMBIGUOUS
        assert edge.confidence == SAME_CATEGORY_CONFIDENCE


def test_llm_judged_edges_preserve_confidence_instead_of_discarding():
    a = _embedded_node("a", [1.0, 0.0])
    b = _embedded_node("b", [0.99, 0.02])  # must clear CONTRADICTION_THRESH (0.85)

    builder = _builder()
    # Two calls per kept edge now: the relationship-detection call, then the
    # independent grounding call (AXIS2_GROUND_LLM_EDGES) -- must be grounded
    # for _build_llm_edges to keep it, so the second response says so.
    builder.client.chat_completion.side_effect = [
        MagicMock(
            choices=[
                MagicMock(
                    message=MagicMock(
                        content='{"relationship": "ELABORATES", "direction": "A_TO_B", "confidence": 0.92, "reason": "expands on the same topic"}'
                    )
                )
            ]
        ),
        MagicMock(
            choices=[MagicMock(message=MagicMock(content='{"grounded": true, "confidence": 0.88}'))]
        ),
    ]

    edges = builder._build_llm_edges([a, b])

    assert len(edges) == 1
    edge = edges[0]
    assert edge.rel_type == RelType.ELABORATES
    assert edge.confidence_tier == EdgeConfidenceTier.INFERRED
    assert edge.confidence == 0.92  # preserved, not discarded after the >=0.7 gate
    assert edge.properties["reason"] == "expands on the same topic"
    assert edge.properties["grounding_checked"] is True
    assert edge.properties["grounding_confidence"] == 0.88


def test_llm_low_confidence_below_threshold_produces_no_edge():
    a = _embedded_node("a", [1.0, 0.0])
    b = _embedded_node("b", [0.99, 0.02])

    builder = _builder()
    builder.client.chat_completion.return_value = MagicMock(
        choices=[
            MagicMock(
                message=MagicMock(
                    content='{"relationship": "ELABORATES", "direction": "A_TO_B", "confidence": 0.4, "reason": "weak"}'
                )
            )
        ]
    )

    edges = builder._build_llm_edges([a, b])

    assert edges == []
