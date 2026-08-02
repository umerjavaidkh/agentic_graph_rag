"""
tests/test_axis2_review_improvements_unit.py — axis2.py Axis-2 quality
improvements from an external design review.

Covers three changes to Axis2Builder, all self-contained inside
src/semantic/axis2.py:
  1. SHARES_ENTITY weight is IDF-rarity-weighted, not a flat shared-entity
     count -- a rarer shared entity produces a higher edge weight.
  2. SAME_CATEGORY clusters on entity co-occurrence (a signal distinct from
     SEMANTICALLY_SIMILAR's raw embeddings) when enough nodes carry
     entities, falling back to embeddings otherwise -- properties["signal"]
     records which path was taken.
  3. The LLM reasoning pass (_build_llm_edges) reserves part of its
     candidate budget for "entity-bridge" pairs (tied by a shared rare
     entity but not already embedding-similar), not just top-cosine pairs.

Run with:
    python -m pytest tests/test_axis2_review_improvements_unit.py -v
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

# sklearn.cluster.KMeans has a real binary-incompatibility issue in this
# dev env (numpy 2.x vs a wheel built for 1.x) -- stub it, same style as
# the other axis2 test files.
if "sklearn" not in sys.modules:
    sys.modules["sklearn"] = types.ModuleType("sklearn")
if "sklearn.cluster" not in sys.modules:
    sys.modules["sklearn.cluster"] = types.ModuleType("sklearn.cluster")


class _FakeKMeans:
    """Deterministic index-modulo split -- ignores vector content, so these
    tests verify ROUTING (which signal/vector-source was used, and whether
    a candidate pair was ever proposed to the LLM), not clustering quality
    itself (already covered by test_axis2_edge_cap_unit.py)."""

    def __init__(self, n_clusters, random_state=None, n_init="auto"):
        self.n_clusters = n_clusters

    def fit_predict(self, vecs):
        n = len(vecs)
        return [i % self.n_clusters for i in range(n)]


sys.modules["sklearn.cluster"].KMeans = _FakeKMeans

from src.models import DKGEdge, DKGNode, NodeType, RelType
from src.semantic.axis2 import Axis2Builder


def _builder() -> Axis2Builder:
    builder = Axis2Builder.__new__(Axis2Builder)
    builder.client = MagicMock()
    return builder


def _node(node_id: str, *, entities=None, embedding=None) -> DKGNode:
    n = DKGNode(id=node_id, type=NodeType.SECTION, title=node_id, text="x", order=0)
    n.entities = entities or []
    n.embedding = embedding
    return n


# ── 1. SHARES_ENTITY rarity weighting ────────────────────────────────────


def test_shares_entity_weight_is_higher_for_rarer_entity():
    """10 entity-bearing nodes (above the genericity-filter's 5-node
    activation floor, max_df = int(10*0.4) = 4). "rare" is shared by only
    2 nodes; "medium" by 4 -- both stay under the informativeness cutoff,
    but "rare" must produce a strictly higher edge weight."""
    a, b = _node("a", entities=["rare"]), _node("b", entities=["rare"])
    c, d = _node("c", entities=["medium"]), _node("d", entities=["medium"])
    e, f = _node("e", entities=["medium"]), _node("f", entities=["medium"])
    padding = [_node(f"p{i}", entities=[f"unique_{i}"]) for i in range(4)]

    builder = _builder()
    edges = builder._build_entity_edges([a, b, c, d, e, f, *padding])

    rare_edge = next(edge for edge in edges if {edge.source_id, edge.target_id} == {"a", "b"})
    medium_edge = next(edge for edge in edges if {edge.source_id, edge.target_id} == {"c", "d"})

    assert rare_edge.weight > medium_edge.weight
    assert rare_edge.properties["rarity_score"] == rare_edge.weight


def test_shares_entity_weight_stays_at_least_one_for_single_shared_entity():
    """Existing invariant from test_axis2_entity_canonicalization_unit.py
    (weight >= 1 for one shared entity) must still hold -- the smoothed
    idf formula is chosen specifically so it never drops below the old
    flat-count floor."""
    a = _node("a", entities=["newton's", "force"])
    b = _node("b", entities=["newton", "energy"])

    builder = _builder()
    edges = builder._build_entity_edges([a, b])

    assert len(edges) == 1
    assert edges[0].weight >= 1


# ── 2. SAME_CATEGORY entity-cooccurrence signal ──────────────────────────


def test_same_category_uses_entity_signal_when_entities_available():
    nodes = [
        _node(f"n{i}", entities=[f"topic_{i % 2}"], embedding=[float(i), 0.0, 0.0])
        for i in range(6)
    ]
    builder = _builder()
    _, edges = builder._build_category_edges(nodes)

    assert edges
    assert all(edge.properties["signal"] == "entity_cooccurrence" for edge in edges)


def test_same_category_falls_back_to_embedding_signal_without_entities():
    nodes = [_node(f"n{i}", embedding=[float(i), 0.0, 0.0]) for i in range(6)]
    builder = _builder()
    _, edges = builder._build_category_edges(nodes)

    assert edges
    assert all(edge.properties["signal"] == "embedding" for edge in edges)


# ── 3. LLM pass entity-bridge candidate pool ─────────────────────────────


def test_llm_pass_includes_entity_bridge_pair_below_similarity_threshold():
    """a/b are embedding-similar (clears CONTRADICTION_THRESH on their
    own); c/d are embedding-DISSIMILAR (orthogonal vectors, far below
    threshold) but linked by a SHARES_ENTITY edge -- without the bridge
    pool, c/d would never be proposed to the LLM at all."""
    a = _node("a", embedding=[1.0, 0.0])
    b = _node("b", embedding=[0.99, 0.02])   # cosine ~0.999, clears 0.85
    c = _node("c", embedding=[0.0, 1.0])
    d = _node("d", embedding=[0.0, -1.0])    # cosine -1.0 with c, far below threshold

    bridge_edge = DKGEdge(
        source_id="c", target_id="d", rel_type=RelType.SHARES_ENTITY,
        weight=2.0, axis=2, properties={"shared_entities": ["rare_thing"]},
    )

    builder = _builder()
    builder.client.chat_completion.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(
            content='{"relationship":"NONE","direction":"A_TO_B","confidence":0.0,"reason":""}'
        ))]
    )

    builder._build_llm_edges([a, b, c, d], entity_edges=[bridge_edge])

    call_count = builder.client.chat_completion.call_count
    assert call_count == 2, f"expected both the similarity pair and the bridge pair to be sent, got {call_count}"

    prompts = [call.kwargs["messages"][0]["content"] for call in builder.client.chat_completion.call_args_list]
    assert any("(c):" in p and "(d):" in p for p in prompts), "bridge pair c/d was never proposed to the LLM"


def test_llm_pass_without_entity_edges_behaves_as_before():
    """entity_edges defaults to None -- existing call sites (and the
    pre-existing test_llm_pair_cap_limits_candidates) must be unaffected."""
    a = _node("a", embedding=[1.0, 0.0])
    b = _node("b", embedding=[0.99, 0.02])

    builder = _builder()
    builder.client.chat_completion.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(
            content='{"relationship":"NONE","direction":"A_TO_B","confidence":0.0,"reason":""}'
        ))]
    )

    edges = builder._build_llm_edges([a, b])
    assert builder.client.chat_completion.call_count == 1
    assert edges == []
