"""
tests/test_axis2_edge_cap_unit.py — Axis2's per-node edge cap.

Regression: a 7,165-node document (a physics textbook, ingested to
stress-test the pipeline against something structurally unlike the SEC
filings the corpus was built around) produced 2.17M semantic edges (~303
per node) instead of the intended max-20-per-node cap. Root cause was in
two places:

- _build_similarity_edges created an edge for every pair above a flat
  cosine threshold, with no per-node bound at all -- O(n^2) edge count.
- _build_category_edges (SAME_CATEGORY) capped cluster COUNT at 10
  regardless of corpus size, so cluster SIZE (and thus fully-connected
  intra-cluster pairs) grew quadratically with the corpus instead of
  staying flat -- the dominant contributor (10 clusters of ~716 members
  each -> C(716,2) * 10 ≈ 2.56M, verified live).
- _build_entity_edges (SHARES_ENTITY) had the identical unbounded-pairwise
  shape (every pair with any shared entity got an edge).

All three are fixed the same way: cap each node to its top-k neighbors
(by cosine similarity, or by shared-entity count) instead of every
qualifying pair. These tests build enough near-duplicate nodes to trigger
the old O(n^2) blowup and assert the per-node degree cap actually holds.

Run with:
    python -m pytest tests/test_axis2_edge_cap_unit.py -v
"""
from __future__ import annotations

import sys
import types
from collections import Counter
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest


# hdbscan isn't installed in this dev env -- stub it so
# _build_category_edges can be exercised without a real install, same
# style as this module's other stubs.
if "hdbscan" not in sys.modules:
    sys.modules["hdbscan"] = types.ModuleType("hdbscan")


class _FakeHDBSCAN:
    """Splits into ~equal-size clusters (not all-one-cluster, and no -1
    noise) so intra-cluster pair counts actually scale the way production
    clustering does. Cluster count derives from len(vecs) with the exact
    formula axis2.py used to pass explicitly to KMeans (max 2, min 10,
    sqrt(n)) -- not because HDBSCAN takes a cluster-count parameter (it
    doesn't), but so this fake behaves identically regardless of which
    test FILE's stub of the shared sys.modules["hdbscan"] happens to be
    active for a given test (axis2._build_category_edges does a fresh
    `import hdbscan` per call, so whichever stub was registered last
    during pytest's collection wins for every test in the session, not
    just this file's own). A fixed constant here previously produced
    3 singleton "clusters" for a 3-node test in a different file
    expecting one shared cluster -- this formula matches what that file's
    own fake already assumes, so the two are interchangeable."""

    def __init__(self, min_cluster_size=5, metric="euclidean"):
        self.min_cluster_size = min_cluster_size

    def fit_predict(self, vecs):
        n = len(vecs)
        n_clusters = max(2, min(10, int(n ** 0.5)))
        return [i % n_clusters for i in range(n)]


sys.modules["hdbscan"].HDBSCAN = _FakeHDBSCAN

from src.shared.config.settings import AXIS2_MAX_SIMILARITY_EDGES_PER_NODE
from src.models import DKGNode, NodeType
from src.semantic.axis2 import Axis2Builder


def _builder() -> Axis2Builder:
    builder = Axis2Builder.__new__(Axis2Builder)
    builder.client = MagicMock()
    return builder


def _near_duplicate_nodes(n: int, dim: int = 32, seed: int = 42) -> list[DKGNode]:
    """n nodes whose embeddings are all near-identical (tiny random noise
    around a fixed vector) -- guaranteed to clear SIMILARITY_THRESHOLD
    (0.75) for every pair, reproducing the old builders' worst case."""
    rng = np.random.default_rng(seed)
    base = rng.normal(size=dim)
    out = []
    for i in range(n):
        vec = base + rng.normal(scale=0.01, size=dim)
        node = DKGNode(id=f"n{i}", type=NodeType.SECTION, title=f"n{i}", text="x", order=i)
        node.embedding = vec.tolist()
        node.entities = []
        out.append(node)
    return out


def _degree_counts(edges) -> Counter:
    deg: Counter = Counter()
    for e in edges:
        deg[e.source_id] += 1
        deg[e.target_id] += 1
    return deg


@pytest.mark.parametrize("n", [50, 300])
def test_similarity_edges_are_capped_per_node(n):
    nodes = _near_duplicate_nodes(n)
    builder = _builder()
    edges = builder._build_similarity_edges(nodes)

    assert len(edges) > 0
    deg = _degree_counts(edges)
    assert max(deg.values()) <= AXIS2_MAX_SIMILARITY_EDGES_PER_NODE
    # The old code produced C(n, 2) edges here (every pair, all near-
    # duplicates) -- confirm the fix is actually bounding count, not just
    # coincidentally under the naive n^2 figure for this small n.
    naive_full_graph = n * (n - 1) // 2
    assert len(edges) < naive_full_graph


def test_same_category_edges_are_capped_per_node():
    # 300 nodes, 10 clusters -> 30/cluster with the fake HDBSCAN above.
    # Old code: C(30, 2) * 10 = 4,350 edges, degree up to 29 per node.
    # New code: capped to AXIS2_MAX_SIMILARITY_EDGES_PER_NODE per node.
    nodes = _near_duplicate_nodes(300)
    builder = _builder()
    _, edges = builder._build_category_edges(nodes)

    assert len(edges) > 0
    deg = _degree_counts(edges)
    assert max(deg.values()) <= AXIS2_MAX_SIMILARITY_EDGES_PER_NODE


def test_entity_edges_are_capped_per_node():
    # A subset of nodes (30, comfortably above the degree cap of 20) shares
    # the same entity -- old code wired up a full clique over that subset;
    # new code must cap per-node degree. Deliberately a MINORITY of the
    # 100 entity-bearing nodes (30%), not all of them: were every node to
    # share it, that's indistinguishable from the entity being generic to
    # the document (see test_axis2_generic_entity_filter_unit.py), and the
    # genericity filter would correctly drop it before capping ever
    # applies -- this test isolates degree-capping from that behavior by
    # giving the remaining 70 nodes each their own distinct entity, so
    # they still count toward the document-frequency denominator without
    # sharing "newton"/"force".
    n = 100
    shared_count = 30
    nodes = []
    for i in range(n):
        node = DKGNode(id=f"n{i}", type=NodeType.SECTION, title=f"n{i}", text="x", order=i)
        node.embedding = None
        node.entities = ["newton", "force"] if i < shared_count else [f"unique_entity_{i}"]
        nodes.append(node)

    builder = _builder()
    edges = builder._build_entity_edges(nodes)

    assert len(edges) > 0
    deg = _degree_counts(edges)
    assert max(deg.values()) <= AXIS2_MAX_SIMILARITY_EDGES_PER_NODE
    naive_full_graph = shared_count * (shared_count - 1) // 2
    assert len(edges) < naive_full_graph


def test_large_document_does_not_blow_up_like_the_physics_textbook():
    """End-to-end shape of the regression: a document large enough that the
    old code's quadratic terms would dominate. Total edges from all three
    cheap builders combined must stay in the O(n*k) ballpark, not O(n^2)."""
    n = 1000
    nodes = _near_duplicate_nodes(n)
    for node in nodes:
        node.entities = ["newton", "force"]

    builder = _builder()
    sim_edges = builder._build_similarity_edges(nodes)
    entity_edges = builder._build_entity_edges(nodes)
    _, category_edges = builder._build_category_edges(nodes)
    total = len(sim_edges) + len(entity_edges) + len(category_edges)

    # O(n*k) bound with slack for the union of three independent builders
    # (each capped at k, but they can pick different neighbor sets).
    k = AXIS2_MAX_SIMILARITY_EDGES_PER_NODE
    assert total <= n * k * 3
    # The actual regression: old code gave ~n^2/2 = 500,000 for n=1000.
    assert total < (n * n) // 10
