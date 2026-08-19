"""
tests/test_axis2_anchor_distinctiveness_unit.py — SHARES_ENTITY edges are
scored by their strongest anchor (not the sum of every shared entity), and
must be anchored on at least one entity that is genuinely distinctive within
this document (not merely under the coarse 40%-document-frequency gate).

Regression: found via sampled LLM-judge audit of a real oil-company 10-K
(scripts/validate_ontology_accuracy.py). Axis-2 idea-linking scored 66.7%
(edge precision 0.40, entity grounding 0.93 -- edge precision was the entire
drag). Every judge-flagged invalid edge was a SHARES_ENTITY edge anchored on
pervasive domain vocabulary that passed the genericity gate but says nothing
specific about the two sections:

  * shared=["u.s. (LOCATION)"]                          rarity 2.70
  * shared=["united states","natural gas","crude oil"]  rarity 9.09
  * shared=["kazakhstan","hess","venezuela"]            rarity 10.50

Two root causes, two universal fixes (see src/semantic/axis2.py):

  Fix A (_ANCHOR_SECONDARY_WEIGHT): the old score summed idf across every
  shared entity, so a laundry list of three medium-frequency terms (idf ~3
  each -> sum ~9) tied or beat a single genuinely rare shared entity (idf
  ~9), the strongest possible signal. Now the strongest anchor dominates and
  each additional shared entity earns only fractional credit -- "one strong"
  beats "many weak", not the reverse.

  Fix B (_ANCHOR_DISTINCTIVENESS_PERCENTILE): the 40% document-frequency gate
  was the ONLY distinctiveness test and is a coarse binary cutoff -- "u.s."
  at ~18% document frequency sailed through. An edge is now admitted only if
  its best anchor's idf clears a per-document percentile (the bottom
  quartile) over the idf distribution of entities that can actually co-occur
  (df >= 2), which self-calibrates per document with no tuned constant. The
  quartile was chosen by measuring precision AND recall across a percentile
  sweep -- see the constant's comment for the numbers.

Run with:
    python -m pytest tests/test_axis2_anchor_distinctiveness_unit.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock


from src.models import DKGNode, NodeType
from src.semantic.axis2 import (
    Axis2Builder,
    _ANCHOR_DISTINCTIVENESS_MIN_ENTITIES,
    _ANCHOR_SECONDARY_WEIGHT,
)


def _builder() -> Axis2Builder:
    builder = Axis2Builder.__new__(Axis2Builder)
    builder.client = MagicMock()
    return builder


def _node(node_id: str, entities: list[str], types: dict[str, str] | None = None) -> DKGNode:
    n = DKGNode(id=node_id, type=NodeType.SECTION, title=node_id, text="x", order=0)
    n.entities = entities
    if types:
        n.entity_types = types
    return n


def _edge_between(edges, a: str, b: str):
    for e in edges:
        if {e.source_id, e.target_id} == {a, b}:
            return e
    return None


# ── Fix A: max-anchor scoring, not sum-of-idf ────────────────────────────────


def test_edge_weight_is_best_anchor_plus_fractional_rest_not_sum():
    """Two nodes sharing three identical entities (each idf 1.0 at N=2) must
    score max + 0.25*rest = 1.0 + 0.25*(1.0+1.0) = 1.5 -- NOT the old sum of
    3.0. Untyped entities so the enumeration cap leaves all three in play,
    isolating the scoring change from the type cap."""
    a = _node("a", ["x1", "x2", "x3"])
    b = _node("b", ["x1", "x2", "x3"])

    edges = _builder()._build_entity_edges([a, b])

    assert len(edges) == 1
    expected = 1.0 + _ANCHOR_SECONDARY_WEIGHT * (1.0 + 1.0)
    assert edges[0].weight == round(expected, 4)
    assert edges[0].weight < 3.0  # the old sum-of-idf value


def test_single_rare_entity_outranks_more_generic_laundry_list():
    """The core inversion the audit exposed: a single genuinely rare shared
    entity must outweigh a pair sharing three more-common terms. Under the
    old sum-of-idf this was reversed (3 * idf beat 1 * higher-idf); under
    max-anchor scoring the rarer single anchor wins.

    20 nodes so document frequencies are meaningful. Only four entities
    co-occur (< the floor's activation minimum), so Fix B's distinctiveness
    floor stays off and this isolates Fix A's ranking."""
    nodes: list[DKGNode] = []
    # rare: shared by exactly 2 nodes (idf highest)
    nodes.append(_node("rare_a", ["rareterm"]))
    nodes.append(_node("rare_b", ["rareterm"]))
    # three medium terms each shared by a broad set of nodes (lower idf)
    g_nodes = [_node(f"g{i}", ["g1", "g2", "g3"]) for i in range(10)]
    nodes.extend(g_nodes)
    # padding singletons to push the medium terms' document frequency down
    nodes.extend(_node(f"p{i}", [f"unique_{i}"]) for i in range(8))

    edges = _builder()._build_entity_edges(nodes)

    rare_edge = _edge_between(edges, "rare_a", "rare_b")
    laundry_edge = _edge_between(edges, "g0", "g1")
    assert rare_edge is not None
    assert laundry_edge is not None
    assert rare_edge.weight > laundry_edge.weight


# ── Fix B: adaptive distinctiveness floor ────────────────────────────────────


def _floored_document() -> list[DKGNode]:
    """14 entity-bearing nodes with 7 co-occurring entities (>= the floor's
    activation minimum): one pervasive generic term "us" (df=6, the lowest
    idf) plus six rare terms (df=2 each, all higher idf). "us" is the sole
    occupant of the bottom of the idf distribution, so the quartile floor
    lands above it and rejects any edge whose best anchor is "us"."""
    assert _ANCHOR_DISTINCTIVENESS_MIN_ENTITIES <= 7
    nodes = [_node(f"n{i}", []) for i in range(14)]
    for i in range(6):  # "us" pervades the first six nodes -> generic
        nodes[i].entities.append("us")
    # six genuinely rare terms, each in its own disjoint pair of nodes
    rare_pairs = {
        "alpha": (0, 1), "beta": (2, 3), "gamma": (4, 5),
        "delta": (6, 7), "epsilon": (8, 9), "zeta": (10, 11),
    }
    for term, (x, y) in rare_pairs.items():
        nodes[x].entities.append(term)
        nodes[y].entities.append(term)
    # a couple more entity-bearing nodes so N counts them, no new co-occurrence
    nodes[12].entities.append("solo_1")
    nodes[13].entities.append("solo_2")
    return nodes


def test_generic_only_pair_dropped_by_distinctiveness_floor():
    """A pair whose ONLY shared entity is the document's most generic term
    ("us", well below the median co-occurring idf) must not form an edge --
    even though "us" passes the 40% genericity gate. Reproduces the flagged
    shared=["u.s. (LOCATION)"] edge."""
    nodes = _floored_document()
    # nodes 0 and 2 share only "us" (0 has us+alpha; 2 has us+beta)
    edges = _builder()._build_entity_edges(nodes)

    generic_only = _edge_between(edges, "n0", "n2")
    assert generic_only is None
    # and no surviving edge is anchored solely on "us"
    for e in edges:
        assert e.properties["shared_entities"] != ["us"]


def test_pair_sharing_a_rare_entity_survives_the_floor():
    """Fix B must not gut recall: a pair sharing a genuinely rare entity
    (above the median) still forms an edge."""
    nodes = _floored_document()
    edges = _builder()._build_entity_edges(nodes)

    rare_edge = _edge_between(edges, "n8", "n9")  # share "epsilon" (df=2)
    assert rare_edge is not None
    assert "epsilon" in rare_edge.properties["shared_entities"]


def test_laundry_list_of_generic_terms_dropped_even_though_sum_ranks_high():
    """Direct reproduction of shared=["united states","natural gas","crude
    oil"] / ["kazakhstan","hess","venezuela"]: a pair sharing several
    pervasive terms whose SUMMED idf would rank the edge highly must still be
    dropped, because none of them clears the distinctiveness floor. Combines
    Fix A (sum no longer inflates the score) and Fix B (best anchor below the
    floor -> no edge)."""
    nodes = _floored_document()
    # Make nodes 0 and 1 also co-mention three more pervasive terms, so their
    # ONLY strong-looking connection is a generic laundry list. All three are
    # spread across many nodes -> low idf, below the floor.
    for i in range(12):
        nodes[i].entities.extend(["united states", "natural gas", "crude oil"])
    for n in nodes:
        for e in n.entities:
            n.entity_types = getattr(n, "entity_types", {})
            n.entity_types.setdefault(
                e,
                "PRODUCT" if e in ("natural gas", "crude oil") else "LOCATION",
            )
    # n0 and n1 no longer share any rare term uniquely between just them...
    # remove "alpha" so their only overlap is the generic laundry list.
    nodes[0].entities.remove("alpha")
    nodes[1].entities.remove("alpha")

    edges = _builder()._build_entity_edges(nodes)

    n0_n1 = _edge_between(edges, "n0", "n1")
    assert n0_n1 is None


# ── floor stays off on small documents (no small-corpus regression) ──────────


def test_floor_removes_generic_edges_without_gutting_specific_ones():
    """Guards the precision/recall balance the percentile was chosen for (see
    _ANCHOR_DISTINCTIVENESS_PERCENTILE's measured sweep): on a corpus shaped
    like the 10-K this was found on -- pervasive domain vocabulary at ~18%
    document frequency plus a long tail of specific terms -- the floor must
    remove ALL generic-only-anchored edges while still retaining a
    substantial population of specific ones. A future percentile increase
    that quietly discards most legitimate edges fails here rather than only
    showing up as thin retrieval much later."""
    import random

    generic = ["united states", "natural gas", "crude oil", "u.s.", "hess"]
    # Multi-word, non-substring-colliding names: entity canonicalization
    # merges substring variants, so "term_1"/"term_11" would collapse into
    # one artificially-frequent entity and mask the behavior under test.
    heads = ["tengiz", "gorgon", "permian", "bakken", "guyana", "stabroek"]
    tails = ["expansion", "impairment", "royalty", "turnaround", "appraisal"]
    rare = [f"{h} {t}" for h in heads for t in tails]

    random.seed(7)
    nodes = []
    for i in range(220):
        ents = [g for g in generic if random.random() < 0.18]
        ents += random.sample(rare, random.choice([1, 1, 2, 3]))
        types = {
            e: (("PRODUCT" if e in ("natural gas", "crude oil")
                 else "ORG" if e == "hess" else "LOCATION")
                if e in generic else "CONCEPT")
            for e in ents
        }
        nodes.append(_node(f"n{i}", ents, types))

    edges = _builder()._build_entity_edges(nodes)

    from src.semantic.axis2 import _entity_base_text
    generic_only = [
        e for e in edges
        if all(_entity_base_text(s) in generic for s in e.properties["shared_entities"])
    ]
    assert generic_only == []
    # Recall guard: the measured sweep retained hundreds of specific edges at
    # this percentile. A large margin below that, so ordinary variation in
    # the builder doesn't make this brittle -- it only catches a collapse.
    assert len(edges) > 150


def test_floor_inactive_below_min_cooccurring_entities():
    """With fewer than _ANCHOR_DISTINCTIVENESS_MIN_ENTITIES co-occurring
    entities the idf distribution is too small to calibrate a floor, so it
    stays off and a generic-ish shared entity can still anchor an edge -- the
    same small-document permissiveness the genericity filter already keeps."""
    a = _node("a", ["shared_term"])
    b = _node("b", ["shared_term"])

    edges = _builder()._build_entity_edges([a, b])

    assert len(edges) == 1
    assert edges[0].properties["shared_entities"] == ["shared_term"]
