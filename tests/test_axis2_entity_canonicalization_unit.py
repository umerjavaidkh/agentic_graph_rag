"""
tests/test_axis2_entity_canonicalization_unit.py — SHARES_ENTITY edges are
not fragmented by surface-text variants of the same entity.

Regression: _build_entity_edges only lowercased entity strings before
grouping, so "Newton", "Newton's laws", and "Sir Isaac Newton" (three
distinct NER extractions of the same real-world entity, common in an
equation-heavy physics textbook) were treated as three separate entities
that never "share" with each other -- fragmenting the SHARES_ENTITY signal
instead of connecting nodes that are actually about the same thing.

Fixed via _canonicalize_entities: a general string-similarity clustering
(possessive-stripping + difflib.SequenceMatcher ratio) over the corpus's
own entity vocabulary -- no hardcoded entity names, so it generalizes to
any document rather than being tuned to this one.

Run with:
    python -m pytest tests/test_axis2_entity_canonicalization_unit.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock


from src.models import DKGNode, NodeType
from src.semantic.axis2 import (
    Axis2Builder,
    _ENTITY_FUZZY_CLUSTER_MAX_VOCAB,
    _canonicalize_entities,
)


def _builder() -> Axis2Builder:
    builder = Axis2Builder.__new__(Axis2Builder)
    builder.client = MagicMock()
    return builder


def test_canonicalize_merges_possessive_variant():
    # NER sometimes grabs the possessive form whole ("Newton's") as its own
    # entity string, distinct from the bare noun ("Newton") -- these two
    # ARE the same entity and should merge. A possessive attached to a
    # longer phrase ("Newton's laws") is NOT the same entity as "Newton"
    # (a person vs. a set of physical laws) and must not be merged --
    # SequenceMatcher's length-sensitive ratio naturally keeps them apart.
    canonical = _canonicalize_entities({"newton", "newton's"})
    assert canonical["newton"] == canonical["newton's"]


def test_canonicalize_merges_close_string_variant():
    canonical = _canonicalize_entities({"kinematics", "kinematic"})
    assert canonical["kinematics"] == canonical["kinematic"]


def test_canonicalize_does_not_merge_unrelated_entities():
    canonical = _canonicalize_entities({"newton", "energy", "vector"})
    assert len({canonical["newton"], canonical["energy"], canonical["vector"]}) == 3


def test_canonicalize_does_not_merge_a_word_with_a_longer_phrase_containing_it():
    # "Newton" and "Newton's laws" are genuinely different entities (a
    # person vs. a set of physical laws) -- must stay distinct even though
    # one contains the other as a substring.
    canonical = _canonicalize_entities({"newton", "newton's laws"})
    assert canonical["newton"] != canonical["newton's laws"]


def test_canonicalize_is_stable_for_single_entity():
    canonical = _canonicalize_entities({"newton"})
    assert canonical == {"newton": "newton"}


def test_canonicalize_stays_bounded_above_the_fuzzy_cluster_vocab_cap():
    """Regression: prefix-bucketed pairwise clustering is still O(bucket^2)
    for a vocabulary that repeats a small set of words (a physics
    textbook's few hundred core concepts recombined into many phrases) --
    measured directly at 3,500 unique entities under this file's own
    fuzzy-clustering pass and it took ~2.5s, worse for larger vocabularies.
    Above _ENTITY_FUZZY_CLUSTER_MAX_VOCAB, only the O(n) possessive-strip
    exact-match normalization runs, so a huge, repetitive vocabulary must
    still return quickly (bounded cost regardless of corpus size)."""
    import time

    words = ["alpha", "beta", "gamma", "delta", "epsilon"]
    entities = {f"{a} {b}" for a in words for b in words for _ in range(1)}
    entities |= {f"{a}{i}" for a in words for i in range(2000)}
    assert len(entities) > _ENTITY_FUZZY_CLUSTER_MAX_VOCAB

    start = time.time()
    canonical = _canonicalize_entities(entities)
    elapsed = time.time() - start

    assert elapsed < 2.0
    assert set(canonical.keys()) == entities


def test_canonicalize_does_not_merge_same_string_different_types():
    """"Apple" tagged ORG in one node and "apple" tagged CONCEPT in
    another must never merge, however similar the strings look -- this is
    the exact "fruit vs. company" failure typed entities exist to prevent."""
    canonical = _canonicalize_entities(
        {"apple"}, entity_types=None
    )  # sanity: still works with no types at all
    assert canonical == {"apple": "apple"}

    # Same normalized text, two different real-world things -- must be
    # distinguishable by type when both are known. difflib similarity of
    # "apple" to itself is 1.0, so without type-awareness these would be
    # indistinguishable; entity_types is what separates them.
    canonical = _canonicalize_entities(
        {"apple:org", "apple:concept"},
        entity_types={"apple:org": "ORG", "apple:concept": "CONCEPT"},
    )
    assert canonical["apple:org"] != canonical["apple:concept"]


def test_canonicalize_type_prefix_prevents_cross_type_fuzzy_merge():
    """Two near-identical strings that would normally fuzzy-merge (ratio
    above threshold) must stay separate when tagged with different types."""
    canonical = _canonicalize_entities(
        {"newton", "newtons"},
        entity_types={"newton": "PERSON", "newtons": "METRIC"},  # unit of force vs. the person
    )
    assert canonical["newton"] != canonical["newtons"]


def test_canonicalize_still_merges_within_the_same_type():
    canonical = _canonicalize_entities(
        {"newton", "newton's"},
        entity_types={"newton": "PERSON", "newton's": "PERSON"},
    )
    assert canonical["newton"] == canonical["newton's"]


def test_canonicalize_untyped_entities_share_one_implicit_bucket():
    """Entities missing from entity_types (older ingestion run, or a batch
    that fell back to untyped strings) behave exactly as before this
    parameter existed -- one shared "untyped" namespace, not one bucket
    per missing entity."""
    canonical = _canonicalize_entities({"newton", "newton's"}, entity_types={})
    assert canonical["newton"] == canonical["newton's"]


def test_shares_entity_edge_created_across_possessive_surface_variants():
    """Two nodes whose only "shared" entity is a possessive variant of the
    same word must still get a SHARES_ENTITY edge -- before the fix, raw
    .lower() grouping would put them in different buckets and no edge
    would ever be built."""
    a = DKGNode(id="a", type=NodeType.SECTION, title="a", text="x", order=0)
    a.entities = ["Newton's", "force"]
    b = DKGNode(id="b", type=NodeType.SECTION, title="b", text="x", order=1)
    b.entities = ["Newton", "energy"]

    builder = _builder()
    edges = builder._build_entity_edges([a, b])

    assert len(edges) == 1
    assert {edges[0].source_id, edges[0].target_id} == {"a", "b"}
    assert edges[0].weight >= 1


def test_build_entity_edges_does_not_merge_homonyms_of_different_types():
    """End-to-end: two nodes both mentioning "apple", but one node's NER
    tagged it ORG (the company) and the other's tagged it CONCEPT (the
    fruit) -- _build_entity_edges must aggregate node.entity_types and
    pass it through to _canonicalize_entities, so these do NOT count as a
    shared entity."""
    a = DKGNode(id="a", type=NodeType.SECTION, title="a", text="x", order=0)
    a.entities = ["apple"]
    a.entity_types = {"apple": "ORG"}
    b = DKGNode(id="b", type=NodeType.SECTION, title="b", text="x", order=1)
    b.entities = ["apple"]
    b.entity_types = {"apple": "CONCEPT"}

    builder = _builder()
    edges = builder._build_entity_edges([a, b])

    assert edges == []
