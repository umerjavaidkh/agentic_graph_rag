"""
tests/test_axis2_acronym_canonicalization_unit.py — an abbreviation and its
expansion ("u.s." / "united states") are one entity, not two.

Regression: found via sampled LLM-judge audit of a real oil-company 10-K.
Two of the three flagged SHARES_ENTITY edges were anchored on what is
actually the SAME real-world entity under two surface forms:

    edge 1: shared=["u.s. (LOCATION)"]
    edge 2: shared=["united states (LOCATION)", "natural gas", "crude oil"]

Counting one entity as two splits its document frequency across both forms,
so each lands under the genericity cutoff that should have excluded it, and
each carries an inflated idf (rarity) weight -- pervasive vocabulary ends up
looking distinctive. This is the root cause beneath the symptom that
tests/test_axis2_anchor_distinctiveness_unit.py suppresses from the scoring
side.

_canonicalize_entities's existing fuzzy pass structurally cannot merge these:
it only compares strings within the same first-N-character prefix bucket, and
an acronym never shares a prefix with its expansion (nor would character
similarity rate "u.s." close to "united states" if it did). So this is a
separate, structural pass: exact initials match, same NER type only, with an
ambiguity guard.

Deliberately limited to orthographically-marked abbreviations (containing a
period). See _acronym_letters for why undotted forms like "sec"/"opec" are
NOT accepted: casing is gone by this stage, so a bare letter run is
indistinguishable from an ordinary short word, and accepting them merged
"oil" into "offshore installation license" on this very corpus.

Run with:
    python -m pytest tests/test_axis2_acronym_canonicalization_unit.py -v
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
    _acronym_letters,
    _canonicalize_entities,
    _expansion_initials,
    _resolve_canonical_entities,
)


def _builder() -> Axis2Builder:
    builder = Axis2Builder.__new__(Axis2Builder)
    builder.client = MagicMock()
    return builder


def _node(node_id: str, entities: list[str], types: dict[str, str]) -> DKGNode:
    n = DKGNode(id=node_id, type=NodeType.SECTION, title=node_id, text="x", order=0)
    n.entities = entities
    n.entity_types = types
    return n


# ── _acronym_letters ─────────────────────────────────────────────────────────


def test_dotted_abbreviation_recognized():
    assert _acronym_letters("u.s.") == "us"
    assert _acronym_letters("u.s.a.") == "usa"


def test_undotted_letter_run_not_treated_as_acronym():
    # The guard that prevents "oil" -> "offshore installation license".
    assert _acronym_letters("oil") is None
    assert _acronym_letters("sec") is None


def test_multi_word_text_is_never_an_acronym():
    assert _acronym_letters("united states") is None


def test_overlong_dotted_string_rejected():
    assert _acronym_letters("a.b.c.d.e.f.g.") is None


# ── _expansion_initials ──────────────────────────────────────────────────────


def test_initials_include_both_conventions():
    # "securities and exchange commission" -> "saec" (all words) and
    # "sec" (skipping the conjunction); real acronyms use either.
    assert _expansion_initials("securities and exchange commission") == {"saec", "sec"}


def test_initials_empty_for_single_word():
    assert _expansion_initials("kazakhstan") == set()


# ── merging behavior ─────────────────────────────────────────────────────────


def test_abbreviation_merges_into_expansion():
    """Direct reproduction of the flagged pair: "u.s." and "united states"
    must resolve to one canonical entity, and the fuller name must be the
    representative (more informative when read back in shared_entities)."""
    types = {"u.s.": "LOCATION", "united states": "LOCATION"}
    result = _canonicalize_entities(set(types), types)
    assert result["u.s."] == result["united states"]
    assert result["u.s."] == "united states (LOCATION)"


def test_abbreviation_merges_across_skip_word_convention():
    types = {"u.s.a.": "LOCATION", "united states of america": "LOCATION"}
    result = _canonicalize_entities(set(types), types)
    assert result["u.s.a."] == result["united states of america"]


def test_ordinary_short_word_not_merged_into_initials_match():
    """The false merge that forced the dotted-only rule: "oil" matches the
    initials of "offshore installation license", and merging them would
    destroy one of an oil filing's most important entities."""
    types = {"oil": "PRODUCT", "offshore installation license": "PRODUCT"}
    result = _canonicalize_entities(set(types), types)
    assert result["oil"] != result["offshore installation license"]


def test_abbreviation_not_merged_across_types():
    """Same 'never merge across NER type' invariant the rest of
    canonicalization already guarantees -- an acronym match is not a reason
    to breach it."""
    types = {"u.s.": "LOCATION", "united states": "ORG"}
    result = _canonicalize_entities(set(types), types)
    assert result["u.s."] != result["united states"]


def test_ambiguous_abbreviation_left_alone():
    """An acronym matching two distinct expansion clusters has no single
    correct answer -- it must stay unmerged rather than be attached to an
    arbitrary one."""
    types = {
        "u.s.": "LOCATION",
        "united states": "LOCATION",
        "united section": "LOCATION",
    }
    result = _canonicalize_entities(set(types), types)
    assert result["u.s."] != result["united states"]
    assert result["u.s."] != result["united section"]


def test_variants_of_one_expansion_are_not_ambiguity():
    """Several surface forms of the SAME expansion already cluster together
    via the fuzzy pass -- that is one cluster, not competing candidates, so
    the ambiguity guard must not be tripped by it."""
    types = {
        "u.s.": "LOCATION",
        "united states": "LOCATION",
        "united states'": "LOCATION",  # possessive variant, same expansion
    }
    result = _canonicalize_entities(set(types), types)
    assert result["u.s."] == result["united states"]


def test_unrelated_entities_unaffected():
    types = {"u.s.": "LOCATION", "united states": "LOCATION", "kazakhstan": "LOCATION"}
    result = _canonicalize_entities(set(types), types)
    assert result["kazakhstan"] == "kazakhstan (LOCATION)"


# ── end-to-end through _resolve_canonical_entities / edge building ──────────


def test_resolve_canonical_entities_unifies_abbreviation_across_nodes():
    """The split this fix exists to remove: one node says "u.s.", another
    says "united states". Both occurrences must resolve to the same
    canonical key, so document frequency is counted once rather than twice."""
    a = _node("a", ["u.s."], {"u.s.": "LOCATION"})
    b = _node("b", ["united states"], {"united states": "LOCATION"})

    canonical = _resolve_canonical_entities([a, b])

    assert canonical[("a", "u.s.")] == canonical[("b", "united states")]


def test_split_abbreviation_no_longer_evades_the_genericity_filter():
    """The measurable consequence, end to end: a term pervasive enough to be
    generic, but written as "u.s." in half the document and "united states"
    in the other half, previously had its document frequency split so each
    form sat under the genericity cutoff and could still anchor edges. Once
    merged, the combined frequency is correctly seen as generic and anchors
    nothing -- while a genuinely specific entity still does."""
    nodes = []
    for i in range(10):
        surface = "u.s." if i % 2 == 0 else "united states"
        nodes.append(_node(f"n{i}", [surface], {surface: "LOCATION"}))
    # one pair additionally shares a genuinely specific entity
    for i in (0, 1):
        nodes[i].entities.append("tengiz expansion")
        nodes[i].entity_types["tengiz expansion"] = "CONCEPT"

    edges = _builder()._build_entity_edges(nodes)

    for e in edges:
        for shared in e.properties["shared_entities"]:
            assert "united states" not in shared
            assert "u.s." not in shared
    assert any(
        any("tengiz expansion" in s for s in e.properties["shared_entities"])
        for e in edges
    )
