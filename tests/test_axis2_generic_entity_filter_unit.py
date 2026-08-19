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

Third regression: found via live verification of a real 264-page 10-K
(223 entity-bearing nodes). "Chevron" (the filer's own name, 57% raw
document frequency) is exactly what this filter exists to catch -- but
type-aware canonicalization (_resolve_canonical_entities) keys document
frequency by TYPE-SUFFIXED identity ("chevron (ORG)"), and the LLM's type
tag isn't perfectly consistent across the ~15 separate NER batch calls a
document this size requires. Chevron's true frequency (127/223) got
fragmented across type buckets, and its dominant "chevron (ORG)" bucket
landed at exactly 95/223 -- just under the adaptive cutoff of 95 -- while
the real combined entity anchored 716 of 1,846 SHARES_ENTITY edges
(38.8%) in the ingested graph. Fixed via _entity_base_text: genericity is
now judged on the entity's base text, aggregating node coverage across
every type variant it was tagged with, not on one type-suffixed key's own
count in isolation.

Run with:
    python -m pytest tests/test_axis2_generic_entity_filter_unit.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock


from src.models import DKGNode, NodeType
from src.semantic.axis2 import (
    Axis2Builder,
    _ENTITY_GENERICITY_DF_RATIO,
    _ENTITY_GENERICITY_DF_RATIO_CEILING,
    _ENTITY_GENERICITY_LARGE_CORPUS_NODES,
    _ENTITY_GENERICITY_MIN_NODES,
    _NON_TOPICAL_CONTINENT_NAMES,
    _NON_TOPICAL_ENTITY_PHRASES,
    _adaptive_genericity_ratio,
    _dedupe_enumeration_types,
    _entity_base_text,
    _entity_type,
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


# ── enumeration-type cap (multi-entity "laundry list" edges) ────────────────
# Regression: found via sampled LLM-judge audit of a real 10-K -- edges
# shared 3-4 LOCATION entities at once ("kazakhstan, u.s., canada, dj
# basin"), the signature of two independent boilerplate enumeration
# paragraphs ("the company operates in X, Y, Z...") rather than a genuine
# topical connection. These scored artificially HIGH (idf summed across
# every listed entity) despite being the judge's most consistently flagged
# failure mode, outranking genuinely specific single-entity edges.


def test_dedupe_enumeration_types_caps_same_type_at_two():
    shared = {"kazakhstan (LOCATION)", "u.s. (LOCATION)", "canada (LOCATION)", "dj basin (LOCATION)"}
    result = _dedupe_enumeration_types(shared)
    assert len(result) == 2
    assert result <= shared


def test_dedupe_enumeration_types_preserves_specific_entity_of_different_type():
    # A soft cap, not a wholesale exclusion -- the one specific, non-
    # repeated-type entity survives alongside the capped locations.
    shared = {"kazakhstan (LOCATION)", "u.s. (LOCATION)", "canada (LOCATION)", "tco (ORG)"}
    result = _dedupe_enumeration_types(shared)
    assert "tco (ORG)" in result
    assert sum(1 for e in result if "(LOCATION)" in e) == 2


def test_dedupe_enumeration_types_two_same_type_unaffected():
    shared = {"kazakhstan (LOCATION)", "canada (LOCATION)"}
    assert _dedupe_enumeration_types(shared) == shared


def test_dedupe_enumeration_types_untyped_entities_not_capped():
    # Legacy/untyped entities have no reliable type signal to group by --
    # left alone rather than risk incorrectly capping them.
    shared = {"alpha", "beta", "gamma", "delta"}
    assert _dedupe_enumeration_types(shared) == shared


def test_build_entity_edges_caps_laundry_list_location_edge():
    """Direct reproduction of the live 10-K bug: two sections sharing 4
    LOCATION entities at once must not get full credit for every one of
    them."""
    a = _node("a", ["kazakhstan", "u.s.", "canada", "dj basin"])
    b = _node("b", ["kazakhstan", "u.s.", "canada", "dj basin"])
    for n in (a, b):
        n.entity_types = {e: "LOCATION" for e in n.entities}

    builder = _builder()
    edges = builder._build_entity_edges([a, b])

    assert len(edges) == 1
    assert len(edges[0].properties["shared_entities"]) == 2  # capped from 4


def test_build_entity_edges_keeps_specific_entity_alongside_capped_locations():
    a = _node("a", ["kazakhstan", "u.s.", "canada", "tco"])
    b = _node("b", ["kazakhstan", "u.s.", "canada", "tco"])
    for n in (a, b):
        n.entity_types = {"kazakhstan": "LOCATION", "u.s.": "LOCATION", "canada": "LOCATION", "tco": "ORG"}

    builder = _builder()
    edges = builder._build_entity_edges([a, b])

    shared = edges[0].properties["shared_entities"]
    assert any("tco" in s.lower() for s in shared)
    assert sum(1 for s in shared if "location" in s.lower()) == 2


def test_build_entity_edges_weight_reflects_capped_entities_only():
    """The reported weight/rarity_score must match the FILTERED shared set,
    not the full uncapped one -- otherwise a laundry-list edge would still
    outrank a genuinely specific single-entity edge despite being capped."""
    a = _node("a", ["kazakhstan", "u.s.", "canada", "dj basin"])
    b = _node("b", ["kazakhstan", "u.s.", "canada", "dj basin"])
    for n in (a, b):
        n.entity_types = {e: "LOCATION" for e in n.entities}

    builder = _builder()
    edges = builder._build_entity_edges([a, b])
    # Both nodes share all 4 entities identically -> each entity's idf =
    # log((2+1)/(2+1)) + 1 = 1.0 exactly; capped to 2 entities. The score is
    # best-anchor-dominant (_ANCHOR_SECONDARY_WEIGHT), not a sum: max(idf) +
    # 0.25 * sum(rest) = 1.0 + 0.25 * 1.0 = 1.25 -- not 2.0 (summing the two
    # capped entities) and certainly not 4.0 (summing all four uncapped).
    assert edges[0].weight == 1.25


# ── _entity_base_text / type-fragmentation regression ───────────────────────


def test_entity_base_text_strips_known_type_suffix():
    assert _entity_base_text("chevron (ORG)") == "chevron"
    assert _entity_base_text("apple (CONCEPT)") == "apple"


def test_entity_base_text_leaves_untyped_key_unchanged():
    assert _entity_base_text("chevron") == "chevron"


def test_entity_base_text_does_not_strip_non_type_parenthetical():
    # A parenthetical that isn't one of the fixed NER types (e.g. a bare
    # year) must not be mistaken for a type suffix.
    assert _entity_base_text("fiscal year (2024)") == "fiscal year (2024)"


def test_type_fragmented_dominant_entity_is_still_excluded():
    """Direct reproduction of the live 10-K bug: an entity whose TRUE
    document frequency (127/223, 57%) is dominant gets tagged with two
    different types across separate NER batch calls, splitting it into
    "chevron (ORG)" (95 nodes) and "chevron (OTHER)" (32 nodes) -- each
    individually at or under the adaptive cutoff (max_df=95 at n=223), so
    the old per-type-key frequency check let the dominant "chevron (ORG)"
    bucket through. Both variants must now be excluded, since genericity
    is judged on their combined base-text frequency (127 > 95)."""
    total = 223
    org_nodes = list(range(95))
    other_nodes = list(range(95, 95 + 32))  # disjoint node set, same real entity
    entity_to_nodes = {
        "chevron (ORG)": org_nodes,
        "chevron (OTHER)": other_nodes,
        "specific finding (CONCEPT)": [0, 1],
    }
    result = _informative_entities(entity_to_nodes, total_entity_nodes=total)
    assert "chevron (ORG)" not in result
    assert "chevron (OTHER)" not in result
    assert "specific finding (CONCEPT)" in result


def test_type_consistent_dominant_entity_still_excluded_as_before():
    # Sanity check that the fix doesn't depend on fragmentation being
    # present -- a single, consistently-typed dominant entity must still
    # be excluded exactly as it was before this fix.
    total = 223
    entity_to_nodes = {
        "chevron (ORG)": list(range(127)),
        "specific finding (CONCEPT)": [0, 1],
    }
    result = _informative_entities(entity_to_nodes, total_entity_nodes=total)
    assert "chevron (ORG)" not in result
    assert "specific finding (CONCEPT)" in result


def test_build_entity_edges_excludes_type_fragmented_dominant_entity():
    """End-to-end: node.entity_types disagreeing across nodes for the same
    real-world entity must not let it anchor an edge. 6 nodes all mention
    "chevron", tagged ORG on 4 of them and OTHER on the other 2 -- close
    to this test's small n, so use a tight corpus where the flat/adaptive
    ratio at min-nodes-threshold scale still excludes a 100%-frequency
    term regardless of how it's split across two types."""
    nodes = []
    for i in range(4):
        n = DKGNode(id=f"org{i}", type=NodeType.SECTION, title=f"org{i}", text="x", order=i)
        n.entities = ["chevron"]
        n.entity_types = {"chevron": "ORG"}
        nodes.append(n)
    for i in range(2):
        n = DKGNode(id=f"other{i}", type=NodeType.SECTION, title=f"other{i}", text="x", order=4 + i)
        n.entities = ["chevron"]
        n.entity_types = {"chevron": "OTHER"}
        nodes.append(n)
    # One pair shares a genuinely rare, consistently-typed entity too.
    nodes[0].entities.append("rare finding")
    nodes[0].entity_types["rare finding"] = "CONCEPT"
    nodes[1].entities.append("rare finding")
    nodes[1].entity_types["rare finding"] = "CONCEPT"

    builder = _builder()
    edges = builder._build_entity_edges(nodes)

    for edge in edges:
        assert "chevron (ORG)" not in edge.properties["shared_entities"]
        assert "chevron (OTHER)" not in edge.properties["shared_entities"]
    assert any("rare finding (CONCEPT)" in e.properties["shared_entities"] for e in edges)


# ── large-corpus floor convergence regression ────────────────────────────────


def test_ratio_reaches_exact_floor_at_large_corpus_size():
    assert _adaptive_genericity_ratio(_ENTITY_GENERICITY_LARGE_CORPUS_NODES) == _ENTITY_GENERICITY_DF_RATIO


def test_ratio_below_large_corpus_threshold_still_has_asymptotic_excess():
    # Just under the hard-floor cutoff, the exponential's residual excess
    # over the floor should still be present (confirms the clamp is doing
    # something, not just always returning the floor).
    ratio = _adaptive_genericity_ratio(_ENTITY_GENERICITY_LARGE_CORPUS_NODES - 1)
    assert ratio > _ENTITY_GENERICITY_DF_RATIO


def test_dominant_term_at_real_document_scale_is_excluded():
    """Direct reproduction of the live 10-K bug's second cause: "Chevron"
    landed at 95/222 = 42.8% document frequency -- comfortably above the
    validated 40% floor, but inside the exponential curve's un-converged
    residual margin (~42.8% at n=222 before this fix), so it wasn't
    excluded. canonicalization correctly merged all its spelling variants
    into one key here (this is NOT the type-fragmentation case above) --
    the curve itself just hadn't reached the floor yet."""
    total = 222
    entity_to_nodes = {
        "chevron (ORG)": list(range(95)),
        "specific finding (CONCEPT)": [0, 1],
    }
    result = _informative_entities(entity_to_nodes, total_entity_nodes=total)
    assert "chevron (ORG)" not in result
    assert "specific finding (CONCEPT)" in result


# ── DATE entities: type-excluded regardless of frequency ────────────────────


def test_entity_type_extracts_tagged_type():
    assert _entity_type("2025 (DATE)") == "DATE"
    assert _entity_type("chevron (ORG)") == "ORG"


def test_entity_type_none_for_untyped_key():
    assert _entity_type("some untyped entity") is None
    assert _entity_type("not a type (whatever)") is None


def test_date_entity_excluded_even_at_low_document_frequency():
    """Direct reproduction of the live 10-K bug's third cause: "2025"
    appeared in only 12.2% of entity-bearing nodes (88/722) -- nowhere
    near the 40% genericity floor -- yet dominated the sampled ontology
    score's flagged invalid edges (all DATE-anchored, score dropped to
    50%). Frequency-based filtering can't catch this: it's a type problem
    (two pages sharing a calendar year says nothing about topical
    relatedness), not a frequency one."""
    total = 722
    entity_to_nodes = {
        "2025 (DATE)": list(range(88)),
        "chevron (ORG)": list(range(2)),
    }
    result = _informative_entities(entity_to_nodes, total_entity_nodes=total)
    assert "2025 (DATE)" not in result
    assert "chevron (ORG)" in result


def test_date_entity_excluded_below_min_nodes_bypass():
    # Below _ENTITY_GENERICITY_MIN_NODES, frequency filtering is bypassed
    # entirely -- but the DATE exclusion is a type judgment, not a
    # frequency one, so it still applies even here.
    entity_to_nodes = {"2024 (DATE)": [0, 1], "specific (CONCEPT)": [0]}
    result = _informative_entities(entity_to_nodes, total_entity_nodes=2)
    assert "2024 (DATE)" not in result
    assert "specific (CONCEPT)" in result


def test_location_entity_not_excluded_by_type():
    # Deliberately scoped to DATE only -- LOCATION can be a genuinely
    # meaningful relation anchor (e.g. "Gulf of America", "Kazakhstan"),
    # so it must still go through ordinary frequency-based filtering
    # rather than being excluded outright.
    total = 722
    entity_to_nodes = {"kazakhstan (LOCATION)": list(range(2))}
    result = _informative_entities(entity_to_nodes, total_entity_nodes=total)
    assert "kazakhstan (LOCATION)" in result


def test_untyped_entity_unaffected_by_date_exclusion():
    total = 722
    entity_to_nodes = {"legacy untyped entity": list(range(2))}
    result = _informative_entities(entity_to_nodes, total_entity_nodes=total)
    assert "legacy untyped entity" in result


# ── SEC-filing boilerplate phrases: excluded regardless of frequency ────────
# Regression: found via sampled LLM-judge audit of a real 10-K -- these
# flagged as the dominant remaining failure mode even after the DATE and
# enumeration-cap fixes. Unlike DATE, they don't share one NER type (OTHER/
# METRIC/CONCEPT all show up), so a type-based exclusion can't catch them --
# and unlike the self-referential-entity bugs, raw document frequency can't
# either: verified live, all of these sat at 0.6%-3.0% document frequency,
# far below any reasonable threshold, yet are standard SEC-filing/accounting
# citation vocabulary any 10-K/10-Q uses regardless of filer.


def test_consolidated_balance_sheet_excluded_even_at_low_frequency():
    total = 636
    entity_to_nodes = {
        "consolidated balance sheet (OTHER)": list(range(19)),
        "tco (ORG)": list(range(2)),
    }
    result = _informative_entities(entity_to_nodes, total_entity_nodes=total)
    assert "consolidated balance sheet (OTHER)" not in result
    assert "tco (ORG)" in result


def test_millions_of_dollars_excluded_regardless_of_type_tag():
    # Same phrase, different type tags across NER batches (the same
    # inconsistency _entity_base_text already accounts for elsewhere) --
    # the phrase-based exclusion must match on base text, not exact key.
    total = 636
    entity_to_nodes = {"millions of dollars (METRIC)": list(range(14))}
    result = _informative_entities(entity_to_nodes, total_entity_nodes=total)
    assert "millions of dollars (METRIC)" not in result


def test_boilerplate_phrase_excluded_below_min_nodes_bypass():
    # A type/frequency judgment is bypassed below the min-nodes floor, but
    # this is neither -- it must still apply.
    entity_to_nodes = {"regulation s-k (OTHER)": [0, 1], "specific (CONCEPT)": [0]}
    result = _informative_entities(entity_to_nodes, total_entity_nodes=2)
    assert "regulation s-k (OTHER)" not in result
    assert "specific (CONCEPT)" in result


def test_untyped_boilerplate_phrase_also_excluded():
    # Unlike DATE (a type judgment), the phrase list works on base text
    # regardless of whether a type tag is present at all.
    total = 636
    entity_to_nodes = {"exchange act": list(range(4))}
    result = _informative_entities(entity_to_nodes, total_entity_nodes=total)
    assert "exchange act" not in result


def test_similar_but_distinct_phrase_not_excluded():
    # Sanity check the fix targets the specific known phrases, not any
    # string that merely overlaps with one -- "balance sheet" alone (a
    # genuinely generic accounting term, arguably, but not one of the
    # phrases actually measured/verified live) must not be swept in by
    # accident via a substring match.
    total = 636
    entity_to_nodes = {"balance sheet": list(range(2))}
    result = _informative_entities(entity_to_nodes, total_entity_nodes=total)
    assert "balance sheet" in result


def test_non_topical_entity_phrases_are_all_lowercase():
    # The exclusion check lowercases the entity's base text before
    # comparing -- the phrase set itself must already be lowercase or the
    # comparison silently never matches.
    assert all(p == p.lower() for p in _NON_TOPICAL_ENTITY_PHRASES)


# ── continent names: excluded regardless of frequency ───────────────────────
# Regression: found via sampled LLM-judge audit of a real 10-K -- "asia
# (LOCATION), africa (LOCATION)" shared between two sections was flagged
# "not meaningfully connected". A continent is categorically different from
# the filing-boilerplate phrases above (a geographic-hierarchy fact, not
# this document TYPE's own vocabulary) and generalizes to every document,
# not just SEC filings -- the same structural argument as DATE, for a much
# smaller, closed, universally recognizable set.


def test_continent_name_excluded_even_at_low_frequency():
    total = 636
    entity_to_nodes = {
        "asia (LOCATION)": list(range(2)),
        "africa (LOCATION)": list(range(2)),
        "kazakhstan (LOCATION)": list(range(2)),
    }
    result = _informative_entities(entity_to_nodes, total_entity_nodes=total)
    assert "asia (LOCATION)" not in result
    assert "africa (LOCATION)" not in result
    assert "kazakhstan (LOCATION)" in result  # a real country, not a continent


def test_continent_name_excluded_below_min_nodes_bypass():
    entity_to_nodes = {"europe (LOCATION)": [0, 1], "guyana (LOCATION)": [0]}
    result = _informative_entities(entity_to_nodes, total_entity_nodes=2)
    assert "europe (LOCATION)" not in result
    assert "guyana (LOCATION)" in result


def test_all_seven_continents_excluded():
    total = 636
    entity_to_nodes = {f"{c} (LOCATION)": list(range(2)) for c in _NON_TOPICAL_CONTINENT_NAMES}
    result = _informative_entities(entity_to_nodes, total_entity_nodes=total)
    assert result == set()


def test_country_within_a_continent_not_excluded():
    # Sanity check the fix targets the continent name specifically, not any
    # LOCATION generally associated with one.
    total = 636
    entity_to_nodes = {"nigeria (LOCATION)": list(range(2))}
    result = _informative_entities(entity_to_nodes, total_entity_nodes=total)
    assert "nigeria (LOCATION)" in result
