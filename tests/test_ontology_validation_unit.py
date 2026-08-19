"""
tests/test_ontology_validation_unit.py — Axis-1/Axis-2 ontology-accuracy
scoring (src/document/ontology_validation.py).

Pure scoring logic, no live Neo4j/blob/LLM — mirrors test_document_
verification_unit.py's FakeModelProvider pattern for the Axis-2 judge.

Run with:
    python -m pytest tests/test_ontology_validation_unit.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest


from src.unstructured.document.ontology_validation import (
    _entity_centered_window,
    _shared_entity_texts,
    ONTOLOGY_ACCURACY_TARGET,
    Axis1Report,
    Axis2Report,
    score_axis1_against_toc,
    score_axis1_structural_invariants,
    score_axis2_idea_linking,
)


class FakeModelProvider:
    def __init__(self, responses: list[str]):
        self._responses = list(responses)
        self.calls: list[dict] = []

    def chat_completion(self, model, messages, **kwargs):
        self.calls.append({"model": model, "messages": messages, **kwargs})
        content = self._responses.pop(0) if self._responses else '{"valid": false}'
        return MagicMock(choices=[MagicMock(message=MagicMock(content=content))])


# ── score_axis1_against_toc ──────────────────────────────────────────────────


def test_axis1_toc_perfect_match_scores_1():
    toc = [(1, "Chapter 1: Introduction", 1), (2, "1.1 Overview", 3)]
    constructed = [
        {"id": "c1", "title": "Chapter 1: Introduction", "depth": 1, "page_start": 1},
        {"id": "s1", "title": "1.1 Overview", "depth": 2, "page_start": 3},
    ]
    report = score_axis1_against_toc(constructed, toc, "doc1")
    assert report.score == 1.0
    assert report.precision == 1.0
    assert report.recall == 1.0
    assert report.matched == 2
    assert report.method == "toc_ground_truth"
    assert report.mismatches == []


def test_axis1_toc_missing_entry_lowers_recall():
    toc = [(1, "Chapter 1", 1), (1, "Chapter 2", 10)]
    constructed = [{"id": "c1", "title": "Chapter 1", "depth": 1, "page_start": 1}]
    report = score_axis1_against_toc(constructed, toc, "doc1")
    assert report.matched == 1
    assert report.recall == 0.5
    assert any("Chapter 2" in m for m in report.mismatches)


def test_axis1_toc_extra_constructed_node_lowers_precision():
    toc = [(1, "Chapter 1", 1)]
    constructed = [
        {"id": "c1", "title": "Chapter 1", "depth": 1, "page_start": 1},
        {"id": "c2", "title": "Spurious duplicate", "depth": 1, "page_start": 50},
    ]
    report = score_axis1_against_toc(constructed, toc, "doc1")
    assert report.recall == 1.0
    assert report.precision == 0.5
    assert report.score < 1.0


def test_axis1_toc_fuzzy_title_match_tolerates_minor_cleanup():
    # Non-breaking space + trailing whitespace cleaned up by the parser.
    toc = [(1, "Chapter\xa01:  Introduction ", 1)]
    constructed = [{"id": "c1", "title": "Chapter 1: Introduction", "depth": 1, "page_start": 1}]
    report = score_axis1_against_toc(constructed, toc, "doc1")
    assert report.matched == 1
    assert report.score == 1.0


def test_axis1_toc_unrelated_title_does_not_match():
    toc = [(1, "Chapter 1: Introduction", 1)]
    constructed = [{"id": "c1", "title": "Appendix B: Glossary", "depth": 1, "page_start": 1}]
    report = score_axis1_against_toc(constructed, toc, "doc1")
    assert report.matched == 0
    assert report.recall == 0.0


def test_axis1_toc_page_tolerance_allows_off_by_one():
    toc = [(1, "Chapter 1", 5)]
    constructed = [{"id": "c1", "title": "Chapter 1", "depth": 1, "page_start": 6}]
    report = score_axis1_against_toc(constructed, toc, "doc1")
    assert report.matched == 1


def test_axis1_toc_page_beyond_tolerance_does_not_match():
    toc = [(1, "Chapter 1", 5)]
    constructed = [{"id": "c1", "title": "Chapter 1", "depth": 1, "page_start": 20}]
    report = score_axis1_against_toc(constructed, toc, "doc1")
    assert report.matched == 0


# ── score_axis1_structural_invariants ────────────────────────────────────────


def test_axis1_invariants_clean_tree_scores_1():
    constructed = [
        {"id": "c1", "parent_id": None, "depth": 1, "page_start": 1, "page_end": 20},
        {"id": "s1", "parent_id": "c1", "depth": 2, "page_start": 1, "page_end": 10},
        {"id": "s2", "parent_id": "c1", "depth": 2, "page_start": 11, "page_end": 20},
    ]
    report = score_axis1_structural_invariants(constructed, "doc1")
    assert report.score == 1.0
    assert report.method == "structural_invariants"
    assert report.mismatches == []


def test_axis1_invariants_child_outside_parent_range_flagged():
    constructed = [
        {"id": "c1", "parent_id": None, "depth": 1, "page_start": 1, "page_end": 10},
        {"id": "s1", "parent_id": "c1", "depth": 2, "page_start": 5, "page_end": 15},
    ]
    report = score_axis1_structural_invariants(constructed, "doc1")
    assert report.score < 1.0
    assert any("page range outside parent" in m for m in report.mismatches)


def test_axis1_invariants_overlapping_siblings_flagged():
    constructed = [
        {"id": "c1", "parent_id": None, "depth": 1, "page_start": 1, "page_end": 20},
        {"id": "s1", "parent_id": "c1", "depth": 2, "page_start": 1, "page_end": 12},
        {"id": "s2", "parent_id": "c1", "depth": 2, "page_start": 10, "page_end": 20},
    ]
    report = score_axis1_structural_invariants(constructed, "doc1")
    assert any("overlapping siblings" in m for m in report.mismatches)


def test_axis1_invariants_orphan_non_root_flagged():
    constructed = [{"id": "s1", "parent_id": "missing", "depth": 2, "page_start": 1, "page_end": 5}]
    report = score_axis1_structural_invariants(constructed, "doc1")
    assert any("orphan node" in m for m in report.mismatches)


def test_axis1_invariants_empty_input_scores_1():
    report = score_axis1_structural_invariants([], "doc1")
    assert report.score == 1.0


def test_axis1_invariants_section_parented_directly_by_document_not_orphan():
    # Regression: scripts/validate_ontology_accuracy.py originally fetched
    # only Chapter/Section nodes into `constructed`, so a Section whose
    # real CONTAINS parent is the Document root (no intervening Chapter --
    # common for SEC filings) had a parent_id the scorer couldn't resolve,
    # and got misread as an orphan. The Document root must be included in
    # `constructed` (depth 0) for this case to score correctly.
    constructed = [
        {"id": "doc1", "parent_id": None, "depth": 0, "page_start": 1, "page_end": 50},
        {"id": "s1", "parent_id": "doc1", "depth": 2, "page_start": 1, "page_end": 1},
    ]
    report = score_axis1_structural_invariants(constructed, "doc1")
    assert report.score == 1.0
    assert report.mismatches == []


# ── score_axis2_idea_linking ─────────────────────────────────────────────────


def test_axis2_all_valid_scores_1():
    provider = FakeModelProvider(['{"valid": true}'] * 4)
    edges = [{"source_text": "a", "target_text": "b", "rel_type": "SHARES_ENTITY", "shared": "x"}] * 2
    entities = [{"source_text": "a", "entity": "x"}] * 2
    report = score_axis2_idea_linking(edges, entities, provider=provider, model="m", logical_doc_id="doc1")
    assert report.score == 1.0
    assert report.edge_precision == 1.0
    assert report.entity_grounding_precision == 1.0
    assert report.invalid_examples == []


def test_axis2_mixed_valid_invalid_computes_precision():
    provider = FakeModelProvider(['{"valid": true}', '{"valid": false}'])
    edges = [{"source_text": "a", "target_text": "b", "rel_type": "SHARES_ENTITY", "shared": "x"}] * 2
    report = score_axis2_idea_linking(edges, [], provider=provider, model="m", logical_doc_id="doc1")
    assert report.edge_precision == 0.5
    assert report.entity_grounding_precision is None
    assert report.score == 0.5
    assert len(report.invalid_examples) == 1


def test_axis2_provider_exception_counts_as_invalid_not_open():
    class RaisingProvider:
        def chat_completion(self, model, messages, **kwargs):
            raise RuntimeError("down")

    edges = [{"source_text": "a", "target_text": "b", "rel_type": "SHARES_ENTITY", "shared": "x"}]
    report = score_axis2_idea_linking(edges, [], provider=RaisingProvider(), model="m", logical_doc_id="doc1")
    assert report.edge_precision == 0.0


def test_axis2_malformed_judge_response_counts_as_invalid():
    provider = FakeModelProvider(["not json at all"])
    edges = [{"source_text": "a", "target_text": "b", "rel_type": "SHARES_ENTITY", "shared": "x"}]
    report = score_axis2_idea_linking(edges, [], provider=provider, model="m", logical_doc_id="doc1")
    assert report.edge_precision == 0.0


def test_axis2_empty_samples_scores_1():
    provider = FakeModelProvider([])
    report = score_axis2_idea_linking([], [], provider=provider, model="m", logical_doc_id="doc1")
    assert report.score == 1.0
    assert report.sampled_edges == 0
    assert report.sampled_entities == 0


# ── target constant + as_dict shape ──────────────────────────────────────────


def test_ontology_accuracy_target_is_90_percent():
    assert ONTOLOGY_ACCURACY_TARGET == 0.90


def test_axis1_report_as_dict_rounds_and_caps_mismatches():
    report = Axis1Report(
        logical_doc_id="doc1",
        method="toc_ground_truth",
        score=0.123456,
        precision=1.0,
        recall=0.5,
        matched=1,
        total_ground_truth=2,
        total_constructed=1,
        mismatches=[f"m{i}" for i in range(20)],
    )
    d = report.as_dict()
    assert d["score"] == 0.1235
    assert len(d["mismatches"]) == 10


def test_axis2_report_as_dict_handles_none_precisions():
    report = Axis2Report(logical_doc_id="doc1", score=1.0)
    d = report.as_dict()
    assert d["edge_precision"] is None
    assert d["entity_grounding_precision"] is None


# ── _entity_centered_window / _shared_entity_texts ───────────────────────────
# Regression: found live via a real 10-K's ontology score -- "tco (ORG)" was
# flagged "not meaningfully connected" by the judge, but TCO (Tengizchevroil)
# genuinely was mentioned in the source section, at character offset 1789 of
# a 3,299-char passage. The judge only ever saw the first 800 chars (a plain
# prefix truncation), so it was judging a passage that -- from its point of
# view -- didn't contain the entity at all. Not a graph-quality problem;
# a measurement artifact in how evidence was shown to the judge.


def test_entity_centered_window_short_text_returned_as_is():
    assert _entity_centered_window("short text", ["text"]) == "short text"


def test_entity_centered_window_centers_on_late_occurrence():
    text = ("padding " * 200) + "TCO operates the Tengiz field." + ("more padding " * 200)
    window = _entity_centered_window(text, ["tco (ORG)"], window=100)
    assert "TCO operates the Tengiz field" in window
    assert len(window) <= 100


def test_entity_centered_window_direct_reproduction_of_live_bug():
    """Direct reproduction: a 3,299-char passage with the claimed entity at
    offset 1789 -- the old plain text[:800] truncation would never include
    it; the fix must."""
    text = ("x" * 1789) + "TCO" + ("y" * (3299 - 1789 - 3))
    assert len(text) == 3299
    old_naive_window = text[:800]
    assert "TCO" not in old_naive_window  # confirms the bug this fix targets

    new_window = _entity_centered_window(text, ["tco (ORG)"], window=800)
    assert "TCO" in new_window


def test_entity_centered_window_falls_back_to_prefix_when_entity_not_found():
    # Paraphrase-only match a substring search can't locate -- same
    # behavior as before this fix for this harder case.
    text = "a" * 1000
    assert _entity_centered_window(text, ["nowhere to be found"], window=800) == text[:800]


def test_entity_centered_window_empty_needles_falls_back_to_prefix():
    text = "a" * 1000
    assert _entity_centered_window(text, [], window=800) == text[:800]


def test_entity_centered_window_uses_first_needle_in_list_that_matches():
    # Needle-list order decides, not textual order -- a SHARES_ENTITY edge's
    # entity list is short and any one being visible is reasonable evidence,
    # so this is a simple, deliberate choice, not an accident.
    text = ("z" * 500) + "SECOND" + ("z" * 500) + "FIRST" + ("z" * 500)
    window = _entity_centered_window(text, ["FIRST", "SECOND"], window=50)
    assert "FIRST" in window
    assert "SECOND" not in window


def test_shared_entity_texts_parses_json_string():
    shared = '{"shared_entities": ["tco (ORG)", "hess (ORG)"], "rarity_score": 3.88}'
    assert _shared_entity_texts(shared) == ["tco (ORG)", "hess (ORG)"]


def test_shared_entity_texts_accepts_dict_directly():
    shared = {"shared_entities": ["kazakhstan (LOCATION)"]}
    assert _shared_entity_texts(shared) == ["kazakhstan (LOCATION)"]


def test_shared_entity_texts_returns_empty_for_unrelated_shape():
    # SEMANTICALLY_SIMILAR edges carry {"score": ...}, no entities to
    # center on -- must not crash, just fall back to no needles.
    assert _shared_entity_texts('{"score": 0.81}') == []


def test_shared_entity_texts_returns_empty_for_malformed_json():
    assert _shared_entity_texts("not json") == []


def test_axis2_same_category_edge_falls_back_to_each_side_own_entities():
    """SAME_CATEGORY (and CONTRADICTS/ELABORATES/PREREQUISITE_OF) have no
    "shared" entity field at all ({cluster_id, signal} only) -- the window
    must fall back to each side's OWN entity list rather than an arbitrary
    first-800-char prefix that might land on boilerplate."""
    provider = FakeModelProvider(['{"valid": true}'])
    source_long = ("boilerplate header text. " * 40) + "Tengizchevroil expansion project details here."
    target_long = ("different boilerplate. " * 40) + "Kazakhstan production volumes discussed here."
    edges = [{
        "source_text": source_long,
        "target_text": target_long,
        "rel_type": "SAME_CATEGORY",
        "shared": '{"cluster_id": 0, "signal": "entity_cooccurrence"}',
        "source_entities": ["Tengizchevroil"],
        "target_entities": ["Kazakhstan"],
    }]
    score_axis2_idea_linking(edges, [], provider=provider, model="m", logical_doc_id="doc1")

    sent_prompt = provider.calls[0]["messages"][0]["content"]
    assert "Tengizchevroil expansion project" in sent_prompt
    assert "Kazakhstan production volumes" in sent_prompt


def test_axis2_edge_missing_entities_fields_degrades_to_prefix_not_crash():
    # Backward-compat: an edges_sample dict without source_entities/
    # target_entities (e.g. a caller not yet updated) must not KeyError.
    provider = FakeModelProvider(['{"valid": true}'])
    edges = [{"source_text": "a" * 1000, "target_text": "b" * 1000, "rel_type": "SAME_CATEGORY", "shared": ""}]
    report = score_axis2_idea_linking(edges, [], provider=provider, model="m", logical_doc_id="doc1")
    assert report.edge_precision == 1.0


def test_axis2_judge_receives_entity_centered_window_not_naive_prefix():
    """End-to-end through score_axis2_idea_linking: the judge's prompt must
    actually contain the entity-centered window, not the first 800 chars."""
    provider = FakeModelProvider(['{"valid": true}'])
    long_text = ("x" * 1789) + "TCO operates the Tengiz field." + ("y" * 2000)
    edges = [{
        "source_text": long_text,
        "target_text": "short target text",
        "rel_type": "SHARES_ENTITY",
        "shared": '{"shared_entities": ["tco (ORG)"], "rarity_score": 3.88}',
    }]
    score_axis2_idea_linking(edges, [], provider=provider, model="m", logical_doc_id="doc1")

    sent_prompt = provider.calls[0]["messages"][0]["content"]
    assert "TCO operates the Tengiz field" in sent_prompt
