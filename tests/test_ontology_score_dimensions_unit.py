"""
tests/test_ontology_score_dimensions_unit.py — a quality score reports its
WORST dimension, not an average that lets a healthy dimension carry a broken
one, and a sampled score reports its uncertainty.

Regression: every gate read green while the system was demonstrably wrong.

  * Axis-1 pooled three unrelated check families into one fraction. Measured
    on a real 264-page 10-K: containment 411/412 (99.8%) and sibling ordering
    100% pooled with title quality 77/172 (44.8%) to report 86.9% -- a
    document whose section titles were more than half junk (wrapped running
    headers used as 25 separate titles, table data rows, 49 synthetic
    "Preamble" catch-alls) read as "pretty good". Worse, adding the title
    check barely moved the number, because 172 title checks were diluted by
    roughly 560 passing ones.

  * Axis-2 averaged edge precision with entity grounding. Measured: 0.26 and
    0.98 averaged to 62%, which reads as middling rather than "one of the two
    halves is broken".

  * Axis-2 reported a bare point estimate from a 15-item sample. Two runs
    against a byte-identical graph returned 63% and 80%, and the difference
    was taken seriously -- their confidence intervals overlap almost
    entirely.

Run with:
    python -m pytest tests/test_ontology_score_dimensions_unit.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

_root = Path(__file__).resolve().parents[1]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from src.document.ontology_validation import (
    _wilson_interval,
    score_axis1_structural_invariants,
)


def _doc(titles: list[str]) -> list[dict]:
    """A well-nested document (containment + sibling checks all pass) whose
    only variable is title quality."""
    nodes = [{"id": "d", "depth": 0, "page_start": 1, "page_end": 3 * len(titles), "title": "f.pdf"}]
    for i, t in enumerate(titles):
        nodes.append({
            "id": f"s{i}", "parent_id": "d", "depth": 1,
            "page_start": 3 * i + 1, "page_end": 3 * i + 3, "title": t,
        })
    return nodes


# ── Axis-1: worst dimension, not pooled average ─────────────────────────────


def test_score_is_the_worst_dimension_not_the_pooled_average():
    """The exact shape of the live failure: structure nests perfectly, titles
    are mostly junk. The score must follow the titles."""
    report = score_axis1_structural_invariants(
        _doc(["Introduction", "Preamble", "| 2025 | 2024 |", "Preamble"]), "doc1"
    )
    pooled = report.matched / report.total_ground_truth
    assert report.dimensions["containment"] == 1.0
    assert report.dimensions["titles"] == 0.25
    assert report.score == 0.25
    assert report.score < pooled  # pooling would have hidden it


def test_all_dimensions_reported_even_when_passing():
    report = score_axis1_structural_invariants(_doc(["Introduction", "Methods"]), "doc1")
    assert set(report.dimensions) == {"containment", "titles", "siblings"}
    assert report.score == 1.0


def test_a_single_broken_dimension_cannot_be_carried_by_the_others():
    """Many passing containment checks must not rescue a failing dimension --
    this is what let 55%-junk titles report 99.76%."""
    report = score_axis1_structural_invariants(_doc(["Preamble"] * 20 + ["Real Heading"]), "doc1")
    assert report.dimensions["containment"] == 1.0
    assert report.score < 0.10


def test_dimension_with_no_checks_is_omitted_not_counted_as_perfect():
    """An absent dimension is not evidence of quality; counting it as 100%
    could make it the reported score."""
    # Single section: no sibling PAIR exists, so that dimension has no checks.
    report = score_axis1_structural_invariants(_doc(["Introduction"]), "doc1")
    assert "siblings" not in report.dimensions
    assert report.score == 1.0


def test_untitled_nodes_do_not_create_a_titles_dimension():
    """Callers that fetch no titles must be unaffected — the dimension simply
    does not exist for them, rather than scoring 0."""
    nodes = [
        {"id": "d", "depth": 0, "page_start": 1, "page_end": 6},
        {"id": "a", "parent_id": "d", "depth": 1, "page_start": 1, "page_end": 3},
        {"id": "b", "parent_id": "d", "depth": 1, "page_start": 4, "page_end": 6},
    ]
    report = score_axis1_structural_invariants(nodes, "doc1")
    assert "titles" not in report.dimensions
    assert report.score == 1.0


def test_containment_failure_still_lowers_the_score():
    """The original invariants must keep working — this is additive."""
    nodes = _doc(["Introduction", "Methods"])
    nodes[1]["page_end"] = 999  # escapes its parent's range
    report = score_axis1_structural_invariants(nodes, "doc1")
    assert report.dimensions["containment"] < 1.0
    assert report.score < 1.0


# ── Wilson confidence intervals ─────────────────────────────────────────────


def test_the_63_vs_80_runs_have_overlapping_intervals():
    """The two live n=15 runs over an identical graph. Their intervals
    overlap heavily, so the difference was never evidence of anything."""
    lo_a, hi_a = _wilson_interval(4, 15)    # 26.7%
    lo_b, hi_b = _wilson_interval(9, 15)    # 60.0%
    assert lo_b < hi_a, "intervals must overlap — the runs are indistinguishable"


def test_larger_sample_tightens_the_interval():
    narrow = _wilson_interval(13, 50)
    wide = _wilson_interval(4, 15)
    assert (narrow[1] - narrow[0]) < (wide[1] - wide[0])


def test_interval_stays_within_zero_and_one_at_the_extremes():
    """Why Wilson and not the normal approximation: at p=0 or p=1 the normal
    interval collapses to zero width (claiming certainty from a handful of
    samples) and can run outside [0, 1]."""
    for successes, total in ((0, 10), (10, 10)):
        lo, hi = _wilson_interval(successes, total)
        assert 0.0 <= lo <= hi <= 1.0
        assert hi - lo > 0.0


def test_no_samples_yields_no_interval():
    assert _wilson_interval(0, 0) is None
