"""tests/unstructured/test_term_stats_unit.py — measuring "generic", not listing it.

A question can match the corpus superbly and still name nothing: "What are
the key findings?" is answerable in every document. Three retrieval
statistics were tried against this and all three failed (see the module
docstring), so the signal here is document frequency instead.
"""
from __future__ import annotations

from src.unstructured.retrieval.services.term_stats import CorpusTermStats


class _Session:
    """Two documents share the structure words; each owns one rare term."""

    def __init__(self, rows):
        self._rows = rows
        self.runs = 0

    def run(self, *_a, **_k):
        self.runs += 1
        return list(self._rows)


def _rows(n_common=10, rare=("chevron", "depreciation")):
    out = []
    for i in range(n_common):
        extra = rare[i] if i < len(rare) else ""
        out.append({"d": f"doc{i}", "t": f"the key findings and conclusion here {extra}"})
    return out


def test_structure_words_are_common_and_rare_terms_are_not():
    stats, s = CorpusTermStats(), _Session(_rows())

    assert stats.document_frequency(s, "findings") == 1.0
    assert stats.document_frequency(s, "chevron") < 0.2


def test_terms_far_below_the_threshold_are_pruned_to_zero():
    """Memory is bounded by dropping rare terms; a lookup miss means rare.

    Only terms near the decision threshold need an accurate figure, so this
    cannot change a verdict -- a pruned term is below the threshold either
    way. Needs a corpus large enough for the floor to engage.
    """
    rows = [{"d": f"doc{i}", "t": "the key findings here"} for i in range(200)]
    rows[0]["t"] += " chevron"
    stats, s = CorpusTermStats(), _Session(rows)

    assert stats.document_frequency(s, "findings") == 1.0
    assert stats.document_frequency(s, "chevron") == 0.0


def test_a_question_of_only_structure_words_scores_high():
    stats, s = CorpusTermStats(), _Session(_rows())

    assert stats.min_term_frequency(s, ["key", "findings", "conclusion"]) == 1.0


def test_one_rare_term_is_enough_to_place_a_question():
    """The minimum is over terms: a single distinctive word makes it answerable."""
    stats, s = CorpusTermStats(), _Session(_rows())

    assert stats.min_term_frequency(s, ["findings", "depreciation"]) < 0.2


def test_morphological_variants_count_once():
    """Keyword extraction emits "challenges" and the stem "challeng" together.

    The stem is absent from prose, so scoring each separately read the pair
    as a rare term on a df of 0.009 and let a generic question through.
    """
    stats = CorpusTermStats()
    s = _Session([{"d": f"doc{i}", "t": "the challenges discussed"} for i in range(10)])

    assert stats.min_term_frequency(s, ["challenges", "challeng"]) == 1.0


def test_table_is_cached_rather_than_rebuilt_per_query():
    stats, s = CorpusTermStats(), _Session(_rows())

    for _ in range(5):
        stats.min_term_frequency(s, ["findings"])

    assert s.runs == 1


def test_empty_corpus_expresses_no_opinion():
    """No table means no verdict -- the gate must not refuse on missing data."""
    stats, s = CorpusTermStats(), _Session([])

    assert stats.min_term_frequency(s, ["findings"]) is None
