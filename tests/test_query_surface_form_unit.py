"""
tests/test_query_surface_form_unit.py — a question finds its answer even when
it words things differently from the document.

Regression, found by probing an ingested 52-page WHO report with questions a
user would actually type. Three failures, all of them the SAME defect class:
every lexical predicate in this pipeline matches with CONTAINS (exact
substring), so any morphological or spelling difference between the question
and the document loses the answer completely — and loses it silently, as
"this document does not cover…", which reads as an authoritative absence.

  * "Which Jordanian hospitals implemented Go.Data in January 2021?"
    -> "this document does not cover" — while the SAME graph, one question
    earlier, had correctly listed all eight of those hospitals out of the
    same table. Two morphological misses in one question were enough:
    "jordanian" vs the document's "Jordan", "hospitals" vs "Hospital".
  * "What does Figure 1 show?" -> not found; "What does Fig. 1 show?" ->
    answered correctly. The region is titled "Fig. 1: ...".
  * "What is in the list of abbreviations?" -> not found; the identical
    question capitalized ("List of Abbreviations") returned all 24 entries.

Fixes, all document-agnostic:
  * query-side suffix stemming (a stem is a PREFIX, so CONTAINS(stem) matches
    a superset — recall can only improve);
  * structural-noun references recognized case-insensitively;
  * both surface forms of abbreviated structural nouns emitted
    ("figure 1" and "fig. 1").

Run with:
    python -m pytest tests/test_query_surface_form_unit.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

_root = Path(__file__).resolve().parents[1]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from src.retrieval.unstructured.services.ranking import RankingService


def _r() -> RankingService:
    return RankingService.__new__(RankingService)


# ── morphological stemming ──────────────────────────────────────────────────


def test_demonym_stems_to_country_name():
    """The exact live failure: the question says "Jordanian", the table says
    "Jordan"."""
    assert _r()._morphological_stem("jordanian") == "jordan"


def test_plural_stems_to_singular():
    assert _r()._morphological_stem("hospitals") == "hospital"
    assert _r()._morphological_stem("abbreviations") == "abbreviation"


def test_verb_inflection_stems_to_base():
    assert _r()._morphological_stem("implemented") == "implement"


def test_ies_plural_stems_to_shared_prefix():
    """"countries" must stem to a prefix of BOTH "country" and "countries" --
    "countri" would match the plural and silently miss the singular."""
    stem = _r()._morphological_stem("countries")
    assert "country".startswith(stem) and "countries".startswith(stem)


def test_short_words_are_left_alone():
    """Below the minimum length, suffix stripping destroys words rather than
    recovering base forms ("data" -> "dat")."""
    for w in ("data", "uses", "site"):
        assert _r()._morphological_stem(w) is None


def test_stem_is_always_a_prefix_of_the_original():
    """The property the whole approach depends on: CONTAINS(stem) can only
    match MORE than CONTAINS(original), never less."""
    for w in ("hospitals", "jordanian", "implemented", "countries", "abbreviations"):
        stem = _r()._morphological_stem(w)
        assert stem and w.startswith(stem)


def test_non_alphabetic_tokens_are_not_stemmed():
    assert _r()._morphological_stem("2021") is None
    assert _r()._morphological_stem("go.data") is None


def test_keywords_include_both_original_and_stem():
    """Both forms are kept so idf can still prefer an exact document match."""
    kws = _r()._content_keywords_from_query(
        "Which Jordanian hospitals implemented Go.Data in January 2021?"
    )
    for expected in ("jordanian", "jordan", "hospitals", "hospital"):
        assert expected in kws, f"missing {expected!r}"


# ── structural references / surface forms ───────────────────────────────────


def test_figure_query_emits_both_surface_forms():
    """A document writes "Fig. 1" and the question says "Figure 1" (or the
    reverse) — both must be searched."""
    phrases = [p.lower() for p in _r().scope_phrases_from_query("What does Figure 1 show?")]
    assert "figure 1" in phrases
    assert "fig. 1" in phrases


def test_abbreviated_figure_query_also_emits_long_form():
    phrases = [p.lower() for p in _r().scope_phrases_from_query("What does Fig. 1 show?")]
    assert "figure 1" in phrases


def test_lowercase_structural_reference_is_recognized():
    """The live failure: capitalized worked, lowercase returned nothing."""
    phrases = [p.lower() for p in _r().scope_phrases_from_query(
        "What is in the list of abbreviations?"
    )]
    assert any("abbreviation" in p for p in phrases)


def test_lowercase_and_capitalized_agree():
    """Same question, different capitalization, same scoping — the defect was
    that these two disagreed."""
    lower = {p.lower() for p in _r().scope_phrases_from_query("what is in the list of abbreviations?")}
    upper = {p.lower() for p in _r().scope_phrases_from_query("What is in the List of Abbreviations?")}
    assert lower & upper


def test_table_reference_with_identifier():
    phrases = [p.lower() for p in _r().scope_phrases_from_query(
        "How many institutions are listed in Table A3?"
    )]
    assert "table a3" in phrases


def test_appendix_and_annex_are_cross_emitted():
    phrases = [p.lower() for p in _r().scope_phrases_from_query("What is in Appendix 2?")]
    assert "annex 2" in phrases


def test_question_without_structure_or_proper_noun_is_still_a_no_op():
    """Ordinary questions must not acquire scope phrases — the scope path has
    to stay inert for them."""
    assert _r().scope_phrases_from_query("what is the total revenue") == []


# ── keyword-leader pin ──────────────────────────────────────────────────────


def _item(cid: str, score: float = 1.0, text: str = "x") -> dict:
    return {"id": cid, "title": cid, "text": text, "score": score}


def test_keyword_leader_is_pinned_when_outranked():
    """The live failure: the page holding the answer ranked FIRST in the
    keyword search and still never reached synthesis, because vector hits
    enter merging at ~4-5 and lexical at ~1."""
    items = [_item(f"vec{i}", 4.6) for i in range(8)]
    lexical = [_item("page_46", 1.2)]

    out = _r()._pin_keyword_leader(items, lexical, limit=8)

    assert out[0]["id"] == "page_46"
    assert "via:keyword_leader" in out[0]["related"]
    assert len(out) == 8


def test_keyword_leader_not_duplicated_when_already_present():
    items = [_item("page_46", 9.0)] + [_item(f"v{i}", 4.0) for i in range(3)]
    out = _r()._pin_keyword_leader(items, [_item("page_46", 1.0)], limit=8)
    assert [i["id"] for i in out].count("page_46") == 1
    assert out == items


def test_only_the_leader_is_pinned_not_the_whole_lexical_list():
    """One slot, not a takeover — the rest of the lexical hits stay subject to
    ordinary ranking."""
    items = [_item(f"v{i}", 4.0) for i in range(6)]
    lexical = [_item("lead", 1.0), _item("second", 0.9), _item("third", 0.8)]
    out = _r()._pin_keyword_leader(items, lexical, limit=6)
    ids = [i["id"] for i in out]
    assert ids[0] == "lead"
    assert "second" not in ids and "third" not in ids


def test_no_lexical_hits_leaves_items_untouched():
    items = [_item("a", 2.0), _item("b", 1.0)]
    assert _r()._pin_keyword_leader(items, [], limit=5) == items


# ── quantity questions ──────────────────────────────────────────────────────
# Regression: "How many countries and institutions used Go.Data?" answered
# "this document does not cover", while the chunk holding both "65 countries"
# and "115 institutions" did not place in the top TEN keyword hits. Not a
# ranking bug: in a document about country implementations those nouns are
# everywhere, so idf correctly finds them uninformative and everything ties.
# What identifies the answer is a numeral sitting next to the counted noun.


def test_quantity_question_detected():
    from src.retrieval.unstructured.services.lexical import _QUANTITY_QUESTION_RE
    for q in ("How many countries used it?", "What is the number of sites?",
              "How much funding?", "total number of users"):
        assert _QUANTITY_QUESTION_RE.search(q), q


def test_non_quantity_question_not_detected():
    """The path must stay completely inert for ordinary questions."""
    from src.retrieval.unstructured.services.lexical import _QUANTITY_QUESTION_RE
    for q in ("What does Figure 1 show?", "Who wrote this report?",
              "Describe the methodology"):
        assert not _QUANTITY_QUESTION_RE.search(q), q


def test_quantity_pattern_matches_number_before_noun():
    """The generalizable shape: a numeral, optionally with a qualifier, then
    the counted noun — true of prose in any domain."""
    import re as _re
    pattern = r"(?s).*\d[\d,.]*\s+(?:\w+\s+){0,2}institutions.*"
    for text in ("used by 115 institutions worldwide",
                 "over 115 institutions",
                 "115 total participating institutions"):
        assert _re.match(pattern, text.lower()), text


def test_quantity_pattern_rejects_noun_without_number():
    import re as _re
    pattern = r"(?s).*\d[\d,.]*\s+(?:\w+\s+){0,2}institutions.*"
    assert not _re.match(pattern, "institutions across the region adopted it")
