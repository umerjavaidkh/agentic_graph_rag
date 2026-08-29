"""
tests/shared/test_arabic_normalization_unit.py — matching in one space,
reading in another.

Arabic writes the same word several ways and a reader does not
distinguish them: hamza carriers vary by convention, teh marbuta and heh
are interchanged in informal and OCR'd text, tashkeel is optional and
usually absent from what someone types, and tatweel is a typographic
stretch with no meaning at all.

Measured on the ingested corpus before this existed: the query "الهويات"
matched 0 nodes while the stored form "الهويّات" matched 38 -- the same
word, one shadda apart.

The fix is deliberately NOT to normalize stored text. Citations have to
stay byte-identical to the PDF and every deterministic eval in eval/
depends on exact spans, so the normalizer produces a separate matching
key and the text a reader sees is never touched.

Run with:
    python -m pytest tests/shared/test_arabic_normalization_unit.py -v
"""
from __future__ import annotations

from src.shared.language import derive_match_text, get_profile, normalize_arabic

AR = get_profile("ar")
EN = get_profile("en")


def test_a_diacritic_does_not_make_a_different_word():
    """The measured failure: 0 hits against 38, one shadda apart."""
    assert AR.normalize("الهويّات") == AR.normalize("الهويات")


def test_hamza_carriers_agree():
    assert AR.normalize("أحمد") == AR.normalize("احمد") == AR.normalize("إحمد")


def test_teh_marbuta_and_heh_agree():
    """The pair OCR most often disagrees with itself about."""
    assert AR.normalize("مدرسة") == AR.normalize("مدرسه")


def test_tatweel_is_typography_and_carries_no_meaning():
    assert AR.normalize("الــــكتاب") == AR.normalize("الكتاب")


def test_normalization_is_idempotent():
    """A key that changes when re-derived would drift from the stored one
    on any later backfill."""
    once = AR.normalize("النَّتائِجُ")
    assert AR.normalize(once) == once


def test_english_normalization_is_the_identity():
    """The whole English guarantee rests on this."""
    for text in ("Revenue increased by 12 percent.", "NIST SP 800-207", ""):
        assert EN.normalize(text) == text


def test_english_derives_no_match_text_at_all():
    """Not a copy -- absent.

    A copy would double the stored text of the entire English corpus to
    say nothing, and `coalesce(match_text, search_text)` falls through to
    precisely the behaviour English already had.
    """
    assert derive_match_text("Revenue increased by 12 percent.", "en") is None


def test_arabic_derives_a_match_text_only_when_it_differs():
    assert derive_match_text("الهويّات", "ar") == "الهويات"
    # already in normal form: nothing to store
    assert derive_match_text("الهويات", "ar") is None


def test_empty_text_derives_nothing():
    assert derive_match_text("", "ar") is None
    assert derive_match_text(None, "ar") is None


def test_normalization_leaves_latin_and_digits_alone():
    """An Arabic document quoting English or figures must not have that
    part rewritten -- the quoted span still has to match the page."""
    mixed = "الهويّات ISO 9001 revenue 2024"
    assert "ISO 9001 revenue 2024" in normalize_arabic(mixed)


def test_the_matching_key_reads_match_text_then_falls_back():
    """The Cypher side of the same guarantee."""
    from src.unstructured.retrieval.cypher_scope import match_key_cypher

    expr = match_key_cypher("n")
    assert "n.match_text" in expr
    assert "n.search_text" in expr
    assert expr.index("match_text") < expr.index("search_text"), (
        "search_text must be the FALLBACK, not the preferred key"
    )


def test_the_english_stemmer_never_fires_on_arabic():
    """Measured, then pinned: it is already correct and must stay absent.

    The design doc lists "retire _STEM_SUFFIXES for Arabic" as work to
    do, implying the English stemmer damages Arabic. It does not: all
    nine suffixes are Latin strings, so an Arabic word can never end with
    one and `_morphological_stem` returns None every time. Removing it
    would be a no-op.

    What this pins is the opposite risk -- that someone reads the design
    doc and ADDS an Arabic stemmer. Arabic morphology is templatic: the
    root sits in the consonant skeleton, not in a strippable suffix, and
    suffix-stripping an Arabic word destroys it. Embeddings already carry
    the morphological relation; lexical matching is for exact anchors.
    """
    from src.unstructured.retrieval.services.ranking import RankingService

    stem = RankingService.__new__(RankingService)._morphological_stem
    for word in ("الهويات", "المعلم", "النتائج", "المدرسة", "الكلمات", "كتب"):
        assert stem(word) is None, f"a stemmer fired on Arabic: {word}"
