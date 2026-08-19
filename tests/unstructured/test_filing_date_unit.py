"""tests/test_filing_date_unit.py — filing-date fast-path.

Guards the fix for a real reported failure: "What is the exact filing date
of this Form 10-Q?" answered with the period-end date (e.g. "March 29,
2026") instead of the actual EDGAR filing/submission date (e.g. "May 5,
2026"). Root cause: the real filing date is usually not printed anywhere in
the PDF body at all (EDGAR's "Filed:" stamp lives in the filing's
HTML/index wrapper, not the document), so retrieval pulled whichever
date-heavy MD&A chunk ranked highest and the LLM picked the most prominent
date in it — a classic date/entity-confusion failure, not a retrieval-
ranking problem, since the correct answer genuinely isn't in the text.

Two pieces:
1. is_filing_date_question() — must fire for "filing date"/"when was this
   filed" phrasing, must NOT fire for "period ended" phrasing (a different,
   legitimately-in-the-text question).
2. extract_filing_date_from_filename() — pulls the trailing YYYY-MM-DD from
   a source_filename following the SEC-EDGAR sample corpus's
   TICKER_FORM_YYYY-MM-DD.ext convention (date sourced from EDGAR's own
   filingDate field, not the period-end date — see
   scripts/fetch_sec_edgar_corpus.py), including filenames with the
   ingestion job_id prefix, and returns None gracefully for filenames with
   no such pattern (non-SEC documents) rather than guessing.

Run with:
    python -m pytest tests/test_filing_date_unit.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest


from src.unstructured.document.versioning import extract_filing_date_from_filename
from src.unstructured.retrieval.query_intent import is_filing_date_question


@pytest.mark.parametrize("query", [
    "What is the exact filing date of this Form 10-Q?",
    "When was this filed?",
    "When was this 10-Q filed?",
    "What is the date filed?",
    "What date was this filed on?",
    "What is the date of filing?",
])
def test_fires_for_filing_date_questions(query):
    assert is_filing_date_question(query) is True


@pytest.mark.parametrize("query", [
    # The exact confusion this fixes: period-end phrasing is a different,
    # legitimately-in-the-text question and must not be fast-pathed here.
    "What is the period ended for this Pfizer 10-Q?",
    "For the quarterly period ended March 29, 2026",
    "What was net income for the quarter ended?",
    "What does this filing say about risk factors?",
    "Tell me about Human Capital Management in this filing.",
])
def test_does_not_fire_for_period_or_unrelated_questions(query):
    assert is_filing_date_question(query) is False


@pytest.mark.parametrize(
    "filename,expected",
    [
        ("PFE_10-Q_2026-05-05.pdf", "2026-05-05"),
        ("fa95b96ca817488198db5bb0a6ee1f40_PFE_10-Q_2026-05-05.pdf", "2026-05-05"),
        ("4eae755b54144273ae9af0f86315222f_AMZN_10-Q_2016-07-29.pdf", "2016-07-29"),
        ("3f8927d2461c44318acbfff465533dbd_WMT_10-K_2026-03-13.pdf", "2026-03-13"),
    ],
)
def test_extracts_filing_date_from_sec_edgar_filenames(filename, expected):
    assert extract_filing_date_from_filename(filename) == expected


@pytest.mark.parametrize(
    "filename",
    [
        "ffa1b256f1024f9b82705af9be892865_rag_document.pdf",  # non-SEC demo doc
        "",
        None,
    ],
)
def test_returns_none_for_non_matching_filenames(filename):
    assert extract_filing_date_from_filename(filename) is None
