"""tests/test_page_numbers_label_detection_unit.py — detect_document_page_label().

Covers a real bug found live on two different documents: the "p."/"pg."
abbreviation pattern had an optional period, so a bare "p" could match the
start of any unrelated word ending a candidate line — a repeated running
header "COMPLIANCE POLICY" has " P" immediately before "OLICY", greedily
captured whole as if "OLICY" were the printed page label. The same class
of bug hit JPMorgan's 10-K too ("PCI" -> "CI"). Also covers the fix's own
follow-on: widening the search window to catch labels that sit a few
lines after a multi-line repeated header, without exposing the weak
bare-digit fallback to unrelated numbers deeper in body content (e.g. a
financial table's "2016" fiscal-year column header).

Run with:
    python -m pytest tests/test_page_numbers_label_detection_unit.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_root = Path(__file__).resolve().parents[1]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from src.document.page_numbers import detect_document_page_label


def test_none_for_empty_text():
    assert detect_document_page_label("") is None
    assert detect_document_page_label("   \n  ") is None
    assert detect_document_page_label(None) is None


def test_plain_bare_digit_footer():
    text = "Some heading\n\nBody text here.\nMore body text.\n\n43"
    assert detect_document_page_label(text) == "43"


def test_explicit_page_prefix():
    text = "Body text.\nMore body.\n\nPage 12"
    assert detect_document_page_label(text) == "12"


def test_roman_numeral_footer():
    text = "Front matter content.\nMore front matter.\n\nxiv"
    assert detect_document_page_label(text) == "xiv"


def test_two_line_header_then_n_of_m_label():
    """The exact STRATEC shape: two-line repeated running header, then the
    "N / M" label, then body — the label must not be missed just because
    it's the 3rd line, past a naively narrow first-N-lines window."""
    text = (
        "COMPLIANCE POLICY\n"
        "CORPORATE COMPLIANCE POLICY 2025\n"
        "6 / 12\n"
        "4.1.1\n"
        "NOTE: HOW CAN I AVOID CONFLICTS WHEN IT COMES TO GIFTS AND BENEFITS?\n"
        "Gifts and benefits are usually non-cash benefits..."
    )
    assert detect_document_page_label(text) == "6"


def test_three_line_header_then_bare_number_label():
    """JPMorgan's actual shape on a subset of pages: three-line repeated
    running header/footer with the bare number as the third line."""
    text = "JPMorgan Chase & Co./2016 Annual\nReport\n37"
    assert detect_document_page_label(text) == "37"


def test_running_header_does_not_produce_garbage_label():
    """Regression: 'COMPLIANCE POLICY' must not match as a fragment
    ('OLICY') via the p./pg. abbreviation pattern. With no real label
    anywhere in the text, the correct answer is None, not a fragment."""
    text = "COMPLIANCE POLICY\nCORPORATE COMPLIANCE POLICY 2025\nSome body text with no footer."
    assert detect_document_page_label(text) != "OLICY"
    assert detect_document_page_label(text) is None


def test_pci_style_running_header_does_not_produce_garbage_label():
    """Regression: the same bug class hit JPMorgan's 10-K too — a line
    ending in '...excluding PCI loans' must not match as 'CI'."""
    text = (
        "Notes to consolidated financial statements\n"
        "Residential real estate - excluding PCI\n"
        "loans\n"
        "December 31,\n"
        "(in millions, except ratios)\n"
        "Home equity"
    )
    assert detect_document_page_label(text) != "CI"


def test_bare_number_deep_in_wide_window_is_not_mistaken_for_label():
    """Regression: widening the search window to catch a 3rd-line label
    (see test_two_line_header_then_n_of_m_label) must not let the weak
    bare-digit fallback pick up an unrelated number from body content that
    only coincidentally falls within the wider window — verified live: a
    financial table's fiscal-year column header ('2016') on a real
    ~164-line JPMorgan 10-K page landed inside a naively widened window and
    got mistaken for a page label. The page must be long enough that the
    table header sits outside the narrow first-2/last-4 window (which
    predates this fix and has its own separate, pre-existing risk on very
    short pages) — this test is specifically about the *wide* window's
    weak-pattern exposure, not the narrow one's."""
    body_filler = "\n".join(f"Body line {i} with unrelated prose content." for i in range(60))
    text = (
        "Noninterest expense\n"
        "Year ended December\n"
        "31,\n"
        "(in millions)\n"
        "2016\n"
        "2015\n"
        f"{body_filler}\n"
        "Final paragraph continues here with no real footer label present."
    )
    assert detect_document_page_label(text) != "2016"


def test_p_dot_abbreviation_still_matches_real_label():
    """The mandatory-period fix must not break the legitimate 'p. 43' /
    'pg. 43' abbreviation forms it was meant to support."""
    text = "Body text.\nMore body.\n\np. 43"
    assert detect_document_page_label(text) == "43"
    text2 = "Body text.\nMore body.\n\npg. 7"
    assert detect_document_page_label(text2) == "7"
