"""
tests/test_document_patterns_unit.py — src/document/patterns.py number
parsing/hierarchy helpers.

Covers a real gap in SEC-filing structure: "Item 1A." headings don't
start with a digit ("Item " comes first), so the pre-existing
NUMBERED_HEADING regex (built for dot-numbered schemes like "4.5.1")
never matched them at all -- section_number came back None, and "Item 1A"
never nested under "Item 1" except by accident of font-size heuristics.
ITEM_HEADING + the letter-suffix-aware number_depth/parent_number fix
this without touching the dot-numbered path's existing behavior.

Run with:
    python -m pytest tests/test_document_patterns_unit.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

_root = Path(__file__).resolve().parents[1]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from src.document.patterns import number_depth, parent_number, parse_numbered_title


# ── Dot-numbered scheme (regression -- must stay unchanged) ──────────────


def test_dot_numbered_heading_unchanged():
    num, title = parse_numbered_title("4.5 ENVIRONMENTAL PROTECTION")
    assert num == "4.5"
    assert title == "ENVIRONMENTAL PROTECTION"


def test_dot_number_depth_unchanged():
    assert number_depth("4") == 1
    assert number_depth("4.5") == 2
    assert number_depth("4.5.1") == 3


def test_dot_number_parent_unchanged():
    assert parent_number("4.5.1") == "4.5"
    assert parent_number("4.5") == "4"
    assert parent_number("4") is None


def test_plain_title_has_no_number():
    num, title = parse_numbered_title("Executive Summary")
    assert num is None
    assert title == "Executive Summary"


# ── "Item N[Letter]" scheme (new) ─────────────────────────────────────────


def test_item_heading_extracts_number_and_title():
    num, title = parse_numbered_title("Item 1A. Risk Factors")
    assert num == "Item 1A"
    assert title == "Risk Factors"


def test_item_heading_normalizes_casing_and_spacing():
    """"item 1a." / "ITEM 1A" / "Item  1A." must all normalize to the same
    section_number key -- number_map lookups depend on exact string match,
    so a filer's own casing/spacing quirks can't be allowed to fragment it."""
    for text in ("item 1a. Risk Factors", "ITEM 1A Risk Factors", "Item 1A Risk Factors"):
        num, _ = parse_numbered_title(text)
        assert num == "Item 1A", f"failed for {text!r}"


def test_item_heading_without_letter_suffix():
    num, title = parse_numbered_title("Item 7. Management's Discussion")
    assert num == "Item 7"
    assert title == "Management's Discussion"


def test_item_number_depth_letter_suffix_is_one_deeper():
    assert number_depth("Item 1") == 1
    assert number_depth("Item 1A") == 2
    assert number_depth("Item 7A") == 2


def test_item_parent_number_strips_letter_suffix():
    assert parent_number("Item 1A") == "Item 1"
    assert parent_number("Item 7A") == "Item 7"
    assert parent_number("Item 1") is None  # top-level, no parent


def test_non_item_text_starting_with_a_number_word_is_not_matched():
    """Guard against false positives: prose that happens to start with a
    number-like token but isn't "Item N" must not be misparsed."""
    num, title = parse_numbered_title("3D printing is transforming manufacturing")
    assert num is None
    assert title == "3D printing is transforming manufacturing"


# ── numbered-heading false positives on table rows ──────────────────────────
# NUMBERED_HEADING accepts anything after a leading number, so a table's first
# data cell was read as a section number and the rest of the row as its title.
# Verified live on a 264-page 10-K: rows like "24 % 14,703 33 % 2,632" became
# Section nodes whose title was the row text.


def test_numeric_table_row_is_not_a_numbered_heading():
    assert parse_numbered_title("24 % 14,703 33 % 2,632 16 % 2,779") == (
        None,
        "24 % 14,703 33 % 2,632 16 % 2,779",
    )


def test_row_of_bare_figures_is_not_a_numbered_heading():
    assert parse_numbered_title("1,865 1,739 1,771")[0] is None
    assert parse_numbered_title("12 5,416 5,494")[0] is None


def test_real_numbered_headings_still_parse():
    assert parse_numbered_title("4.5 ENVIRONMENTAL PROTECTION") == ("4.5", "ENVIRONMENTAL PROTECTION")
    assert parse_numbered_title("1. Introduction") == ("1", "Introduction")
    assert parse_numbered_title("2.3.1 OpenWHO") == ("2.3.1", "OpenWHO")


def test_heading_containing_numbers_still_parses():
    """Only essentially-numeric rows are rejected — a title that merely
    contains figures is still a title."""
    num, title = parse_numbered_title("4 Revenue in 2024 and 2025")
    assert num == "4" and title == "Revenue in 2024 and 2025"
