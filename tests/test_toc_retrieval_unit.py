"""Offline tests for TOC heuristics."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.retrieval.unstructured.toc_retrieval import (
    include_in_outline_fallback,
    score_page_text_as_toc,
    section_title_is_toc,
)


def test_score_toc_page():
    toc = """
    Table of Contents
    1. Introduction ................ 7
    2. Methods ..................... 12
    3. Results ..................... 18
    """
    assert score_page_text_as_toc(toc) >= 0.5
    body = "Box 1\nThis box describes proximity tracing in detail across many pages."
    assert score_page_text_as_toc(body) < 0.4


def test_score_toc_page_number_on_own_line():
    # Common layout: entry title then page number (arabic/roman) on the next line.
    toc = (
        "Contents\n"
        "acknowledgements\nv\n"
        "List of abbreviations\nvi\n"
        "EXECUTIVE SUMMARY\nvii\n"
        "1. Introduction\n2\n"
        "1.1. Goal of this report\n5\n"
        "2. Go.Data activities and objectives\n9\n"
    )
    assert score_page_text_as_toc(toc) >= 0.6


def test_score_toc_page_form_10k_index_heading():
    # "Form 10-K Index" is a real, standard alternate TOC heading used by
    # many actual filers (banks in particular) instead of "Table of
    # Contents" — real text pattern from a JPMorgan 10-K.
    toc = (
        "Form 10-K Index\n\n"
        "Part I\nPage\n\n"
        "Item 1. Business.\n1\n\n"
        "Overview\n1\n\n"
        "Business segments\n1\n\n"
        "Competition\n1\n\n"
        "Supervision and regulation\n1\n\n"
    )
    assert score_page_text_as_toc(toc) > 0.42


def test_running_header_page_loses_to_real_toc_with_no_heading():
    """Regression: many SEC filings print "Table of Contents" as a running
    header on every page that follows the real TOC, not just the TOC page
    itself. Verified live on Tesla's 10-K: the real Item/page-number
    listing has no heading at all (its first line is the column header
    "Page"), while the very next page -- pure "Forward-Looking Statements"
    prose with zero real entries -- carries that running header and used
    to win via the heading bonus alone, so "what is the table of contents"
    returned Forward-Looking-Statements boilerplate instead of the actual
    TOC."""
    real_toc_no_heading = (
        "Page\nPART I.\nItem 1.\nBusiness\n2\nItem 1A.\nRisk Factors\n12\n"
        "Item 1B.\nUnresolved Staff Comments\n27\nItem 1C.\nCybersecurity\n27\n"
        "Item 2.\nProperties\n28\nItem 3.\nLegal Proceedings\n28\n"
    )
    running_header_decoy = (
        "Table of Contents\nForward-Looking Statements\n"
        "The discussions in this Annual Report on Form 10-K contain forward-looking "
        "statements within the meaning of the Private Securities Litigation Reform Act "
        "of 1995. Forward-looking statements are based on assumptions with respect to "
        "the future and management's current expectations, involve certain risks and "
        "uncertainties and are not guarantees. These forward-looking statements include, "
        "but are not limited to, statements concerning supply chain constraints."
    )
    real_score = score_page_text_as_toc(real_toc_no_heading)
    decoy_score = score_page_text_as_toc(running_header_decoy)
    assert real_score > decoy_score
    assert real_score > 0.42  # must clear _toc_find_best_page's selection threshold


def test_outline_filters_boxes():
    assert not include_in_outline_fallback("Box 8", 2, "Section")
    assert include_in_outline_fallback("1. Introduction", 2, "Section")
    assert include_in_outline_fallback("EXECUTIVE SUMMARY", 1, "Section")
    assert section_title_is_toc("Table of Contents")


def main() -> None:
    test_score_toc_page()
    test_score_toc_page_number_on_own_line()
    test_outline_filters_boxes()
    print("toc retrieval unit checks: OK")


if __name__ == "__main__":
    main()
