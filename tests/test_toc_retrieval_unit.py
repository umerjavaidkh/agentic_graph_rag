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


def test_outline_filters_boxes():
    assert not include_in_outline_fallback("Box 8", 2, "Section")
    assert include_in_outline_fallback("1. Introduction", 2, "Section")
    assert include_in_outline_fallback("EXECUTIVE SUMMARY", 1, "Section")
    assert section_title_is_toc("Table of Contents")


def main() -> None:
    test_score_toc_page()
    test_outline_filters_boxes()
    print("toc retrieval unit checks: OK")


if __name__ == "__main__":
    main()
