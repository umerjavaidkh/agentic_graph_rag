"""tests/unstructured/test_region_description_unit.py — a table must say what it is.

Tables arrived from the parser as pipe grids and were stored, titled and
embedded from exactly that: the title was the grid's first row, and what
went into the vector was punctuation, `<br>` markers and column padding.
Fully embedded, effectively unsearchable -- which is why every recall miss
was a Region node.
"""
from __future__ import annotations

from src.unstructured.graph.region_description import (
    captions_on_page,
    describe,
    parse_grid,
    region_title,
)

_GRID = (
    "| Tier | Nodes | Server |\n"
    "|---|---|---|\n"
    "| Tier 1 | 10 | alpha |\n"
    "| Tier 2 | 20 | beta |"
)


def test_grid_parses_into_header_and_body():
    header, body = parse_grid(_GRID)

    assert header == ["Tier", "Nodes", "Server"]
    assert len(body) == 2  # the |---| separator row is not a row of data


def test_br_markers_are_not_left_inside_cells():
    header, _ = parse_grid("| No<br>des | Server |\n|---|---|\n| 1 | a |")

    assert header == ["No des", "Server"]


def test_description_names_the_columns_a_question_would_use():
    out = describe("table", _GRID, "Table 3.2. Tiers of the framework")

    assert "Table 3.2. Tiers of the framework." in out
    assert "Tier, Nodes, Server" in out
    assert "2 rows" in out
    assert "Tier 1" in out  # a real value, so a row lookup can match


def test_caption_is_recovered_from_the_page_not_the_grid():
    page = "prose above\nTable 3.2. Tiers of the framework\nprose below"

    assert captions_on_page(page, "table") == ["Table 3.2. Tiers of the framework"]


def test_figure_captions_do_not_answer_for_tables():
    page = "Figure 1: CSF Core structure\nTable 2. Something else"

    assert captions_on_page(page, "figure") == ["Figure 1. CSF Core structure"]
    assert captions_on_page(page, "table") == ["Table 2. Something else"]


def test_title_is_the_caption_never_the_first_grid_row():
    """The observed bug: titles like "|  | Interest Expenses . . . 45<br>"."""
    assert region_title("table", "Table 3.2. Tiers", 1, 12) == "Table 3.2. Tiers"


def test_title_falls_back_to_a_label_when_the_page_has_no_caption():
    assert region_title("table", "", 2, 12) == "Table 2 (PDF page 12)"


def test_uncaptioned_table_still_describes_its_shape():
    out = describe("table", _GRID, "", index=2, page=12)

    assert "Table 2 on page 12." in out
    assert "Tier, Nodes, Server" in out


def test_figure_keeps_its_own_text_since_it_has_no_grid():
    out = describe("figure", "CSF Core: Govern, Identify, Protect", "Figure 1. CSF Core")

    assert "Figure 1. CSF Core." in out
    assert "Govern, Identify, Protect" in out


def test_description_is_bounded():
    huge = "| a | b |\n|---|---|\n" + "\n".join(f"| row{i} | {i} |" for i in range(5000))

    assert len(describe("table", huge, "Table 1. Big")) <= 900


def test_a_table_without_a_grid_keeps_its_prose():
    """Parsers disagree: rtldoc emits pipe grids, LightPdfParser emits prose.

    Describing only the grid replaced a prose table's contents with the
    sentence "Table 3 on page 12." -- strictly worse than the raw text this
    change exists to improve on.
    """
    out = describe("table", "Withholding depends on filing status and allowances", "",
                   index=3, page=12)

    assert "Table 3 on page 12." in out
    assert "filing status and allowances" in out


def test_a_reference_to_a_table_in_body_text_is_not_a_caption():
    """Documents refer to their own tables mid-sentence.

    Taking the rest of that line produced the title
    "Table 1-1. 4. There are changes in the tax law that af-".
    """
    page = "as shown in Table 1-1. 4. There are changes in the tax law that af-"

    assert captions_on_page(page, "table") == []


def test_a_real_caption_still_reads_as_one():
    page = "Table 1-1. Personal and Financial Changes"

    assert captions_on_page(page, "table") == ["Table 1-1. Personal and Financial Changes"]
