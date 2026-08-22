"""Sections must be nameable and distinguishable without a model.

Two failures this guards, both found on the BERT paper:

Diagram labels became headings. rtldoc classifies the Figure 1/3 token
boxes as `figure`/`activity_marker`, but the role was dropped on the way
into the IR, so the font-size rescue saw short, large, isolated text and
promoted it -- 103 sections for a 16-page paper.

Every unnamed section was called "Preamble". The name is assigned when the
section is created, before any of its content exists, and never revisited.
A document whose headings the parser cannot classify ends up with a dozen
identically named sections: useless in a citation, and worse in retrieval,
where they all carry the same search_text.
"""
import pytest

from src.unstructured.graph.axis1_structural import (
    _NON_HEADING_ROLES,
    _name_untitled_sections,
    _title_from_body,
)
from src.unstructured.models import DKGNode, NodeType


def _section(title="Preamble", text="", search_text="Preamble"):
    return DKGNode(
        id="s1", type=NodeType.SECTION, title=title,
        text=text, search_text=search_text, order=1, depth=2,
    )


# ── naming from content ───────────────────────────────────────────────────

def test_the_placeholder_is_not_read_back_as_the_new_name():
    """The section body is written with its own title as the first line, so
    reading line one naively renames "Preamble" to "Preamble" -- a no-op
    that still reports success."""
    node = _section(text="Preamble\nAbstract\nWe introduce a new model.")
    assert _name_untitled_sections([node]) == 1
    assert node.title == "Abstract"


def test_search_text_is_renamed_too():
    """Left as "Preamble", every such section is identical to the ranker."""
    node = _section(text="Preamble\n1 Introduction\nLanguage model pre-training…")
    _name_untitled_sections([node])
    assert node.search_text == "1 Introduction"


@pytest.mark.parametrize(
    "body, expected",
    [
        ("Abstract\nWe introduce BERT.", "Abstract"),
        ("[Table] Table 8: Ablation\nrows", "Table 8: Ablation"),
        # Page furniture names nothing.
        ("7\nAbstract", "Abstract"),
        ("Page 12\n1 Introduction", "1 Introduction"),
        ("iv\nContents", "Contents"),
        # A rendered table row is data that happens to be first.
        ("| Input | [CLS] | my | dog |\nInput/Output Representations", "Input/Output Representations"),
        # A line starting mid-sentence is the previous page's paragraph.
        ("sequences * 512 tokens = 128,000\nHyperparams", "Hyperparams"),
    ],
)
def test_a_name_is_taken_from_the_first_line_that_names_something(body, expected):
    assert _title_from_body(body) == expected


def test_a_long_opening_paragraph_is_clipped_on_a_word_boundary():
    body = "Language model pre-training has been shown to be effective for improving many natural language processing tasks"
    title = _title_from_body(body)
    assert len(title) <= 71
    assert title.endswith("…")
    assert not title[:-1].endswith(" ")


def test_a_section_with_nothing_usable_keeps_its_placeholder():
    """Better an honest placeholder than a name invented from page numbers."""
    node = _section(text="Preamble\n1\n2\n3")
    assert _name_untitled_sections([node]) == 0
    assert node.title == "Preamble"


def test_only_placeholder_sections_are_touched():
    named = _section(title="3.1 Pre-training BERT", text="3.1 Pre-training BERT\nWe pre-train…")
    assert _name_untitled_sections([named]) == 0
    assert named.title == "3.1 Pre-training BERT"


def test_naming_is_deterministic():
    """Runs must be comparable across ingests; a model in this path would
    make the same document produce different names each time."""
    body = "Preamble\nAbstract\nWe introduce a new language representation model."
    assert _title_from_body(body, skip="Preamble") == _title_from_body(body, skip="Preamble")


# ── figure text is not a heading ──────────────────────────────────────────

def test_figure_and_table_roles_can_never_be_headings():
    """The information was always there -- nothing read it until now."""
    assert {"figure", "table", "activity_marker"} <= _NON_HEADING_ROLES
