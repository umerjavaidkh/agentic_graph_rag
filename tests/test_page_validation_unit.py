"""tests/test_page_validation_unit.py — pure page-level coverage scoring
(src/document/page_validation.py). page_report.py's Neo4j/parser wiring is
integration-level and not covered here, same boundary as
ontology_report.py/ontology_validation.py and graph_snapshot.py's live
query functions.

Run with:
    python -m pytest tests/test_page_validation_unit.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

_root = Path(__file__).resolve().parents[1]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from src.document.ir import DocumentIR, PageBlock
from src.document.page_validation import check_construction_coverage, score_page, summarize_pages
from src.models import DKGNode, NodeType


def test_full_coverage_page_passes():
    r = score_page(
        page_number=1,
        raw_text="the quick brown fox jumps over the lazy dog",
        graph_text="the quick brown fox jumps over the lazy dog",
        entity_count=2,
        edge_count=1,
    )
    assert r.status == "PASS"
    assert r.coverage == 1.0
    assert r.entity_count == 2
    assert r.edge_count == 1


def test_low_coverage_page_flagged_for_reextraction():
    r = score_page(
        page_number=5,
        raw_text=" ".join(["word"] * 100),
        graph_text=" ".join(["word"] * 10),  # 10% of the raw content made it into the graph
        entity_count=0,
        edge_count=0,
    )
    assert r.coverage == 0.1
    assert r.status == "RE_EXTRACT_REQUIRED"


def test_coverage_at_exact_threshold_passes():
    r = score_page(
        page_number=1,
        raw_text=" ".join(["word"] * 10),
        graph_text=" ".join(["word"] * 5),  # exactly PAGE_COVERAGE_PASS_THRESHOLD (0.5)
        entity_count=0,
        edge_count=0,
    )
    assert r.coverage == 0.5
    assert r.status == "PASS"


def test_empty_raw_and_graph_text_passes_trivially():
    """An image-only page with no extractable text on either side isn't a
    failure -- there's nothing to have lost."""
    r = score_page(page_number=3, raw_text="", graph_text="", entity_count=0, edge_count=0)
    assert r.status == "PASS"
    assert r.coverage == 1.0


def test_graph_text_with_no_raw_text_flagged_as_reparse_mismatch():
    """Graph has content for a page the re-parse says is empty -- a
    disagreement between the stored source and the ingested revision
    (different page count/offset), not a coverage gap to silently pass."""
    r = score_page(page_number=7, raw_text="", graph_text="some content", entity_count=1, edge_count=0)
    assert r.status == "REPARSE_MISMATCH"


def test_coverage_caps_at_one_even_if_graph_text_is_longer():
    r = score_page(
        page_number=1,
        raw_text="a b c",
        graph_text="a b c d e f g h",  # graph text expanded beyond the raw page (surrounding context, etc.)
        entity_count=0,
        edge_count=0,
    )
    assert r.coverage == 1.0
    assert r.status == "PASS"


def test_summarize_pages_empty_list():
    s = summarize_pages([])
    assert s == {"page_count": 0, "avg_coverage": 0.0, "pages_failing": 0, "requires_reprocessing": False}


def test_summarize_pages_all_passing():
    pages = [
        score_page(page_number=i, raw_text="a b c d", graph_text="a b c d", entity_count=1, edge_count=1)
        for i in range(1, 4)
    ]
    s = summarize_pages(pages)
    assert s["page_count"] == 3
    assert s["avg_coverage"] == 1.0
    assert s["pages_failing"] == 0
    assert s["requires_reprocessing"] is False


def test_summarize_pages_flags_reprocessing_when_any_page_fails():
    pages = [
        score_page(page_number=1, raw_text="a b c d", graph_text="a b c d", entity_count=1, edge_count=1),
        score_page(page_number=2, raw_text="a b c d e f g h", graph_text="a", entity_count=0, edge_count=0),
    ]
    s = summarize_pages(pages)
    assert s["pages_failing"] == 1
    assert s["requires_reprocessing"] is True


# ── check_construction_coverage ─────────────────────────────────────────
# Runs immediately after Axis1StructuralBuilder.build(), comparing the raw
# DocumentIR pages against whatever Page nodes the structural builder
# actually produced -- catches a heading-detection collapse at ingestion
# time instead of only on an on-demand re-parse later.


def _ir(pages: dict[int, str], page_count: int | None = None) -> DocumentIR:
    return DocumentIR(
        source_name="doc",
        page_count=page_count or max(pages, default=0),
        pages=[PageBlock(page=n, text=t) for n, t in pages.items()],
    )


def _page_node(page_no: int, text: str) -> DKGNode:
    return DKGNode(
        id=f"doc_page_{page_no}", type=NodeType.PAGE, title=f"Page {page_no}",
        text=text, order=page_no, page_start=page_no, page_end=page_no, depth=99,
    )


def test_construction_coverage_all_pages_pass():
    ir = _ir({1: "the quick brown fox", 2: "jumps over the lazy dog"})
    nodes = [_page_node(1, "the quick brown fox"), _page_node(2, "jumps over the lazy dog")]

    report = check_construction_coverage(ir, nodes)

    assert report["summary"]["requires_reprocessing"] is False
    assert report["summary"]["avg_coverage"] == 1.0
    assert [p["page_number"] for p in report["pages"]] == [1, 2]


def test_construction_coverage_detects_collapsed_document():
    """Simulates a heading-detection collapse: every page has real raw
    text, but the structural builder produced Page nodes with no text at
    all for two of the three pages."""
    ir = _ir({
        1: " ".join(["word"] * 50),
        2: " ".join(["word"] * 50),
        3: " ".join(["word"] * 50),
    })
    nodes = [_page_node(1, " ".join(["word"] * 50)), _page_node(2, ""), _page_node(3, "")]

    report = check_construction_coverage(ir, nodes)

    assert report["summary"]["requires_reprocessing"] is True
    assert report["summary"]["pages_failing"] == 2
    statuses = {p["page_number"]: p["status"] for p in report["pages"]}
    assert statuses[1] == "PASS"
    assert statuses[2] == "RE_EXTRACT_REQUIRED"
    assert statuses[3] == "RE_EXTRACT_REQUIRED"


def test_construction_coverage_ignores_non_page_nodes():
    """A Chapter/Section node happening to share a page_start with a page
    number must not be mistaken for that page's graph text -- only actual
    Page-type nodes count."""
    ir = _ir({1: "real page content here"})
    section = DKGNode(
        id="doc_section_1", type=NodeType.SECTION, title="Section 1",
        text="real page content here and much more besides", order=1,
        page_start=1, page_end=1, depth=2,
    )
    page = _page_node(1, "")  # the actual Page node has no text (collapsed)

    report = check_construction_coverage(ir, [section, page])

    assert report["pages"][0]["status"] == "RE_EXTRACT_REQUIRED"


def test_construction_coverage_page_missing_from_graph_entirely():
    """A page the raw extraction saw but the structural builder never
    emitted a Page node for at all (not even an empty one) -- must still
    be reported and flagged, not silently skipped."""
    ir = _ir({1: "some content", 2: "more content over here"})
    nodes = [_page_node(1, "some content")]  # page 2 never built

    report = check_construction_coverage(ir, nodes)

    statuses = {p["page_number"]: p["status"] for p in report["pages"]}
    assert statuses[1] == "PASS"
    assert statuses[2] == "RE_EXTRACT_REQUIRED"
