"""tests/test_parser_table_aware_unit.py — TableAwarePdfParser's heading veto
and its parser_registry wiring.

Run with:
    python -m pytest tests/test_parser_table_aware_unit.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_root = Path(__file__).resolve().parents[1]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

def _drop_fake_document_stubs() -> None:
    """Drop stale fake stubs another test file left in sys.modules (e.g.
    test_scalable_pipeline_unit.py stubs src.document.parser_registry as a
    bare types.ModuleType with a mocked get_parser) — only if they're fake,
    never a genuinely-imported real module. Pytest collects every test
    file's module-level code before running any test function, so this
    must run at TEST-CALL time (inside each test that imports src.document.*
    locally), not just once at this file's own import time.
    """
    for _mod_name in list(sys.modules):
        if _mod_name == "src.document" or _mod_name.startswith("src.document."):
            _mod = sys.modules[_mod_name]
            if not hasattr(_mod, "__file__") and not hasattr(_mod, "__path__"):
                del sys.modules[_mod_name]


_drop_fake_document_stubs()

from src.document.table_aware.parser import TableAwarePdfParser


@pytest.mark.parametrize(
    "text,expected",
    [
        ("2027\n1,674\n59\n1,733", True),  # lease/debt schedule row
        ("8-K\n4.1\n5/3/13", True),  # exhibit index row
        ("$ 10,930", True),
        ("Item 7A. Quantitative and Qualitative Disclosures About Market Risk", False),
        ("September 30,", False),  # column header — letter-dominant, not vetoed
        ("Net sales:", False),
        ("PART II", False),
        ("Risk Factors", False),
        ("", False),
    ],
)
def test_looks_like_table_fragment(text, expected):
    assert TableAwarePdfParser._looks_like_table_fragment(text) is expected


def test_registered_under_table_aware_backend():
    # Deliberately checks type name/module rather than isinstance()/type() is:
    # _drop_fake_document_stubs() forces a fresh re-import of src.document.*,
    # which can produce a TableAwarePdfParser class object distinct (by
    # identity) from the one imported at this file's top — a duplicate-
    # module-object artifact of cross-file sys.modules stub pollution during
    # pytest collection, not a real behavioral difference.
    _drop_fake_document_stubs()
    from src.document.parser_registry import get_parser

    parser = get_parser("doc.pdf", backend="table-aware")
    assert type(parser).__name__ == "TableAwarePdfParser"
    assert type(parser).__module__ == "src.document.table_aware.parser"


def test_light_backend_unaffected():
    # Deliberately passes backend="light" explicitly rather than relying on
    # get_parser()'s ambient PDF_PARSER_BACKEND fallback — that env var is a
    # deployment/session setting (e.g. temporarily set to "table-aware" for
    # a live re-ingestion comparison) and asserting against its current
    # value would make this test depend on ambient state instead of the
    # registry's own explicit-backend resolution behavior.
    _drop_fake_document_stubs()
    from src.document.parser_registry import get_parser

    parser = get_parser("doc.pdf", backend="light")
    assert type(parser).__name__ == "LightPdfParser"
    assert type(parser).__module__ == "src.document.light.parser"


# ── geometric table-region veto ──────────────────────────────────────────


@pytest.mark.parametrize(
    "bbox,regions,expected",
    [
        # Fully inside a single region.
        ([10, 10, 20, 20], [(0, 0, 100, 100)], True),
        # Outside every region.
        ([200, 200, 210, 210], [(0, 0, 100, 100)], False),
        # No regions at all.
        ([10, 10, 20, 20], [], False),
        # Inside the second of several regions.
        ([55, 55, 60, 60], [(0, 0, 10, 10), (50, 50, 70, 70)], True),
        # Straddles a region's edge — not fully contained, so not "in" it.
        ([95, 95, 105, 105], [(0, 0, 100, 100)], False),
    ],
)
def test_bbox_in_any(bbox, regions, expected):
    assert TableAwarePdfParser._bbox_in_any(bbox, regions) is expected


def test_padded_table_bboxes_pads_by_row_height_not_a_fixed_value():
    """Padding must scale with the table's own geometry (row height), not a
    hardcoded point value — otherwise it wouldn't generalize across
    documents with different font sizes/table densities."""

    class _FakeTable:
        def __init__(self, bbox, row_count):
            self.bbox = bbox
            self.row_count = row_count

    class _FakeTables:
        def __init__(self, tables):
            self.tables = tables

    class _FakePage:
        def __init__(self, tables):
            self._tables = tables

        def get_drawings(self):
            return []

        def find_tables(self):
            return _FakeTables(self._tables)

    # Table A: short rows (row_height=5) -> small pad.
    page_a = _FakePage([_FakeTable((0, 100, 50, 150), row_count=10)])  # row_height=5
    (x0, y0, x1, y1) = TableAwarePdfParser._padded_table_bboxes(page_a)[0]
    assert y0 == pytest.approx(100 - 5 * 3)

    # Table B: tall rows -> pad hits the safety ceiling instead of scaling unbounded.
    page_b = _FakePage([_FakeTable((0, 100, 50, 500), row_count=2)])  # row_height=200
    (x0, y0, x1, y1) = TableAwarePdfParser._padded_table_bboxes(page_b)[0]
    assert y0 == pytest.approx(100 - 60)  # capped at _MAX_TABLE_PAD_PT


def test_find_tables_failure_degrades_to_no_veto(monkeypatch):
    """A page find_tables() can't parse shouldn't crash ingestion — just
    skip the geometric veto for that page (digit-ratio veto still applies)."""

    class _BrokenPage:
        def get_drawings(self):
            return []

        def find_tables(self):
            raise RuntimeError("layout analysis failed")

    assert TableAwarePdfParser._padded_table_bboxes(_BrokenPage()) == []


def test_dense_graphics_page_skips_find_tables_entirely(monkeypatch):
    """Regression: PyMuPDF's find_tables() (default "lines" strategy) does
    graphics-based table inference, not just literal ruling-line detection —
    its internal neighbor-checking hung indefinitely on a real page with
    23k+ vector drawing primitives (a dense chart/map, not an actual table;
    a real table page — even a dense financial-filing one — had ~194).
    Skip find_tables() entirely above a generous ceiling instead of calling
    into a known-pathological path."""
    from src.document.table_aware.parser import _MAX_PAGE_DRAWINGS_FOR_TABLE_DETECTION

    class _DenseGraphicsPage:
        def get_drawings(self):
            return [object()] * (_MAX_PAGE_DRAWINGS_FOR_TABLE_DETECTION + 1)

        def find_tables(self):
            raise AssertionError("find_tables() must not be called on a dense-graphics page")

    assert TableAwarePdfParser._padded_table_bboxes(_DenseGraphicsPage()) == []


def test_normal_graphics_count_still_calls_find_tables():
    class _FakeTable:
        bbox = (0, 100, 50, 150)
        row_count = 10

    class _FakeTables:
        tables = [_FakeTable()]

    class _NormalPage:
        def get_drawings(self):
            return [object()] * 194  # observed count on a real filing table page

        def find_tables(self):
            return _FakeTables()

    assert TableAwarePdfParser._padded_table_bboxes(_NormalPage()) != []


def test_parses_real_sample_pdf_without_error():
    """Light regression check against a real, small, committed sample PDF —
    not asserting exact structure (that's content-specific), just that the
    table-region pass runs end-to-end without crashing."""
    parser = TableAwarePdfParser()
    nodes, edges = parser.parse("sample_data_to_test/unstructured/rag_document.pdf")
    assert nodes
    assert any((n.type.value if hasattr(n.type, "value") else n.type) == "Page" for n in nodes)


# ── repeated running header/footer veto ──────────────────────────────────


def _make_extract(page_num, blocks_data):
    from src.document.light.parser import _PageExtract, _PdfBlock

    blocks = [
        _PdfBlock(text=text, page=page_num, bbox=[10.0, y0, 200.0, y0 + 10.0])
        for text, y0 in blocks_data
    ]
    return _PageExtract(page=page_num, text="\n".join(t for t, _ in blocks_data), blocks=blocks)


def test_flags_text_repeated_across_many_pages_at_consistent_position():
    # "Table of Contents" on 5 of 6 pages, always near y0=20 (a running header).
    extracts = [
        _make_extract(i, [("Table of Contents", 20.0), (f"Real content {i}", 100.0)])
        if i != 3
        else _make_extract(i, [(f"Real content {i}", 100.0)])
        for i in range(1, 7)
    ]
    parser = TableAwarePdfParser()
    parser._flag_repeated_headers(extracts)

    flagged = [
        b.text
        for e in extracts
        for b in e.blocks
        if b.is_repeated_header
    ]
    assert flagged == ["Table of Contents"] * 5
    # Real per-page content must never be flagged.
    assert all(not b.is_repeated_header for e in extracts for b in e.blocks if b.text.startswith("Real content"))


def test_does_not_flag_a_heading_reused_a_few_times():
    # "Overview" appears as a real subsection heading in only 2 of 20 pages —
    # far below both the count and percentage thresholds.
    extracts = [_make_extract(i, [(f"Page {i} body", 100.0)]) for i in range(1, 21)]
    extracts[0].blocks.append(_make_extract(1, [("Overview", 50.0)]).blocks[0])
    extracts[10].blocks.append(_make_extract(11, [("Overview", 50.0)]).blocks[0])

    parser = TableAwarePdfParser()
    parser._flag_repeated_headers(extracts)

    assert not any(b.text == "Overview" and b.is_repeated_header for e in extracts for b in e.blocks)


def test_does_not_flag_repeated_text_at_varying_vertical_position():
    # Same text repeated often, but position drifts a lot each time — not
    # consistent with a running header/footer, so it's left alone.
    extracts = [
        _make_extract(i, [("Confidential", 20.0 + i * 15.0)])
        for i in range(1, 8)
    ]
    parser = TableAwarePdfParser()
    parser._flag_repeated_headers(extracts)

    assert not any(b.is_repeated_header for e in extracts for b in e.blocks)


def test_repeated_header_veto_wired_into_is_heading():
    """is_repeated_header now travels via Block.extra (_block_to_ir),
    read by Axis1StructuralBuilder's single collapsed _is_heading rather
    than a per-class TableAwarePdfParser._is_heading override."""
    from src.document.light.parser import _PdfBlock
    from src.graph.axis1_structural import Axis1StructuralBuilder

    block = _PdfBlock(text="Table of Contents", page=5, bbox=[10.0, 20.0, 200.0, 30.0])
    block.is_repeated_header = True
    parser = TableAwarePdfParser()
    ir_block = parser._block_to_ir(block)
    assert Axis1StructuralBuilder()._is_heading(ir_block, font_threshold=999.0) is False


def test_short_document_skips_repeat_detection():
    """Fewer pages than _REPEAT_MIN_PAGES: nothing to compare against, so
    the pass is a no-op rather than flagging on thin evidence."""
    extracts = [_make_extract(1, [("Header", 20.0)]), _make_extract(2, [("Header", 20.0)])]
    parser = TableAwarePdfParser()
    parser._flag_repeated_headers(extracts)
    assert not any(b.is_repeated_header for e in extracts for b in e.blocks)
