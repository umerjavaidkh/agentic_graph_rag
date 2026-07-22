"""document/table_aware/parser.py — LightPdfParser variant that vetoes
table-row/column-header fragments and repeated running headers/footers
from heading detection.

Registered as a separate backend (".pdf:table-aware") via parser_registry.py
so it can be A/B compared against the default "light" backend on the same
document using scripts/validate_ingestion.py — no ingestion pipeline or
retrieval changes needed to compare them.
"""
from __future__ import annotations

import re
from collections import defaultdict

import fitz

from ..light.parser import LightPdfParser, _PageExtract, _PdfBlock

# Safety ceiling on how far a table region gets padded upward to catch
# header rows / stub labels sitting above the detector's strict boundary
# (see _padded_table_bboxes). Bounds the *primary* signal (a multiple of
# the table's own row height, which scales with the document's own
# typography) rather than replacing it — this stops an outlier table from
# padding far enough to swallow an unrelated heading several lines above it.
_MAX_TABLE_PAD_PT = 60.0

# Repeated-header/footer detection thresholds — both required, so a real
# heading reused a few times across independent chapters of a long
# document (e.g. "Overview" appearing 3 times in a 200-page report) isn't
# mistaken for a running header. Running headers/footers overwhelmingly
# clear both: printed on most pages, at a near-identical vertical position
# every time (a real heading's page position varies with content flow).
_REPEAT_MIN_PAGES = 3
_REPEAT_MIN_PAGE_FRACTION = 0.15
_REPEAT_Y_TOLERANCE_PT = 20.0

_WS_RE = re.compile(r"\s+")


class TableAwarePdfParser(LightPdfParser):
    """Vetoes non-heading content from heading detection via three
    complementary, schema-agnostic signals — none reference this or any
    specific document's vocabulary:

    1. Digit-dominance: a real heading is virtually always letter-dominant;
       a merged table-row fragment ("8-K 4.1 5/3/13", "2027 1,674 59 1,733")
       is usually digit-dominant.
    2. Geometric table-region membership: PyMuPDF's built-in find_tables()
       (already a dependency, no new package) detects real table bounding
       boxes via ruling lines. Padded upward by a multiple of the table's
       own row height — not a fixed point value — to also catch header
       rows and first-column labels ("Net sales:", "September 30,") that
       line-based table detectors commonly place just outside the strict
       boundary, since there's often no visible rule above them.
    3. Repeated running headers/footers: text that recurs verbatim across
       many pages at a consistent vertical position (e.g. a "Table of
       Contents" navigation label printed atop most pages of a filing) is
       a page-furniture artifact, not real section content — a standard
       PDF-preprocessing technique, not specific to any document or
       vocabulary. Needs document-wide knowledge (unlike signals 1-2,
       which only need one block's own text/bbox), so it's a separate pass
       over all pages rather than a per-block check.

    Digit-dominance alone doesn't catch letter-dominant column headers or
    running headers; table-region membership alone doesn't catch a
    misparsed heading near but not inside a detected table, or a header
    repeated outside any table. Together the three cover the distinct
    failure modes observed across real SEC filings from different filers.
    """

    def _extract_pages(self, source, doc: fitz.Document) -> list[_PageExtract]:
        extracts = super()._extract_pages(source, doc)
        for extract, page in zip(extracts, doc):
            table_bboxes = self._padded_table_bboxes(page)
            if not table_bboxes:
                continue
            for block in extract.blocks:
                if block.bbox and self._bbox_in_any(block.bbox, table_bboxes):
                    block.in_table_region = True
        self._flag_repeated_headers(extracts)
        return extracts

    @staticmethod
    def _normalize_for_repeat_check(text: str) -> str:
        joined = " ".join(ln.strip() for ln in (text or "").splitlines() if ln.strip())
        return _WS_RE.sub(" ", joined).strip().lower()

    def _flag_repeated_headers(self, extracts: list[_PageExtract]) -> None:
        total_pages = len(extracts)
        if total_pages < _REPEAT_MIN_PAGES:
            return

        # normalized text -> list of (page, y0, block)
        groups: dict[str, list[tuple[int, float, _PdfBlock]]] = defaultdict(list)
        for extract in extracts:
            for block in extract.blocks:
                if block.kind != "text" or block.low_confidence or not block.bbox:
                    continue
                norm = self._normalize_for_repeat_check(block.text)
                if not (3 <= len(norm) <= 160) or len(norm.split()) > 18:
                    continue
                groups[norm].append((extract.page, block.bbox[1], block))

        min_pages_needed = max(_REPEAT_MIN_PAGES, int(total_pages * _REPEAT_MIN_PAGE_FRACTION))
        for occurrences in groups.values():
            distinct_pages = {p for p, _, _ in occurrences}
            if len(distinct_pages) < min_pages_needed:
                continue
            y_values = [y for _, y, _ in occurrences]
            if max(y_values) - min(y_values) > _REPEAT_Y_TOLERANCE_PT:
                continue  # position varies too much to be a running header/footer
            for _, _, block in occurrences:
                block.is_repeated_header = True

    @staticmethod
    def _padded_table_bboxes(page: fitz.Page) -> list[tuple[float, float, float, float]]:
        try:
            tables = page.find_tables()
        except Exception:
            return []
        padded: list[tuple[float, float, float, float]] = []
        for t in tables.tables:
            x0, y0, x1, y1 = t.bbox
            row_height = (y1 - y0) / max(t.row_count, 1)
            pad = min(row_height * 3, _MAX_TABLE_PAD_PT)
            padded.append((x0, max(0.0, y0 - pad), x1, y1))
        return padded

    @staticmethod
    def _bbox_in_any(
        bbox: list[float], regions: list[tuple[float, float, float, float]], tolerance: float = 1.0
    ) -> bool:
        bx0, by0, bx1, by1 = bbox
        for rx0, ry0, rx1, ry1 in regions:
            if rx0 - tolerance <= bx0 and ry0 - tolerance <= by0 and rx1 + tolerance >= bx1 and ry1 + tolerance >= by1:
                return True
        return False

    def _is_heading(self, block: _PdfBlock, font_threshold: float) -> bool:
        if block.in_table_region:
            return False
        if block.is_repeated_header:
            return False
        if self._looks_like_table_fragment(block.text):
            return False
        return super()._is_heading(block, font_threshold)

    @staticmethod
    def _looks_like_table_fragment(text: str) -> bool:
        letters = sum(1 for ch in text if ch.isalpha())
        digits = sum(1 for ch in text if ch.isdigit())
        if digits == 0:
            return False
        return digits >= letters and digits >= 2
