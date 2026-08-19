"""document/rtldoc_backend/parser.py — LightPdfParser variant that sources
its blocks from rtldoc (https://github.com/umerjavaidkh/rtldoc) instead of
raw PyMuPDF text/font heuristics.

rtldoc reconstructs reading order and bidi text from vector glyph
coordinates (no OCR, no model) and classifies each block's role (heading,
table, figure, passage, ...) directly, and detects table grids from the
PDF's own vector ruling lines rather than a font-size/regex heuristic. That
role classification replaces this repo's own heading-detection heuristics
(font size vs. median, bold, uppercase ratio, numbered-title regex) for any
page rtldoc successfully parses — headings are known outright, not
inferred from typography.

Registered as a separate backend (".pdf:rtldoc") via parser_registry.py,
same pattern as TableAwarePdfParser: only _extract_pages and _block_to_ir
are overridden — chapter/section nesting, page-node building, region
linking, sequential edges, reference detection, and number-hierarchy
linking all live in Axis1StructuralBuilder (src/graph/axis1_structural.py)
now, reached via the same DocumentIR every backend produces.
"""
from __future__ import annotations

from pathlib import Path

import fitz

from ..ir import Block
from ..light.parser import LightPdfParser, _PageExtract, _PdfBlock


class RtldocPdfParser(LightPdfParser):
    def __init__(self) -> None:
        # Per-parse() call, tells _is_heading which rtldoc-sourced blocks
        # rtldoc itself classified as headings — keyed by id() rather than
        # a new _PdfBlock field so the base dataclass (shared with
        # TableAwarePdfParser) doesn't need touching for this parser's needs.
        self._heading_block_ids: set[int] = set()

    def _extract_pages(self, source: Path, doc: fitz.Document) -> list[_PageExtract]:
        try:
            from rtldoc.pipeline import parse_document
        except ImportError:
            print("   ⚠ rtldoc not installed; falling back to lightweight parser entirely")
            return super()._extract_pages(source, doc)

        self._heading_block_ids = set()
        try:
            results = parse_document(str(source))
        except Exception as exc:
            print(f"   ⚠ rtldoc failed to parse the document ({exc}); "
                  "falling back to lightweight parser entirely")
            return super()._extract_pages(source, doc)

        extracts: list[_PageExtract] = []
        for i, result in enumerate(results):
            page_no = result.page
            page = doc[i] if i < doc.page_count else None

            # rtldoc declined this page outright (its own born_digital
            # check failed, e.g. a scanned page, or a rare false negative
            # on a page with very sparse but genuine text) — fall back to
            # this repo's own PyMuPDF extraction for just this one page
            # rather than silently losing its content. Same per-page
            # fallback philosophy as the base parser's own pdfplumber/OCR
            # fallback chain.
            if page is not None and (not result.born_digital or not result.blocks):
                extracts.append(self._extract_page_via_base(page, page_no))
                continue

            blocks, regions = self._convert_blocks(result.blocks, page_no, page)
            text = self._join_blocks(blocks)
            extracts.append(
                _PageExtract(
                    page=page_no,
                    text=text,
                    blocks=blocks,
                    regions=self._dedupe_regions(regions),
                    # rtldoc's own audit() flags low-confidence pages via
                    # reversed-line/presentation-form repair counts and
                    # empty-block ratio; a coarser proxy (any real text at
                    # all) is enough here since the base pipeline only uses
                    # `low_confidence` to decide whether to append the
                    # "[Low confidence extract]" marker text, not to trigger
                    # a further fallback.
                    confidence=1.0 if text.strip() else 0.0,
                    low_confidence=not text.strip(),
                )
            )
        return extracts

    def _extract_page_via_base(self, page: fitz.Page, page_no: int) -> _PageExtract:
        blocks = self._extract_pymupdf_blocks(page, page_no)
        text = self._join_blocks(blocks)
        confidence = self._extraction_confidence(page, text, blocks)
        return _PageExtract(
            page=page_no,
            text=text,
            blocks=blocks,
            regions=self._dedupe_regions(self._regions_from_blocks(blocks)),
            confidence=confidence,
            low_confidence=confidence < 0.35,
        )

    @staticmethod
    def _parse_style(style: str | None) -> tuple[float, bool]:
        """rtldoc's Block.style is "{font}|{size}|{hex_color}|{flags}"
        (flags contains "B" when bold, e.g. "Helvetica-Bold|11.0|000000|B").
        Returns (font_size, bold), defaulting to (0.0, False) on anything
        malformed rather than raising -- this is a best-effort geometric
        signal, not required for correctness elsewhere."""
        if not style:
            return 0.0, False
        parts = style.split("|")
        if len(parts) < 4:
            return 0.0, False
        try:
            size = float(parts[1])
        except ValueError:
            size = 0.0
        bold = "B" in parts[3]
        return size, bold

    def _convert_blocks(
        self, rtl_blocks: list, page_no: int, page: fitz.Page | None
    ) -> tuple[list[_PdfBlock], list[_PdfBlock]]:
        page_size = [float(page.rect.width), float(page.rect.height)] if page is not None else None
        blocks: list[_PdfBlock] = []
        regions: list[_PdfBlock] = []
        for b in rtl_blocks:
            text = (b.text or "").strip()
            if not text:
                continue
            if b.role == "table":
                kind = "table"
            elif b.role == "figure":
                kind = "figure"
            else:
                kind = "text"
            font_size, bold = self._parse_style(getattr(b, "style", None))
            pdf_block = _PdfBlock(
                text=text,
                page=page_no,
                bbox=list(b.bbox) if b.bbox else None,
                page_size=page_size,
                # Populated from rtldoc's own Block.style so _is_heading's
                # geometric rescue (below) has real data to check, not the
                # dataclass defaults (0.0/False) every rtldoc-sourced block
                # silently carried before this -- which made that rescue a
                # permanent no-op.
                max_font_size=font_size,
                avg_font_size=font_size,
                bold=bold,
                source="rtldoc",
                kind=kind,
            )
            if b.role == "heading":
                self._heading_block_ids.add(id(pdf_block))
            blocks.append(pdf_block)
            if kind in ("table", "figure"):
                regions.append(pdf_block)
        return blocks, regions

    def _block_to_ir(self, b: _PdfBlock) -> Block:
        """rtldoc's own role classification, stamped as a hint rather than
        decided here outright -- Axis1StructuralBuilder._is_heading (src/
        graph/axis1_structural.py) is what actually trusts it, falling back
        to the same font-size/bold heuristic used for PyMuPDF-sourced
        blocks when rtldoc says "not a heading" but the block's own
        geometry strongly disagrees (rtldoc's classifier isn't infallible:
        verified live on a synthetic test PDF where "Section 1:
        Introduction & Overview" -- 11pt bold vs. a 9pt non-bold body
        baseline, an unambiguous heading by font size/weight alone -- was
        classified role="passage", collapsing the whole 10-page document
        into one undifferentiated section).

        This used to be a per-class `_is_heading` override reading
        self._heading_block_ids directly; moved here (construction no
        longer lives on this class, see docs/DESIGN_unstructured_graph_v2.md
        phase 2) since instance state like that set can't survive into a
        freshly-instantiated Axis1StructuralBuilder -- Block.extra carries
        the same signal instead."""
        ir_block = super()._block_to_ir(b)
        if b.source == "rtldoc" and id(b) in self._heading_block_ids:
            ir_block.extra["heading_hint"] = "heading"
        return ir_block
