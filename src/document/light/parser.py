"""
parser.py — Lightweight PDF → DKGNode tree.

Lightweight parser built on PyMuPDF with targeted pdfplumber fallback for pages
where text/table extraction confidence is low.
"""
from __future__ import annotations

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from dataclasses import dataclass, field
from pathlib import Path
import re

import fitz

from ...config.settings import (
    PDF_ENABLE_OCR,
    PDF_ENABLE_PDFPLUMBER,
    PDF_LOW_TEXT_CHARS,
    PDF_OCR_BACKEND,
    PDF_OCR_DPI,
    PDF_OCR_LANG,
    PDF_PLUMBER_PAGE_TIMEOUT_SEC,
)
from ...models import DKGEdge, DKGNode
from ..ir import Block, DocumentIR, PageBlock
from ..ocr import get_ocr_backend

try:
    import pdfplumber
except ImportError:  # pragma: no cover - optional dependency at import time
    pdfplumber = None

_TABLE_OR_FIGURE = re.compile(
    r"\b(?:table|figure|fig\.|box)\s+[a-z]?\d+(?:\.\d+)?\b|\[(?:table|figure)\]",
    re.I,
)
_TABLE_DENSITY = re.compile(r"\|.+\||\s{3,}\S|\b\d+(?:[,.]\d+)*\b")


@dataclass
class _PdfBlock:
    text: str
    page: int
    bbox: list[float] | None = None
    page_size: list[float] | None = None
    max_font_size: float = 0.0
    avg_font_size: float = 0.0
    bold: bool = False
    source: str = "pymupdf"
    kind: str = "text"  # text | table | region
    low_confidence: bool = False
    in_table_region: bool = False  # set only by TableAwarePdfParser; unused here
    is_repeated_header: bool = False  # set only by TableAwarePdfParser; unused here


@dataclass
class _PageExtract:
    page: int
    text: str
    blocks: list[_PdfBlock] = field(default_factory=list)
    regions: list[_PdfBlock] = field(default_factory=list)
    confidence: float = 0.0
    low_confidence: bool = False
    used_pdfplumber: bool = False


# Repeated-header/footer detection thresholds — both required, so a real
# heading reused a few times across independent chapters of a long
# document (e.g. "Overview" appearing 3 times in a 200-page report) isn't
# mistaken for a running header. Running headers/footers overwhelmingly
# clear both: printed on most pages, at a near-identical vertical position
# every time (a real heading's page position varies with content flow).
# Moved here from table_aware/parser.py (where this was originally built
# and used only to veto heading misclassification) so every backend gets
# it, and so the flagged text is actually excluded from page/section
# bodies -- not just from heading detection. Previously the flag was set
# AFTER each page's .text had already been joined, so it never actually
# kept repeated headers out of what gets embedded/NER'd; only out of what
# got classified as a heading.
_REPEAT_MIN_PAGES = 3
_REPEAT_MIN_PAGE_FRACTION = 0.15
_REPEAT_Y_TOLERANCE_PT = 20.0
_REPEAT_HEADER_WS_RE = re.compile(r"\s+")


class LightPdfParser:
    """Converts PDFs into the internal document graph."""

    def parse(self, source: str | Path) -> tuple[list[DKGNode], list[DKGEdge]]:
        """Extraction (this class) + construction (Axis1StructuralBuilder,
        src/graph/axis1_structural.py -- see docs/DESIGN_unstructured_
        graph_v2.md phase 2). Kept as a single call so every existing
        caller/test is unaffected by the construction-logic extraction.

        Local import: axis1_structural.py imports _TABLE_OR_FIGURE from
        this module at ITS top level, so a top-level import here would be
        circular. Module-load order is fine either way (only one side's
        import is deferred to call time); this mirrors the same lazy-
        import convention already used elsewhere for cross-package
        dependencies (e.g. ingestion/service.py's Axis2Builder import)."""
        from ...graph.axis1_structural import Axis1StructuralBuilder
        from ...graph.chunker import StructuralChunker

        print(f"   Parsing PDF via lightweight PyMuPDF parser: {Path(source).name}")
        ir = self.parse_ir(source)
        chunks = StructuralChunker().chunk(ir)
        return Axis1StructuralBuilder().build(ir, chunks)

    def parse_ir(self, source: str | Path) -> DocumentIR:
        """Produce this document's storage-agnostic IR -- extraction phase
        only, no graph construction (see docs/DESIGN_unstructured_graph_v2.md
        §3). `parse()` above calls this directly and hands the result to
        Axis1StructuralBuilder (src/graph/axis1_structural.py); exposed as
        its own method so a caller that wants the IR without immediately
        building a graph from it (e.g. GraphConstructionService) can call
        it independently of parse()."""
        source = Path(source)
        if source.suffix.lower() != ".pdf":
            raise ValueError("Only PDF ingestion is supported by the lightweight parser.")

        doc = fitz.open(str(source))
        try:
            extracts = self._extract_pages(source, doc)
            self._flag_repeated_headers(extracts)
            toc = self._usable_toc(doc)
            return self._to_document_ir(extracts, toc, source.stem, len(doc)).finalize()
        finally:
            doc.close()

    def _block_to_ir(self, b: _PdfBlock) -> Block:
        """Backend hook: convert one extracted _PdfBlock to its IR Block.
        Base impl copies matching fields plus the two generic veto flags
        already present on _PdfBlock (in_table_region/is_repeated_header)
        into `extra`. Subclasses override to add backend-specific hints
        (rtldoc's role classification, table_aware's table-fragment text
        heuristic) that today live in per-class `_is_heading` overrides --
        see RtldocPdfParser._block_to_ir / TableAwarePdfParser._block_to_ir."""
        extra: dict = {}
        if b.in_table_region:
            extra["in_table_region"] = True
        if b.is_repeated_header:
            extra["is_repeated_header"] = True
        return Block(
            text=b.text,
            page=b.page,
            bbox=b.bbox,
            page_size=b.page_size,
            max_font_size=b.max_font_size,
            avg_font_size=b.avg_font_size,
            bold=b.bold,
            source=b.source,
            kind=b.kind,
            low_confidence=b.low_confidence,
            extra=extra,
        )

    def _to_document_ir(
        self,
        extracts: list[_PageExtract],
        toc: list[tuple[int, str, int]] | None,
        source_name: str,
        page_count: int,
    ) -> DocumentIR:
        pages: list[PageBlock] = []
        for extract in extracts:
            pages.append(PageBlock(
                page=extract.page,
                text=extract.text,
                # regions are independently-constructed _PdfBlock copies
                # (see _regions_from_blocks), not the same objects as
                # entries in .blocks -- converted separately here for the
                # same reason, matching existing _PageExtract semantics.
                blocks=[self._block_to_ir(b) for b in extract.blocks],
                regions=[self._block_to_ir(r) for r in extract.regions],
                confidence=extract.confidence,
                low_confidence=extract.low_confidence,
            ))
        return DocumentIR(
            source_name=source_name,
            page_count=page_count,
            pages=pages,
            toc=toc,
        )

    def _usable_toc(self, doc: fitz.Document) -> list[tuple[int, str, int]] | None:
        """Return the PDF's own embedded outline/bookmarks (level, title,
        1-indexed page) if there's a real chapter/section structure in it,
        else None.

        This exists because font-size/bold/regex heading heuristics
        (_is_heading and friends below) are fundamentally guesswork, and
        guesswork always has another edge case: fixing one document's
        false-positive heading pattern (a repeated running header, an
        exercise question, an equation fragment misread as a heading) never
        generalizes to the next document's DIFFERENT false positive —
        confirmed live across three distinct failure modes on one single
        physics textbook alone before this method existed. A PDF's embedded
        outline, when present, is ground truth written by whatever tool
        produced the PDF — not a guess, so it sidesteps the entire class of
        heuristic failures for any document that has one, without touching
        behavior for documents that don't.

        This is genuinely additive, not a replacement: verified live that
        real SEC filings (converted from HTML) carry NO embedded outline at
        all (0 entries) and fall through to the existing heuristic path
        completely unchanged, while a professionally-produced textbook PDF
        carries a full, accurate outline (167 entries, exact chapter/
        section titles and page numbers) that this uses directly instead.
        """
        try:
            toc = doc.get_toc()
        except Exception:
            return None
        # A lone bookmark or two (e.g. just a title-page entry some PDFs
        # carry) isn't a real chapter/section structure to build a graph
        # from — require a nontrivial outline AND at least one top-level
        # (chapter-like) entry.
        if len(toc) < 5:
            return None
        if not any(level == 1 for level, _title, _page in toc):
            return None
        return toc

    def _extract_pages(self, source: Path, doc: fitz.Document) -> list[_PageExtract]:
        plumber_doc = None
        if PDF_ENABLE_PDFPLUMBER and pdfplumber is not None:
            try:
                plumber_doc = pdfplumber.open(str(source))
            except Exception:
                plumber_doc = None

        extracts: list[_PageExtract] = []
        try:
            for idx, page in enumerate(doc, start=1):
                blocks = self._extract_pymupdf_blocks(page, idx)
                text = self._join_blocks(blocks)
                confidence = self._extraction_confidence(page, text, blocks)
                regions = self._regions_from_blocks(blocks)
                used_pdfplumber = False

                if self._needs_pdfplumber(text, confidence) and plumber_doc is not None:
                    p_blocks, p_regions = self._run_pdfplumber_page(
                        plumber_doc.pages[idx - 1],
                        idx,
                    )
                    p_text = self._join_blocks(p_blocks)
                    p_confidence = self._extraction_confidence(page, p_text, p_blocks)
                    if p_confidence > confidence or len(p_text) > len(text):
                        blocks = p_blocks or blocks
                        text = p_text or text
                        confidence = p_confidence
                    if p_regions:
                        regions.extend(p_regions)
                    used_pdfplumber = bool(p_blocks or p_regions)

                low_confidence = confidence < 0.35 or len(text.strip()) < PDF_LOW_TEXT_CHARS
                ocr_error = None
                if low_confidence and PDF_ENABLE_OCR:
                    ocr_block, ocr_error = self._try_ocr(page, idx)
                    if ocr_block is not None:
                        blocks.append(ocr_block)
                        text = self._join_blocks(blocks)
                        confidence = self._extraction_confidence(page, text, blocks)
                        low_confidence = confidence < 0.35 or len(text.strip()) < PDF_LOW_TEXT_CHARS
                if low_confidence:
                    blocks.append(
                        self._low_confidence_marker(page, idx, bool(text.strip()), ocr_error)
                    )

                extracts.append(
                    _PageExtract(
                        page=idx,
                        text=text,
                        blocks=blocks,
                        regions=self._dedupe_regions(regions),
                        confidence=confidence,
                        low_confidence=low_confidence,
                        used_pdfplumber=used_pdfplumber,
                    )
                )
        finally:
            if plumber_doc is not None:
                plumber_doc.close()

        return extracts

    def _extract_pymupdf_blocks(self, page: fitz.Page, page_no: int) -> list[_PdfBlock]:
        page_size = [float(page.rect.width), float(page.rect.height)]
        out: list[_PdfBlock] = []
        try:
            data = page.get_text("dict")
        except Exception:
            data = {}

        for block in data.get("blocks", []):
            if block.get("type") != 0:
                continue
            texts: list[str] = []
            sizes: list[float] = []
            bold = False
            for line in block.get("lines", []):
                line_text = "".join(
                    (span.get("text") or "") for span in line.get("spans", [])
                ).strip()
                if line_text:
                    texts.append(line_text)
                for span in line.get("spans", []):
                    if span.get("size"):
                        sizes.append(float(span["size"]))
                    font = (span.get("font") or "").lower()
                    if "bold" in font or "black" in font or "semibold" in font:
                        bold = True

            text = "\n".join(texts).strip()
            if not text:
                continue
            bbox = [float(v) for v in block.get("bbox", [])] or None
            out.append(
                _PdfBlock(
                    text=self._normalize_text(text),
                    page=page_no,
                    bbox=bbox,
                    page_size=page_size,
                    max_font_size=max(sizes) if sizes else 0.0,
                    avg_font_size=sum(sizes) / len(sizes) if sizes else 0.0,
                    bold=bold,
                )
            )

        if not out:
            text = self._normalize_text(page.get_text("text") or "")
            if text:
                out.append(_PdfBlock(text=text, page=page_no, page_size=page_size))
        return out

    def _run_pdfplumber_page(
        self, page, page_no: int
    ) -> tuple[list[_PdfBlock], list[_PdfBlock]]:
        timeout = PDF_PLUMBER_PAGE_TIMEOUT_SEC
        if timeout <= 0:
            return self._extract_pdfplumber_page(page, page_no)

        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(self._extract_pdfplumber_page, page, page_no)
            try:
                return future.result(timeout=timeout)
            except FuturesTimeout:
                print(
                    f"   ⚠ pdfplumber timed out on page {page_no} "
                    f"(>{timeout}s); skipping fallback"
                )
                return [], []

    def _extract_pdfplumber_page(
        self, page, page_no: int
    ) -> tuple[list[_PdfBlock], list[_PdfBlock]]:
        """Text-only fallback (no extract_tables/find_tables — those can hang)."""
        page_size = [float(page.width), float(page.height)]
        try:
            text = self._normalize_text(page.extract_text() or "")
        except Exception:
            text = ""
        if not text:
            return [], []
        block = _PdfBlock(
            text=text,
            page=page_no,
            page_size=page_size,
            source="pdfplumber",
        )
        return [block], []

    def _needs_pdfplumber(self, text: str, confidence: float) -> bool:
        if not PDF_ENABLE_PDFPLUMBER or pdfplumber is None:
            return False
        return len(text.strip()) < PDF_LOW_TEXT_CHARS or confidence < 0.55

    def _extraction_confidence(
        self, page: fitz.Page, text: str, blocks: list[_PdfBlock]
    ) -> float:
        stripped = text.strip()
        if not stripped:
            return 0.0
        printable = sum(1 for ch in stripped if ch.isprintable())
        printable_ratio = printable / max(1, len(stripped))
        text_score = min(1.0, len(stripped) / max(1, PDF_LOW_TEXT_CHARS * 2))
        block_score = min(1.0, len(blocks) / 6)
        image_penalty = 0.0
        try:
            if page.get_images(full=False) and len(stripped) < PDF_LOW_TEXT_CHARS:
                image_penalty = 0.2
        except Exception:
            image_penalty = 0.0
        return max(0.0, min(1.0, (0.55 * text_score) + (0.25 * printable_ratio) + (0.2 * block_score) - image_penalty))

    def _try_ocr(self, page: fitz.Page, page_no: int) -> tuple[_PdfBlock | None, str | None]:
        """Attempt OCR on a low-confidence page. Returns (block, error) —
        at most one is non-None. Never raises."""
        backend = get_ocr_backend()
        if backend is None:
            return None, None
        try:
            pix = page.get_pixmap(dpi=PDF_OCR_DPI)
            text = backend.recognize(pix.tobytes("png"), lang=PDF_OCR_LANG)
        except Exception as exc:
            print(f"   ⚠ OCR failed on page {page_no}: {exc}")
            return None, str(exc)
        if not text or not text.strip():
            return None, None
        return (
            _PdfBlock(
                text=text,
                page=page_no,
                page_size=[float(page.rect.width), float(page.rect.height)],
                source="ocr",
            ),
            None,
        )

    def _low_confidence_marker(
        self, page: fitz.Page, page_no: int, has_text: bool, ocr_error: str | None = None
    ) -> _PdfBlock:
        if ocr_error:
            note = (
                "[Low confidence extract] OCR was attempted for this page but failed: "
                f"{ocr_error}"
            )
        elif PDF_ENABLE_OCR and PDF_OCR_BACKEND != "none" and get_ocr_backend() is None:
            note = (
                "[Low confidence extract] Text extraction was weak. "
                f"Configured OCR backend '{PDF_OCR_BACKEND}' is unavailable in this "
                "environment (missing package or system binary)."
            )
        elif has_text:
            note = "[Low confidence extract] Text may be incomplete; use the saved full-page image if needed."
        else:
            note = "[Low confidence extract] No reliable text extracted; use the saved full-page image or enable page vision/OCR."
        return _PdfBlock(
            text=note,
            page=page_no,
            page_size=[float(page.rect.width), float(page.rect.height)],
            source="confidence",
            low_confidence=True,
        )

    def _regions_from_blocks(self, blocks: list[_PdfBlock]) -> list[_PdfBlock]:
        regions: list[_PdfBlock] = []
        for block in blocks:
            if block.kind == "table" or _TABLE_OR_FIGURE.search(block.text):
                kind = "table" if re.search(r"\btable\b|\[table\]", block.text, re.I) else "figure"
                regions.append(
                    _PdfBlock(
                        **{
                            **block.__dict__,
                            "kind": kind,
                            "source": block.source,
                        }
                    )
                )
        return regions

    def _dedupe_regions(self, regions: list[_PdfBlock]) -> list[_PdfBlock]:
        seen: set[tuple[int, str]] = set()
        out: list[_PdfBlock] = []
        for r in regions:
            key = (r.page, r.text[:200])
            if key in seen:
                continue
            seen.add(key)
            out.append(r)
        return out

    def _join_blocks(self, blocks: list[_PdfBlock]) -> str:
        return "\n\n".join(b.text.strip() for b in blocks if b.text.strip())

    @staticmethod
    def _normalize_for_repeat_check(text: str) -> str:
        joined = " ".join(ln.strip() for ln in (text or "").splitlines() if ln.strip())
        return _REPEAT_HEADER_WS_RE.sub(" ", joined).strip().lower()

    def _flag_repeated_headers(self, extracts: list[_PageExtract]) -> None:
        """Flag blocks that repeat across most of the document at a
        consistent vertical position (running headers/footers -- a company
        name, document title, or confidentiality notice printed on every
        page) and strip them out of each page's .text so they don't
        pollute NER/embeddings, not just heading detection."""
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
        touched_pages: set[int] = set()
        for occurrences in groups.values():
            distinct_pages = {p for p, _, _ in occurrences}
            if len(distinct_pages) < min_pages_needed:
                continue
            y_values = [y for _, y, _ in occurrences]
            if max(y_values) - min(y_values) > _REPEAT_Y_TOLERANCE_PT:
                continue  # position varies too much to be a running header/footer
            for page, _, block in occurrences:
                block.is_repeated_header = True
                touched_pages.add(page)

        if not touched_pages:
            return
        for extract in extracts:
            if extract.page in touched_pages:
                extract.text = self._join_blocks(
                    [b for b in extract.blocks if not b.is_repeated_header]
                )

    def _normalize_text(self, text: str) -> str:
        lines = [re.sub(r"[ \t]+", " ", ln).strip() for ln in (text or "").splitlines()]
        out: list[str] = []
        for ln in lines:
            if ln or (out and out[-1]):
                out.append(ln)
        return "\n".join(out).strip()

    def _table_to_markdown(self, table: list[list[object]]) -> str:
        rows = [
            [self._normalize_text(str(cell or "")) for cell in row]
            for row in table
            if row and any(str(cell or "").strip() for cell in row)
        ]
        if not rows:
            return ""
        width = max(len(r) for r in rows)
        rows = [r + [""] * (width - len(r)) for r in rows]
        header = rows[0]
        body = rows[1:] if len(rows) > 1 else []
        lines = [
            "| " + " | ".join(header) + " |",
            "| " + " | ".join("---" for _ in header) + " |",
        ]
        lines.extend("| " + " | ".join(row) + " |" for row in body)
        return "[Table]\n" + "\n".join(lines)

