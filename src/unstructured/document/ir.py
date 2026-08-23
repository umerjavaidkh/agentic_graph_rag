"""ir.py — Document IR: the storage-agnostic representation a parser
produces, before any graph, blob, or vector concern touches it.

Grounded directly in what already exists: all three registered parser
backends (LightPdfParser, RtldocPdfParser, TableAwarePdfParser) already
split cleanly into two phases —

  1. extraction   — `_extract_pages()` (overridden per backend) turns raw
                     PDF bytes into `_PageExtract`/`_PdfBlock` objects.
  2. construction — `_usable_toc()` + `_build_from_toc()`/
                     `_build_from_extracts()` (never overridden — lives
                     only in LightPdfParser, inherited unchanged by the
                     other two) turns those into `(DKGNode, DKGEdge)`.

`Block`/`PageBlock`/`DocumentIR` here are that same phase-1 output,
promoted from parser-private dataclasses (`_PdfBlock`/`_PageExtract`,
light/parser.py) to a public, parser-independent contract — so phase 2
(now GraphConstructionService, see docs/DESIGN_unstructured_graph_v2.md)
can be extracted out of LightPdfParser and shared, instead of only being
reachable by subclassing it.

No Neo4j/blob/vector types are imported here. A parser emitting a
DocumentIR knows nothing about how it will be stored.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Optional

from ...shared.text_sanitize import sanitize_text


@dataclass
class Block:
    """One extracted content unit on a page — a paragraph, heading, table,
    or figure. Mirrors light/parser.py's `_PdfBlock` field-for-field
    (`max_font_size`/`avg_font_size`/`bold` included: the non-TOC heading
    heuristic in the current `_build_from_extracts` reads them, and that
    heuristic moves into Axis1StructuralBuilder unchanged in the first
    migration pass — dropping them here would silently break it).

    `extra` is an open bag for backend-specific extraction hints that
    don't need a named field on every backend's behalf (e.g. rtldoc's
    role classification, table_aware's repeated-header veto) — new parser
    backends attach what they know without widening this dataclass, which
    is the same "swappable module, stable contract" reasoning as the rest
    of docs/DESIGN_unstructured_graph_v2.md's module map.
    """

    text: str
    page: int  # 1-indexed, matches DKGNode.pdf_page convention
    bbox: Optional[list[float]] = None  # [l, t, r, b], parser page units
    page_size: Optional[list[float]] = None  # [width, height]
    max_font_size: float = 0.0
    avg_font_size: float = 0.0
    bold: bool = False
    source: str = "pymupdf"  # which extractor produced this block: pymupdf | rtldoc | pdfplumber
    kind: str = "text"  # text | table | figure
    low_confidence: bool = False
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # See shared/text_sanitize.py: extraction can emit lone UTF-16
        # surrogates that are unencodable downstream. Cleaning them at
        # the IR boundary keeps content_hash stable too -- it is
        # computed from this text, so sanitizing later would change the
        # idempotency key and re-ingest documents that had not changed.
        self.text = sanitize_text(self.text)


@dataclass
class PageBlock:
    """All content extracted from one page. Mirrors `_PageExtract`."""

    page: int
    text: str
    blocks: list[Block] = field(default_factory=list)
    regions: list[Block] = field(default_factory=list)  # table/figure blocks, subset of `blocks`
    confidence: float = 0.0
    low_confidence: bool = False

    def __post_init__(self) -> None:
        self.text = sanitize_text(self.text)


@dataclass
class DocumentIR:
    """Whole-document extraction output — everything a graph-construction
    stage needs, and nothing about how that graph gets stored.

    `toc`: the PDF's own embedded outline (level, title, 1-indexed page),
    when it has a real chapter/section structure — same gate as
    light/parser.py's `_usable_toc` (>=5 entries, at least one top-level
    entry). None means "no usable outline", not "not checked yet";
    extraction always resolves this once, up front, so both
    GraphConstructionService and scripts/validate_ontology_accuracy.py
    read the exact same ground truth rather than each re-deriving it.
    """

    source_name: str  # source.stem — becomes the document node's title/id seed
    page_count: int
    pages: list[PageBlock] = field(default_factory=list)
    toc: Optional[list[tuple[int, str, int]]] = None
    content_hash: str = ""  # set by `finalize()`; empty until then

    def finalize(self) -> "DocumentIR":
        """Compute `content_hash` from extracted text (idempotency key for
        re-ingestion, mirrors the existing content_hash use in versioning.py
        — same purpose, computed one phase earlier). Call once after all
        pages are populated; returns self for chaining."""
        digest = hashlib.sha256()
        for page in self.pages:
            digest.update(page.text.encode("utf-8", errors="ignore"))
            digest.update(b"\x00")
        self.content_hash = digest.hexdigest()
        return self
