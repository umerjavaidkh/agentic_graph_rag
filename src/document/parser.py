"""
parser.py — Lightweight PDF → DKGNode tree.

Lightweight parser built on PyMuPDF with targeted pdfplumber fallback for pages
where text/table extraction confidence is low.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import re

import fitz

from ..config.settings import (
    PDF_ENABLE_OCR,
    PDF_ENABLE_PDFPLUMBER,
    PDF_LOW_TEXT_CHARS,
    PDF_OCR_BACKEND,
)
from ..models import DKGEdge, DKGNode, NodeType, RelType
from .page_numbers import enrich_page_nodes
from .patterns import (
    REFERENCE_PATTERN,
    is_standalone_number,
    number_depth,
    parse_numbered_title,
    slug,
)

try:
    import pdfplumber
except ImportError:  # pragma: no cover - optional dependency at import time
    pdfplumber = None

_TABLE_OR_FIGURE = re.compile(
    r"\b(?:table|figure|fig\.|box)\s+[a-z]?\d+(?:\.\d+)?\b|\[(?:table|figure)\]",
    re.I,
)
_FIGURE_REF = re.compile(r"\bfigure\s+(\d+(?:\.\d+)?)\b", re.I)
_TABLE_REF = re.compile(r"\btable\s+([a-z]?\d+(?:\.\d+)?)\b", re.I)
_TABLE_DENSITY = re.compile(r"\|.+\||\s{3,}\S|\b\d+(?:[,.]\d+)*\b")
_COMMON_HEADING = re.compile(
    r"^(?:chapter|section|part|appendix|annex|box|table|figure)\b",
    re.I,
)


def _build_region_tags(
    kind: str,
    text: str,
    pdf_page: int,
    index: int,
    document_page: str | None = None,
) -> list[str]:
    tags = [f"kind:{kind}", f"pdf:{pdf_page}", f"region:{pdf_page}:{index}"]
    if document_page:
        tags.append(f"doc:{document_page.strip()}")

    blob = (text or "").lower()
    for ref in _TABLE_REF.findall(blob):
        tags.append(f"table:{ref.lower()}")
    for ref in _FIGURE_REF.findall(blob):
        tags.append(f"figure:{ref.lower()}")
    if kind == "table" and "table:" not in " ".join(tags):
        tags.extend(["table", f"table:{index}"])
    if kind == "figure":
        tags.extend(["figure", f"figure:{index}"])
    return list(dict.fromkeys(tags))


def _region_title(kind: str, text: str, pdf_page: int, index: int) -> str:
    first_line = (text or "").strip().splitlines()[0][:120] if text else ""
    if first_line:
        return first_line
    label = "Table" if kind == "table" else "Figure"
    return f"{label} {index} (PDF page {pdf_page})"


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


@dataclass
class _PageExtract:
    page: int
    text: str
    blocks: list[_PdfBlock] = field(default_factory=list)
    regions: list[_PdfBlock] = field(default_factory=list)
    confidence: float = 0.0
    low_confidence: bool = False
    used_pdfplumber: bool = False


class LightPdfParser:
    """Converts PDFs into the internal document graph."""

    def parse(self, source: str | Path) -> tuple[list[DKGNode], list[DKGEdge]]:
        source = Path(source)
        if source.suffix.lower() != ".pdf":
            raise ValueError("Only PDF ingestion is supported by the lightweight parser.")

        print(f"   Parsing PDF via lightweight PyMuPDF parser: {source.name}")
        doc = fitz.open(str(source))
        try:
            extracts = self._extract_pages(source, doc)
            return self._build_from_extracts(extracts, source.stem, len(doc))
        finally:
            doc.close()

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
                    p_blocks, p_regions = self._extract_pdfplumber_page(
                        plumber_doc.pages[idx - 1], idx
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
                if low_confidence:
                    blocks.append(self._low_confidence_marker(page, idx, bool(text.strip())))

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

    def _extract_pdfplumber_page(self, page, page_no: int) -> tuple[list[_PdfBlock], list[_PdfBlock]]:
        page_size = [float(page.width), float(page.height)]
        blocks: list[_PdfBlock] = []
        regions: list[_PdfBlock] = []

        try:
            text = self._normalize_text(page.extract_text(layout=True) or "")
        except Exception:
            text = ""
        if text:
            blocks.append(
                _PdfBlock(
                    text=text,
                    page=page_no,
                    page_size=page_size,
                    source="pdfplumber",
                )
            )

        try:
            tables = page.extract_tables() or []
        except Exception:
            tables = []
        try:
            table_objs = page.find_tables() or []
        except Exception:
            table_objs = []

        for idx, table in enumerate(tables):
            md = self._table_to_markdown(table)
            if not md:
                continue
            bbox = None
            if idx < len(table_objs) and getattr(table_objs[idx], "bbox", None):
                bbox = [float(v) for v in table_objs[idx].bbox]
            block = _PdfBlock(
                text=md,
                page=page_no,
                bbox=bbox,
                page_size=page_size,
                source="pdfplumber",
                kind="table",
            )
            blocks.append(block)
            regions.append(block)

        return blocks, regions

    def _needs_pdfplumber(self, text: str, confidence: float) -> bool:
        if not PDF_ENABLE_PDFPLUMBER or pdfplumber is None:
            return False
        if len(text.strip()) < PDF_LOW_TEXT_CHARS or confidence < 0.55:
            return True
        lines = [ln for ln in text.splitlines() if ln.strip()]
        if len(lines) >= 3:
            tableish = sum(1 for ln in lines if _TABLE_DENSITY.search(ln))
            return tableish / max(1, len(lines)) > 0.35
        return False

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

    def _low_confidence_marker(self, page: fitz.Page, page_no: int, has_text: bool) -> _PdfBlock:
        if PDF_ENABLE_OCR and PDF_OCR_BACKEND != "none":
            note = (
                "[Low confidence extract] Text extraction was weak. "
                f"Configured OCR backend '{PDF_OCR_BACKEND}' is not implemented in this lightweight parser yet."
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

    def _heading_font_threshold(self, blocks: list[_PdfBlock]) -> float:
        sizes = sorted(
            b.avg_font_size or b.max_font_size
            for b in blocks
            if b.kind == "text" and (b.avg_font_size or b.max_font_size)
        )
        if not sizes:
            return 999.0
        median = sizes[len(sizes) // 2]
        return median + 1.5

    def _is_heading(self, block: _PdfBlock, font_threshold: float) -> bool:
        if block.kind != "text" or block.low_confidence:
            return False

        text = " ".join(ln.strip() for ln in block.text.splitlines() if ln.strip())
        if not (3 <= len(text) <= 160):
            return False
        if len(text.split()) > 18:
            return False
        if _TABLE_OR_FIGURE.search(text) and len(text) > 80:
            return False

        section_number, title = parse_numbered_title(text)
        if section_number and len(title) >= 2:
            return True
        if _COMMON_HEADING.search(text) and len(text.split()) <= 12:
            return True
        if block.bold and block.max_font_size >= font_threshold and len(text) <= 120:
            return True
        if block.max_font_size >= font_threshold + 1 and len(text.split()) <= 12:
            return True
        if self._uppercase_ratio(text) > 0.72 and 2 <= len(text.split()) <= 10:
            return True
        return False

    def _uppercase_ratio(self, text: str) -> float:
        letters = [ch for ch in text if ch.isalpha()]
        if not letters:
            return 0.0
        return sum(1 for ch in letters if ch.isupper()) / len(letters)

    def _build_from_extracts(
        self, extracts: list[_PageExtract], doc_name: str, page_count: int
    ) -> tuple[list[DKGNode], list[DKGEdge]]:
        nodes: list[DKGNode] = []
        edges: list[DKGEdge] = []

        document_id = f"doc_{slug(doc_name)}"
        document_node = DKGNode(
            id=document_id,
            type=NodeType.DOCUMENT,
            title=doc_name,
            text=doc_name,
            order=0,
            page_start=1,
            page_end=max(1, page_count),
            depth=0,
        )
        nodes.append(document_node)

        blocks = self._resolve_number_prefixes(
            [b for page in extracts for b in page.blocks if b.text.strip()]
        )

        chapter_nodes: list[DKGNode] = []
        section_nodes: list[DKGNode] = []
        page_buckets: dict[int, list[str]] = {
            page.page: [page.text] if page.text else [] for page in extracts
        }

        current_chapter: DKGNode | None = None
        current_section: DKGNode | None = None
        current_section_texts: list[str] = []

        chapter_idx = 0
        global_section_idx = 0
        doc_order = 0

        # (structural_level, node_id) — parent = nearest entry with lower level
        heading_stack: list[tuple[int, str]] = [(0, document_id)]
        number_map: dict[str, str] = {}

        def finalize_section() -> None:
            nonlocal current_section, current_section_texts
            if current_section and current_section_texts:
                body = "\n\n".join(current_section_texts)
                current_section.text = (
                    f"{current_section.title}\n\n{body}"
                    if body.strip()
                    else current_section.title
                )
                current_section_texts = []

        def link_contains(parent_id: str, child_id: str) -> None:
            edges.append(DKGEdge(parent_id, child_id, RelType.CONTAINS, axis=1))
            edges.append(DKGEdge(child_id, parent_id, RelType.PART_OF, axis=1))

        def structural_level(
            is_chapter: bool,
            title: str,
            section_number: str | None,
        ) -> int:
            """Map heading to a comparable depth for parent stack (higher = deeper)."""
            if is_chapter:
                return 1
            if section_number:
                # e.g. "4.5" → depth 2; under chapter (+1) → at least 2
                nd = number_depth(section_number)
                base = nd + (1 if current_chapter else 0)
                return max(2, base)
            return 2

        def parent_id_for_level(level: int) -> str:
            while len(heading_stack) > 1 and heading_stack[-1][0] >= level:
                heading_stack.pop()
            return heading_stack[-1][1]

        heading_font_threshold = self._heading_font_threshold(blocks)

        for block in blocks:
            text = block.text
            page_no = block.page
            is_heading = self._is_heading(block, heading_font_threshold)

            if is_heading:
                finalize_section()

                section_number, title = parse_numbered_title(text)
                is_chapter = bool(section_number and number_depth(section_number) == 1)
                level = structural_level(is_chapter, title, section_number)

                # Nest under the current section when depth would otherwise make them siblings.
                if (
                    not is_chapter
                    and current_section is not None
                    and heading_stack
                    and level <= heading_stack[-1][0]
                    and heading_stack[-1][1] == current_section.id
                ):
                    sn_parent, _ = parse_numbered_title(current_section.title)
                    if section_number and sn_parent:
                        if number_depth(section_number) > number_depth(sn_parent):
                            level = heading_stack[-1][0] + 1
                    else:
                        level = heading_stack[-1][0] + 1

                parent_id = parent_id_for_level(level)
                doc_order += 1

                if is_chapter:
                    chapter_idx += 1
                    global_section_idx = 0
                    node_id = f"{document_id}_chapter_{chapter_idx}"
                    node = DKGNode(
                        id=node_id,
                        type=NodeType.CHAPTER,
                        title=title,
                        text=title,
                        order=doc_order,
                        page_start=page_no,
                        page_end=page_no,
                        depth=1,
                    )
                    nodes.append(node)
                    chapter_nodes.append(node)
                    current_chapter = node
                    current_section = None
                    heading_stack = [(0, document_id), (level, node_id)]
                    link_contains(document_id, node_id)
                else:
                    global_section_idx += 1
                    node_id = f"{document_id}_section_{chapter_idx}_{global_section_idx}"
                    full_title = (
                        f"{section_number} {title}".strip()
                        if section_number
                        else title
                    )
                    node = DKGNode(
                        id=node_id,
                        type=NodeType.SECTION,
                        title=full_title,
                        text=full_title,
                        order=doc_order,
                        page_start=page_no,
                        page_end=page_no,
                        depth=level,
                    )
                    nodes.append(node)
                    section_nodes.append(node)
                    current_section = node
                    heading_stack.append((level, node_id))
                    link_contains(parent_id, node_id)

                    if section_number:
                        number_map[section_number] = node_id
                        for i in range(1, len(section_number.split("."))):
                            prefix = ".".join(section_number.split(".")[:i])
                            if prefix not in number_map:
                                number_map[prefix] = node_id

                continue

            clean = text.strip()
            if not clean:
                continue

            if current_section is None:
                doc_order += 1
                global_section_idx += 1
                node_id = f"{document_id}_section_{chapter_idx}_{global_section_idx}"
                parent_id = current_chapter.id if current_chapter else document_id
                node = DKGNode(
                    id=node_id,
                    type=NodeType.SECTION,
                    title="Preamble",
                    text="",
                    order=doc_order,
                    page_start=page_no,
                    page_end=page_no,
                    depth=2,
                )
                nodes.append(node)
                section_nodes.append(node)
                current_section = node
                heading_stack.append((2, node_id))
                link_contains(parent_id, node_id)

            prefix = "[Table] " if block.kind == "table" else ""
            if block.low_confidence:
                prefix = "[Low confidence extract] " + prefix
            current_section_texts.append(prefix + clean)

            if current_section and page_no > current_section.page_end:
                current_section.page_end = page_no
            if current_chapter and page_no > current_chapter.page_end:
                current_chapter.page_end = page_no

        finalize_section()

        all_page_numbers = set(range(1, page_count + 1)) | set(page_buckets.keys())
        page_nodes = self._build_page_nodes(
            page_buckets,
            edges,
            chapter_nodes,
            section_nodes,
            document_id,
            all_page_numbers=all_page_numbers,
        )
        enrich_page_nodes(page_nodes, section_nodes)
        nodes.extend(page_nodes)

        region_nodes = self._build_region_nodes(
            extracts, page_nodes, edges, document_id
        )
        nodes.extend(region_nodes)

        self._add_sequential_edges(chapter_nodes, edges)
        self._add_sequential_edges(section_nodes, edges)
        self._add_sequential_edges(page_nodes, edges)
        self._detect_reference_edges(nodes, edges)
        self._link_number_hierarchy(number_map, edges)

        print(
            f"   📐 Structure: {len(chapter_nodes)} chapters, "
            f"{len(section_nodes)} sections (nested CONTAINS), {len(page_nodes)} pages, "
            f"{len(region_nodes)} regions"
        )
        return nodes, edges

    def _build_region_nodes(
        self,
        extracts: list[_PageExtract],
        page_nodes: list[DKGNode],
        edges: list[DKGEdge],
        document_id: str,
    ) -> list[DKGNode]:
        """Table/figure/box hints from PyMuPDF/pdfplumber → Region nodes linked to Page."""
        page_by_pdf: dict[int, DKGNode] = {}
        for pn in page_nodes:
            key = pn.pdf_page or pn.order
            page_by_pdf[key] = pn

        region_nodes: list[DKGNode] = []
        index_by_page: dict[int, int] = {}

        for page in extracts:
            for region in page.regions:
                page_no = region.page
                index_by_page[page_no] = index_by_page.get(page_no, 0) + 1
                idx = index_by_page[page_no]
                kind = "table" if region.kind == "table" else "figure"
                text = region.text

                page_node = page_by_pdf.get(page_no)
                doc_page = page_node.document_page if page_node else None
                tags = _build_region_tags(kind, text, page_no, idx, doc_page)
                title = _region_title(kind, text, page_no, idx)
                node_id = f"{document_id}_region_{page_no}_{idx}"

                node = DKGNode(
                    id=node_id,
                    type=NodeType.REGION,
                    title=title,
                    text=text or title,
                    order=idx,
                    page_start=page_no,
                    page_end=page_no,
                    pdf_page=page_no,
                    document_page=doc_page,
                    depth=100,
                    region_kind=kind,
                    region_tags=tags,
                    bbox=region.bbox,
                    bbox_page_size=region.page_size,
                )
                region_nodes.append(node)

                page_id = page_node.id if page_node else f"{document_id}_page_{page_no}"
                edges.append(DKGEdge(page_id, node_id, RelType.CONTAINS, axis=1))
                edges.append(DKGEdge(node_id, page_id, RelType.PART_OF, axis=1))

        return region_nodes

    def _resolve_number_prefixes(self, blocks: list[_PdfBlock]) -> list[_PdfBlock]:
        """Merge '4.5' + 'ENVIRONMENTAL PROTECTION' into one heading when needed."""
        resolved: list[_PdfBlock] = []
        pending_number: str | None = None

        for block in blocks:
            if is_standalone_number(block.text):
                pending_number = block.text.rstrip(".")
                continue

            if pending_number is not None:
                if not is_standalone_number(block.text):
                    block = _PdfBlock(**{**block.__dict__, "text": f"{pending_number} {block.text}"})
                pending_number = None

            resolved.append(block)

        return resolved

    def _link_number_hierarchy(
        self, number_map: dict[str, str], edges: list[DKGEdge]
    ) -> None:
        """Ensure numbered parent sections CONTAINS numbered children when inferred."""
        existing = {
            (e.source_id, e.target_id)
            for e in edges
            if e.rel_type == RelType.CONTAINS
        }
        for num, node_id in number_map.items():
            parts = num.split(".")
            if len(parts) < 2:
                continue
            parent_num = ".".join(parts[:-1])
            parent_id = number_map.get(parent_num)
            if parent_id and parent_id != node_id:
                key = (parent_id, node_id)
                if key not in existing:
                    edges.append(DKGEdge(parent_id, node_id, RelType.CONTAINS, axis=1))
                    edges.append(DKGEdge(node_id, parent_id, RelType.PART_OF, axis=1))
                    existing.add(key)

    def _build_page_nodes(
        self,
        page_buckets: dict[int, list[str]],
        edges: list[DKGEdge],
        chapter_nodes: list[DKGNode],
        section_nodes: list[DKGNode],
        document_id: str,
        all_page_numbers: set[int] | None = None,
    ) -> list[DKGNode]:
        page_nodes: list[DKGNode] = []
        page_nos = sorted(set(page_buckets.keys()) | (all_page_numbers or set()))
        for page_no in page_nos:
            texts = page_buckets.get(page_no, [])
            node_id = f"{document_id}_page_{page_no}"
            node = DKGNode(
                id=node_id,
                type=NodeType.PAGE,
                title=f"Page {page_no}",
                text="\n".join(texts) if texts else "",
                order=page_no,
                page_start=page_no,
                page_end=page_no,
                depth=99,
            )
            page_nodes.append(node)
            parent_id = self._find_page_parent(
                page_no, section_nodes, chapter_nodes, document_id
            )
            edges.append(DKGEdge(parent_id, node_id, RelType.CONTAINS, axis=1))
            edges.append(DKGEdge(node_id, parent_id, RelType.PART_OF, axis=1))
        return page_nodes

    def _find_page_parent(
        self,
        page_no: int,
        section_nodes: list[DKGNode],
        chapter_nodes: list[DKGNode],
        document_id: str,
    ) -> str:
        """Prefer deepest section spanning this page (nested structure)."""
        candidates = [
            s
            for s in section_nodes
            if s.page_start <= page_no <= s.page_end
        ]
        if candidates:
            return max(candidates, key=lambda s: (s.depth, s.order)).id
        for c in reversed(chapter_nodes):
            if c.page_start <= page_no <= c.page_end:
                return c.id
        return document_id

    def _add_sequential_edges(
        self, nodes: list[DKGNode], edges: list[DKGEdge]
    ) -> None:
        ordered = sorted(nodes, key=lambda n: n.order)
        for a, b in zip(ordered, ordered[1:]):
            edges.append(DKGEdge(a.id, b.id, RelType.PRECEDES, axis=1))
            edges.append(DKGEdge(b.id, a.id, RelType.FOLLOWS, axis=1))

    def _detect_reference_edges(
        self, nodes: list[DKGNode], edges: list[DKGEdge]
    ) -> None:
        title_lookup: dict[str, str] = {}
        for n in nodes:
            title_lookup[n.title.strip().lower()] = n.id
            if n.type in (NodeType.CHAPTER, NodeType.SECTION):
                title_lookup[f"{n.type.value.lower()} {n.order}"] = n.id

        for node in nodes:
            for match in REFERENCE_PATTERN.findall(node.text):
                ref_key = match.strip().lower()
                for key, target_id in title_lookup.items():
                    if key in ref_key and target_id != node.id:
                        edges.append(
                            DKGEdge(
                                source_id=node.id,
                                target_id=target_id,
                                rel_type=RelType.REFERENCES,
                                axis=2,
                                properties={"matched_text": match.strip()},
                            )
                        )
                        break

