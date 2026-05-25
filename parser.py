"""
parser.py — Docling → DKGNode tree.

Walks Docling's DoclingDocument and builds the document hierarchy:
    Book → Chapter → Section → Page → Concept

Axis 1 structural edges are derived FREE from the tree structure.
Text content is preserved at every level for embedding later.
"""
import re
import uuid
from pathlib import Path
from typing import Generator

from docling.document_converter import DocumentConverter
from docling.datamodel.base_models import InputFormat
from docling.datamodel.document import DoclingDocument
from docling_core.types.doc import DoclingDocument, DocItemLabel

from models import DKGNode, DKGEdge, NodeType, RelType


# ─────────────────────────────────────────
# DOCLING LABEL → NODE TYPE MAPPING
# ─────────────────────────────────────────
CHAPTER_LABELS = {DocItemLabel.TITLE}
SECTION_LABELS = {DocItemLabel.SECTION_HEADER}
TEXT_LABELS    = {
    DocItemLabel.TEXT,
    DocItemLabel.LIST_ITEM,
    DocItemLabel.TABLE,
    DocItemLabel.CAPTION,
    DocItemLabel.FOOTNOTE,
}


# ─────────────────────────────────────────
# REFERENCE PATTERN  (for REFERENCES edge)
# ─────────────────────────────────────────
REFERENCE_PATTERN = re.compile(
    r"(?:see|refer(?:s)? to|as (?:discussed|described|shown) in|"
    r"described in|mentioned in|in)\s+"
    r"(?:Chapter|Section|Appendix|Figure|Table|Part)\s+[\w\d\.]+",
    re.IGNORECASE,
)


# ─────────────────────────────────────────
# MAIN PARSER
# ─────────────────────────────────────────
class DoclingParser:
    """
    Converts any document (PDF, DOCX, PPTX, HTML…) via Docling
    into a flat list of DKGNode objects + Axis-1 structural edges.

    Usage:
        parser = DoclingParser()
        nodes, edges = parser.parse("path/to/book.pdf")
    """

    def __init__(self):
        self.converter = DocumentConverter()

    def parse(self, source: str | Path) -> tuple[list[DKGNode], list[DKGEdge]]:
        source   = Path(source)
        doc_name = source.stem

        if source.suffix.lower() == ".json":
            print(f"   📂 Loading pre-parsed Docling JSON: {source.name}")
            doc = DoclingDocument.model_validate_json(source.read_text(encoding="utf-8"))
        else:
            print(f"   🔍 Parsing document via Docling: {source.name}")
            result = self.converter.convert(str(source))
            doc    = result.document

        nodes: list[DKGNode] = []
        edges: list[DKGEdge] = []

        # ── Book root node ───────────────────────────────────────
        book_id   = f"book_{_slug(doc_name)}"
        book_node = DKGNode(
            id         = book_id,
            type       = NodeType.BOOK,
            title      = doc_name,
            text       = doc_name,
            order      = 0,
            page_start = 1,
            page_end   = self._max_page(doc),
            depth      = 0,
        )
        nodes.append(book_node)

        # ── Walk document tree ───────────────────────────────────
        chapter_nodes: list[DKGNode] = []
        section_nodes: list[DKGNode] = []
        page_buckets:  dict[int, list[str]] = {}  # page_no → [text chunks]

        current_chapter: DKGNode | None = None
        current_section: DKGNode | None = None
        chapter_idx = 0
        section_idx = 0

        # Track text accumulating under current section
        current_section_texts: list[str] = []

        for item, _level in doc.iterate_items():
            label    = getattr(item, "label", None)
            text     = getattr(item, "text", "") or ""
            page_no  = self._get_page(item)

            if not text.strip():
                continue

            # ── Chapter ──────────────────────────────────────────
            if label in CHAPTER_LABELS:
                # Finalize previous section's text before starting new chapter
                if current_section and current_section_texts:
                    current_section.text = current_section.title + "\n\n" + "\n".join(current_section_texts)
                    current_section_texts = []

                chapter_idx += 1
                section_idx  = 0
                node_id      = f"chapter_{chapter_idx}"
                node = DKGNode(
                    id         = node_id,
                    type       = NodeType.CHAPTER,
                    title      = text.strip(),
                    text       = text.strip(),
                    order      = chapter_idx,
                    page_start = page_no,
                    page_end   = page_no,
                    depth      = 1,
                )
                nodes.append(node)
                chapter_nodes.append(node)
                current_chapter = node
                current_section = None

                edges.append(DKGEdge(book_id, node_id, RelType.CONTAINS, axis=1))
                edges.append(DKGEdge(node_id, book_id, RelType.PART_OF,  axis=1))

            # ── Section ──────────────────────────────────────────
            elif label in SECTION_LABELS:
                # Finalize previous section's text before starting new section
                if current_section and current_section_texts:
                    current_section.text = current_section.title + "\n\n" + "\n".join(current_section_texts)
                    current_section_texts = []

                section_idx += 1
                parent_id    = current_chapter.id if current_chapter else book_id
                node_id      = f"section_{chapter_idx}_{section_idx}"
                node = DKGNode(
                    id         = node_id,
                    type       = NodeType.SECTION,
                    title      = text.strip(),
                    text       = text.strip(),  # Will be overwritten with body text when finalized
                    order      = section_idx,
                    page_start = page_no,
                    page_end   = page_no,
                    depth      = 2,
                )
                nodes.append(node)
                section_nodes.append(node)
                current_section = node

                edges.append(DKGEdge(parent_id, node_id, RelType.CONTAINS, axis=1))
                edges.append(DKGEdge(node_id, parent_id, RelType.PART_OF,  axis=1))

            # ── Text → accumulate into current section AND page buckets ─
            elif label in TEXT_LABELS:
                clean_text = text.strip()
                
                # Accumulate under current section
                if current_section:
                    current_section_texts.append(clean_text)
                
                # Also keep for page nodes
                if page_no not in page_buckets:
                    page_buckets[page_no] = []
                page_buckets[page_no].append(clean_text)

                # Track page_end on parent nodes
                if current_section and page_no > current_section.page_end:
                    current_section.page_end = page_no
                if current_chapter and page_no > current_chapter.page_end:
                    current_chapter.page_end = page_no

        # ── Finalize last section's text ─────────────────────────
        if current_section and current_section_texts:
            current_section.text = current_section.title + "\n\n" + "\n".join(current_section_texts)

        # ── Build Page nodes from page buckets ───────────────────
        page_nodes = self._build_page_nodes(
            page_buckets, nodes, edges, chapter_nodes, section_nodes
        )
        nodes.extend(page_nodes)

        # ── Axis 1: PRECEDES / FOLLOWS chains ────────────────────
        self._add_sequential_edges(chapter_nodes, edges)
        self._add_sequential_edges(section_nodes, edges)
        self._add_sequential_edges(page_nodes,    edges)

        # ── Axis 2 (partial): REFERENCES from text patterns ──────
        self._detect_reference_edges(nodes, edges)

        return nodes, edges

    # ─────────────────────────────────────────────────────────────
    # PRIVATE HELPERS
    # ─────────────────────────────────────────────────────────────

    def _build_page_nodes(
        self,
        page_buckets:   dict[int, list[str]],
        existing_nodes: list[DKGNode],
        edges:          list[DKGEdge],
        chapter_nodes:  list[DKGNode],
        section_nodes:  list[DKGNode],
    ) -> list[DKGNode]:
        page_nodes = []
        for page_no in sorted(page_buckets.keys()):
            texts    = page_buckets[page_no]
            full_text = "\n".join(texts)
            node_id  = f"page_{page_no}"
            node = DKGNode(
                id         = node_id,
                type       = NodeType.PAGE,
                title      = f"Page {page_no}",
                text       = full_text,
                order      = page_no,
                page_start = page_no,
                page_end   = page_no,
                depth      = 3,
            )
            page_nodes.append(node)

            # Find the most specific parent (section > chapter > book)
            parent_id = self._find_page_parent(
                page_no, section_nodes, chapter_nodes
            )
            edges.append(DKGEdge(parent_id, node_id, RelType.CONTAINS, axis=1))
            edges.append(DKGEdge(node_id, parent_id, RelType.PART_OF,  axis=1))

        return page_nodes

    def _find_page_parent(
        self,
        page_no:       int,
        section_nodes: list[DKGNode],
        chapter_nodes: list[DKGNode],
    ) -> str:
        # Section that spans this page
        for s in reversed(section_nodes):
            if s.page_start <= page_no <= s.page_end:
                return s.id
        # Chapter that spans this page
        for c in reversed(chapter_nodes):
            if c.page_start <= page_no <= c.page_end:
                return c.id
        return "book_root"

    def _add_sequential_edges(
        self, nodes: list[DKGNode], edges: list[DKGEdge]
    ) -> None:
        sorted_nodes = sorted(nodes, key=lambda n: n.order)
        for i in range(len(sorted_nodes) - 1):
            a, b = sorted_nodes[i], sorted_nodes[i + 1]
            edges.append(DKGEdge(a.id, b.id, RelType.PRECEDES, axis=1))
            edges.append(DKGEdge(b.id, a.id, RelType.FOLLOWS,  axis=1))

    def _detect_reference_edges(
        self, nodes: list[DKGNode], edges: list[DKGEdge]
    ) -> None:
        """
        Regex scan for explicit cross-references in text.
        e.g. 'as discussed in Chapter 3', 'see Section 2.1'
        """
        id_map = {n.id: n for n in nodes}
        # Build a lookup: "chapter 3" → node_id
        title_lookup: dict[str, str] = {}
        for n in nodes:
            key = n.title.strip().lower()
            title_lookup[key] = n.id
            # Also index by type+order: "chapter 1"
            if n.type in (NodeType.CHAPTER, NodeType.SECTION):
                type_key = f"{n.type.value.lower()} {n.order}"
                title_lookup[type_key] = n.id

        for node in nodes:
            matches = REFERENCE_PATTERN.findall(node.text)
            for match in matches:
                ref_key = match.strip().lower()
                # Normalise: "chapter 3" → look up
                for key, target_id in title_lookup.items():
                    if key in ref_key and target_id != node.id:
                        edges.append(DKGEdge(
                            source_id  = node.id,
                            target_id  = target_id,
                            rel_type   = RelType.REFERENCES,
                            axis       = 2,
                            properties = {"matched_text": match.strip()},
                        ))
                        break

    def _get_page(self, item) -> int:
        try:
            return item.prov[0].page_no
        except Exception:
            return 0

    def _max_page(self, doc: DoclingDocument) -> int:
        max_p = 0
        for item, _ in doc.iterate_items():
            p = self._get_page(item)
            if p > max_p:
                max_p = p
        return max_p


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
