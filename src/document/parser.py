"""
parser.py — Structure-aware Docling → DKGNode tree.

Builds a real hierarchy from Docling reading order and item levels:
    Book → Chapter (TITLE) → Section (nested via SECTION_HEADER levels / numbering)
         → Page

Section → Section CONTAINS edges are created when Docling depth or numbered
headings indicate subsections (fixes flat sibling-only graphs).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from docling.document_converter import DocumentConverter
from docling_core.types.doc import DocItemLabel, DoclingDocument

from ..models import DKGEdge, DKGNode, NodeType, RelType
from .page_numbers import enrich_page_nodes
from .patterns import (
    REFERENCE_PATTERN,
    is_standalone_number,
    number_depth,
    parse_numbered_title,
    slug,
)

CHAPTER_LABELS = {DocItemLabel.TITLE}
SECTION_LABELS = {DocItemLabel.SECTION_HEADER}
TEXT_LABELS = {
    DocItemLabel.TEXT,
    DocItemLabel.LIST_ITEM,
    DocItemLabel.TABLE,
    DocItemLabel.CAPTION,
    DocItemLabel.FOOTNOTE,
}
HEADING_LABELS = CHAPTER_LABELS | SECTION_LABELS


class DoclingParser:
    """
    Converts PDF, DOCX, PPTX, HTML, and pre-exported Docling JSON into a
    hierarchical DKG (nodes + Axis-1 structural edges).
    """

    def __init__(self):
        self.converter = DocumentConverter()

    def parse(self, source: str | Path) -> tuple[list[DKGNode], list[DKGEdge]]:
        source = Path(source)
        doc_name = source.stem

        if source.suffix.lower() == ".json":
            print(f"   📂 Loading pre-parsed Docling JSON: {source.name}")
            doc = DoclingDocument.model_validate_json(
                source.read_text(encoding="utf-8")
            )
        else:
            print(f"   🔍 Parsing document via Docling: {source.name}")
            result = self.converter.convert(str(source))
            doc = result.document

        return self._build_from_docling(doc, doc_name)

    def _build_from_docling(
        self, doc: DoclingDocument, doc_name: str
    ) -> tuple[list[DKGNode], list[DKGEdge]]:
        nodes: list[DKGNode] = []
        edges: list[DKGEdge] = []

        book_id = f"book_{slug(doc_name)}"
        book_node = DKGNode(
            id=book_id,
            type=NodeType.BOOK,
            title=doc_name,
            text=doc_name,
            order=0,
            page_start=1,
            page_end=self._max_page(doc),
            depth=0,
        )
        nodes.append(book_node)

        raw_items = self._extract_raw_items(doc)
        items = self._resolve_number_prefixes(raw_items)

        chapter_nodes: list[DKGNode] = []
        section_nodes: list[DKGNode] = []
        page_buckets: dict[int, list[str]] = {}

        current_chapter: DKGNode | None = None
        current_section: DKGNode | None = None
        current_section_texts: list[str] = []

        chapter_idx = 0
        global_section_idx = 0
        doc_order = 0

        # (structural_level, node_id) — parent = nearest entry with lower level
        heading_stack: list[tuple[int, str]] = [(0, book_id)]
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
            label: DocItemLabel | None,
            title: str,
            docling_level: int,
            section_number: str | None,
        ) -> int:
            """Map heading to a comparable depth for parent stack (higher = deeper)."""
            if label in CHAPTER_LABELS:
                return max(1, docling_level) if docling_level > 0 else 1

            if docling_level > 1:
                return docling_level

            if section_number:
                # e.g. "4.5" → depth 2; under chapter (+1) → at least 2
                nd = number_depth(section_number)
                base = nd + (1 if current_chapter else 0)
                return max(2, base)

            return max(2, docling_level) if docling_level > 0 else 2

        def parent_id_for_level(level: int) -> str:
            while len(heading_stack) > 1 and heading_stack[-1][0] >= level:
                heading_stack.pop()
            return heading_stack[-1][1]

        for item in items:
            label = item["label"]
            text = item["text"]
            page_no = item["page"]
            docling_level = item["level"]

            if label in HEADING_LABELS:
                finalize_section()

                section_number, title = parse_numbered_title(text)
                level = structural_level(label, title, docling_level, section_number)

                # Docling often emits subsections at the same level as their parent.
                # Nest under the current section when depth would otherwise make them siblings.
                if (
                    label in SECTION_LABELS
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

                if label in CHAPTER_LABELS:
                    chapter_idx += 1
                    global_section_idx = 0
                    node_id = f"chapter_{chapter_idx}"
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
                    heading_stack = [(0, book_id), (level, node_id)]
                    link_contains(book_id, node_id)
                else:
                    global_section_idx += 1
                    node_id = f"section_{chapter_idx}_{global_section_idx}"
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

            if label in TEXT_LABELS:
                clean = text.strip()
                if not clean:
                    continue

                if current_section is None:
                    doc_order += 1
                    global_section_idx += 1
                    node_id = f"section_{chapter_idx}_{global_section_idx}"
                    parent_id = (
                        current_chapter.id if current_chapter else book_id
                    )
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

                prefix = "[Table] " if label == DocItemLabel.TABLE else ""
                current_section_texts.append(prefix + clean)
                page_buckets.setdefault(page_no, []).append(clean)

                if current_section and page_no > current_section.page_end:
                    current_section.page_end = page_no
                if current_chapter and page_no > current_chapter.page_end:
                    current_chapter.page_end = page_no

        finalize_section()

        page_nodes = self._build_page_nodes(
            page_buckets, edges, chapter_nodes, section_nodes, book_id
        )
        enrich_page_nodes(page_nodes, section_nodes)
        nodes.extend(page_nodes)

        self._add_sequential_edges(chapter_nodes, edges)
        self._add_sequential_edges(section_nodes, edges)
        self._add_sequential_edges(page_nodes, edges)
        self._detect_reference_edges(nodes, edges)
        self._link_number_hierarchy(number_map, edges)

        print(
            f"   📐 Structure: {len(chapter_nodes)} chapters, "
            f"{len(section_nodes)} sections (nested CONTAINS), {len(page_nodes)} pages"
        )
        return nodes, edges

    def _extract_raw_items(self, doc: DoclingDocument) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        for item, level in doc.iterate_items():
            text = self._item_text(doc, item)
            if not text:
                continue
            items.append({
                "text": text,
                "label": getattr(item, "label", None),
                "page": self._get_page(item),
                "level": int(level) if level is not None else 0,
            })
        return items

    def _item_text(self, doc: DoclingDocument, item) -> str:
        """Plain text or markdown table export when Docling leaves item.text empty."""
        text = (getattr(item, "text", "") or "").strip()
        if text:
            return text
        if getattr(item, "label", None) != DocItemLabel.TABLE:
            return ""
        for exporter in ("export_to_markdown", "export_to_html"):
            fn = getattr(item, exporter, None)
            if not callable(fn):
                continue
            try:
                exported = fn(doc=doc)
                if isinstance(exported, str) and exported.strip():
                    return exported.strip()
            except TypeError:
                try:
                    exported = fn()
                    if isinstance(exported, str) and exported.strip():
                        return exported.strip()
                except Exception:
                    pass
            except Exception:
                pass
        df_fn = getattr(item, "export_to_dataframe", None)
        if callable(df_fn):
            try:
                df = df_fn(doc=doc)
                return df.to_markdown(index=False)
            except TypeError:
                try:
                    df = df_fn()
                    return df.to_markdown(index=False)
                except Exception:
                    pass
            except Exception:
                pass
        return ""

    def _resolve_number_prefixes(self, items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Merge '4.5' + 'ENVIRONMENTAL PROTECTION' into one heading when needed."""
        resolved: list[dict[str, Any]] = []
        pending_number: str | None = None

        for item in items:
            text = item["text"]
            label = item["label"]

            if is_standalone_number(text):
                pending_number = text.rstrip(".")
                continue

            if pending_number is not None:
                if label in HEADING_LABELS or not is_standalone_number(text):
                    text = f"{pending_number} {text}"
                    item = {**item, "text": text}
                pending_number = None

            resolved.append(item)

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
        book_id: str,
    ) -> list[DKGNode]:
        page_nodes: list[DKGNode] = []
        for page_no in sorted(page_buckets.keys()):
            texts = page_buckets[page_no]
            node_id = f"page_{page_no}"
            node = DKGNode(
                id=node_id,
                type=NodeType.PAGE,
                title=f"Page {page_no}",
                text="\n".join(texts),
                order=page_no,
                page_start=page_no,
                page_end=page_no,
                depth=99,
            )
            page_nodes.append(node)
            parent_id = self._find_page_parent(
                page_no, section_nodes, chapter_nodes, book_id
            )
            edges.append(DKGEdge(parent_id, node_id, RelType.CONTAINS, axis=1))
            edges.append(DKGEdge(node_id, parent_id, RelType.PART_OF, axis=1))
        return page_nodes

    def _find_page_parent(
        self,
        page_no: int,
        section_nodes: list[DKGNode],
        chapter_nodes: list[DKGNode],
        book_id: str,
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
        return book_id

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

    def _get_page(self, item) -> int:
        try:
            return int(item.prov[0].page_no)
        except Exception:
            return 1

    def _max_page(self, doc: DoclingDocument) -> int:
        max_p = 1
        for item, _ in doc.iterate_items():
            max_p = max(max_p, self._get_page(item))
        return max_p
