"""
Cheap vision pass per PDF page: tables, charts, diagrams, shapes, and other visuals → text.

Stored on Page.visual_content for search/retrieval when normal text parsing is incomplete.
"""
from __future__ import annotations

import base64
import re
from pathlib import Path

import fitz  # PyMuPDF

from ..config.settings import (
    MODEL_PROVIDER,
    OPENAI_API_KEY,
    VISION_DPI,
    VISION_IMAGE_DETAIL,
    VISION_MAX_PAGES_PER_DOC,
    VISION_MIN_TEXT_CHARS,
    VISION_MODEL,
    VISION_SELECTIVE,
)
from ..model_providers.factory import get_model_provider
from ..models import DKGNode, NodeType

VISION_SYSTEM = """You are a document page analyst. Describe ONLY what is visible on the page image.
Do not invent numbers, names, or data that are not clearly readable.

Structure your reply exactly as:

## Tables
- For each table: caption/title if visible, then a complete markdown pipe table with every row and column you can read.

## Charts and graphs
- Chart type (bar, line, pie, map, etc.), title, axis labels, legend, series names, and key values or trends.

## Diagrams and flowcharts
- Type of diagram, boxes/nodes, arrows, labels, and sequence or relationships.

## Shapes, maps, and other figures
- What the figure shows, labels, regions, icons, and captions.

## Other visible content
- Headings, footnotes, page numbers not covered above.

If something is unreadable, write "unclear". Prefer completeness over brevity for tables."""

# Pages likely to contain non-text visuals (not only tables)
VISUAL_PAGE_HINTS = re.compile(
    r"\btable\b|\bfigure\b|\bfig\.\b|\bannex\b|\[table\]|\[figure\]"
    r"|chart|diagram|graph\b|flowchart|flow\s+chart|map\b|illustration"
    r"|screenshot|infographic|box\s+\d+|appendix",
    re.IGNORECASE,
)


class PageVisionEnricher:
    def __init__(self, api_key: str | None = None):
        self.provider = get_model_provider(MODEL_PROVIDER, api_key or OPENAI_API_KEY)

    def enrich_document(
        self,
        pdf_path: str | Path,
        nodes: list[DKGNode],
    ) -> int:
        """Run vision on selected Page nodes. Returns count of pages enriched."""
        pdf_path = Path(pdf_path)
        page_nodes = [
            n for n in nodes
            if n.type == NodeType.PAGE or n.type == NodeType.PAGE.value
        ]
        if not page_nodes:
            return 0

        section_nodes = [
            n for n in nodes
            if n.type in (NodeType.SECTION, NodeType.SECTION.value)
        ]
        targets = self._select_pages(page_nodes, section_nodes)
        if VISION_MAX_PAGES_PER_DOC > 0:
            targets = targets[:VISION_MAX_PAGES_PER_DOC]

        if not targets:
            return 0

        doc = fitz.open(str(pdf_path))
        enriched = 0
        try:
            for page_node in targets:
                page_no = page_node.pdf_page or page_node.page_start or page_node.order
                if page_no < 1 or page_no > len(doc):
                    continue
                description = self._describe_page_image(doc[page_no - 1])
                if description:
                    page_node.visual_content = description.strip()
                    enriched += 1
        finally:
            doc.close()

        return enriched

    def _select_pages(
        self,
        page_nodes: list[DKGNode],
        section_nodes: list[DKGNode],
    ) -> list[DKGNode]:
        if not VISION_SELECTIVE:
            return sorted(page_nodes, key=lambda p: p.order)

        visual_pages: set[int] = set()
        for sec in section_nodes:
            blob = f"{sec.title}\n{sec.text}"
            if VISUAL_PAGE_HINTS.search(blob):
                start = sec.page_start or 1
                end = sec.page_end or start
                for pno in range(start, end + 1):
                    visual_pages.add(pno)

        selected: list[DKGNode] = []
        for pn in sorted(page_nodes, key=lambda p: p.order):
            page_no = pn.page_start or pn.order
            text_len = len((pn.text or "").strip())
            if page_no in visual_pages:
                selected.append(pn)
            elif text_len < VISION_MIN_TEXT_CHARS:
                selected.append(pn)
            elif VISUAL_PAGE_HINTS.search(pn.text or ""):
                selected.append(pn)

        return selected

    def _describe_page_image(self, page: fitz.Page) -> str:
        pix = page.get_pixmap(dpi=VISION_DPI)
        image_b64 = base64.standard_b64encode(pix.tobytes("png")).decode("ascii")

        response = self.provider.chat_completion(
            model=VISION_MODEL,
            messages=[
                {"role": "system", "content": VISION_SYSTEM},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "Describe all tables, charts, graphs, diagrams, maps, "
                                "shapes, and figures on this page."
                            ),
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{image_b64}",
                                "detail": VISION_IMAGE_DETAIL,
                            },
                        },
                    ],
                },
            ],
            temperature=0.0,
            max_tokens=2000,
        )
        return (response.choices[0].message.content or "").strip()
