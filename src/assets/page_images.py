"""Render PDF pages to compressed JPEG and store via AssetStore."""
from __future__ import annotations

import io
from pathlib import Path
from typing import Optional

import fitz
from PIL import Image

from ..config.settings import (
    ENABLE_PAGE_IMAGES,
    PAGE_IMAGE_JPEG_QUALITY,
    PAGE_IMAGE_MAX_PAGES,
    PAGE_IMAGE_SELECTIVE,
    PAGE_IMAGE_SKIP_WHEN_REGIONS,
    VISION_MIN_TEXT_CHARS,
)
from ..document.page_vision import VISUAL_PAGE_HINTS
from ..models import DKGNode, NodeType
from .factory import get_asset_store
from .image_keys import page_full_image_key


def _select_pages_for_images(
    page_nodes: list[DKGNode],
    section_nodes: list[DKGNode],
) -> list[DKGNode]:
    if not PAGE_IMAGE_SELECTIVE:
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
        pdf_page = pn.pdf_page or pn.order
        text_len = len((pn.text or "").strip())
        if pdf_page in visual_pages or text_len < VISION_MIN_TEXT_CHARS:
            selected.append(pn)
        elif VISUAL_PAGE_HINTS.search(pn.text or ""):
            selected.append(pn)
    return selected


def _render_page_jpeg(page: fitz.Page, quality: int) -> bytes:
    pix = page.get_pixmap(dpi=120)
    img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality, optimize=True)
    return buf.getvalue()


def save_document_page_images(
    pdf_path: str | Path,
    book_id: str,
    nodes: list[DKGNode],
) -> int:
    if not ENABLE_PAGE_IMAGES:
        return 0

    pdf_path = Path(pdf_path)
    page_nodes = [
        n for n in nodes
        if n.type in (NodeType.PAGE, NodeType.PAGE.value)
    ]
    section_nodes = [
        n for n in nodes
        if n.type in (NodeType.SECTION, NodeType.SECTION.value)
    ]
    targets = _select_pages_for_images(page_nodes, section_nodes)
    # Always keep full-page JPEGs (even when region crops exist) for whole-page queries.
    if PAGE_IMAGE_MAX_PAGES > 0:
        targets = targets[:PAGE_IMAGE_MAX_PAGES]

    store = get_asset_store()
    doc = fitz.open(str(pdf_path))
    saved = 0
    try:
        for pn in targets:
            pdf_page = pn.pdf_page or pn.page_start or pn.order
            if pdf_page < 1 or pdf_page > len(doc):
                continue
            key = page_full_image_key(book_id, pdf_page)
            data = _render_page_jpeg(doc[pdf_page - 1], PAGE_IMAGE_JPEG_QUALITY)
            store.put(key, data)
            pn.image_key = key
            saved += 1
    finally:
        doc.close()
    return saved


def resolve_image_url(image_key: Optional[str]) -> Optional[str]:
    if not image_key:
        return None
    store = get_asset_store()
    return store.public_url(image_key)
