"""Crop TABLE / FIGURE regions from PDF using Docling bboxes."""
from __future__ import annotations

import io
import re
from collections import defaultdict
from pathlib import Path

import fitz
from PIL import Image

from ..config.settings import (
    ENABLE_REGION_IMAGES,
    PAGE_IMAGE_JPEG_QUALITY,
)
from ..models import DKGNode, NodeType
from .factory import get_asset_store
from .image_keys import region_image_key

_FIGURE_REF = re.compile(r"\bfigure\s+(\d+(?:\.\d+)?)\b", re.I)
_TABLE_REF = re.compile(r"\btable\s+([a-z]?\d+(?:\.\d+)?)\b", re.I)


def _crop_page_jpeg(
    page: fitz.Page,
    bbox: list[float],
    page_size: list[float],
    quality: int,
    padding: float = 4.0,
) -> bytes | None:
    if len(bbox) != 4 or len(page_size) != 2:
        return None
    pw, ph = float(page_size[0]), float(page_size[1])
    if pw <= 0 or ph <= 0:
        return None

    rect = page.rect
    sx = rect.width / pw
    sy = rect.height / ph
    l, t, r, b = (float(v) for v in bbox)
    x0 = max(0, l * sx - padding)
    y0 = max(0, t * sy - padding)
    x1 = min(rect.width, r * sx + padding)
    y1 = min(rect.height, b * sy + padding)
    if x1 <= x0 or y1 <= y0:
        return None

    clip = fitz.Rect(x0, y0, x1, y1)
    pix = page.get_pixmap(dpi=120, clip=clip)
    img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality, optimize=True)
    return buf.getvalue()


def save_region_images(
    pdf_path: str | Path,
    book_id: str,
    nodes: list[DKGNode],
) -> int:
    if not ENABLE_REGION_IMAGES:
        return 0

    region_nodes = [
        n for n in nodes
        if n.type in (NodeType.REGION, NodeType.REGION.value)
        and n.bbox and n.bbox_page_size
    ]
    if not region_nodes:
        return 0

    store = get_asset_store()
    doc = fitz.open(str(pdf_path))
    saved = 0
    try:
        for rn in sorted(region_nodes, key=lambda n: (n.pdf_page or 0, n.order)):
            pdf_page = rn.pdf_page or rn.page_start or rn.order
            if pdf_page < 1 or pdf_page > len(doc):
                continue
            kind = rn.region_kind or "region"
            key = region_image_key(book_id, pdf_page, kind, rn.order)
            data = _crop_page_jpeg(
                doc[pdf_page - 1],
                rn.bbox or [],
                rn.bbox_page_size or [],
                PAGE_IMAGE_JPEG_QUALITY,
            )
            if not data:
                continue
            store.put(key, data)
            rn.image_key = key
            saved += 1
    finally:
        doc.close()
    return saved


def build_region_tags(
    kind: str,
    text: str,
    pdf_page: int,
    index: int,
    document_page: str | None = None,
) -> list[str]:
    tags = [
        f"kind:{kind}",
        f"pdf:{pdf_page}",
        f"region:{pdf_page}:{index}",
    ]
    if document_page:
        tags.append(f"doc:{document_page.strip()}")

    blob = (text or "").lower()
    for ref in _TABLE_REF.findall(blob):
        tags.append(f"table:{ref.lower()}")
    for ref in _FIGURE_REF.findall(blob):
        tags.append(f"figure:{ref.lower()}")

    if kind == "table" and "table:" not in " ".join(tags):
        tags.append("table")
        tags.append(f"table:{index}")
    if kind == "figure":
        tags.append("figure")
        tags.append(f"figure:{index}")

    return list(dict.fromkeys(tags))


def region_title(kind: str, text: str, pdf_page: int, index: int) -> str:
    first_line = (text or "").strip().splitlines()[0][:120] if text else ""
    if first_line:
        return first_line
    label = "Table" if kind == "table" else "Figure"
    return f"{label} {index} (PDF page {pdf_page})"
