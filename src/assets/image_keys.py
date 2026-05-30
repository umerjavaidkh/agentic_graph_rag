"""Unified object-storage keys — all document images live under {book}/images/."""
from __future__ import annotations


def _safe_book_id(book_id: str) -> str:
    return book_id.replace("/", "_")


def images_dir(book_id: str) -> str:
    return f"{_safe_book_id(book_id)}/images"


def page_full_image_key(book_id: str, pdf_page: int) -> str:
    return f"{images_dir(book_id)}/page_{pdf_page:04d}_full.jpg"


def region_image_key(book_id: str, pdf_page: int, kind: str, index: int) -> str:
    return f"{images_dir(book_id)}/page_{pdf_page:04d}_{kind}_{index:02d}.jpg"


# Backward-compatible alias used by page_images module
def page_image_key(book_id: str, pdf_page: int) -> str:
    return page_full_image_key(book_id, pdf_page)
