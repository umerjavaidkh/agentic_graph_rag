"""Remove stale JPEG crops when a book is re-ingested or the DB is reset."""
from __future__ import annotations

from .factory import get_asset_store
from .image_keys import _safe_book_id


def cleanup_book_assets(book_id: str) -> int:
    """Delete all images for one book (page + region crops)."""
    prefix = f"{_safe_book_id(book_id)}/"
    return get_asset_store().delete_prefix(prefix)


def cleanup_all_book_assets() -> int:
    """Delete every object under the asset store root (pair with Neo4j wipe)."""
    return get_asset_store().delete_prefix("")
