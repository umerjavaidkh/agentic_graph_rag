from .factory import get_asset_store
from .image_keys import _safe_document_id


def cleanup_document_assets(document_id: str) -> int:
    store = get_asset_store()
    prefix = f"{_safe_document_id(document_id)}/"
    return store.delete_prefix(prefix)


def cleanup_all_document_assets() -> int:
    """Delete every object under the asset store root (pair with Neo4j wipe)."""
    return get_asset_store().delete_prefix("")


# Deprecated aliases
cleanup_book_assets = cleanup_document_assets
cleanup_all_book_assets = cleanup_all_document_assets
