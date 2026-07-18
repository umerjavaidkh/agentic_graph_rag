"""storage/blob/factory.py — process-level BlobStore singleton, keyed off settings."""
from __future__ import annotations

from .base import BlobStore

_store_singleton: BlobStore | None = None


def get_blob_store() -> BlobStore:
    """Return the process-level BlobStore singleton (backend chosen via BLOB_STORE_BACKEND)."""
    global _store_singleton
    if _store_singleton is not None:
        return _store_singleton

    from ...config.settings import BLOB_STORE_BACKEND, LOCAL_BLOB_STORE_DIR

    if BLOB_STORE_BACKEND == "minio":
        from .minio_store import MinioBlobStore
        from ...config.settings import (
            MINIO_ENDPOINT,
            MINIO_ACCESS_KEY,
            MINIO_SECRET_KEY,
            MINIO_BUCKET,
            MINIO_SECURE,
        )

        _store_singleton = MinioBlobStore(
            endpoint=MINIO_ENDPOINT,
            access_key=MINIO_ACCESS_KEY,
            secret_key=MINIO_SECRET_KEY,
            bucket=MINIO_BUCKET,
            secure=MINIO_SECURE,
        )
        return _store_singleton

    from .local_store import LocalFsBlobStore

    _store_singleton = LocalFsBlobStore(LOCAL_BLOB_STORE_DIR)
    return _store_singleton
