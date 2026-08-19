"""storage/vector/factory.py — process-level VectorStore singleton, keyed off settings."""
from __future__ import annotations

from .base import VectorStore

_store_singleton: VectorStore | None = None


def get_vector_store() -> VectorStore:
    """Return the process-level VectorStore singleton (backend chosen via VECTOR_STORE_BACKEND)."""
    global _store_singleton
    if _store_singleton is not None:
        return _store_singleton

    from ...config.settings import VECTOR_STORE_BACKEND

    if VECTOR_STORE_BACKEND == "qdrant":
        from .qdrant_store import QdrantVectorStore
        from ...config.settings import QDRANT_URL, QDRANT_COLLECTION, QDRANT_API_KEY, VECTOR_DIM

        _store_singleton = QdrantVectorStore(
            url=QDRANT_URL,
            collection_name=QDRANT_COLLECTION,
            api_key=QDRANT_API_KEY or None,
            dim=VECTOR_DIM,
        )
        return _store_singleton

    from .memory_store import InMemoryVectorStore

    _store_singleton = InMemoryVectorStore()
    return _store_singleton
