"""storage/vector/memory_store.py — zero-dependency dev/test VectorStore backend."""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from .base import VectorStore

logger = logging.getLogger(__name__)

# query() is a brute-force O(n) scan with no spatial index — fine for tests
# and small local corpora, but silently expensive if this backend ends up
# serving real traffic (e.g. VECTOR_STORE_BACKEND misconfigured in a
# deployment that meant to run Qdrant). Warn once, not every query, past
# this size so a misconfiguration surfaces in logs instead of just getting
# slower and slower unnoticed.
_SIZE_WARNING_THRESHOLD = 10_000


class InMemoryVectorStore(VectorStore):
    """Plain dict + numpy cosine similarity. For tests and single-process dev."""

    def __init__(self):
        self._vectors: dict[str, np.ndarray] = {}
        self._metadata: dict[str, dict] = {}
        self._size_warned = False

    def upsert(self, id: str, embedding: list[float], *, metadata: Optional[dict] = None) -> None:
        self._vectors[id] = np.asarray(embedding, dtype=np.float32)
        self._metadata[id] = metadata or {}
        if not self._size_warned and len(self._vectors) > _SIZE_WARNING_THRESHOLD:
            self._size_warned = True
            logger.warning(
                "InMemoryVectorStore holds %d vectors (> %d) — this backend is a "
                "brute-force O(n) scan meant for tests/single-process dev, not "
                "production scale. If this is a real deployment, set "
                "VECTOR_STORE_BACKEND=qdrant.",
                len(self._vectors),
                _SIZE_WARNING_THRESHOLD,
            )

    def upsert_batch(self, items: list[tuple[str, list[float], Optional[dict]]]) -> None:
        for id, embedding, metadata in items:
            self.upsert(id, embedding, metadata=metadata)

    def query(
        self, embedding: list[float], top_k: int = 10, *, filters: Optional[dict] = None
    ) -> list[tuple[str, float]]:
        if not self._vectors:
            return []
        query_vec = np.asarray(embedding, dtype=np.float32)
        query_norm = np.linalg.norm(query_vec) or 1.0

        candidates = self._vectors.keys()
        if filters:
            candidates = [
                id
                for id in candidates
                if all(self._metadata.get(id, {}).get(k) == v for k, v in filters.items())
            ]

        scored: list[tuple[str, float]] = []
        for id in candidates:
            vec = self._vectors[id]
            denom = (np.linalg.norm(vec) or 1.0) * query_norm
            score = float(np.dot(query_vec, vec) / denom)
            scored.append((id, score))

        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored[:top_k]

    def delete(self, id: str) -> None:
        self._vectors.pop(id, None)
        self._metadata.pop(id, None)

    def delete_by_filter(self, filters: dict) -> None:
        to_delete = [
            id
            for id, meta in self._metadata.items()
            if all(meta.get(k) == v for k, v in filters.items())
        ]
        for id in to_delete:
            self.delete(id)
