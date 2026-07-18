"""
storage/vector/base.py — interface for embedding storage/similarity search.

Replaces Neo4j native vector-index properties/queries so embeddings can live
in a dedicated vector DB while Neo4j keeps only structure + a reference id.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional


class VectorStore(ABC):
    @abstractmethod
    def upsert(self, id: str, embedding: list[float], *, metadata: Optional[dict] = None) -> None: ...

    @abstractmethod
    def upsert_batch(self, items: list[tuple[str, list[float], Optional[dict]]]) -> None:
        """Batch upsert — implementations should use this for bulk ingest writes."""

    @abstractmethod
    def query(
        self, embedding: list[float], top_k: int = 10, *, filters: Optional[dict] = None
    ) -> list[tuple[str, float]]:
        """Return [(id, score), ...] ordered by descending similarity."""

    @abstractmethod
    def delete(self, id: str) -> None: ...

    @abstractmethod
    def delete_by_filter(self, filters: dict) -> None:
        """Used for revision-expiry purge (parallel to Neo4j's DETACH DELETE on supersede)."""
