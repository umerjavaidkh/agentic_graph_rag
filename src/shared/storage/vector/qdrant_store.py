"""storage/vector/qdrant_store.py — production VectorStore backend (Qdrant)."""
from __future__ import annotations

import uuid
from typing import Optional

from .base import VectorStore

# Qdrant point IDs must be an unsigned int or a UUID — our DKGNode ids are
# arbitrary strings (slugs), so we derive a stable UUID5 per id and keep the
# original id in the point payload for round-trip on query results.
_ID_NAMESPACE = uuid.UUID("5c1f6a1e-9b2e-4f8a-9d3a-9b6f0f6b7a10")


def _point_id(id: str) -> str:
    return str(uuid.uuid5(_ID_NAMESPACE, id))


class QdrantVectorStore(VectorStore):
    """Wraps `qdrant_client`. Requires the `qdrant-client` package (see requirements.txt)."""

    def __init__(
        self,
        url: str,
        collection_name: str,
        *,
        api_key: Optional[str] = None,
        dim: int = 1536,
        distance: str = "Cosine",
    ):
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, VectorParams

        self._client = QdrantClient(url=url, api_key=api_key)
        self.collection_name = collection_name
        if not self._client.collection_exists(collection_name):
            self._client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=dim, distance=Distance(distance)),
            )

    def upsert(self, id: str, embedding: list[float], *, metadata: Optional[dict] = None) -> None:
        self.upsert_batch([(id, embedding, metadata)])

    def upsert_batch(self, items: list[tuple[str, list[float], Optional[dict]]]) -> None:
        from qdrant_client.models import PointStruct

        points = [
            PointStruct(
                id=_point_id(id),
                vector=embedding,
                payload={**(metadata or {}), "_source_id": id},
            )
            for id, embedding, metadata in items
        ]
        self._client.upsert(collection_name=self.collection_name, points=points)

    def query(
        self, embedding: list[float], top_k: int = 10, *, filters: Optional[dict] = None
    ) -> list[tuple[str, float]]:
        from qdrant_client.models import FieldCondition, Filter, MatchValue

        query_filter = None
        if filters:
            query_filter = Filter(
                must=[FieldCondition(key=k, match=MatchValue(value=v)) for k, v in filters.items()]
            )

        # `.search()` was removed in newer qdrant-client releases (>=1.10) in
        # favor of `.query_points()`, which wraps hits in a QueryResponse
        # instead of returning a bare list.
        response = self._client.query_points(
            collection_name=self.collection_name,
            query=embedding,
            limit=top_k,
            query_filter=query_filter,
        )
        return [(hit.payload.get("_source_id", str(hit.id)), hit.score) for hit in response.points]

    def query_with_docs(
        self, embedding: list[float], top_k: int = 10, *, filters: Optional[dict] = None
    ) -> list[tuple[str, float, Optional[str]]]:
        """As `query`, plus the document each hit belongs to.

        Read from the payload `_write_lean_storage` already writes, rather
        than parsed back out of the node id: the payload is what the
        exporter actually recorded, so it stays right for any id shape.
        """
        from qdrant_client.models import FieldCondition, Filter, MatchValue

        query_filter = None
        if filters:
            query_filter = Filter(
                must=[FieldCondition(key=k, match=MatchValue(value=v)) for k, v in filters.items()]
            )
        response = self._client.query_points(
            collection_name=self.collection_name,
            query=embedding,
            limit=top_k,
            query_filter=query_filter,
        )
        out: list[tuple[str, float, Optional[str]]] = []
        for hit in response.points:
            payload = hit.payload or {}
            node_id = payload.get("_source_id", str(hit.id))
            doc_id = payload.get("logical_doc_id") or self.doc_id_for_node_id(node_id)
            out.append((node_id, hit.score, doc_id))
        return out

    def delete(self, id: str) -> None:
        self._client.delete(
            collection_name=self.collection_name,
            points_selector=[_point_id(id)],
        )

    def delete_by_filter(self, filters: dict) -> None:
        from qdrant_client.models import FieldCondition, Filter, FilterSelector, MatchValue

        query_filter = Filter(
            must=[FieldCondition(key=k, match=MatchValue(value=v)) for k, v in filters.items()]
        )
        self._client.delete(
            collection_name=self.collection_name,
            points_selector=FilterSelector(filter=query_filter),
        )

    def point_id_for(self, node_id: str) -> str:
        return _point_id(node_id)
