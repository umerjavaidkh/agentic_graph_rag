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

    def query_with_docs(
        self, embedding: list[float], top_k: int = 10, *, filters: Optional[dict] = None
    ) -> list[tuple[str, float, Optional[str]]]:
        """Return [(id, score, logical_doc_id), ...] ordered by descending similarity.

        Which documents a question is about is a question the vector index
        can already answer, and answer in time that does not grow with the
        corpus. The graph-side alternative walks every document's subtree
        applying a regex per node; this is one ANN lookup.

        The default implementation derives the document from the node id,
        which is correct for any backend whose ids carry the logical id as a
        prefix, and returns None when it cannot. A backend that stores the
        document alongside the vector should override and read it back
        rather than parse it.
        """
        return [
            (id, score, self.doc_id_for_node_id(id))
            for id, score in self.query(embedding, top_k, filters=filters)
        ]

    @staticmethod
    def doc_id_for_node_id(node_id: str) -> Optional[str]:
        """The logical document id embedded in a content node's id, if any.

        Revision-scoped ids look like ``<logical_id>:<revision>::<rest>``.
        Mirrors DocumentResolver._logical_id_from_node_id -- kept here too so
        a store can answer without importing retrieval code.
        """
        if not node_id:
            return None
        return node_id.split(":", 1)[0] or None

    @abstractmethod
    def delete(self, id: str) -> None: ...

    @abstractmethod
    def delete_by_filter(self, filters: dict) -> None:
        """Used for revision-expiry purge (parallel to Neo4j's DETACH DELETE on supersede)."""

    @abstractmethod
    def set_payload_by_filter(self, filters: dict, payload: dict) -> int:
        """Add or overwrite payload fields on every point matching `filters`.

        For migrations that add a scoping field to points already written.
        A vector is expensive to recompute and none of this needs to: the
        embedding is unchanged, only what the store knows about it.

        A filter value that is a list matches any of its members, so a
        migration can name every document of one language in one call
        instead of one call per document.

        Returns the number of points updated where the backend reports it,
        and 0 where it does not.
        """

    def point_id_for(self, node_id: str) -> str:
        """The id this store actually keys a node's vector under -- may
        differ from `node_id` itself (e.g. Qdrant requires an unsigned int
        or UUID, so it derives one deterministically). Default: identity,
        correct for any backend (like InMemoryVectorStore) that keys
        directly off the node's own id. Persisted on DKGNode.vector_id at
        write time (docs/DESIGN_unstructured_graph_v2.md phase 3) so
        callers don't have to re-derive it."""
        return node_id
