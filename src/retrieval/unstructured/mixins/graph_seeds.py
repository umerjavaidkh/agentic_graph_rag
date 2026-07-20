"""Document RAG retriever — graph seeds."""
from __future__ import annotations

from ....config.settings import EMBEDDING_MODEL, VECTOR_STORE_BACKEND
from ....graph.tenancy import tenant_filter
from ....graph.versioning import lifecycle_active
from ....storage.vector.factory import get_vector_store
from ..constants import _GRAPH_REL_TYPES, _TEXT_NODE_LABELS
from ..model_provider import provider


class GraphSeedsMixin:
    def _vector_seed(
        self, session, embedding: list[float], limit: int, tenant_id: str = ""
    ) -> list[dict]:
        # The in-process "memory" VectorStore doesn't survive across worker/API
        # process boundaries, so only a real shared external store (Qdrant) is
        # trustworthy as the similarity-search read path; otherwise fall back
        # to Neo4j's native vector index, which dual-write keeps populated.
        if VECTOR_STORE_BACKEND == "qdrant":
            return self._vector_seed_via_vector_store(session, embedding, limit, tenant_id)
        try:
            rows = session.run(
                f"""
                CALL db.index.vector.queryNodes('section_embedding', $limit, $embedding)
                YIELD node AS n, score
                WHERE coalesce(n.text, '') <> ''
                  AND {lifecycle_active("n")}
                  AND {tenant_filter("n")}
                RETURN
                  coalesce(n.id, '') AS id,
                  coalesce(n.title, '') AS title,
                  coalesce(n.text, '') AS text,
                  coalesce(labels(n)[0], '') AS node_label,
                  score
                ORDER BY score DESC
                """,
                limit=max(1, limit),
                embedding=embedding,
                tenant_id=tenant_id,
            )
            return [
                {
                    "id": r["id"],
                    "title": r["title"] or r["id"],
                    "text": r["text"],
                    "node_label": r.get("node_label") or "",
                    "score": float(r["score"] or 0.0),
                    "related": [],
                }
                for r in rows
                if r["id"]
            ]
        except Exception:
            return []

    def _vector_seed_via_vector_store(
        self, session, embedding: list[float], limit: int, tenant_id: str = ""
    ) -> list[dict]:
        """Similarity search against the external VectorStore, hydrated from Neo4j."""
        try:
            filters = {"tenant_id": tenant_id} if tenant_id else None
            hits = get_vector_store().query(embedding, top_k=max(1, limit), filters=filters)
            if not hits:
                return []
            scores = dict(hits)
            rows = session.run(
                f"""
                UNWIND $ids AS nid
                MATCH (n) WHERE n.id = nid
                WHERE coalesce(n.text, '') <> ''
                  AND {lifecycle_active("n")}
                  AND {tenant_filter("n")}
                RETURN
                  coalesce(n.id, '') AS id,
                  coalesce(n.title, '') AS title,
                  coalesce(n.text, '') AS text,
                  coalesce(labels(n)[0], '') AS node_label
                """,
                ids=list(scores.keys()),
                tenant_id=tenant_id,
            )
            by_id = {r["id"]: r for r in rows if r["id"]}
            results = [
                {
                    "id": id,
                    "title": row["title"] or id,
                    "text": row["text"],
                    "node_label": row.get("node_label") or "",
                    "score": float(scores.get(id, 0.0)),
                    "related": [],
                }
                for id, row in by_id.items()
            ]
            results.sort(key=lambda r: r["score"], reverse=True)
            return results
        except Exception:
            return []

    def _fulltext_seed(self, session, query: str, limit: int, tenant_id: str = "") -> list[dict]:
        lucene_q = self._fulltext_query(query)
        if not lucene_q:
            return []
        try:
            rows = session.run(
                f"""
                CALL db.index.fulltext.queryNodes('node_text_index', $q, {{limit: $limit}})
                YIELD node AS n, score
                WHERE coalesce(n.text, '') <> ''
                  AND {lifecycle_active("n")}
                  AND {tenant_filter("n")}
                  AND any(l IN labels(n) WHERE l IN $labels)
                RETURN
                  coalesce(n.id, '') AS id,
                  coalesce(n.title, '') AS title,
                  coalesce(n.text, '') AS text,
                  coalesce(labels(n)[0], '') AS node_label,
                  score
                ORDER BY score DESC
                """,
                q=lucene_q,
                limit=max(1, limit),
                labels=list(_TEXT_NODE_LABELS),
                tenant_id=tenant_id,
            )
            return [
                {
                    "id": r["id"],
                    "title": r["title"] or r["id"],
                    "text": r["text"],
                    "node_label": r.get("node_label") or "",
                    "score": float(r["score"] or 0.0),
                    "related": [],
                }
                for r in rows
                if r["id"]
            ]
        except Exception:
            return []

    def _graph_expand(
        self,
        session,
        seed_ids: list[str],
        *,
        hops: int,
        limit: int,
        tenant_id: str = "",
    ) -> list[dict]:
        if hops == 1:
            cypher = f"""
                UNWIND $seed_ids AS sid
                MATCH (seed:Section {{id: sid}})
                WHERE {lifecycle_active("seed")}
                  AND {tenant_filter("seed")}
                MATCH (seed)-[r]-(related)
                WHERE type(r) IN $rel_types
                  AND any(l IN labels(related) WHERE l IN $node_labels)
                  AND coalesce(related.text, '') <> ''
                  AND {lifecycle_active("related")}
                  AND {tenant_filter("related")}
                RETURN DISTINCT
                  coalesce(related.id, '') AS id,
                  coalesce(related.title, '') AS title,
                  coalesce(related.text, '') AS text,
                  coalesce(labels(related)[0], '') AS node_label,
                  type(r) AS rel_type,
                  coalesce(r.weight, 0.75) AS edge_weight,
                  sid AS seed_id,
                  1 AS hops
                LIMIT $limit
            """
        else:
            cypher = f"""
                UNWIND $seed_ids AS sid
                MATCH (seed:Section {{id: sid}})
                WHERE {lifecycle_active("seed")}
                  AND {tenant_filter("seed")}
                MATCH (seed)-[r1]-(mid)-[r2]-(related)
                WHERE type(r1) IN $rel_types
                  AND type(r2) IN $rel_types
                  AND any(l IN labels(related) WHERE l IN $node_labels)
                  AND coalesce(related.text, '') <> ''
                  AND {lifecycle_active("related")}
                  AND {tenant_filter("related")}
                  AND related.id <> sid
                RETURN DISTINCT
                  coalesce(related.id, '') AS id,
                  coalesce(related.title, '') AS title,
                  coalesce(related.text, '') AS text,
                  coalesce(labels(related)[0], '') AS node_label,
                  type(r1) + '->' + type(r2) AS rel_type,
                  coalesce(r2.weight, 0.75) AS edge_weight,
                  sid AS seed_id,
                  2 AS hops
                LIMIT $limit
            """
        try:
            rows = session.run(
                cypher,
                seed_ids=seed_ids,
                rel_types=list(_GRAPH_REL_TYPES),
                node_labels=list(_TEXT_NODE_LABELS),
                limit=max(1, limit),
                tenant_id=tenant_id,
            )
            return [
                {
                    "id": r["id"],
                    "title": r["title"] or r["id"],
                    "text": r["text"],
                    "node_label": r.get("node_label") or "",
                    "rel_type": r["rel_type"],
                    "edge_weight": float(r["edge_weight"] or 0.75),
                    "seed_id": r["seed_id"],
                    "hops": int(r["hops"] or hops),
                    "related": [r["rel_type"]] if r.get("rel_type") else [],
                }
                for r in rows
                if r["id"]
            ]
        except Exception:
            return []

    def _get_embedding(self, text: str) -> list[float]:
        resp = provider.embeddings(model=EMBEDDING_MODEL, input=(text or "")[:8000])
        return list(resp.data[0].embedding)

    def _fulltext_query(self, question: str) -> str:
        """Build a Lucene query from question terms (document-agnostic)."""
        phrases = self._search_phrases_from_query(question)
        if phrases:
            quoted = [f'"{p}"' for p in phrases[:5] if " " in p]
            terms = self._query_keywords(question)[:6]
            parts = quoted + terms
            if parts:
                return " OR ".join(parts)
        keywords = self._query_keywords(question)
        extra_stop = {"employees", "employee", "company", "corporate", "policy"}
        keywords = [k for k in keywords if k not in extra_stop][:14]
        if not keywords:
            return (question or "")[:120]
        return " OR ".join(keywords)

