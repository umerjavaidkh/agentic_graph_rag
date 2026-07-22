"""graph_seeds.py — vector/fulltext seeding and graph expansion, shared by retrieval strategies.

Extracted from mixins/graph_seeds.py (GraphSeedsMixin). Depends on
RankingService for query-keyword/phrase extraction used in Lucene query
building — constructed after RankingService in the service bundle.
"""
from __future__ import annotations

from ....config.settings import EMBEDDING_MODEL, VECTOR_STORE_BACKEND
from ....graph.tenancy import tenant_filter
from ....graph.versioning import lifecycle_active
from ....storage.vector.factory import get_vector_store
from ..constants import _GRAPH_REL_TYPES, _TEXT_NODE_LABELS
from ..model_provider import provider
from .ranking import RankingService


def _document_filter(alias: str, document_id: str) -> str:
    """WHERE-clause conjunct scoping `alias` to one resolved document.

    Degrades to harmless "true" when no document was confidently resolved
    (ambiguous or single-document corpus) — same idiom as tenant_filter(),
    so every call site can splice `AND {_document_filter(...)}`
    unconditionally with zero behavior change when scoping isn't possible.
    """
    if not document_id:
        return "true"
    return f"{alias}.logical_doc_id = $document_id"


class GraphSeedService:
    def __init__(self, ranking: RankingService):
        self._ranking = ranking

    def vector_seed(
        self,
        session,
        embedding: list[float],
        limit: int,
        tenant_id: str = "",
        document_id: str = "",
    ) -> list[dict]:
        # The in-process "memory" VectorStore doesn't survive across worker/API
        # process boundaries, so only a real shared external store (Qdrant) is
        # trustworthy as the similarity-search read path; otherwise fall back
        # to Neo4j's native vector index, which dual-write keeps populated.
        if VECTOR_STORE_BACKEND == "qdrant":
            return self.vector_seed_via_vector_store(session, embedding, limit, tenant_id, document_id)
        try:
            # Neo4j's vector index returns its top-K globally BEFORE the WHERE
            # filter runs, so a document filter applied on the same $limit
            # could starve the target document's own matches if they're not
            # already in the unfiltered global top-K (e.g. a larger, more
            # topically-similar document crowds them out). Over-fetch from
            # the index when scoping, then filter and cap to the originally
            # requested limit — a fixed multiplier, not tuned to one corpus.
            index_limit = min(limit * 8, 200) if document_id else limit
            rows = session.run(
                f"""
                CALL db.index.vector.queryNodes('section_embedding', $index_limit, $embedding)
                YIELD node AS n, score
                WHERE coalesce(n.text, '') <> ''
                  AND {lifecycle_active("n")}
                  AND {tenant_filter("n")}
                  AND {_document_filter("n", document_id)}
                RETURN
                  coalesce(n.id, '') AS id,
                  coalesce(n.title, '') AS title,
                  coalesce(n.text, '') AS text,
                  coalesce(labels(n)[0], '') AS node_label,
                  score
                ORDER BY score DESC
                LIMIT $limit
                """,
                index_limit=max(1, index_limit),
                limit=max(1, limit),
                embedding=embedding,
                tenant_id=tenant_id,
                document_id=document_id,
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

    def vector_seed_via_vector_store(
        self,
        session,
        embedding: list[float],
        limit: int,
        tenant_id: str = "",
        document_id: str = "",
    ) -> list[dict]:
        """Similarity search against the external VectorStore, hydrated from Neo4j."""
        try:
            filters: dict = {}
            if tenant_id:
                filters["tenant_id"] = tenant_id
            if document_id:
                filters["logical_doc_id"] = document_id
            hits = get_vector_store().query(embedding, top_k=max(1, limit), filters=filters or None)
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

    def fulltext_seed(
        self, session, query: str, limit: int, tenant_id: str = "", document_id: str = ""
    ) -> list[dict]:
        lucene_q = self.fulltext_query(query)
        if not lucene_q:
            return []
        try:
            # Same over-fetch-then-filter reasoning as vector_seed: the
            # fulltext index's own $limit caps results BEFORE the WHERE
            # document filter runs.
            index_limit = min(limit * 8, 200) if document_id else limit
            rows = session.run(
                f"""
                CALL db.index.fulltext.queryNodes('node_text_index', $q, {{limit: $index_limit}})
                YIELD node AS n, score
                WHERE coalesce(n.text, '') <> ''
                  AND {lifecycle_active("n")}
                  AND {tenant_filter("n")}
                  AND {_document_filter("n", document_id)}
                  AND any(l IN labels(n) WHERE l IN $labels)
                RETURN
                  coalesce(n.id, '') AS id,
                  coalesce(n.title, '') AS title,
                  coalesce(n.text, '') AS text,
                  coalesce(labels(n)[0], '') AS node_label,
                  score
                ORDER BY score DESC
                LIMIT $limit
                """,
                q=lucene_q,
                index_limit=max(1, index_limit),
                limit=max(1, limit),
                labels=list(_TEXT_NODE_LABELS),
                tenant_id=tenant_id,
                document_id=document_id,
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

    def graph_expand(
        self,
        session,
        seed_ids: list[str],
        *,
        hops: int,
        limit: int,
        tenant_id: str = "",
        document_id: str = "",
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
                  AND {_document_filter("related", document_id)}
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
                  AND {_document_filter("related", document_id)}
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
                document_id=document_id,
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

    def get_embedding(self, text: str) -> list[float]:
        resp = provider.embeddings(model=EMBEDDING_MODEL, input=(text or "")[:8000])
        return list(resp.data[0].embedding)

    @staticmethod
    def _with_singular_variants(terms: list[str]) -> list[str]:
        """Add a naive singular form for plural-looking terms ("directors"
        -> also "director") — the fulltext index (standard-no-stop-words
        analyzer) does no stemming, so a question using a plural noun would
        otherwise get zero credit for a document that only uses the
        singular (e.g. a signature page listing "Director" as a title next
        to each name). Conservative: only strips a bare trailing 's' on
        words long enough that it's unlikely to mangle a non-plural word,
        and never touches the original term list's order/membership.
        """
        expanded = list(terms)
        seen = set(terms)
        for term in terms:
            if len(term) > 4 and term.endswith("s") and not term.endswith("ss"):
                singular = term[:-1]
                if singular not in seen:
                    expanded.append(singular)
                    seen.add(singular)
        return expanded

    def fulltext_query(self, question: str) -> str:
        """Build a Lucene query from question terms (document-agnostic)."""
        phrases = self._ranking._search_phrases_from_query(question)
        if phrases:
            quoted = [f'"{p}"' for p in phrases[:5] if " " in p]
            terms = self._with_singular_variants(self._ranking._query_keywords(question)[:6])
            parts = quoted + terms
            if parts:
                return " OR ".join(parts)
        keywords = self._ranking._query_keywords(question)
        extra_stop = {"employees", "employee", "company", "corporate", "policy"}
        keywords = self._with_singular_variants([k for k in keywords if k not in extra_stop][:14])
        if not keywords:
            return (question or "")[:120]
        return " OR ".join(keywords)
