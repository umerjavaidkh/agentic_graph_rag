"""
retrieval/unstructured/retriever.py — Unstructured document retriever (simple).

Flow:
- RBAC gate (KnowledgeArea: esg)
- embed(query)
- Neo4j vector search on `Section.embedding` using `section_embedding` index
- return top-k chunks
"""

from typing import Any, Dict, Optional

from neo4j import GraphDatabase

from ...auth.rbac_setup import GraphRBAC
from ...auth.roles import DEFAULT_PUBLIC_CONTEXT, UserContext
from ...config.settings import (
    EMBEDDING_MODEL,
    MODEL_PROVIDER,
    NEO4J_PASSWORD,
    NEO4J_URI,
    NEO4J_USER,
    OPENAI_API_KEY,
    RETRIEVAL_FINAL_LIMIT,
)
from ...model_providers.factory import get_model_provider

provider = get_model_provider(MODEL_PROVIDER, OPENAI_API_KEY)


class DocumentRAGRetriever:
    def __init__(
        self,
        uri: str = NEO4J_URI,
        user: str = NEO4J_USER,
        password: str = NEO4J_PASSWORD,
        user_context: Optional[UserContext] = None,
    ):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.user_context = user_context or DEFAULT_PUBLIC_CONTEXT
        self.rbac = GraphRBAC(uri, user, password)

    def close(self) -> None:
        self.driver.close()

    def semantic_retrieve(
        self,
        query: str,
        limit: int = RETRIEVAL_FINAL_LIMIT,
        user_context: Optional[UserContext] = None,
    ) -> Dict[str, Any]:
        return self.hybrid_retrieve(query, limit=limit, user_context=user_context)

    def hybrid_retrieve(
        self,
        query: str,
        limit: int = RETRIEVAL_FINAL_LIMIT,
        user_context: Optional[UserContext] = None,
    ) -> Dict[str, Any]:
        ctx = user_context or self.user_context
        denied = self._access_denied_response(query, ctx)
        if denied:
            return denied

        embedding = self._get_embedding(query)
        with self.driver.session() as session:
            rows = session.run(
                """
                CALL db.index.vector.queryNodes('section_embedding', $limit, $embedding)
                YIELD node AS n, score
                RETURN
                  coalesce(n.id, '') AS id,
                  coalesce(n.title, '') AS title,
                  coalesce(n.text, '') AS text,
                  score
                ORDER BY score DESC
                """,
                limit=max(1, int(limit)),
                embedding=embedding,
            )
            items = []
            for r in rows:
                items.append(
                    {
                        "id": r["id"],
                        "title": r["title"] or r["id"],
                        "text": r["text"],
                        "score": float(r["score"] or 0.0),
                        "related": [],
                    }
                )
        return self._format_response(query, items, user_context=ctx)

    # Minimal helper still used by follow-up machinery
    def _resolve_document_id(self, session, name: str) -> Optional[str]:
        if not name:
            return None
        row = session.run(
            """
            MATCH (d:Document)
            WHERE d.title IS NOT NULL AND toLower(d.title) CONTAINS toLower($name)
            RETURN d.id AS id
            LIMIT 1
            """,
            name=name.strip(),
        ).single()
        return str(row["id"]) if row and row.get("id") else None

    def _get_embedding(self, text: str) -> list[float]:
        resp = provider.embeddings(model=EMBEDDING_MODEL, input=(text or "")[:8000])
        return list(resp.data[0].embedding)

    def _access_denied_response(self, query: str, ctx: UserContext) -> Optional[Dict[str, Any]]:
        if self.rbac.can_query_knowledge_area(ctx.user_id, "esg"):
            return None
        return {
            "query": query,
            "chunks": [
                {
                    "id": "access_denied",
                    "title": "Access Denied",
                    "text": f"User {ctx.user_id} does not have permission to query Agentic Graph RAG data.",
                    "score": 0.0,
                    "related": [],
                }
            ],
            "total_available": 0,
            "_access_level": ctx.role.value,
            "_user_id": ctx.user_id,
        }

    def _format_response(
        self,
        query: str,
        items: list[dict],
        user_context: Optional[UserContext] = None,
    ) -> Dict[str, Any]:
        ctx = user_context or self.user_context
        return {
            "query": query,
            "chunks": [
                {
                    "id": r.get("id", ""),
                    "title": r.get("title", ""),
                    "text": r.get("text", ""),
                    "score": round(float(r.get("score", 0.0)), 3),
                    "related": r.get("related", []),
                }
                for r in items
            ],
            "total_available": len(items),
            "_access_level": ctx.role.value,
            "_user_id": ctx.user_id,
        }


ESGComplianceRetriever = DocumentRAGRetriever
RAGDataRetriever = DocumentRAGRetriever

