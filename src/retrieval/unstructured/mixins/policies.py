"""Document RAG retriever — policies."""
from __future__ import annotations

from typing import Any, Optional

from ....auth.roles import UserContext


class PoliciesMixin:
    def _access_denied_response(self, query: str, ctx: UserContext) -> Optional[dict[str, Any]]:
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
    ) -> dict[str, Any]:
        ctx = user_context or self.user_context
        _passthrough = ("pdf_page", "document_page", "region_kind", "visual_content")
        return {
            "query": query,
            "chunks": [
                {
                    "id": r.get("id", ""),
                    "title": r.get("title", ""),
                    "text": r.get("text", ""),
                    "score": round(float(r.get("score", 0.0)), 3),
                    "related": r.get("related", []),
                    **{k: r[k] for k in _passthrough if r.get(k) is not None},
                }
                for r in items
            ],
            "total_available": len(items),
            "_access_level": ctx.role.value,
            "_user_id": ctx.user_id,
        }
