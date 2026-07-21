"""formatter.py — response formatting + RBAC access gate, shared by retrieval strategies.

Extracted from mixins/policies.py (PoliciesMixin). The access-denied check is
kept as a plain function (not a method on the formatter) — it's called once
by the orchestrator before any strategy runs, not pushed into every strategy,
matching today's behavior where `_access_denied_response` gates
`hybrid_retrieve()` as a whole rather than being re-checked per branch.
"""
from __future__ import annotations

from typing import Any, Optional

from ....auth.rbac_setup import GraphRBAC
from ....auth.roles import UserContext
from ....config.settings import DOCUMENT_KNOWLEDGE_AREA_ID


def access_denied_response(rbac: GraphRBAC, query: str, ctx: UserContext) -> Optional[dict[str, Any]]:
    if rbac.can_query_knowledge_area(ctx.user_id, DOCUMENT_KNOWLEDGE_AREA_ID):
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


class ResponseFormatter:
    _PASSTHROUGH = ("pdf_page", "document_page", "region_kind", "visual_content")

    def format(
        self,
        query: str,
        items: list[dict],
        *,
        ctx: UserContext,
    ) -> dict[str, Any]:
        return {
            "query": query,
            "chunks": [
                {
                    "id": r.get("id", ""),
                    "title": r.get("title", ""),
                    "text": r.get("text", ""),
                    "score": round(float(r.get("score", 0.0)), 3),
                    "related": r.get("related", []),
                    **{k: r[k] for k in self._PASSTHROUGH if r.get(k) is not None},
                }
                for r in items
            ],
            "total_available": len(items),
            "_access_level": ctx.role.value,
            "_user_id": ctx.user_id,
        }
