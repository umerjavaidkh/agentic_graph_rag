"""chapter_summary.py — fetch chapter-level rollup summaries for a document.

Query-side counterpart to src/semantic/chapter_summary.py (which writes
Chapter.summary at ingestion time). Only meaningful for synthesis-shaped
questions ("what does this document/chapter discuss") — callers should
gate on is_synthesis_question() before calling this, same as the ingestion
side only runs when OPENAI_API_KEY is set.
"""
from __future__ import annotations

from ....graph.constants import DOCUMENT_ROOT_CYPHER
from ....graph.tenancy import tenant_filter
from ..cypher_scope import _doc_scope_cypher


class ChapterSummaryService:
    def fetch_for_document(
        self, session, document_id: str, tenant_id: str = ""
    ) -> list[dict]:
        if not document_id:
            return []
        rows = session.run(
            f"""
            MATCH (d:{DOCUMENT_ROOT_CYPHER})
            WHERE {_doc_scope_cypher("d")}
              AND {tenant_filter("d")}
            MATCH (n:Chapter)
            WHERE coalesce(n.summary, '') <> ''
              AND (
                EXISTS {{ MATCH (d)-[:CONTAINS*0..6]->(n) }}
                OR n.id STARTS WITH d.id + '_'
              )
            RETURN
              coalesce(n.id, '') AS id,
              coalesce(n.title, '') AS title,
              n.summary AS summary,
              coalesce(n.order, 0) AS order
            ORDER BY order ASC
            LIMIT 12
            """,
            doc_id=document_id,
            tenant_id=tenant_id,
        )
        items: list[dict] = []
        for r in rows:
            if not r.get("id") or not r.get("summary"):
                continue
            title = r.get("title") or r["id"]
            items.append({
                "id": r["id"],
                "title": title,
                "text": r["summary"],
                "score": 0.9,
                "related": ["via:chapter_summary"],
            })
        return items
