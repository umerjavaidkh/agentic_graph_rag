"""chapter_summary.py — fetch chapter-level rollup summaries for a document.

This ran the `EXISTS {{ MATCH (d)-[:CONTAINS*0..6]->(n) }}` membership test
that lexical.py was migrated off, and it is reached only by overview-shaped
questions -- which is why exactly one question in a nine-question set timed
out while the other eight answered in seconds. Knowing the document did not
help: the cost is O(corpus), not O(document), because the traversal is
re-explored per candidate node regardless of scope. Measured before the fix:
still running after 120s for a single, already-resolved document.

Query-side counterpart to src/semantic/chapter_summary.py (which writes
Chapter.summary at ingestion time). Only meaningful for synthesis-shaped
questions ("what does this document/chapter discuss") — callers should
gate on is_synthesis_question() before calling this, same as the ingestion
side only runs when OPENAI_API_KEY is set.
"""
from __future__ import annotations

from ....shared.config.settings import DEFAULT_LANGUAGE
from ....shared.neo4j.tenancy import language_filter, tenant_filter
from ..cypher_scope import content_scope_where


class ChapterSummaryService:
    def fetch_for_document(
        self, session, document_id: str, tenant_id: str = "", language: str = DEFAULT_LANGUAGE
    ) -> list[dict]:
        if not document_id:
            return []
        rows = session.run(
            f"""
            MATCH (n:Chapter)
            WHERE {content_scope_where("n")}
              AND {tenant_filter("n")} AND {language_filter("n")}
              AND coalesce(n.summary, '') <> ''
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
            language=language,
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
