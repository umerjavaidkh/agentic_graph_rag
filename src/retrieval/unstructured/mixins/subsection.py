"""Document RAG retriever — subsection."""
from __future__ import annotations

from ....graph.constants import DOCUMENT_ROOT_CYPHER
from ..cypher_scope import _doc_scope_cypher


class SubsectionMixin:
    def _structural_subsections(
        self,
        session,
        query: str,
        sec_num: str,
    ) -> tuple[list[dict], dict]:
        """Return (children items, parent item) for a numbered section like 2.5."""
        doc_id, _ = self._resolve_document_for_query(session, query)
        row = session.run(
            f"""
            MATCH (d:{DOCUMENT_ROOT_CYPHER})
            WHERE {_doc_scope_cypher("d")}
            MATCH (s:Section)
            WHERE (s.id STARTS WITH d.id + '_' OR EXISTS {{ MATCH (d)-[:CONTAINS*1..6]->(s) }})
              AND s.title IS NOT NULL
              AND trim(s.title) <> ''
              AND toLower(s.title) STARTS WITH toLower($sec_num)
            WITH s
            OPTIONAL MATCH (s)-[:CONTAINS]->(c:Section)
            WHERE c.title IS NOT NULL AND trim(c.title) <> ''
            RETURN
              s.id AS sid,
              s.title AS stitle,
              coalesce(s.text,'') AS stext,
              collect({{id: c.id, title: c.title, text: coalesce(c.text,'')}}) AS children
            LIMIT 1
            """,
            doc_id=doc_id,
            sec_num=sec_num,
        ).single()
        if not row:
            return [], {}

        parent = {
            "id": row.get("sid") or "",
            "title": (row.get("stitle") or "").strip(),
            "text": (row.get("stext") or "").strip(),
            "score": 1.0,
            "related": ["via:section_lookup"],
        }

        children = row.get("children") or []
        items: list[dict] = []
        for c in children:
            if not c or not c.get("id") or not c.get("title"):
                continue
            items.append({
                "id": c["id"],
                "title": c["title"],
                "text": (c.get("text") or "").strip(),
                "score": 1.0,
                "related": ["via:subsections"],
            })
        return items, parent

