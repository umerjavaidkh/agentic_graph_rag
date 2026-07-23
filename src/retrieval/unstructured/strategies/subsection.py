"""subsection.py — numbered-section child/parent lookup strategy.

Extracted from mixins/subsection.py (SubsectionMixin) plus the doc-choice
clarification logic that previously lived inline in mixins/hybrid.py's
ladder (the two were split across files for the same "subsection request"
concern — consolidated here into one self-contained strategy).
"""
from __future__ import annotations

from typing import Any, Optional

from ....auth.roles import UserContext
from ....graph.constants import DOCUMENT_ROOT_CYPHER
from ....graph.tenancy import tenant_filter
from ..cypher_scope import _doc_scope_cypher
from ..executor import DocumentQueryExecutor
from ..services.document_resolver import DocumentResolver
from ..services.formatter import ResponseFormatter


class SubsectionStrategy:
    name = "subsection_tree"

    def __init__(
        self,
        document_resolver: DocumentResolver,
        formatter: ResponseFormatter,
        exec_: DocumentQueryExecutor,
    ):
        self._document_resolver = document_resolver
        self._formatter = formatter
        self._exec = exec_

    def retrieve(
        self,
        session: Any,
        query: str,
        *,
        tenant_id: str,
        limit: int,
        ctx: UserContext,
        document_id_hint: str = "",
    ) -> Optional[dict[str, Any]]:
        if not self._exec.is_subsection_request(query):
            return None
        sec_num = self._exec.parse_section_number(query)
        if not sec_num:
            return None

        # If the user didn't name a document and multiple documents exist,
        # ask them to pick rather than silently guessing the biggest one.
        if not self._document_resolver.document_match_terms(query):
            docs = self._document_resolver.list_documents(session, limit=5, tenant_id=tenant_id)
            if len(docs) > 1:
                clar = self._exec.build_doc_choice_clarification(
                    original_question=query,
                    documents=docs,
                )
                return {
                    "query": query,
                    "strategy": "graph_rag",
                    "mode": "needs_clarification",
                    "original_question": query,
                    "clarification_kind": clar.kind,
                    "clarification_options": clar.options,
                    "chunks": [{
                        "id": "clarification",
                        "title": "Clarification",
                        "text": clar.prompt,
                        "score": 1.0,
                        "related": [],
                    }],
                    "total_available": 1,
                }

        items, parent, doc_id, doc_title = self._structural_subsections(
            session, query, sec_num, tenant_id, document_id_hint
        )
        if items:
            response = self._formatter.format(query, items, ctx=ctx)
            response["mode"] = "subsection_tree"
            response["strategy"] = "graph_rag"
            response["parent_id"] = parent.get("id")
            response["parent_title"] = parent.get("title")
            response["vector_seeds"] = 0
            response["fulltext_hits"] = 0
            response["graph_expanded"] = len(items)
            response["document_id"] = doc_id
            response["document_title"] = doc_title
            return response

        if parent and parent.get("text"):
            response = self._formatter.format(query, [parent], ctx=ctx)
            response["mode"] = "section_detail"
            response["strategy"] = "graph_rag"
            response["parent_id"] = parent.get("id")
            response["parent_title"] = parent.get("title")
            response["vector_seeds"] = 0
            response["fulltext_hits"] = 0
            response["graph_expanded"] = 1
            response["document_id"] = doc_id
            response["document_title"] = doc_title
            return response

        return None

    def _structural_subsections(
        self,
        session: Any,
        query: str,
        sec_num: str,
        tenant_id: str = "",
        document_id_hint: str = "",
    ) -> tuple[list[dict], dict, Optional[str], Optional[str]]:
        """Return (children items, parent item, doc_id, doc_title) for a numbered section like 2.5."""
        doc_id, doc_title = self._document_resolver.resolve_document_for_query(
            session, query, tenant_id, document_id_hint=document_id_hint
        )
        row = session.run(
            f"""
            MATCH (d:{DOCUMENT_ROOT_CYPHER})
            WHERE {_doc_scope_cypher("d")}
              AND {tenant_filter("d")}
            MATCH (s:Section)
            WHERE (s.id STARTS WITH d.id + '_' OR EXISTS {{ MATCH (d)-[:CONTAINS*1..6]->(s) }})
              AND s.title IS NOT NULL
              AND trim(s.title) <> ''
              AND toLower(s.title) STARTS WITH toLower($sec_num)
              AND {tenant_filter("s")}
            WITH s
            OPTIONAL MATCH (s)-[:CONTAINS]->(c:Section)
            WHERE c.title IS NOT NULL AND trim(c.title) <> ''
              AND (c IS NULL OR {tenant_filter("c")})
            RETURN
              s.id AS sid,
              s.title AS stitle,
              coalesce(s.text,'') AS stext,
              collect({{id: c.id, title: c.title, text: coalesce(c.text,'')}}) AS children
            LIMIT 1
            """,
            doc_id=doc_id,
            sec_num=sec_num,
            tenant_id=tenant_id,
        ).single()
        if not row:
            return [], {}, doc_id, doc_title

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
        return items, parent, doc_id, doc_title
