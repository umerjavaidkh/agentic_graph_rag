"""toc.py — table-of-contents retrieval strategy.

Extracted from mixins/toc_strategy.py (TocStrategyMixin) plus the doc-choice
clarification logic that previously lived inline in mixins/hybrid.py's
ladder (same split-across-files pattern as Subsection — consolidated here).
"""
from __future__ import annotations

from typing import Any, Optional

from ....auth.roles import UserContext
from ....graph.tenancy import tenant_filter
from ....graph.versioning import lifecycle_active
from ..cypher_scope import _node_scope_cypher
from ..executor import DocumentQueryExecutor
from ..query_intent import is_toc_question
from ..services.document_resolver import DocumentResolver
from ..services.formatter import ResponseFormatter
from .page import parse_page_targets
from ..toc_retrieval import (
    format_outline_chunk,
    format_toc_chunk,
    include_in_outline_fallback,
    score_page_text_as_toc,
    section_title_is_toc,
)


class TocStrategy:
    name = "structural_toc"

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
    ) -> Optional[dict[str, Any]]:
        if not is_toc_question(query):
            return None

        # If the user named a specific document but we cannot find it,
        # return a clarification rather than silently using the wrong doc.
        doc_terms = self._document_resolver.doc_name_terms(query)
        if doc_terms:
            doc_id, _ = self._document_resolver.resolve_document_for_query_strict(session, query, tenant_id)
            if doc_id is None:
                docs = self._document_resolver.list_documents(session, limit=8, tenant_id=tenant_id)
                if docs:
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

        toc_items = self._structural_toc_retrieve(session, query, tenant_id)
        if toc_items:
            response = self._formatter.format(query, toc_items, ctx=ctx)
            response["mode"] = "structural_toc"
            response["strategy"] = "graph_rag"
            response["vector_seeds"] = 0
            response["fulltext_hits"] = 0
            response["graph_expanded"] = len(toc_items)
            return response
        return None

    def _structural_toc_retrieve(self, session: Any, query: str, tenant_id: str = "") -> list[dict]:
        """
        1) TOC page text (printed/PDF page if named in query, else best-scoring early page).
        2) Section titled Table of Contents / Contents.
        3) Outline from chapter + major section headings (not boxes/regions).
        """
        # Prefer strict resolution when the user named a specific document, so a
        # generic term (e.g. "all") can't rank a bigger unrelated doc above it.
        doc_id, doc_title = self._document_resolver.resolve_document_for_query_strict(session, query, tenant_id)
        if doc_id is None:
            doc_id, doc_title = self._document_resolver.resolve_document_for_query(session, query, tenant_id)
        label = doc_title or doc_id or "ingested document"

        pdf_page, doc_page = parse_page_targets(query)
        if pdf_page is not None or doc_page:
            page_hit = self._toc_fetch_page(
                session, doc_id, pdf_page=pdf_page, doc_page=doc_page, tenant_id=tenant_id
            )
            if page_hit and (page_hit.get("text") or "").strip():
                return [
                    format_toc_chunk(
                        body=(page_hit["text"] or "").strip(),
                        doc_title=page_hit.get("doc_title") or label,
                        source="Table of contents (from requested document page):",
                        pdf_page=page_hit.get("pdf_page"),
                        document_page=page_hit.get("document_page"),
                    )
                ]

        page_hit = self._toc_find_best_page(session, doc_id, tenant_id)
        if page_hit:
            return [
                format_toc_chunk(
                    body=(page_hit["text"] or "").strip(),
                    doc_title=page_hit.get("doc_title") or label,
                    source="Table of contents (from document TOC page text):",
                    pdf_page=page_hit.get("pdf_page"),
                    document_page=page_hit.get("document_page"),
                )
            ]

        section_hit = self._toc_find_section(session, doc_id, tenant_id)
        if section_hit and (section_hit.get("text") or "").strip():
            return [
                format_toc_chunk(
                    body=(section_hit["text"] or "").strip(),
                    doc_title=section_hit.get("doc_title") or label,
                    source="Table of contents (from Contents section):",
                )
            ]

        outline = self._toc_outline_fallback(session, doc_id, tenant_id)
        if outline:
            return [format_outline_chunk(outline, doc_title=label)]
        return []

    def _toc_fetch_page(
        self,
        session: Any,
        doc_id: Optional[str],
        *,
        pdf_page: Optional[int],
        doc_page: Optional[str],
        tenant_id: str = "",
    ) -> Optional[dict]:
        row = session.run(
            f"""
            MATCH (p:Page)
            WHERE {_node_scope_cypher("p")}
              AND {lifecycle_active("p")}
              AND {tenant_filter("p")}
              AND trim(coalesce(p.text, '')) <> ''
              AND (
                ($pdf_page IS NOT NULL AND p.pdf_page = $pdf_page)
                OR (
                  $doc_page IS NOT NULL
                  AND toLower(coalesce(p.document_page, '')) = toLower($doc_page)
                )
              )
            RETURN
              coalesce(p.text, '') AS text,
              p.pdf_page AS pdf_page,
              p.document_page AS document_page
            ORDER BY p.order
            LIMIT 1
            """,
            doc_id=doc_id,
            pdf_page=pdf_page,
            doc_page=doc_page,
            tenant_id=tenant_id,
        ).single()
        return dict(row) if row else None

    def _toc_find_best_page(
        self, session: Any, doc_id: Optional[str], tenant_id: str = ""
    ) -> Optional[dict]:
        rows = session.run(
            f"""
            MATCH (p:Page)
            WHERE {_node_scope_cypher("p")}
              AND {lifecycle_active("p")}
              AND {tenant_filter("p")}
              AND trim(coalesce(p.text, '')) <> ''
            RETURN
              coalesce(p.text, '') AS text,
              p.pdf_page AS pdf_page,
              p.document_page AS document_page,
              coalesce(p.pdf_page, p.order, 9999) AS sort_key
            ORDER BY sort_key
            LIMIT 40
            """,
            doc_id=doc_id,
            tenant_id=tenant_id,
        )
        best: Optional[dict] = None
        best_score = 0.42
        for r in rows:
            text = (r.get("text") or "").strip()
            if not text:
                continue
            s = score_page_text_as_toc(text)
            if s > best_score:
                best_score = s
                best = dict(r)
        return best

    def _toc_find_section(
        self, session: Any, doc_id: Optional[str], tenant_id: str = ""
    ) -> Optional[dict]:
        rows = session.run(
            f"""
            MATCH (s:Section)
            WHERE {_node_scope_cypher("s")}
              AND {lifecycle_active("s")}
              AND {tenant_filter("s")}
              AND trim(coalesce(s.title, '')) <> ''
            RETURN
              trim(s.title) AS title,
              coalesce(s.text, '') AS text,
              coalesce(s.order, 0) AS ord
            ORDER BY ord
            """,
            doc_id=doc_id,
            tenant_id=tenant_id,
        )
        for r in rows:
            if section_title_is_toc(r.get("title") or ""):
                body = (r.get("text") or "").strip()
                if len(body) >= 30:
                    return dict(r)
        return None

    def _toc_outline_fallback(
        self, session: Any, doc_id: Optional[str], tenant_id: str = ""
    ) -> list[str]:
        rows = session.run(
            f"""
            MATCH (n)
            WHERE (n:Chapter OR n:Section)
              AND {_node_scope_cypher("n")}
              AND {lifecycle_active("n")}
              AND {tenant_filter("n")}
            WITH n,
                 trim(coalesce(n.title, '')) AS title,
                 coalesce(n.order, 0) AS ord,
                 coalesce(n.depth, 99) AS depth,
                 labels(n)[0] AS label
            WHERE title <> ''
            RETURN title, ord, depth, label
            ORDER BY ord, depth, title
            """,
            doc_id=doc_id,
            tenant_id=tenant_id,
        )
        seen: set[str] = set()
        entries: list[str] = []
        for r in rows:
            title = (r.get("title") or "").strip()
            if not include_in_outline_fallback(
                title, int(r.get("depth") or 99), str(r.get("label") or "")
            ):
                continue
            key = title.casefold()
            if key in seen:
                continue
            seen.add(key)
            entries.append(title)
        return entries
