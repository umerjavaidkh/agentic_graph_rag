"""Document RAG retriever — toc strategy."""
from __future__ import annotations

from typing import Optional

from ....graph.versioning import lifecycle_active
from ..cypher_scope import _node_scope_cypher
from ..toc_retrieval import (
    format_outline_chunk,
    format_toc_chunk,
    include_in_outline_fallback,
    score_page_text_as_toc,
    section_title_is_toc,
)


class TocStrategyMixin:
    def _structural_toc_retrieve(self, session, query: str) -> list[dict]:
        """
        1) TOC page text (printed/PDF page if named in query, else best-scoring early page).
        2) Section titled Table of Contents / Contents.
        3) Outline from chapter + major section headings (not boxes/regions).
        """
        # Prefer strict resolution when the user named a specific document, so a
        # generic term (e.g. "all") can't rank a bigger unrelated doc above it.
        doc_id, doc_title = self._resolve_document_for_query_strict(session, query)
        if doc_id is None:
            doc_id, doc_title = self._resolve_document_for_query(session, query)
        label = doc_title or doc_id or "ingested document"

        pdf_page, doc_page = self._parse_page_targets(query)
        if pdf_page is not None or doc_page:
            page_hit = self._toc_fetch_page(
                session, doc_id, pdf_page=pdf_page, doc_page=doc_page
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

        page_hit = self._toc_find_best_page(session, doc_id)
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

        section_hit = self._toc_find_section(session, doc_id)
        if section_hit and (section_hit.get("text") or "").strip():
            return [
                format_toc_chunk(
                    body=(section_hit["text"] or "").strip(),
                    doc_title=section_hit.get("doc_title") or label,
                    source="Table of contents (from Contents section):",
                )
            ]

        outline = self._toc_outline_fallback(session, doc_id)
        if outline:
            return [format_outline_chunk(outline, doc_title=label)]
        return []

    def _toc_fetch_page(
        self,
        session,
        doc_id: Optional[str],
        *,
        pdf_page: Optional[int],
        doc_page: Optional[str],
    ) -> Optional[dict]:
        row = session.run(
            f"""
            MATCH (p:Page)
            WHERE {_node_scope_cypher("p")}
              AND {lifecycle_active("p")}
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
        ).single()
        return dict(row) if row else None

    def _toc_find_best_page(
        self, session, doc_id: Optional[str]
    ) -> Optional[dict]:
        rows = session.run(
            f"""
            MATCH (p:Page)
            WHERE {_node_scope_cypher("p")}
              AND {lifecycle_active("p")}
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
        self, session, doc_id: Optional[str]
    ) -> Optional[dict]:
        rows = session.run(
            f"""
            MATCH (s:Section)
            WHERE {_node_scope_cypher("s")}
              AND {lifecycle_active("s")}
              AND trim(coalesce(s.title, '')) <> ''
            RETURN
              trim(s.title) AS title,
              coalesce(s.text, '') AS text,
              coalesce(s.order, 0) AS ord
            ORDER BY ord
            """,
            doc_id=doc_id,
        )
        for r in rows:
            if section_title_is_toc(r.get("title") or ""):
                body = (r.get("text") or "").strip()
                if len(body) >= 30:
                    return dict(r)
        return None

    def _toc_outline_fallback(
        self, session, doc_id: Optional[str]
    ) -> list[str]:
        rows = session.run(
            f"""
            MATCH (n)
            WHERE (n:Chapter OR n:Section)
              AND {_node_scope_cypher("n")}
              AND {lifecycle_active("n")}
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

