"""page.py — PDF/printed page text and page-visual (figure/image) retrieval strategy.

Extracted from mixins/page_strategy.py (PageStrategyMixin). Handles both
page-visual and plain page-text retrieval — the two were already one
mixin/one concern, kept as one strategy here too. Preserves the original
independent-check behavior exactly: `is_visual_page_question` and
`is_page_question` are checked unconditionally in sequence (not if/elif),
matching mixins/hybrid.py's original two separate if-blocks — a query that
matches the visual check but returns no hits still falls through to the
plain-page check, it does not short-circuit like Box's list/content split.
"""
from __future__ import annotations

import re
from typing import Any, Optional

from ....auth.roles import UserContext
from ....document.page_numbers import parse_page_number_from_query
from ....document.page_vision import compact_visual_content
from ....graph.constants import DOCUMENT_ROOT_CYPHER
from ....graph.tenancy import tenant_filter
from ....storage.hydrator import get_hydrator
from ..cypher_scope import _doc_scope_cypher
from ..query_intent import FIG_CAPTION_RE as _FIG_CAPTION_RE
from ..query_intent import is_page_question, is_visual_page_question
from ..services.document_resolver import DocumentResolver
from ..services.formatter import ResponseFormatter
from ..visual_retrieval import parse_visual_intent


def parse_page_targets(query: str) -> tuple[Optional[int], Optional[str]]:
    """Resolve PDF page index vs printed document page label from the question.

    Module-level (not a PageStrategy method) since it's pure and also needed
    by TocStrategy (a TOC lookup can be scoped to a specific requested page).
    """
    pdf_page, doc_page = parse_page_number_from_query(query)
    # Bare "page 29" → match document_page footer label, not PDF file page 29.
    if doc_page and str(doc_page).isdigit() and pdf_page is None:
        if re.search(r"\bpdf\b", (query or "").lower()) and re.search(
            r"\b(?:pdf\s+page|page\s+\d+\s+(?:of|in|from)\s+(?:the\s+)?pdf)\b",
            query or "",
            re.I,
        ):
            pdf_page = int(doc_page)
            doc_page = None
    return pdf_page, doc_page


class PageStrategy:
    name = "structural_page"

    def __init__(self, document_resolver: DocumentResolver, formatter: ResponseFormatter):
        self._document_resolver = document_resolver
        self._formatter = formatter

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
        if is_visual_page_question(query):
            visual_items = self._structural_page_visual_retrieve(
                session, query, tenant_id, document_id_hint
            )
            if visual_items:
                pdf_page, doc_page = parse_page_targets(query)
                response = self._formatter.format(query, visual_items, ctx=ctx)
                if self._query_wants_all_page_visuals(query) and any(
                    (c.get("visual_content") or "").strip() for c in visual_items
                ):
                    response["mode"] = "page_visual_list"
                else:
                    response["mode"] = "structural_page_visual"
                response["pdf_page"] = pdf_page
                response["document_page"] = doc_page
                response["strategy"] = "graph_rag"
                response["vector_seeds"] = 0
                response["fulltext_hits"] = 0
                response["graph_expanded"] = len(visual_items)
                doc_id, doc_title = self._document_resolver.resolve_document_for_query(
                    session, query, tenant_id, document_id_hint=document_id_hint
                )
                response["document_id"] = doc_id
                response["document_title"] = doc_title
                return response

        if is_page_question(query):
            page_items = self._structural_page_retrieve(session, query, tenant_id, document_id_hint)
            if page_items:
                response = self._formatter.format(query, page_items, ctx=ctx)
                response["mode"] = "structural_page"
                response["strategy"] = "graph_rag"
                response["vector_seeds"] = 0
                response["fulltext_hits"] = 0
                response["graph_expanded"] = len(page_items)
                doc_id, doc_title = self._document_resolver.resolve_document_for_query(
                    session, query, tenant_id, document_id_hint=document_id_hint
                )
                response["document_id"] = doc_id
                response["document_title"] = doc_title
                return response

        return None

    @staticmethod
    def _extract_figure_captions(page_text: str) -> dict[str, str]:
        """Map document figure number → caption line from page OCR text."""
        caps: dict[str, str] = {}
        for m in _FIG_CAPTION_RE.finditer(page_text or ""):
            num, title = m.group(1), m.group(2).strip()
            caps[num] = title
            caps[num.lstrip("0") or num] = title
        return caps

    @staticmethod
    def _figure_number_from_query(query: str) -> Optional[str]:
        for pat in (
            r"\b(?:fig\.?|figure)\s*(\d+(?:\.\d+)?)\b",
            r"\b(?:image|diagram)\s+(?:of|for|showing)?\s*(?:fig\.?|figure)?\s*(\d+)\b",
        ):
            m = re.search(pat, query, re.I)
            if m:
                return m.group(1)
        return None

    def _query_wants_all_page_visuals(self, query: str) -> bool:
        return parse_visual_intent(query).list_all

    def _resolve_pdf_page_by_figure_number(
        self, session: Any, doc_id: Optional[str], tenant_id: str, fig_num: str
    ) -> Optional[int]:
        """Find the PDF page whose OCR'd text names "Figure N" (no page number given).

        _structural_page_visual_retrieve otherwise requires a page number up
        front; this lets a bare "what does Figure 1 show" question reach the
        visual-content pathway instead of falling through to plain-text
        search, which finds nothing since the figure's actual description
        lives in visual_content, not in page.text.
        """
        if not doc_id:
            return None
        rows = session.run(
            f"""
            MATCH (d:{DOCUMENT_ROOT_CYPHER})
            WHERE {_doc_scope_cypher("d")}
              AND {tenant_filter("d")}
            MATCH (p:Page)
            WHERE p.id STARTS WITH d.id + '_page_'
              AND {tenant_filter("p")}
              AND toLower(coalesce(p.search_text, '')) CONTAINS 'fig'
            RETURN p.pdf_page AS pdf_page, p.search_text AS page_text
            ORDER BY p.order
            """,
            doc_id=doc_id,
            tenant_id=tenant_id,
        )
        pat = re.compile(rf"\b(?:fig\.?|figure)\s*0*{re.escape(fig_num)}\b", re.I)
        for r in rows:
            if pat.search(r.get("page_text") or ""):
                return r.get("pdf_page")
        return None

    def _structural_page_visual_retrieve(
        self, session: Any, query: str, tenant_id: str = "", document_id_hint: str = ""
    ) -> list[dict]:
        """Page figures/diagrams via stored visual_content text (ingest vision enrichment)."""
        pdf_page, doc_page = parse_page_targets(query)
        doc_id, doc_title = self._document_resolver.resolve_document_for_query(
            session, query, tenant_id, document_id_hint=document_id_hint
        )
        if pdf_page is None and not doc_page:
            want_fig = self._figure_number_from_query(query)
            if want_fig:
                pdf_page = self._resolve_pdf_page_by_figure_number(session, doc_id, tenant_id, want_fig)
            if pdf_page is None:
                return []

        list_all_visuals = self._query_wants_all_page_visuals(query)
        row = session.run(
            f"""
            MATCH (d:{DOCUMENT_ROOT_CYPHER})
            WHERE {_doc_scope_cypher("d")}
              AND {tenant_filter("d")}
            MATCH (p:Page)
            WHERE p.id STARTS WITH d.id + '_page_'
              AND (
                ($pdf_page IS NOT NULL AND p.pdf_page = $pdf_page)
                OR (
                  $doc_page IS NOT NULL
                  AND toLower(coalesce(p.document_page, '')) = toLower($doc_page)
                )
              )
              AND {tenant_filter("p")}
            WITH p, d
            ORDER BY
              CASE WHEN $doc_page IS NOT NULL
                AND toLower(coalesce(p.document_page, '')) = toLower($doc_page) THEN 0
              WHEN $pdf_page IS NOT NULL AND p.pdf_page = $pdf_page THEN 1
              ELSE 2 END,
              p.order
            LIMIT 1
            OPTIONAL MATCH (p)-[:CONTAINS]->(r:Region)
            RETURN
              p.id AS page_id,
              coalesce(p.title, '') AS page_title,
              coalesce(p.search_text, '') AS page_text,
              coalesce(p.visual_content, '') AS page_visual,
              p.pdf_page AS pdf_page,
              p.document_page AS document_page,
              coalesce(d.title, d.id) AS doc_title,
              collect(DISTINCT {{
                id: r.id,
                title: coalesce(r.title, ''),
                text: coalesce(r.search_text, ''),
                kind: coalesce(r.region_kind, ''),
                visual_content: coalesce(r.visual_content, ''),
                order: coalesce(r.order, 0)
              }}) AS regions
            """,
            doc_id=doc_id,
            pdf_page=pdf_page,
            doc_page=doc_page,
            tenant_id=tenant_id,
        ).single()

        if not row or not row.get("page_id"):
            return []

        page_text = (row.get("page_text") or "").strip()
        captions = self._extract_figure_captions(page_text)
        want_fig = self._figure_number_from_query(query)
        regions = [
            r for r in (row.get("regions") or [])
            if r and (r.get("visual_content") or "").strip()
        ]
        regions.sort(key=lambda r: r.get("order") or 0)

        if not regions and (row.get("page_visual") or "").strip():
            regions = [{
                "id": row["page_id"],
                "title": row.get("page_title") or f"Page {pdf_page}",
                "text": "",
                "kind": "page",
                "visual_content": row.get("page_visual") or "",
                "order": 0,
            }]

        chunks: list[dict] = []
        doc_label = row.get("doc_title") or doc_title or doc_id or "ingested document"

        if not regions:
            return []

        for idx, reg in enumerate(regions):
            stored_visual = compact_visual_content((reg.get("visual_content") or "").strip())
            page_visual = compact_visual_content((row.get("page_visual") or "").strip())

            cap_num = want_fig
            if not cap_num and len(captions) == 1:
                cap_num = next(iter(captions))
            elif not cap_num and captions:
                cap_num = sorted(captions.keys(), key=lambda x: float(x) if x.replace(".", "").isdigit() else x)[-1]

            caption_title = ""
            if cap_num and cap_num in captions:
                caption_title = f"Fig. {cap_num}: {captions[cap_num]}"
            elif captions:
                first_num = sorted(captions.keys(), key=lambda x: float(x) if x.replace(".", "").isdigit() else x)[0]
                caption_title = f"Fig. {first_num}: {captions[first_num]}"

            reg_title = (reg.get("title") or "").strip()
            display_title = caption_title or reg_title or f"Figure on PDF page {pdf_page}"

            vision_desc = stored_visual or page_visual

            parts = [
                f"Document: {doc_label}",
                f"PDF page {row.get('pdf_page') or pdf_page}",
                "",
                f"## {display_title}",
            ]
            if vision_desc:
                parts.extend(["", "[Visual description]", vision_desc])
            elif page_text:
                parts.extend([
                    "",
                    "[Caption from page text only — vision description unavailable]",
                    display_title,
                ])

            snippet = page_text[:1200] if page_text else ""
            if snippet:
                parts.extend(["", "## Surrounding page text", snippet])

            body = "\n".join(parts).strip()
            if not body:
                continue

            chunk: dict[str, Any] = {
                "id": reg.get("id") or row["page_id"],
                "title": display_title,
                "text": body,
                "score": 1.0 - idx * 0.01,
                "related": ["via:structural_page_visual"],
                "pdf_page": row.get("pdf_page") or pdf_page,
                "document_page": row.get("document_page"),
                "region_kind": reg.get("kind") or "figure",
                "visual_content": vision_desc,
            }
            chunks.append(chunk)

        if not list_all_visuals and want_fig and chunks:
            return chunks[:1]
        return chunks

    def _structural_page_retrieve(
        self, session: Any, query: str, tenant_id: str = "", document_id_hint: str = ""
    ) -> list[dict]:
        """
        Fetch Page node content by pdf_page / document_page for a resolved document.
        Works for any ingested PDF with Page nodes in the graph.
        """
        pdf_page, doc_page = parse_page_targets(query)
        if pdf_page is None and not doc_page:
            return []

        doc_id, doc_title = self._document_resolver.resolve_document_for_query(
            session, query, tenant_id, document_id_hint=document_id_hint
        )
        row = session.run(
            f"""
            MATCH (d:{DOCUMENT_ROOT_CYPHER})
            WHERE {_doc_scope_cypher("d")}
              AND {tenant_filter("d")}
            MATCH (p:Page)
            WHERE p.id STARTS WITH d.id + '_page_'
              AND (
                ($pdf_page IS NOT NULL AND p.pdf_page = $pdf_page)
                OR (
                  $doc_page IS NOT NULL
                  AND toLower(coalesce(p.document_page, '')) = toLower($doc_page)
                )
              )
              AND {tenant_filter("p")}
            WITH p, d
            ORDER BY
              CASE WHEN $pdf_page IS NOT NULL AND p.pdf_page = $pdf_page THEN 0 ELSE 1 END,
              p.order
            LIMIT 1
            OPTIONAL MATCH (p)-[:CONTAINS]->(r:Region)
            OPTIONAL MATCH (s:Section)-[:CONTAINS]->(p)
            RETURN
              p.id AS id,
              coalesce(p.title, '') AS title,
              coalesce(p.search_text, '') AS text,
              coalesce(p.visual_content, '') AS visual_content,
              p.pdf_page AS pdf_page,
              p.document_page AS document_page,
              coalesce(d.title, d.id) AS doc_title,
              collect(DISTINCT {{
                title: coalesce(r.title, ''),
                text: coalesce(r.search_text, ''),
                kind: coalesce(r.region_kind, '')
              }}) AS regions,
              collect(DISTINCT {{
                title: coalesce(s.title, ''),
                blob_key_text: s.blob_key_text,
                search_text: coalesce(s.search_text, '')
              }}) AS sections
            """,
            doc_id=doc_id,
            pdf_page=pdf_page,
            doc_page=doc_page,
            tenant_id=tenant_id,
        ).single()

        if not row or not row.get("id"):
            return []

        parts: list[str] = [
            f"Document: {row.get('doc_title') or doc_title or doc_id or 'ingested document'}",
            f"Page: {row.get('title') or 'Page'} (PDF page {row.get('pdf_page')}, "
            f"printed label {row.get('document_page') or 'n/a'})",
            "",
            "## Page text",
            (row.get("text") or "").strip(),
        ]
        visual = (row.get("visual_content") or "").strip()
        if visual:
            parts.extend(["", "## Visual content (tables/figures)", visual])

        hydrator = get_hydrator()
        for sec in row.get("sections") or []:
            if not sec or not sec.get("title"):
                continue
            body = hydrator.hydrate(sec.get("blob_key_text"), sec.get("search_text") or "").strip()
            if body:
                parts.extend(["", f"## Related section: {sec['title']}", body[:2500]])

        for reg in row.get("regions") or []:
            if not reg or not reg.get("text"):
                continue
            label = reg.get("title") or reg.get("kind") or "Region"
            parts.extend(["", f"## Region: {label}", (reg.get("text") or "")[:1500]])

        page_text = "\n".join(parts).strip()
        if not page_text:
            return []

        return [{
            "id": row["id"],
            "title": row.get("title") or f"Page {pdf_page or doc_page}",
            "text": page_text,
            "score": 1.0,
            "related": ["via:structural_page"],
        }]
