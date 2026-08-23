"""financial_summary.py — fetch a document's firmwide financial-summary sections.

Query-side companion to chapter_summary.py. For a firmwide financial-metric
question (net earnings, revenue, EPS, ROE, book value, …) the authoritative
figure lives in a summary section — Executive/Financial Overview, Financial
Highlights, Selected Financial Data, or the Consolidated Statements. Those
long narrative/table sections lose vector-cosine ranking to short segment
tables that repeat the metric name as a row label, so the plain question
drifts to a *segment's* figure (e.g. Global Banking & Markets net earnings)
instead of the firm total.

This service fetches those sections by title so full_hybrid.py can pin them
into the answer context. Callers must gate on
is_firmwide_financial_metric_question() before calling — for every other
question shape these sections are noise.
"""
from __future__ import annotations

from ....shared.neo4j.tenancy import tenant_filter
from ..cypher_scope import content_scope_where

# Case-insensitive title fragments that mark an authoritative firmwide summary
# section in a 10-K/annual report. Segment/geographic sections never carry
# these exact headings, so matching on title alone keeps the fetch precise.
_SUMMARY_TITLE_FRAGMENTS = [
    "executive overview",
    "financial overview",
    "financial highlights",
    "selected financial data",
    "consolidated statements of earnings",
    "consolidated statements of operations",
    "consolidated statements of income",
]


class FinancialSummaryService:
    def fetch_for_document(
        self, session, document_id: str, tenant_id: str = ""
    ) -> list[dict]:
        if not document_id:
            return []
        title_pred = " OR ".join(
            f"toLower(coalesce(n.title,'')) CONTAINS '{frag}'"
            for frag in _SUMMARY_TITLE_FRAGMENTS
        )
        rows = session.run(
            f"""
            MATCH (n:Section|Chapter)
            WHERE {content_scope_where("n")}
              AND {tenant_filter("n")}
              AND coalesce(n.text, '') <> ''
              // A genuine Executive Overview/Financial Highlights section
              // runs to thousands of characters; a Table of Contents entry
              // that merely lists Executive Overview as a cross-reference
              // (page number, title, page number, next title, all on
              // separate lines) is under 100 chars. Without this floor, the
              // TOC fragment title-matches too and, being short, wins
              // _pin_firmwide_summary_chunks' prefer-compact tiebreak over
              // the real section, so the answer sees page numbers instead
              // of the actual overview text.
              AND size(n.text) > 200
              AND ({title_pred})
            RETURN
              coalesce(n.id, '') AS id,
              coalesce(n.title, '') AS title,
              n.text AS text,
              n.page_start AS page_start,
              n.document_page AS document_page,
              coalesce(n.order, 0) AS order
            ORDER BY order ASC
            LIMIT 5
            """,
            doc_id=document_id,
            tenant_id=tenant_id,
        )
        items: list[dict] = []
        for r in rows:
            if not r.get("id") or not r.get("text"):
                continue
            title = r.get("title") or r["id"]
            items.append({
                "id": r["id"],
                "title": title,
                "text": r["text"],
                "page_start": r.get("page_start"),
                "score": 1.0,
                "related": ["via:financial_summary"],
            })
        return items

    def fetch_quarterly_for_document(
        self, session, document_id: str, tenant_id: str = ""
    ) -> list[dict]:
        """Fetch the "Selected Quarterly Financial Data (Unaudited)" table.

        Unlike the firmwide-summary sections above, this table's *wrapping*
        section title is a generic "Supplementary information" — the
        identifying phrase lives in its TEXT, not its title — so this
        matches on content instead. Same size floor as fetch_for_document
        and for the same reason: the filing's own Table of Contents entry
        for this section is a short cross-reference fragment that contains
        the identical phrase and would otherwise title/text-match too.
        """
        if not document_id:
            return []
        rows = session.run(
            f"""
            MATCH (n:Section|Chapter)
            WHERE {content_scope_where("n")}
              AND {tenant_filter("n")}
              AND coalesce(n.text, '') <> ''
              AND size(n.text) > 200
              AND toLower(n.text) CONTAINS 'quarterly financial data'
            RETURN
              coalesce(n.id, '') AS id,
              coalesce(n.title, '') AS title,
              n.text AS text,
              n.page_start AS page_start,
              n.document_page AS document_page,
              coalesce(n.order, 0) AS order
            ORDER BY order ASC
            LIMIT 5
            """,
            doc_id=document_id,
            tenant_id=tenant_id,
        )
        items: list[dict] = []
        for r in rows:
            if not r.get("id") or not r.get("text"):
                continue
            title = r.get("title") or r["id"]
            items.append({
                "id": r["id"],
                "title": title,
                "text": r["text"],
                "page_start": r.get("page_start"),
                "score": 1.0,
                "related": ["via:quarterly_financial_data"],
            })
        return items
