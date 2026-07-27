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

from ....graph.constants import DOCUMENT_ROOT_CYPHER
from ....graph.tenancy import tenant_filter
from ..cypher_scope import _doc_scope_cypher

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
            MATCH (d:{DOCUMENT_ROOT_CYPHER})
            WHERE {_doc_scope_cypher("d")}
              AND {tenant_filter("d")}
            MATCH (n)
            WHERE (n:Section OR n:Chapter)
              AND coalesce(n.text, '') <> ''
              AND ({title_pred})
              AND (
                EXISTS {{ MATCH (d)-[:CONTAINS*0..6]->(n) }}
                OR n.id STARTS WITH d.id + '_'
              )
            RETURN
              coalesce(n.id, '') AS id,
              coalesce(n.title, '') AS title,
              n.text AS text,
              n.page_start AS page_start,
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
