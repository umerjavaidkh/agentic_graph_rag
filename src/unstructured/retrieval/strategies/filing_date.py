"""filing_date.py — answers "when was this filed" from ingestion metadata,
not document text.

The actual SEC filing/submission date is usually not printed anywhere in a
10-K/10-Q's own PDF body — EDGAR's "Filed:" stamp lives in the filing's
HTML/index wrapper, not the document itself. The document typically only
states the PERIOD it covers ("for the quarterly period ended March 29,
2026"), which is a different date. Retrieval/synthesis has no way to find a
fact that genuinely isn't in the text: the general hybrid strategy would
pull whichever date-heavy MD&A chunk ranked highest, and the LLM would pick
the most prominent date in it — almost always the period-end date, not the
filing date. See query_intent.is_filing_date_question's docstring.

This strategy sidesteps the problem entirely by answering from
DocRevision.source_filename instead of guessing from prose: every document
in the SEC-EDGAR sample corpus is named TICKER_FORM_YYYY-MM-DD.ext, where
the date is EDGAR's own filingDate field (see
scripts/fetch_sec_edgar_corpus.py) — a real, authoritative fact, just one
that lives in ingestion metadata rather than the parsed text.

Returns None (falls through to the next strategy) when the filename doesn't
carry an extractable date — e.g. a non-SEC document — rather than claiming
an answer that isn't grounded in anything.
"""
from __future__ import annotations

from typing import Any, Optional

from ....shared.auth.roles import UserContext
from ...document.versioning import extract_filing_date_from_filename
from ...graph.constants import DOC_REVISION_LABEL, DOCUMENT_LOGICAL_LABEL
from ....shared.neo4j.tenancy import tenant_filter
from ..services.document_resolver import DocumentResolver
from ..services.formatter import ResponseFormatter


class FilingDateStrategy:
    name = "structural_filing_date"

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
        doc_id, doc_title = self._document_resolver.resolve_document_for_query(
            session, query, tenant_id, document_id_hint=document_id_hint
        )
        if not doc_id:
            return None

        row = session.run(
            f"""
            MATCH (dl:{DOCUMENT_LOGICAL_LABEL} {{logical_id: $doc_id}})-[:ACTIVE_REVISION]->(rev:{DOC_REVISION_LABEL})
            WHERE {tenant_filter("rev")}
            RETURN rev.source_filename AS source_filename
            LIMIT 1
            """,
            doc_id=doc_id,
            tenant_id=tenant_id,
        ).single()
        source_filename = row.get("source_filename") if row else None
        filing_date = extract_filing_date_from_filename(source_filename or "")
        if not filing_date:
            return None

        label = doc_title or doc_id
        chunk = {
            "id": f"{doc_id}:filing_date",
            "title": "Filing date",
            "text": (
                f"{label} was filed on {filing_date}, per the source filing's "
                f"metadata (SEC EDGAR filing date). Note this is the actual "
                f"submission date, not the period the filing covers, which the "
                f"document text may state separately (e.g. \"for the quarterly "
                f"period ended ...\")."
            ),
            "score": 1.0,
            "related": ["via:filing_date_metadata"],
        }
        response = self._formatter.format(query, [chunk], ctx=ctx)
        response["mode"] = "structural_filing_date"
        response["strategy"] = "graph_rag"
        response["vector_seeds"] = 0
        response["fulltext_hits"] = 0
        response["graph_expanded"] = 0
        response["document_id"] = doc_id
        response["document_title"] = doc_title
        return response
