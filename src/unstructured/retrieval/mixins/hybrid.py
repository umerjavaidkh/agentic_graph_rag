"""Document RAG retriever — hybrid."""
from __future__ import annotations

from typing import Any, Optional

from ....shared.auth.rbac_setup import GraphRBAC
from ....shared.auth.roles import DEFAULT_PUBLIC_CONTEXT, UserContext
from ....shared.config.settings import (
    NEO4J_PASSWORD,
    NEO4J_URI,
    NEO4J_USER,
    RETRIEVAL_FINAL_LIMIT,
)
from ....shared.neo4j.driver import get_neo4j_driver
from ....shared.telemetry.context import TelemetryEvent, get_telemetry
from ....shared.registries.strategy_registry import get_unstructured
from ..executor import DocumentQueryExecutor
from ..query_intent import (
    is_filing_date_question,
    is_page_question,
    is_toc_question,
    is_visual_page_question,
)
from ..services.formatter import access_denied_response
from ..strategies import registration as _strategy_registration  # noqa: F401  (side-effect: registers strategies)


class HybridRetrieveMixin:
    def semantic_retrieve(
        self,
        query: str,
        limit: int = RETRIEVAL_FINAL_LIMIT,
        user_context: Optional[UserContext] = None,
        document_id_hint: str = "",
    ) -> dict[str, Any]:
        return self.hybrid_retrieve(
            query, limit=limit, user_context=user_context, document_id_hint=document_id_hint
        )

    def hybrid_retrieve(
        self,
        query: str,
        limit: int = RETRIEVAL_FINAL_LIMIT,
        user_context: Optional[UserContext] = None,
        document_id_hint: str = "",
    ) -> dict[str, Any]:
        """
        Neo4j Graph RAG (all run together for normal queries):
        1. Semantic — embed query, vector search on Section embeddings
        2. Full-text — Lucene on node_text_index
        3. Graph — 1–2 hop expand from vector seeds via structural/semantic edges
        4. Lexical — phrase CONTAINS + keyword overlap (merged in ranker, not a bypass)

        Early exit (no semantic): TOC, PDF page, page visual lookups only.

        `document_id_hint`: the document the current conversation thread was
        already discussing (see conversation/thread_memory.py) — passed
        through to every strategy, used only as a fallback when the query
        has no stronger document signal of its own.
        """
        ctx = user_context or self.user_context
        tenant_id = ctx.tenant_id
        denied = access_denied_response(self.rbac, query, ctx)
        if denied:
            return denied

        tel = get_telemetry()

        # Box request (heading list or specific box content) — migrated to a
        # registered strategy; see strategies/box.py.
        if self._exec.is_box_list_request(query) or self._exec.parse_box_number(query) is not None:
            with self.driver.session() as session:
                response = get_unstructured("structural_box_list").retrieve(
                    session, query, tenant_id=tenant_id, limit=limit, ctx=ctx,
                    document_id_hint=document_id_hint,
                )
            if response:
                if tel is not None:
                    tel.add(TelemetryEvent(kind="unstructured_retrieve", meta={"mode": response.get("mode")}))
                return response

        # Subsection request (section listing, section detail, or doc-choice
        # clarification when multiple documents exist and none was named) —
        # migrated to a registered strategy; see strategies/subsection.py.
        if self._exec.is_subsection_request(query) and self._exec.parse_section_number(query):
            with self.driver.session() as session:
                response = get_unstructured("subsection_tree").retrieve(
                    session, query, tenant_id=tenant_id, limit=limit, ctx=ctx,
                    document_id_hint=document_id_hint,
                )
            if response:
                if tel is not None:
                    tel.add(TelemetryEvent(kind="unstructured_retrieve", meta={"mode": response.get("mode")}))
                return response

        # TOC request (table of contents, or doc-choice clarification when the
        # named document can't be resolved) — migrated to a registered
        # strategy; see strategies/toc.py.
        if is_toc_question(query):
            with self.driver.session() as session:
                response = get_unstructured("structural_toc").retrieve(
                    session, query, tenant_id=tenant_id, limit=limit, ctx=ctx,
                    document_id_hint=document_id_hint,
                )
            if response:
                if tel is not None:
                    tel.add(TelemetryEvent(kind="unstructured_retrieve", meta={"mode": response.get("mode")}))
                return response

        # Filing-date request — answered from ingestion metadata
        # (DocRevision.source_filename), not document text; see
        # strategies/filing_date.py for why guessing from prose can't work
        # here (the real filing date usually isn't in the PDF body at all).
        if is_filing_date_question(query):
            with self.driver.session() as session:
                response = get_unstructured("structural_filing_date").retrieve(
                    session, query, tenant_id=tenant_id, limit=limit, ctx=ctx,
                    document_id_hint=document_id_hint,
                )
            if response:
                if tel is not None:
                    tel.add(TelemetryEvent(kind="unstructured_retrieve", meta={"mode": response.get("mode")}))
                return response

        # Page request (figure/visual page or plain page text) — migrated to
        # a registered strategy; see strategies/page.py.
        if is_visual_page_question(query) or is_page_question(query):
            with self.driver.session() as session:
                response = get_unstructured("structural_page").retrieve(
                    session, query, tenant_id=tenant_id, limit=limit, ctx=ctx,
                    document_id_hint=document_id_hint,
                )
            if response:
                if tel is not None:
                    tel.add(TelemetryEvent(kind="unstructured_retrieve", meta={"mode": response.get("mode")}))
                return response

        # Terminal fallthrough: concurrent vector+lexical+graph fetch, merge,
        # and pin — migrated to a registered strategy; see
        # strategies/full_hybrid.py. No session is opened here: this
        # strategy opens its own per-task sessions internally (each
        # concurrently-submitted fetch needs its own, since Neo4j sessions
        # aren't thread-safe) — `session=None` is passed only for Protocol
        # conformance with the other strategies, which do use it.
        return get_unstructured("graph_rag_hybrid").retrieve(
            None, query, tenant_id=tenant_id, limit=limit, ctx=ctx,
            document_id_hint=document_id_hint,
        )

    def close(self) -> None:
        """No-op: driver is process-wide; use close_neo4j_driver() on shutdown."""

    def __init__(
        self,
        uri: str = NEO4J_URI,
        user: str = NEO4J_USER,
        password: str = NEO4J_PASSWORD,
        user_context: Optional[UserContext] = None,
    ):
        self.driver = get_neo4j_driver(uri, user, password)
        self.user_context = user_context or DEFAULT_PUBLIC_CONTEXT
        self.rbac = GraphRBAC(uri, user, password, driver=self.driver)
        self._exec = DocumentQueryExecutor()

