"""Document RAG retriever — hybrid."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Optional, TypeVar

from ....auth.rbac_setup import GraphRBAC
from ....auth.roles import DEFAULT_PUBLIC_CONTEXT, UserContext
from ....config.settings import (
    NEO4J_PASSWORD,
    NEO4J_URI,
    NEO4J_USER,
    RETRIEVAL_CANDIDATE_POOL,
    RETRIEVAL_FINAL_LIMIT,
)
from ....graph.driver import get_neo4j_driver
from ....telemetry.context import TelemetryEvent, get_telemetry
from ....feedback_loop import get_feedback_routing
from ....telemetry.pipeline import record_pipeline_step
from ..constants import (
    _FULLTEXT_LIMIT,
    _GRAPH_1HOP_LIMIT,
    _GRAPH_2HOP_LIMIT,
    _VECTOR_SEED_LIMIT,
)
from ...strategy_registry import get_unstructured
from ..executor import DocumentQueryExecutor
from ..query_intent import (
    is_enumeration_question,
    is_page_question,
    is_synthesis_question,
    is_toc_question,
    is_visual_page_question,
)
from ..strategies import registration as _strategy_registration  # noqa: F401  (side-effect: registers strategies)

_T = TypeVar("_T")


class HybridRetrieveMixin:
    def _neo4j_session_call(self, fn: Callable[..., _T], /, *args: Any, **kwargs: Any) -> _T:
        """Run a Neo4j read in its own session (safe for thread-pool parallelism)."""
        with self.driver.session() as session:
            return fn(session, *args, **kwargs)

    def semantic_retrieve(
        self,
        query: str,
        limit: int = RETRIEVAL_FINAL_LIMIT,
        user_context: Optional[UserContext] = None,
    ) -> dict[str, Any]:
        return self.hybrid_retrieve(query, limit=limit, user_context=user_context)

    def hybrid_retrieve(
        self,
        query: str,
        limit: int = RETRIEVAL_FINAL_LIMIT,
        user_context: Optional[UserContext] = None,
    ) -> dict[str, Any]:
        """
        Neo4j Graph RAG (all run together for normal queries):
        1. Semantic — embed query, vector search on Section embeddings
        2. Full-text — Lucene on node_text_index
        3. Graph — 1–2 hop expand from vector seeds via structural/semantic edges
        4. Lexical — phrase CONTAINS + keyword overlap (merged in ranker, not a bypass)

        Early exit (no semantic): TOC, PDF page, page visual lookups only.
        """
        ctx = user_context or self.user_context
        tenant_id = ctx.tenant_id
        denied = self._access_denied_response(query, ctx)
        if denied:
            return denied

        tel = get_telemetry()

        # Box request (heading list or specific box content) — migrated to a
        # registered strategy; see strategies/box.py.
        if self._exec.is_box_list_request(query) or self._exec.parse_box_number(query) is not None:
            with self.driver.session() as session:
                response = get_unstructured("structural_box_list").retrieve(
                    session, query, tenant_id=tenant_id, limit=limit, ctx=ctx
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
                    session, query, tenant_id=tenant_id, limit=limit, ctx=ctx
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
                    session, query, tenant_id=tenant_id, limit=limit, ctx=ctx
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
                    session, query, tenant_id=tenant_id, limit=limit, ctx=ctx
                )
            if response:
                if tel is not None:
                    tel.add(TelemetryEvent(kind="unstructured_retrieve", meta={"mode": response.get("mode")}))
                return response

        synthesis = is_synthesis_question(query)
        enumeration = is_enumeration_question(query)
        fetch_limit = limit
        if synthesis:
            fetch_limit = max(limit, 16)
        if enumeration:
            fetch_limit = max(fetch_limit, 18)
        vector_limit = min(RETRIEVAL_CANDIDATE_POOL, 16) if synthesis else _VECTOR_SEED_LIMIT
        graph_1hop = 32 if synthesis else _GRAPH_1HOP_LIMIT
        graph_2hop = 24 if synthesis else _GRAPH_2HOP_LIMIT

        mode_hint = get_feedback_routing().document_mode(query)
        skip_vector = mode_hint == "graph_rag_lexical"

        embedding = None if skip_vector else self._get_embedding(query)

        with ThreadPoolExecutor(max_workers=4, thread_name_prefix="hybrid_seed") as pool:
            phrase_future = pool.submit(
                self._neo4j_session_call,
                self._structural_phrase_retrieve,
                query,
                tenant_id=tenant_id,
            )
            keyword_future = pool.submit(
                self._neo4j_session_call,
                self._structural_keyword_retrieve,
                query,
                tenant_id=tenant_id,
            )
            if skip_vector:
                vector_future = None
                vector_hits: list[dict] = []
            else:
                vector_future = pool.submit(
                    self._neo4j_session_call,
                    self._vector_seed,
                    embedding,
                    vector_limit,
                    tenant_id=tenant_id,
                )
            fulltext_future = pool.submit(
                self._neo4j_session_call,
                self._fulltext_seed,
                query,
                _FULLTEXT_LIMIT,
                tenant_id=tenant_id,
            )
            phrase_hits = phrase_future.result()
            keyword_hits = keyword_future.result()
            if vector_future is not None:
                vector_hits = vector_future.result()
            fulltext_hits = fulltext_future.result()

        lexical_hits = self._merge_retrieval_chunks(phrase_hits, keyword_hits)
        seed_ids = [h["id"] for h in vector_hits if h.get("id")]
        seed_scores = {h["id"]: float(h["score"]) for h in vector_hits if h.get("id")}

        graph_hits: list[dict] = []
        if seed_ids:
            with ThreadPoolExecutor(max_workers=2, thread_name_prefix="hybrid_graph") as pool:
                hop1_future = pool.submit(
                    self._neo4j_session_call,
                    self._graph_expand,
                    seed_ids,
                    hops=1,
                    limit=graph_1hop,
                    tenant_id=tenant_id,
                )
                hop2_future = pool.submit(
                    self._neo4j_session_call,
                    self._graph_expand,
                    seed_ids,
                    hops=2,
                    limit=graph_2hop,
                    tenant_id=tenant_id,
                )
                graph_hits = hop1_future.result() + hop2_future.result()

        items = self._merge_and_rank(
            query,
            vector_hits,
            fulltext_hits,
            graph_hits,
            seed_scores,
            lexical_hits=lexical_hits,
            synthesis=synthesis,
            limit=max(1, int(fetch_limit)),
        )
        if lexical_hits:
            items = self._pin_precision_lexical_chunks(
                query, items, lexical_hits, limit=max(1, int(fetch_limit))
            )
        if synthesis and lexical_hits:
            items = self._pin_contrast_lexical_chunks(
                query, items, lexical_hits, limit=max(1, int(fetch_limit))
            )

        response = self._format_response(query, items, user_context=ctx)
        if mode_hint == "graph_rag_lexical":
            response["mode"] = "graph_rag_lexical"
        elif mode_hint == "graph_rag":
            response["mode"] = "graph_rag"
        else:
            response["mode"] = "graph_rag"
            if lexical_hits and vector_hits:
                response["mode"] = "graph_rag_hybrid"
            elif lexical_hits:
                response["mode"] = "graph_rag_lexical"
        response["strategy"] = "graph_rag"
        response["vector_seeds"] = len(vector_hits)
        response["fulltext_hits"] = len(fulltext_hits)
        response["graph_expanded"] = len(graph_hits)
        record_pipeline_step(
            "document.hybrid.merge",
            meta={
                "mode": response.get("mode"),
                "vector_seeds": len(vector_hits),
                "fulltext_hits": len(fulltext_hits),
                "graph_expanded": len(graph_hits),
                "returned": len(items),
            },
        )
        return response

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

