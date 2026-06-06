"""Document RAG retriever — hybrid."""
from __future__ import annotations

from typing import Any, Optional

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
from ..constants import (
    _FULLTEXT_LIMIT,
    _GRAPH_1HOP_LIMIT,
    _GRAPH_2HOP_LIMIT,
    _VECTOR_SEED_LIMIT,
)
from ..executor import DocumentQueryExecutor
from ..query_intent import (
    is_enumeration_question,
    is_page_question,
    is_synthesis_question,
    is_toc_question,
    is_visual_page_question,
)


class HybridRetrieveMixin:
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
        denied = self._access_denied_response(query, ctx)
        if denied:
            return denied

        tel = get_telemetry()

        # Box headings listing (generic): "List all Box headings" should enumerate Box 1..N.
        if self._exec.is_box_list_request(query):
            with self.driver.session() as session:
                items = self._structural_box_headings(session, query)
            if items:
                response = self._format_response(query, items, user_context=ctx)
                response["mode"] = "structural_box_list"
                response["strategy"] = "graph_rag"
                response["vector_seeds"] = 0
                response["fulltext_hits"] = 0
                response["graph_expanded"] = len(items)
                if tel is not None:
                    tel.add(TelemetryEvent(kind="unstructured_retrieve", meta={"mode": response["mode"]}))
                return response

        # Box content fetch (generic): "Box 5" / "Box 10" should retrieve that box text.
        box_n = self._exec.parse_box_number(query)
        if box_n is not None and not self._exec.is_box_list_request(query):
            with self.driver.session() as session:
                items = self._structural_box_content(session, query, box_n)
            if items:
                response = self._format_response(query, items, user_context=ctx)
                response["mode"] = "structural_box_content"
                response["strategy"] = "graph_rag"
                response["vector_seeds"] = 0
                response["fulltext_hits"] = 0
                response["graph_expanded"] = len(items)
                if tel is not None:
                    tel.add(TelemetryEvent(kind="unstructured_retrieve", meta={"mode": response["mode"], "box": box_n}))
                return response

        # If the user asks about a specific section/subsections and multiple documents exist,
        # ask them to pick the document rather than guessing.
        if self._exec.is_subsection_request(query) and self._exec.parse_section_number(query):
            with self.driver.session() as session:
                doc_id, doc_title = self._resolve_document_for_query(session, query)
                # If user didn't mention any doc terms, this resolver can "guess" the biggest doc.
                # When multiple docs exist, prefer explicit clarification.
                if not self._document_match_terms(query):
                    docs = self._list_documents(session, limit=5)
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

        # Subsection listing: if a section number is requested, return child headings if present.
        if self._exec.is_subsection_request(query):
            sec_num = self._exec.parse_section_number(query)
            if sec_num:
                with self.driver.session() as session:
                    items, parent = self._structural_subsections(session, query, sec_num)
                if items:
                    response = self._format_response(query, items, user_context=ctx)
                    response["mode"] = "subsection_tree"
                    response["strategy"] = "graph_rag"
                    response["parent_id"] = parent.get("id")
                    response["parent_title"] = parent.get("title")
                    response["vector_seeds"] = 0
                    response["fulltext_hits"] = 0
                    response["graph_expanded"] = len(items)
                    if tel is not None:
                        tel.add(TelemetryEvent(kind="unstructured_retrieve", meta={"mode": response["mode"]}))
                    return response
                if parent and parent.get("text"):
                    response = self._format_response(query, [parent], user_context=ctx)
                    response["mode"] = "section_detail"
                    response["strategy"] = "graph_rag"
                    response["parent_id"] = parent.get("id")
                    response["parent_title"] = parent.get("title")
                    response["vector_seeds"] = 0
                    response["fulltext_hits"] = 0
                    response["graph_expanded"] = 1
                    if tel is not None:
                        tel.add(TelemetryEvent(kind="unstructured_retrieve", meta={"mode": response["mode"]}))
                    return response

        if is_toc_question(query):
            with self.driver.session() as session:
                # If the user named a specific document but we cannot find it,
                # return a clarification rather than silently using the wrong doc.
                doc_terms = self._doc_name_terms(query)
                if doc_terms:
                    doc_id, _ = self._resolve_document_for_query_strict(session, query)
                    if doc_id is None:
                        docs = self._list_documents(session, limit=8)
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
                toc_items = self._structural_toc_retrieve(session, query)
            if toc_items:
                response = self._format_response(query, toc_items, user_context=ctx)
                response["mode"] = "structural_toc"
                response["strategy"] = "graph_rag"
                response["vector_seeds"] = 0
                response["fulltext_hits"] = 0
                response["graph_expanded"] = len(toc_items)
                return response

        if is_visual_page_question(query):
            with self.driver.session() as session:
                visual_items = self._structural_page_visual_retrieve(session, query)
            if visual_items:
                pdf_page, doc_page = self._parse_page_targets(query)
                response = self._format_response(query, visual_items, user_context=ctx)
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
                return response

        if is_page_question(query):
            with self.driver.session() as session:
                page_items = self._structural_page_retrieve(session, query)
            if page_items:
                response = self._format_response(query, page_items, user_context=ctx)
                response["mode"] = "structural_page"
                response["strategy"] = "graph_rag"
                response["vector_seeds"] = 0
                response["fulltext_hits"] = 0
                response["graph_expanded"] = len(page_items)
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

        embedding = self._get_embedding(query)
        with self.driver.session() as session:
            lexical_hits = self._merge_retrieval_chunks(
                self._structural_phrase_retrieve(session, query),
                self._structural_keyword_retrieve(session, query),
            )
            vector_hits = self._vector_seed(session, embedding, vector_limit)
            fulltext_hits = self._fulltext_seed(session, query, _FULLTEXT_LIMIT)
            seed_ids = [h["id"] for h in vector_hits if h.get("id")]
            seed_scores = {h["id"]: float(h["score"]) for h in vector_hits if h.get("id")}

            graph_hits: list[dict] = []
            if seed_ids:
                graph_hits.extend(
                    self._graph_expand(session, seed_ids, hops=1, limit=graph_1hop)
                )
                graph_hits.extend(
                    self._graph_expand(session, seed_ids, hops=2, limit=graph_2hop)
                )

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
        response["mode"] = "graph_rag"
        if lexical_hits and vector_hits:
            response["mode"] = "graph_rag_hybrid"
        elif lexical_hits:
            response["mode"] = "graph_rag_lexical"
        response["strategy"] = "graph_rag"
        response["vector_seeds"] = len(vector_hits)
        response["fulltext_hits"] = len(fulltext_hits)
        response["graph_expanded"] = len(graph_hits)
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

