"""
retrieval/unstructured/retriever.py — Neo4j Graph RAG for documents.

Flow (document-agnostic — works for any ingested PDF):
1. Vector seed — embed query, find entry Section nodes
2. Full-text seed — keyword lookup on node_text_index
3. Graph expand — 1–2 hop traversal via structural + semantic edges
4. Merge, rank, return top-k chunks for LLM synthesis
"""

from __future__ import annotations

import re
from typing import Any, Optional

from ...auth.rbac_setup import GraphRBAC
from ...graph.driver import get_neo4j_driver
from ...auth.roles import DEFAULT_PUBLIC_CONTEXT, UserContext
from ...config.settings import (
    EMBEDDING_MODEL,
    MODEL_PROVIDER,
    NEO4J_PASSWORD,
    NEO4J_URI,
    NEO4J_USER,
    OPENAI_API_KEY,
    RETRIEVAL_CANDIDATE_POOL,
    RETRIEVAL_FINAL_LIMIT,
)
from ...document.page_numbers import parse_page_number_from_query
from ...document.page_vision import compact_visual_content
from ...graph.constants import (
    DOC_REVISION_LABEL,
    DOCUMENT_LOGICAL_LABEL,
    DOCUMENT_ROOT_CYPHER,
)
from ...graph.versioning import lifecycle_active
from .toc_retrieval import (
    format_outline_chunk,
    format_toc_chunk,
    include_in_outline_fallback,
    score_page_text_as_toc,
    section_title_is_toc,
)
from .visual_retrieval import parse_visual_intent
from ...model_providers.factory import get_model_provider
from .executor import DocumentQueryExecutor
from ...telemetry.context import TelemetryEvent, get_telemetry

provider = get_model_provider(MODEL_PROVIDER, OPENAI_API_KEY)


def _doc_scope_cypher(alias: str = "d") -> str:
    """Match logical document id or revision-prefixed content root id."""
    return (
        f"($doc_id IS NULL OR {alias}.logical_doc_id = $doc_id "
        f"OR {alias}.id = $doc_id "
        f"OR ($doc_id IS NOT NULL AND {alias}.id STARTS WITH $doc_id + ':'))"
    )


_JOB_PREFIX_RE = re.compile(r"^[0-9a-f]{32}_", re.I)


def _clean_doc_title(title: Optional[str]) -> str:
    """Strip a leading 32-hex job-id prefix left on older ingests' titles."""
    t = (title or "").strip()
    return _JOB_PREFIX_RE.sub("", t) or t


def _node_scope_cypher(alias: str = "n") -> str:
    """
    Scope any content node (Page/Section/...) to a document without relying on
    variable-length CONTAINS paths (parsers can nest sections far deeper than a
    fixed hop budget). Matches the resolved logical id or a content-root id prefix.

    The current id format is "<logical_id>:r<rev>::doc_<hash>...", so the ':'
    prefix is the canonical scope. The '_' prefix is only used as a legacy
    fallback for nodes that predate logical_doc_id — and is guarded by
    `logical_doc_id IS NULL` so it cannot leak across sibling documents whose
    logical id is a prefix of another (e.g. "doc_x" vs "doc_x_2").
    """
    return (
        f"($doc_id IS NULL "
        f"OR {alias}.logical_doc_id = $doc_id "
        f"OR {alias}.id STARTS WITH $doc_id + ':' "
        f"OR ({alias}.logical_doc_id IS NULL "
        f"AND {alias}.id STARTS WITH $doc_id + '_'))"
    )

# Structural + semantic edges created at ingest (Axis 1 & Axis 2).
_GRAPH_REL_TYPES = (
    "SEMANTICALLY_SIMILAR",
    "SHARES_ENTITY",
    "REFERENCES",
    "ELABORATES",
    "SAME_CATEGORY",
    "FOLLOWS",
    "PRECEDES",
    "CONTAINS",
    "PART_OF",
)

# Node labels that carry answer text during graph expansion.
_TEXT_NODE_LABELS = ("Section", "Page", "Chapter", "Region")

_VECTOR_SEED_LIMIT = min(RETRIEVAL_CANDIDATE_POOL, 12)
_FULLTEXT_LIMIT = 8
_GRAPH_1HOP_LIMIT = 24
_GRAPH_2HOP_LIMIT = 16

_SYNTHESIS_RE = re.compile(
    r"\b(synthesi[sz]|structural map|escalat|pathway|flowchart|flow chart|"
    r"compare|contrast|relationship between|trace how|build a .{0,20}map|"
    r"how .{0,40} connect|map showing)\b",
    re.I,
)

_ENUMERATION_RE = re.compile(
    r"\b(list\s+all|enumerate|name\s+all|distinct)\b",
    re.I,
)

_CONTRAST_COMPARE_RE = re.compile(
    r"\b(contrast|compare|comparison|versus|vs\.?)\b",
    re.I,
)

_KEYWORD_STOP = frozenset({
    "what", "which", "where", "when", "that", "this", "with", "from", "into",
    "have", "been", "were", "they", "their", "there", "about", "under", "based",
    "specific", "according", "should", "would", "could", "document", "text",
    "showing", "single", "show", "build", "does", "explicitly", "detailed",
})


def _query_anchor_terms(query: str) -> list[str]:
    """Proper names and dotted tokens from the user question (corpus-agnostic)."""
    terms: list[str] = []
    seen: set[str] = set()

    def _add(raw: str) -> None:
        for variant in (
            raw,
            raw.replace(".", ""),
            raw.replace(".", " "),
            raw.replace("-", " "),
        ):
            tl = variant.lower().strip()
            if tl and tl not in seen and len(tl) >= 2:
                seen.add(tl)
                terms.append(tl)

    for m in re.finditer(r"\b[A-Za-z][\w]*(?:\.[\w]+)+\b", query or ""):
        _add(m.group(0))

    for m in re.finditer(r"\b[A-Z][A-Z0-9]{2,}\b", query or ""):
        _add(m.group(0))

    for m in re.finditer(r"\b[A-Z][a-z][A-Za-z0-9]{2,}\b", query or ""):
        _add(m.group(0))

    return terms[:10]


def is_synthesis_question(query: str) -> bool:
    return bool(_SYNTHESIS_RE.search(query or ""))


def is_enumeration_question(query: str) -> bool:
    return bool(_ENUMERATION_RE.search(query or ""))


_TOC_RE = re.compile(
    r"\b(table\s+of\s+contents?|\btoc\b|list\s+(?:all\s+)?(?:the\s+)?contents?|"
    r"show\s+(?:me\s+)?(?:the\s+)?contents?|provide\s+(?:me\s+)?(?:all\s+)?(?:the\s+)?toc)\b",
    re.I,
)


def is_toc_question(query: str) -> bool:
    return bool(_TOC_RE.search(query or ""))


_PAGE_QUERY_RE = re.compile(
    r"\b(?:fetch|get|show|retrieve|read|content|text|everything|all)\b.{0,50}\bpage\b|"
    r"\bpage\s+[\wivxlcdm\-]+\s+(?:of|from|in)\b|"
    r"\bcontent\s+(?:from|on|of)\s+(?:pdf\s+)?page\b|"
    r"\bwhat\s+(?:is|does)\s+(?:pdf\s+)?page\s+",
    re.I,
)


def is_page_question(query: str) -> bool:
    pdf_page, doc_page = parse_page_number_from_query(query)
    if pdf_page is not None or doc_page:
        return True
    return bool(_PAGE_QUERY_RE.search(query or ""))


_VISUAL_PAGE_RE = re.compile(
    r"\bvisual\s+content\b|"
    r"\b(?:all\s+)?(?:the\s+)?(?:images?|figures?|figs?\.?|diagrams?|charts?|photos?|pictures?|visuals?)\b.{0,40}\bpage\b|"
    r"\bpage\b.{0,40}\b(?:images?|figures?|visual|diagram)\b|"
    r"\b(?:tell\s+me|describe|explain).{0,60}\b(?:image|figure|diagram)\b|"
    r"\babout\s+(?:that|the)\s+(?:image|figure|diagram)\b",
    re.I,
)

_FIG_CAPTION_RE = re.compile(
    r"(?:Fig\.?|Figure)\s*(\d+(?:\.\d+)?)\s*[:.]\s*([^\n]+)",
    re.I,
)


_FACT_LOOKUP_RE = re.compile(
    r"\b(?:url|link|website|web\s*site|portal|email|e-mail|hyperlink)\b|"
    r"\bwhat\s+is\s+the\s+(?:url|link|website|address|portal)\b|"
    r"\b(?:which|into\s+which|how\s+many|when\s+did|who\s+hosted)\b|"
    r"\b(?:translated|translation|languages?|hosted|host|workshop)\b",
    re.I,
)

_PHRASE_STOP = _KEYWORD_STOP | frozenset({
    "url", "link", "website", "portal", "email", "address", "http", "https",
    "into", "which", "what", "when", "who", "how", "many", "much",
    "the", "for", "has", "been", "was", "were", "does", "did", "are", "any",
    "whose", "that", "this", "with", "from", "than", "then", "also", "only",
    "name", "list", "give", "tell", "say", "ask",
})

_MONTH_YEAR_RE = re.compile(
    r"\b(january|february|march|april|may|june|july|august|september|october|"
    r"november|december)\s+(20\d{2}|19\d{2})\b",
    re.I,
)

_BROKEN_URL_RE = re.compile(r"https?://[^\s\)\]>\"']+(?:\s+[^\s\)\]>\"']+)+", re.I)


def is_fact_lookup_question(query: str) -> bool:
    return bool(_FACT_LOOKUP_RE.search(query or ""))


def _normalize_broken_urls(text: str) -> str:
    """Fix PDF line-breaks/spaces inside URLs (e.g. 'https://example. org/path' -> 'https://example.org/path')."""

    def _fix(match: re.Match) -> str:
        return re.sub(r"\s+", "", match.group(0))

    return _BROKEN_URL_RE.sub(_fix, text or "")


def _extract_urls(text: str) -> list[str]:
    normalized = _normalize_broken_urls(text)
    return list(dict.fromkeys(re.findall(r"https?://[^\s\)\]>\"']+", normalized)))


def is_visual_page_question(query: str) -> bool:
    """Page-scoped question focused on figures/images, not plain page text."""
    if not _VISUAL_PAGE_RE.search(query or ""):
        return False
    pdf_page, doc_page = parse_page_number_from_query(query)
    if pdf_page is not None or doc_page:
        return True
    intent = parse_visual_intent(query)
    return intent.wants_image or intent.pdf_page is not None


class DocumentRAGRetriever:
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

    def close(self) -> None:
        """No-op: driver is process-wide; use close_neo4j_driver() on shutdown."""

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

    def _vector_seed(self, session, embedding: list[float], limit: int) -> list[dict]:
        try:
            rows = session.run(
                f"""
                CALL db.index.vector.queryNodes('section_embedding', $limit, $embedding)
                YIELD node AS n, score
                WHERE coalesce(n.text, '') <> ''
                  AND {lifecycle_active("n")}
                RETURN
                  coalesce(n.id, '') AS id,
                  coalesce(n.title, '') AS title,
                  coalesce(n.text, '') AS text,
                  coalesce(labels(n)[0], '') AS node_label,
                  score
                ORDER BY score DESC
                """,
                limit=max(1, limit),
                embedding=embedding,
            )
            return [
                {
                    "id": r["id"],
                    "title": r["title"] or r["id"],
                    "text": r["text"],
                    "node_label": r.get("node_label") or "",
                    "score": float(r["score"] or 0.0),
                    "related": [],
                }
                for r in rows
                if r["id"]
            ]
        except Exception:
            return []

    def _fulltext_seed(self, session, query: str, limit: int) -> list[dict]:
        lucene_q = self._fulltext_query(query)
        if not lucene_q:
            return []
        try:
            rows = session.run(
                f"""
                CALL db.index.fulltext.queryNodes('node_text_index', $q, {{limit: $limit}})
                YIELD node AS n, score
                WHERE coalesce(n.text, '') <> ''
                  AND {lifecycle_active("n")}
                  AND any(l IN labels(n) WHERE l IN $labels)
                RETURN
                  coalesce(n.id, '') AS id,
                  coalesce(n.title, '') AS title,
                  coalesce(n.text, '') AS text,
                  coalesce(labels(n)[0], '') AS node_label,
                  score
                ORDER BY score DESC
                """,
                q=lucene_q,
                limit=max(1, limit),
                labels=list(_TEXT_NODE_LABELS),
            )
            return [
                {
                    "id": r["id"],
                    "title": r["title"] or r["id"],
                    "text": r["text"],
                    "node_label": r.get("node_label") or "",
                    "score": float(r["score"] or 0.0),
                    "related": [],
                }
                for r in rows
                if r["id"]
            ]
        except Exception:
            return []

    def _graph_expand(
        self,
        session,
        seed_ids: list[str],
        *,
        hops: int,
        limit: int,
    ) -> list[dict]:
        if hops == 1:
            cypher = f"""
                UNWIND $seed_ids AS sid
                MATCH (seed:Section {{id: sid}})
                WHERE {lifecycle_active("seed")}
                MATCH (seed)-[r]-(related)
                WHERE type(r) IN $rel_types
                  AND any(l IN labels(related) WHERE l IN $node_labels)
                  AND coalesce(related.text, '') <> ''
                  AND {lifecycle_active("related")}
                RETURN DISTINCT
                  coalesce(related.id, '') AS id,
                  coalesce(related.title, '') AS title,
                  coalesce(related.text, '') AS text,
                  coalesce(labels(related)[0], '') AS node_label,
                  type(r) AS rel_type,
                  coalesce(r.weight, 0.75) AS edge_weight,
                  sid AS seed_id,
                  1 AS hops
                LIMIT $limit
            """
        else:
            cypher = f"""
                UNWIND $seed_ids AS sid
                MATCH (seed:Section {{id: sid}})
                WHERE {lifecycle_active("seed")}
                MATCH (seed)-[r1]-(mid)-[r2]-(related)
                WHERE type(r1) IN $rel_types
                  AND type(r2) IN $rel_types
                  AND any(l IN labels(related) WHERE l IN $node_labels)
                  AND coalesce(related.text, '') <> ''
                  AND {lifecycle_active("related")}
                  AND related.id <> sid
                RETURN DISTINCT
                  coalesce(related.id, '') AS id,
                  coalesce(related.title, '') AS title,
                  coalesce(related.text, '') AS text,
                  coalesce(labels(related)[0], '') AS node_label,
                  type(r1) + '->' + type(r2) AS rel_type,
                  coalesce(r2.weight, 0.75) AS edge_weight,
                  sid AS seed_id,
                  2 AS hops
                LIMIT $limit
            """
        try:
            rows = session.run(
                cypher,
                seed_ids=seed_ids,
                rel_types=list(_GRAPH_REL_TYPES),
                node_labels=list(_TEXT_NODE_LABELS),
                limit=max(1, limit),
            )
            return [
                {
                    "id": r["id"],
                    "title": r["title"] or r["id"],
                    "text": r["text"],
                    "node_label": r.get("node_label") or "",
                    "rel_type": r["rel_type"],
                    "edge_weight": float(r["edge_weight"] or 0.75),
                    "seed_id": r["seed_id"],
                    "hops": int(r["hops"] or hops),
                    "related": [r["rel_type"]] if r.get("rel_type") else [],
                }
                for r in rows
                if r["id"]
            ]
        except Exception:
            return []

    def _merge_and_rank(
        self,
        query: str,
        vector_hits: list[dict],
        fulltext_hits: list[dict],
        graph_hits: list[dict],
        seed_scores: dict[str, float],
        limit: int,
        *,
        lexical_hits: Optional[list[dict]] = None,
        synthesis: bool = False,
    ) -> list[dict]:
        merged: dict[str, dict] = {}

        def _upsert(item: dict, score: float, source: str, related: Optional[list] = None) -> None:
            cid = item.get("id") or ""
            if not cid:
                return
            rel = related or item.get("related") or []
            if cid in merged:
                merged[cid]["score"] = max(float(merged[cid]["score"]), score)
                merged[cid]["sources"].add(source)
                for r in rel:
                    if r and r not in merged[cid]["related"]:
                        merged[cid]["related"].append(r)
            else:
                merged[cid] = {
                    "id": cid,
                    "title": item.get("title") or cid,
                    "text": item.get("text") or "",
                    "score": score,
                    "related": list(rel),
                    "sources": {source},
                }

        is_contrast = bool(_CONTRAST_COMPARE_RE.search(query or ""))
        vector_weight = 1.15 if synthesis and not is_contrast else 1.0
        graph_weight = 1.2 if synthesis and not is_contrast else 1.0
        if is_contrast:
            lexical_weight = 1.1
        elif synthesis:
            lexical_weight = 0.82
        else:
            lexical_weight = 1.0

        for item in vector_hits:
            _upsert(item, float(item.get("score", 0.0)) * vector_weight, "vector")

        max_ft = max((float(h.get("score", 0.0)) for h in fulltext_hits), default=1.0) or 1.0
        for item in fulltext_hits:
            norm = float(item.get("score", 0.0)) / max_ft
            _upsert(item, norm * 0.92, "fulltext")

        for item in graph_hits:
            seed_id = item.get("seed_id") or ""
            base = seed_scores.get(seed_id, 0.55)
            hop_decay = 0.88 ** int(item.get("hops", 1))
            edge_w = float(item.get("edge_weight", 0.75))
            rel = item.get("rel_type")
            _upsert(
                item,
                base * hop_decay * edge_w * graph_weight,
                "graph",
                [rel] if rel else [],
            )

        for item in lexical_hits or []:
            src = "phrase" if "phrase_search" in (item.get("related") or []) else "keyword"
            _upsert(item, float(item.get("score", 0.85)) * lexical_weight, src)

        keywords = self._query_keywords(query)
        for item in merged.values():
            item["score"] = float(item["score"]) * self._relevance_boost(
                item.get("title") or "",
                item.get("text") or "",
                keywords,
            )

        ranked = sorted(merged.values(), key=lambda x: x["score"], reverse=True)
        out: list[dict] = []
        for item in ranked[:limit]:
            sources = sorted(item.pop("sources", {"graph"}))
            item["related"] = item.get("related") or []
            if sources:
                item["related"] = list(dict.fromkeys([*item["related"], f"via:{','.join(sources)}"]))
            out.append(item)
        return out

    def _contrast_term_groups(self, query: str) -> list[list[str]]:
        """For compare/contrast questions, one keyword group per side of the comparison."""
        if not _CONTRAST_COMPARE_RE.search(query or ""):
            return []
        parts = re.split(
            r"\b(?:versus|vs\.?|compared\s+to|against)\b",
            query or "",
            maxsplit=1,
            flags=re.I,
        )
        if len(parts) >= 2:
            groups: list[list[str]] = []
            for part in parts[:2]:
                kws = [
                    k
                    for k in self._content_keywords_from_query(part)
                    if len(k) >= 4 and k not in _KEYWORD_STOP
                ]
                if kws:
                    groups.append(kws[:4])
            if len(groups) >= 2:
                return groups
        q = (query or "").lower()
        groups = []
        for token in self._query_keywords(query):
            if len(token) >= 5 and token not in {
                "contrast", "compare", "comparison", "between", "versus",
            }:
                groups.append([token])
        return groups[:2] if len(groups) >= 2 else []

    @staticmethod
    def _text_matches_term_groups(text: str, groups: list[list[str]]) -> bool:
        if len(groups) < 2:
            return False
        norm = (text or "").lower().replace(" ", "").replace(".", "")
        for group in groups:
            if not any(g.lower().replace(" ", "").replace(".", "") in norm for g in group):
                return False
        return True

    def _precision_pin_patterns(self, query: str) -> list[str]:
        """Long query-derived phrases used to pin compact high-signal chunks."""
        min_len = 10 if is_enumeration_question(query) else 8
        patterns = [
            p for p in self._search_phrases_from_query(query) if len(p) >= min_len
        ]
        return list(dict.fromkeys(patterns))[:8]

    def _pin_precision_lexical_chunks(
        self,
        query: str,
        items: list[dict],
        lexical_hits: list[dict],
        *,
        limit: int,
    ) -> list[dict]:
        """
        Pin compact, high-signal lexical hits (e.g. Region facts, network lists)
        that vector search often ranks below broad sections.
        """
        patterns = self._precision_pin_patterns(query)
        if not patterns:
            return items

        pinned: list[dict] = []
        for hit in lexical_hits:
            text = (hit.get("text") or "").lower()
            if any(p in text for p in patterns):
                pinned.append(hit)
        if not pinned:
            return items

        pinned.sort(key=lambda h: len(h.get("text") or ""))

        seen: set[str] = set()
        out: list[dict] = []
        for hit in pinned[:3]:
            cid = hit.get("id")
            if not cid or cid in seen:
                continue
            seen.add(cid)
            out.append(
                {
                    "id": cid,
                    "title": hit.get("title") or cid,
                    "text": hit.get("text") or "",
                    "score": float(hit.get("score", 1.5)) + 0.55,
                    "related": list(
                        dict.fromkeys([*(hit.get("related") or []), "via:precision_pin"])
                    ),
                }
            )

        for item in items:
            cid = item.get("id")
            if cid and cid not in seen:
                out.append(item)
            if len(out) >= limit:
                break
        return out[:limit]

    def _pin_contrast_lexical_chunks(
        self,
        query: str,
        items: list[dict],
        lexical_hits: list[dict],
        *,
        limit: int,
    ) -> list[dict]:
        """
        Contrast questions need chunks that mention BOTH sides named in the query.
        Vector-only ranking often returns executive-summary pages and drops the intro contrast.
        """
        groups = self._contrast_term_groups(query)
        if len(groups) < 2:
            return items

        pinned: list[dict] = []
        for hit in lexical_hits:
            if self._text_matches_term_groups(hit.get("text") or "", groups):
                pinned.append(hit)

        if not pinned:
            return items

        # Prefer the smallest Section chunk (figure callouts are often on one intro section).
        pinned.sort(
            key=lambda h: (
                0 if (h.get("related") or []) and "keyword" in str(h.get("related")) else 1,
                len(h.get("text") or ""),
            )
        )

        seen: set[str] = set()
        out: list[dict] = []
        for hit in pinned[:2]:
            cid = hit.get("id")
            if not cid or cid in seen:
                continue
            seen.add(cid)
            out.append(
                {
                    "id": cid,
                    "title": hit.get("title") or cid,
                    "text": hit.get("text") or "",
                    "score": float(hit.get("score", 1.5)) + 0.5,
                    "related": list(
                        dict.fromkeys([*(hit.get("related") or []), "via:contrast_pin"])
                    ),
                }
            )

        for item in items:
            cid = item.get("id")
            if cid and cid not in seen:
                out.append(item)
            if len(out) >= limit:
                break
        return out[:limit]

    def _search_phrases_from_query(self, query: str) -> list[str]:
        """
        Build document-agnostic search phrases from the question (dates + word n-grams).
        Keeps light stopwords (of, the, at) so phrases align with PDF sentence wording.
        """
        q = (query or "").lower()
        phrases: list[str] = []

        for m in _MONTH_YEAR_RE.finditer(q):
            phrases.append(f"{m.group(1).lower()} {m.group(2)}")

        _light_stop = _PHRASE_STOP - frozenset({
            "of", "at", "in", "on", "to", "and", "or", "for", "by", "with", "from",
        })
        tokens: list[str] = []
        for anchor in _query_anchor_terms(query):
            for w in re.findall(r"[\w']+", anchor):
                if len(w) >= 2 and w not in tokens:
                    tokens.append(w)
        for w in re.findall(r"[\w']+", q):
            if len(w) <= 2 or w in _light_stop:
                continue
            if w not in tokens:
                tokens.append(w)

        for n in range(min(7, len(tokens)), 2, -1):
            for i in range(len(tokens) - n + 1):
                phrase = " ".join(tokens[i : i + n])
                if len(phrase) >= 8:
                    phrases.append(phrase)

        for w in tokens:
            if len(w) >= 5:
                phrases.append(w)

        seen: set[str] = set()
        ordered: list[str] = []
        for p in sorted(phrases, key=len, reverse=True):
            pl = p.lower().strip()
            if pl and pl not in seen:
                seen.add(pl)
                ordered.append(pl)
        return ordered[:14]

    def _content_keywords_from_query(self, query: str) -> list[str]:
        """
        Distinct content terms for AND-style overlap scoring (corpus-agnostic).

        Derived entirely from the question: proper-noun/acronym anchors, month-year
        dates, content tokens, hyphen/space variants, and adjacent bigrams. No
        per-document or per-topic vocabulary is injected here.
        """
        q = (query or "").lower()
        keywords: list[str] = []

        for anchor in _query_anchor_terms(query):
            if anchor not in keywords:
                keywords.append(anchor)

        for m in _MONTH_YEAR_RE.finditer(q):
            keywords.append(f"{m.group(1).lower()} {m.group(2)}")
            keywords.append(m.group(2))

        # Hyphenated terms in the query: add joined / spaced variants generically
        # (e.g. "case-control" → "case control", "casecontrol") to survive PDF wording.
        for hyph in re.findall(r"[a-z]+(?:-[a-z]+)+", q):
            keywords.append(hyph)
            keywords.append(hyph.replace("-", " "))
            keywords.append(hyph.replace("-", ""))

        for w in re.findall(r"[\w']+", q):
            if len(w) <= 2 or w in _PHRASE_STOP:
                continue
            if w not in keywords:
                keywords.append(w)

        words = [
            w
            for w in re.findall(r"[\w']+", q)
            if len(w) >= 4 and w not in _KEYWORD_STOP
        ]
        for i in range(len(words) - 1):
            bigram = f"{words[i]} {words[i + 1]}"
            if bigram not in keywords:
                keywords.append(bigram)

        return list(dict.fromkeys(keywords))[:18]

    @staticmethod
    def _merge_retrieval_chunks(primary: list[dict], extra: list[dict]) -> list[dict]:
        merged = list(primary)
        seen = {c["id"] for c in merged if c.get("id")}
        for item in extra:
            cid = item.get("id")
            if not cid or cid in seen:
                continue
            seen.add(cid)
            merged.append(item)
        merged.sort(key=lambda x: float(x.get("score", 0)), reverse=True)
        return merged[:8]

    def _structural_keyword_retrieve(self, session, query: str) -> list[dict]:
        """
        Rank nodes by how many distinct query keywords appear in text (robust to PDF spacing).
        """
        keywords = self._content_keywords_from_query(query)
        if len(keywords) < 2:
            return []

        min_hits = max(2, min(4, len(keywords) // 3))
        doc_id, _ = self._resolve_document_for_query(session, query)
        rows = session.run(
            f"""
            MATCH (d:{DOCUMENT_ROOT_CYPHER})
            WHERE {_doc_scope_cypher("d")}
            MATCH (n)
            WHERE any(l IN labels(n) WHERE l IN $labels)
              AND coalesce(n.text, '') <> ''
              AND (
                EXISTS {{ MATCH (d)-[:CONTAINS*0..6]->(n) }}
                OR n.id STARTS WITH d.id + '_'
              )
            WITH n,
              [k IN $keywords WHERE toLower(n.text) CONTAINS k] AS matched
            WHERE size(matched) >= $min_hits
            RETURN
              coalesce(n.id, '') AS id,
              coalesce(n.title, '') AS title,
              coalesce(n.text, '') AS text,
              size(matched) AS keyword_hits
            ORDER BY keyword_hits DESC, size(coalesce(n.text, '')) ASC
            LIMIT 6
            """,
            doc_id=doc_id,
            keywords=[k.lower() for k in keywords],
            min_hits=min_hits,
            labels=list(_TEXT_NODE_LABELS),
        )

        items: list[dict] = []
        for r in rows:
            if not r.get("id"):
                continue
            title = r.get("title") or r["id"]
            items.append({
                "id": r["id"],
                "title": title,
                "text": self._enrich_chunk_text_for_facts(title, r.get("text") or ""),
                "score": 0.88 + 0.06 * int(r.get("keyword_hits") or 0),
                "related": ["via:keyword_search"],
            })
        return items

    def _enrich_chunk_text_for_facts(self, title: str, text: str) -> str:
        body = (text or "").strip()
        urls = _extract_urls(body)
        if not urls:
            return body
        url_block = "\n".join(f"- {u}" for u in urls)
        return f"{body}\n\n[Extracted URLs]\n{url_block}".strip()

    def _structural_phrase_retrieve(self, session, query: str) -> list[dict]:
        """
        Direct phrase CONTAINS search for fact/URL questions vector search often misses.
        """
        phrases = self._search_phrases_from_query(query)
        if not phrases:
            return []

        doc_id, _ = self._resolve_document_for_query(session, query)
        rows = session.run(
            f"""
            MATCH (d:{DOCUMENT_ROOT_CYPHER})
            WHERE {_doc_scope_cypher("d")}
            MATCH (n)
            WHERE any(l IN labels(n) WHERE l IN $labels)
              AND coalesce(n.text, '') <> ''
              AND (
                EXISTS {{ MATCH (d)-[:CONTAINS*0..6]->(n) }}
                OR n.id STARTS WITH d.id + '_'
              )
              AND any(phrase IN $phrases WHERE toLower(n.text) CONTAINS phrase)
            WITH n, d,
              size([p IN $phrases WHERE toLower(n.text) CONTAINS p]) AS phrase_hits
            RETURN
              coalesce(n.id, '') AS id,
              coalesce(n.title, '') AS title,
              coalesce(n.text, '') AS text,
              phrase_hits,
              coalesce(d.title, d.id) AS doc_title
            ORDER BY phrase_hits DESC, size(coalesce(n.text, '')) ASC
            LIMIT 6
            """,
            doc_id=doc_id,
            phrases=[p.lower() for p in phrases],
            labels=list(_TEXT_NODE_LABELS),
        )

        items: list[dict] = []
        for r in rows:
            if not r.get("id"):
                continue
            title = r.get("title") or r["id"]
            text = self._enrich_chunk_text_for_facts(title, r.get("text") or "")
            score = 0.9 + 0.08 * int(r.get("phrase_hits") or 0)
            if len(text) < 1200:
                score += 0.12
            tl = text.lower()
            if re.search(r"language|translat", (query or "").lower()) and "available in" in tl:
                score += 0.15
            items.append({
                "id": r["id"],
                "title": title,
                "text": text,
                "score": score,
                "related": ["via:phrase_search"],
            })
        return items

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

    def _structural_page_visual_retrieve(self, session, query: str) -> list[dict]:
        """Page figures/diagrams via stored visual_content text (ingest vision enrichment)."""
        pdf_page, doc_page = self._parse_page_targets(query)
        if pdf_page is None and not doc_page:
            return []

        list_all_visuals = self._query_wants_all_page_visuals(query)
        doc_id, doc_title = self._resolve_document_for_query(session, query)
        row = session.run(
            f"""
            MATCH (d:{DOCUMENT_ROOT_CYPHER})
            WHERE {_doc_scope_cypher("d")}
            MATCH (p:Page)
            WHERE p.id STARTS WITH d.id + '_page_'
              AND (
                ($pdf_page IS NOT NULL AND p.pdf_page = $pdf_page)
                OR (
                  $doc_page IS NOT NULL
                  AND toLower(coalesce(p.document_page, '')) = toLower($doc_page)
                )
              )
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
              coalesce(p.text, '') AS page_text,
              coalesce(p.visual_content, '') AS page_visual,
              p.pdf_page AS pdf_page,
              p.document_page AS document_page,
              coalesce(d.title, d.id) AS doc_title,
              collect(DISTINCT {{
                id: r.id,
                title: coalesce(r.title, ''),
                text: coalesce(r.text, ''),
                kind: coalesce(r.region_kind, ''),
                visual_content: coalesce(r.visual_content, ''),
                order: coalesce(r.order, 0)
              }}) AS regions
            """,
            doc_id=doc_id,
            pdf_page=pdf_page,
            doc_page=doc_page,
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

    def _parse_page_targets(self, query: str) -> tuple[Optional[int], Optional[str]]:
        """Resolve PDF page index vs printed document page label from the question."""
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

    def _structural_page_retrieve(self, session, query: str) -> list[dict]:
        """
        Fetch Page node content by pdf_page / document_page for a resolved document.
        Works for any ingested PDF with Page nodes in the graph.
        """
        pdf_page, doc_page = self._parse_page_targets(query)
        if pdf_page is None and not doc_page:
            return []

        doc_id, doc_title = self._resolve_document_for_query(session, query)
        row = session.run(
            f"""
            MATCH (d:{DOCUMENT_ROOT_CYPHER})
            WHERE {_doc_scope_cypher("d")}
            MATCH (p:Page)
            WHERE p.id STARTS WITH d.id + '_page_'
              AND (
                ($pdf_page IS NOT NULL AND p.pdf_page = $pdf_page)
                OR (
                  $doc_page IS NOT NULL
                  AND toLower(coalesce(p.document_page, '')) = toLower($doc_page)
                )
              )
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
              coalesce(p.text, '') AS text,
              coalesce(p.visual_content, '') AS visual_content,
              p.pdf_page AS pdf_page,
              p.document_page AS document_page,
              coalesce(d.title, d.id) AS doc_title,
              collect(DISTINCT {{
                title: coalesce(r.title, ''),
                text: coalesce(r.text, ''),
                kind: coalesce(r.region_kind, '')
              }}) AS regions,
              collect(DISTINCT {{
                title: coalesce(s.title, ''),
                text: coalesce(s.text, '')
              }}) AS sections
            """,
            doc_id=doc_id,
            pdf_page=pdf_page,
            doc_page=doc_page,
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

        for sec in row.get("sections") or []:
            if not sec or not sec.get("title"):
                continue
            body = (sec.get("text") or "").strip()
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

    def _resolve_document_for_query_strict(
        self, session, query: str
    ) -> tuple[Optional[str], Optional[str]]:
        """
        Resolve the document a user named, scoring each logical document by how
        distinctively its content matches the query's document-name terms.

        Returns (None, None) when the query names a document that cannot be
        confidently resolved (no match, or an ambiguous near-tie), so the caller
        can ask the user to choose instead of silently guessing.

        Scoring (document-agnostic, no per-document special-casing):
          - For each term, count the DISTINCT content nodes per document whose
            title/text contains it (counts, not boolean — a doc that mentions a
            term many times beats one that mentions it once).
          - Weight each term by inverse document frequency: terms appearing in
            only one document are highly distinctive; terms appearing in every
            document (e.g. "data", "report") contribute almost nothing.
          - A title / logical-id match is treated as a very strong signal.
          - The winner must lead the runner-up clearly, else we return None.
        """
        terms = self._doc_name_terms(query)
        if not terms:
            return None, None

        lc = lifecycle_active("d")
        lc_n = lifecycle_active("n")
        rows = session.run(
            f"""
            MATCH (dl:{DOCUMENT_LOGICAL_LABEL})-[:ACTIVE_REVISION]
                  ->(:{DOC_REVISION_LABEL})-[:ROOT]->(d:{DOCUMENT_ROOT_CYPHER})
            WHERE {lc}
            WITH dl, d
            UNWIND $terms AS term
            OPTIONAL MATCH (d)-[:CONTAINS*1..6]->(n)
            WHERE {lc_n}
              AND (toLower(coalesce(n.title, '')) CONTAINS term
                   OR toLower(coalesce(n.text, '')) CONTAINS term)
            WITH dl, term, count(DISTINCT n) AS cnt,
                 (toLower(coalesce(dl.title, '')) CONTAINS term
                  OR toLower(dl.logical_id) CONTAINS term) AS title_match
            RETURN dl.logical_id AS id,
                   coalesce(dl.title, dl.logical_id) AS title,
                   collect({{term: term, cnt: cnt, title_match: title_match}}) AS term_hits
            """,
            terms=terms,
        )

        docs: list[dict] = [dict(r) for r in rows]
        if not docs:
            return None, None

        # Document frequency per term (how many docs contain it at all).
        term_doc_freq: dict[str, int] = {t: 0 for t in terms}
        for d in docs:
            for h in d["term_hits"]:
                if h["cnt"] > 0 or h["title_match"]:
                    term_doc_freq[h["term"]] = term_doc_freq.get(h["term"], 0) + 1

        total_docs = len(docs)

        def term_weight(term: str) -> float:
            df = term_doc_freq.get(term, 0)
            if df <= 0:
                return 0.0
            # Inverse document frequency: distinctive terms (df=1) weigh most.
            return float(total_docs) / float(df)

        scored: list[tuple[float, str, str]] = []
        for d in docs:
            score = 0.0
            for h in d["term_hits"]:
                w = term_weight(h["term"])
                if h["title_match"]:
                    score += 1000.0 * w  # title/id match dominates
                score += float(h["cnt"]) * w
            scored.append((score, str(d["id"]), _clean_doc_title(str(d["title"] or d["id"]))))

        scored.sort(key=lambda x: x[0], reverse=True)
        top = scored[0]
        if top[0] <= 0.0:
            return None, None  # no real match → caller clarifies

        # Require a clear lead over the runner-up to avoid guessing on near-ties.
        if len(scored) > 1:
            runner = scored[1][0]
            if runner > 0.0 and top[0] < runner * 1.5:
                return None, None  # ambiguous → caller clarifies

        return top[1], top[2]

    @staticmethod
    def _logical_id_from_node_id(node_id: str) -> Optional[str]:
        """Extract the logical document id prefix from a content node id."""
        if not node_id:
            return None
        # Revision-scoped ids look like "<logical_id>:<rev>::<...>".
        return node_id.split(":", 1)[0] or None

    def _resolve_document_by_vector(
        self, session, query: str
    ) -> tuple[Optional[str], Optional[str]]:
        """
        Resolve the target document via semantic similarity (corpus-agnostic):
        embed the query, take the top vector seeds, and pick the logical document
        that owns a clear majority of them. No per-document or per-topic terms.
        """
        try:
            embedding = self._get_embedding(query)
        except Exception:
            return None, None
        if not embedding:
            return None, None

        seeds = self._vector_seed(session, embedding, 12)
        if len(seeds) < 3:
            return None, None

        counts: dict[str, int] = {}
        for seed in seeds:
            lid = self._logical_id_from_node_id(seed.get("id") or "")
            if lid:
                counts[lid] = counts.get(lid, 0) + 1
        if not counts:
            return None, None

        ranked = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
        top_id, top_n = ranked[0]
        runner_n = ranked[1][1] if len(ranked) > 1 else 0
        # Require a clear majority over the runner-up to avoid guessing.
        if runner_n > 0 and top_n < runner_n * 1.5:
            return None, None
        if top_n < max(3, len(seeds) // 2):
            return None, None

        title = self._document_title_for_logical_id(session, top_id)
        return top_id, title

    def _document_title_for_logical_id(
        self, session, logical_id: str
    ) -> Optional[str]:
        if not logical_id:
            return None
        row = session.run(
            f"""
            MATCH (dl:{DOCUMENT_LOGICAL_LABEL})
            WHERE dl.logical_id = $lid
            RETURN coalesce(dl.title, dl.logical_id) AS title
            LIMIT 1
            """,
            lid=logical_id,
        ).single()
        if row and row.get("title"):
            return _clean_doc_title(str(row["title"]))
        return _clean_doc_title(logical_id)

    def _resolve_document_for_query(self, session, query: str) -> tuple[Optional[str], Optional[str]]:
        """Return logical document id (preferred) and display title for doc-scoped retrieval."""
        strict_id, strict_title = self._resolve_document_for_query_strict(session, query)
        if strict_id:
            return strict_id, strict_title

        vector_id, vector_title = self._resolve_document_by_vector(session, query)
        if vector_id:
            return vector_id, vector_title

        terms = self._document_match_terms(query)
        lc = lifecycle_active("d")
        lc_n = lifecycle_active("n")
        if terms:
            row = session.run(
                f"""
                UNWIND $terms AS term
                MATCH (dl:{DOCUMENT_LOGICAL_LABEL})
                WHERE toLower(coalesce(dl.title, '')) CONTAINS term
                   OR toLower(dl.logical_id) CONTAINS term
                   OR EXISTS {{
                     MATCH (dl)-[:ACTIVE_REVISION]->(:{DOC_REVISION_LABEL})
                           -[:ROOT]->(d:{DOCUMENT_ROOT_CYPHER})
                     WHERE {lc}
                     MATCH (d)-[:CONTAINS*1..5]->(n)
                     WHERE {lc_n}
                       AND (toLower(coalesce(n.title, '')) CONTAINS term
                            OR toLower(coalesce(n.text, '')) CONTAINS term)
                   }}
                RETURN dl.logical_id AS id, coalesce(dl.title, dl.logical_id) AS title,
                       count(*) AS hits
                ORDER BY hits DESC
                LIMIT 1
                """,
                terms=terms,
            ).single()
            if row and row.get("id"):
                return str(row["id"]), _clean_doc_title(str(row.get("title") or row["id"]))

            row = session.run(
                f"""
                UNWIND $terms AS term
                MATCH (d:{DOCUMENT_ROOT_CYPHER})
                WHERE {lc}
                  AND (toLower(coalesce(d.title, '')) CONTAINS term
                   OR EXISTS {{
                     MATCH (d)-[:CONTAINS*1..5]->(n)
                     WHERE {lc_n}
                       AND (toLower(coalesce(n.title, '')) CONTAINS term
                            OR toLower(coalesce(n.text, '')) CONTAINS term)
                   }})
                RETURN coalesce(d.logical_doc_id, d.id) AS id,
                       coalesce(d.title, d.id) AS title,
                       count(*) AS hits
                ORDER BY hits DESC
                LIMIT 1
                """,
                terms=terms,
            ).single()
            if row and row.get("id"):
                return str(row["id"]), _clean_doc_title(str(row.get("title") or row["id"]))

        row = session.run(
            f"""
            MATCH (dl:{DOCUMENT_LOGICAL_LABEL})-[:ACTIVE_REVISION]->(:{DOC_REVISION_LABEL})
                  -[:ROOT]->(d:{DOCUMENT_ROOT_CYPHER})-[:CONTAINS*1..4]->(s:Section)
            WHERE {lc} AND {lifecycle_active("s")}
            WITH dl, count(s) AS n
            ORDER BY n DESC
            LIMIT 1
            RETURN dl.logical_id AS id, coalesce(dl.title, dl.logical_id) AS title
            """
        ).single()
        if row and row.get("id"):
            return str(row["id"]), _clean_doc_title(str(row.get("title") or row["id"]))

        row = session.run(
            f"""
            MATCH (d:{DOCUMENT_ROOT_CYPHER})-[:CONTAINS*1..4]->(s:Section)
            WHERE {lc} AND {lifecycle_active("s")}
            WITH d, count(s) AS n
            ORDER BY n DESC
            LIMIT 1
            RETURN coalesce(d.logical_doc_id, d.id) AS id, coalesce(d.title, d.id) AS title
            """
        ).single()
        if row and row.get("id"):
            return str(row["id"]), _clean_doc_title(str(row.get("title") or row["id"]))
        return None, None

    def _document_match_terms(self, query: str) -> list[str]:
        terms: list[str] = list(_query_anchor_terms(query))
        for t in re.findall(r"[\w'-]{3,}", (query or "").lower()):
            if t in _KEYWORD_STOP:
                continue
            if t in {"table", "contents", "content", "provide", "list", "show", "give", "from", "form", "page", "fetch", "document"}:
                continue
            if t not in terms:
                terms.append(t)
        return terms[:6]

    def _doc_name_terms(self, query: str) -> list[str]:
        """
        Return only the high-confidence document-name tokens from a query.

        Unlike _document_match_terms (which adds generic keywords for broad matching),
        this returns anchor tokens and proper nouns from the question only.

        Used by the strict document resolver to avoid matching the wrong document
        via common words like "all", "toc", etc.
        """
        q_lower = (query or "").lower()
        terms: list[str] = list(_query_anchor_terms(query))

        # Tokens that are capitalised mid-sentence are likely proper nouns / doc names
        words = re.findall(r"[A-Za-z][\w'-]*", query or "")
        for i, w in enumerate(words):
            t = w.lower()
            if i == 0:
                continue  # skip sentence-start capitalisation
            if w[0].isupper() and len(t) >= 3 and t not in _KEYWORD_STOP and t not in terms:
                terms.append(t)

        # Long tokens (≥6 chars) that survived stop-word filtering are also good candidates
        _generic = {
            "table", "contents", "content", "provide", "list", "show", "give",
            "from", "form", "page", "fetch", "document", "summary", "about",
            "please", "could", "would", "should", "entire", "complete", "whole",
            "languages", "language", "online", "training", "course", "translated",
            "translation", "international", "regional", "national", "network",
            "networks", "epidemiology", "distinct", "explicitly", "mentioned",
            "partners", "collaborators", "document", "report", "annual",
        }
        for t in re.findall(r"[\w'-]{6,}", q_lower):
            if t in _KEYWORD_STOP or t in _generic:
                continue
            if t not in terms:
                terms.append(t)

        return terms[:6]

    def _query_keywords(self, question: str) -> list[str]:
        terms = re.findall(r"[\w'-]{3,}", (question or "").lower())
        return [t for t in terms if t not in _KEYWORD_STOP][:18]

    def _relevance_boost(self, title: str, text: str, keywords: list[str]) -> float:
        """Boost named sections and chunks that match more query terms."""
        boost = 1.0
        if title.strip() and not re.match(r"^Page\s+\d+$", title.strip(), re.I):
            boost *= 1.08
        hay = f"{title} {text}".lower()
        if keywords:
            hits = sum(1 for k in keywords if k in hay)
            boost *= 1.0 + min(0.45, 0.07 * hits)
        return boost

    def _fulltext_query(self, question: str) -> str:
        """Build a Lucene query from question terms (document-agnostic)."""
        phrases = self._search_phrases_from_query(question)
        if phrases:
            quoted = [f'"{p}"' for p in phrases[:5] if " " in p]
            terms = self._query_keywords(question)[:6]
            parts = quoted + terms
            if parts:
                return " OR ".join(parts)
        keywords = self._query_keywords(question)
        extra_stop = {"employees", "employee", "company", "corporate", "policy"}
        keywords = [k for k in keywords if k not in extra_stop][:14]
        if not keywords:
            return (question or "")[:120]
        return " OR ".join(keywords)

    def _resolve_document_id(self, session, name: str) -> Optional[str]:
        if not name:
            return None
        row = session.run(
            f"""
            MATCH (dl:{DOCUMENT_LOGICAL_LABEL})
            WHERE toLower(coalesce(dl.title, '')) CONTAINS toLower($name)
               OR toLower(dl.logical_id) CONTAINS toLower($name)
            RETURN dl.logical_id AS id
            LIMIT 1
            """,
            name=name.strip(),
        ).single()
        if row and row.get("id"):
            return str(row["id"])
        row = session.run(
            f"""
            MATCH (d:{DOCUMENT_ROOT_CYPHER})
            WHERE {lifecycle_active("d")}
              AND d.title IS NOT NULL
              AND toLower(d.title) CONTAINS toLower($name)
            RETURN coalesce(d.logical_doc_id, d.id) AS id
            LIMIT 1
            """,
            name=name.strip(),
        ).single()
        return str(row["id"]) if row and row.get("id") else None

    def _list_documents(self, session, limit: int = 5) -> list[dict[str, str]]:
        rows = session.run(
            f"""
            MATCH (dl:{DOCUMENT_LOGICAL_LABEL})
            RETURN dl.logical_id AS id, coalesce(dl.title, dl.logical_id) AS title
            ORDER BY title
            LIMIT $limit
            """,
            limit=max(1, int(limit)),
        )
        out: list[dict[str, str]] = []
        for r in rows:
            if r.get("id"):
                out.append({"id": str(r["id"]), "title": str(r.get("title") or r["id"])})
        if out:
            return out
        rows = session.run(
            f"""
            MATCH (d:{DOCUMENT_ROOT_CYPHER})
            WHERE {lifecycle_active("d")}
            RETURN coalesce(d.logical_doc_id, d.id) AS id, coalesce(d.title, d.id) AS title
            ORDER BY title
            LIMIT $limit
            """,
            limit=max(1, int(limit)),
        )
        out: list[dict[str, str]] = []
        for r in rows:
            if not r.get("id"):
                continue
            out.append({"id": str(r["id"]), "title": str(r.get("title") or r["id"])})
        return out

    def _structural_subsections(
        self,
        session,
        query: str,
        sec_num: str,
    ) -> tuple[list[dict], dict]:
        """Return (children items, parent item) for a numbered section like 2.5."""
        doc_id, _ = self._resolve_document_for_query(session, query)
        row = session.run(
            f"""
            MATCH (d:{DOCUMENT_ROOT_CYPHER})
            WHERE {_doc_scope_cypher("d")}
            MATCH (s:Section)
            WHERE (s.id STARTS WITH d.id + '_' OR EXISTS {{ MATCH (d)-[:CONTAINS*1..6]->(s) }})
              AND s.title IS NOT NULL
              AND trim(s.title) <> ''
              AND toLower(s.title) STARTS WITH toLower($sec_num)
            WITH s
            OPTIONAL MATCH (s)-[:CONTAINS]->(c:Section)
            WHERE c.title IS NOT NULL AND trim(c.title) <> ''
            RETURN
              s.id AS sid,
              s.title AS stitle,
              coalesce(s.text,'') AS stext,
              collect({{id: c.id, title: c.title, text: coalesce(c.text,'')}}) AS children
            LIMIT 1
            """,
            doc_id=doc_id,
            sec_num=sec_num,
        ).single()
        if not row:
            return [], {}

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
        return items, parent

    def _structural_box_headings(self, session, query: str) -> list[dict]:
        """
        Enumerate Box headings (e.g. "Box 10") inside a document.
        Generic: works for any document that contains "Box <number>" in titles or text.
        """
        doc_id, doc_title = self._resolve_document_for_query(session, query)
        rows = session.run(
            f"""
            MATCH (d:{DOCUMENT_ROOT_CYPHER})
            WHERE {_doc_scope_cypher("d")}
            MATCH (n)
            WHERE any(l IN labels(n) WHERE l IN $labels)
              AND (
                EXISTS {{ MATCH (d)-[:CONTAINS*0..6]->(n) }}
                OR n.id STARTS WITH d.id + '_'
              )
              AND (
                (n.title IS NOT NULL AND toLower(n.title) CONTAINS 'box')
                OR (n.text IS NOT NULL AND toLower(n.text) CONTAINS 'box')
              )
            RETURN
              coalesce(n.id,'') AS id,
              coalesce(n.title,'') AS title,
              coalesce(n.text,'') AS text
            LIMIT 250
            """,
            doc_id=doc_id,
            labels=list(_TEXT_NODE_LABELS),
        )

        found: dict[int, dict] = {}
        for r in rows:
            rid = r.get("id") or ""
            title = (r.get("title") or "").strip()
            text = (r.get("text") or "").strip()
            hay = f"{title}\n{text}"
            for num in self._exec.extract_box_numbers(hay):
                if num in found:
                    continue
                # Prefer title if it contains Box, else synthesize a heading.
                heading = title if re.search(rf"(?i)\\bbox\\s+{num}\\b", title) else f"Box {num}"
                snippet = ""
                if text:
                    # keep a compact preview
                    snippet = text[:800]
                found[num] = {
                    "id": rid or f"box_{num}",
                    "title": heading,
                    "text": snippet,
                    "score": 1.0,
                    "related": [f"doc:{doc_title}" if doc_title else "doc:unknown", "via:box_scan"],
                }

        return [found[k] for k in sorted(found.keys())]

    def _structural_box_content(self, session, query: str, box_n: int) -> list[dict]:
        """
        Retrieve content for a specific Box N (e.g. Box 5).
        Looks for nodes whose title/text mention the box, then returns the best matches.
        """
        doc_id, doc_title = self._resolve_document_for_query(session, query)
        box_phrase = f"box {int(box_n)}"
        rows = session.run(
            f"""
            MATCH (d:{DOCUMENT_ROOT_CYPHER})
            WHERE {_doc_scope_cypher("d")}
            MATCH (n)
            WHERE any(l IN labels(n) WHERE l IN $labels)
              AND (
                EXISTS {{ MATCH (d)-[:CONTAINS*0..6]->(n) }}
                OR n.id STARTS WITH d.id + '_'
              )
              AND (
                (n.title IS NOT NULL AND toLower(n.title) CONTAINS $box_phrase)
                OR (n.text IS NOT NULL AND toLower(n.text) CONTAINS $box_phrase)
              )
            RETURN
              coalesce(n.id,'') AS id,
              coalesce(n.title,'') AS title,
              coalesce(n.text,'') AS text
            LIMIT 20
            """,
            doc_id=doc_id,
            box_phrase=box_phrase,
            labels=list(_TEXT_NODE_LABELS),
        )

        items: list[dict] = []
        for r in rows:
            rid = r.get("id") or ""
            title = (r.get("title") or "").strip()
            text = (r.get("text") or "").strip()
            if not rid or not (title or text):
                continue
            # Prefer chunks whose title explicitly contains Box N.
            score = 1.0
            if re.search(rf"(?i)\\bbox\\s+{box_n}\\b", title):
                score = 1.08
            elif re.search(rf"(?i)\\bbox\\s+{box_n}\\b", text[:200]):
                score = 1.02
            # Keep a larger snippet since user asked "all the data".
            snippet = text[:2500] if text else ""
            items.append({
                "id": rid,
                "title": title or f"Box {box_n}",
                "text": snippet,
                "score": score,
                "related": [f"doc:{doc_title}" if doc_title else "doc:unknown", "via:box_content"],
            })

        items.sort(
            key=lambda x: (float(x.get("score", 0.0)), len(x.get("text") or "")),
            reverse=True,
        )
        if items and len((items[0].get("text") or "")) < 200:
            page_items = self._box_content_from_page_text(session, query, box_n, doc_id)
            if page_items:
                return page_items
        return items[:6]

    def _box_content_from_page_text(
        self,
        session,
        query: str,
        box_n: int,
        doc_id: Optional[str],
    ) -> list[dict]:
        """Fallback when Box sections in Neo4j only store the label (pre-fix ingest)."""
        _, doc_title = self._resolve_document_for_query(session, query)
        rows = session.run(
            f"""
            MATCH (d:{DOCUMENT_ROOT_CYPHER})
            WHERE {_doc_scope_cypher("d")}
            MATCH (p:Page)
            WHERE p.id STARTS WITH d.id + '_page_'
              AND toLower(coalesce(p.text, '')) CONTAINS $box_phrase
            RETURN p.id AS id, p.title AS title, p.text AS text, p.pdf_page AS pdf_page
            ORDER BY size(coalesce(p.text, '')) DESC
            LIMIT 3
            """,
            doc_id=doc_id,
            box_phrase=f"box {int(box_n)}",
        )
        for r in rows:
            page_text = (r.get("text") or "").strip()
            if not page_text:
                continue
            extracted = self._extract_box_snippet_from_page(page_text, box_n)
            if len(extracted) < 80:
                continue
            return [{
                "id": r.get("id") or f"box_{box_n}_page",
                "title": f"Box {box_n}",
                "text": f"Box {box_n}\n\n{extracted}"[:4000],
                "score": 1.1,
                "related": [
                    f"doc:{doc_title}" if doc_title else "doc:unknown",
                    "via:box_page_fallback",
                ],
                "pdf_page": r.get("pdf_page"),
            }]
        return []

    @staticmethod
    def _extract_box_snippet_from_page(page_text: str, box_n: int) -> str:
        target = str(int(box_n))
        lines = page_text.splitlines()
        start: int | None = None
        for i, ln in enumerate(lines):
            m = re.match(r"^\s*Box\s+(\d+(?:\.\d+)?)", ln.strip(), re.I)
            if m and m.group(1).split(".")[0] == target:
                start = i
                break
        if start is None:
            return ""
        body: list[str] = []
        for ln in lines[start + 1 :]:
            if re.match(r"^\s*Box\s+\d+", ln.strip(), re.I):
                break
            body.append(ln)
        return "\n".join(body).strip()

    def _get_embedding(self, text: str) -> list[float]:
        resp = provider.embeddings(model=EMBEDDING_MODEL, input=(text or "")[:8000])
        return list(resp.data[0].embedding)

    def _access_denied_response(self, query: str, ctx: UserContext) -> Optional[dict[str, Any]]:
        if self.rbac.can_query_knowledge_area(ctx.user_id, "esg"):
            return None
        return {
            "query": query,
            "chunks": [
                {
                    "id": "access_denied",
                    "title": "Access Denied",
                    "text": f"User {ctx.user_id} does not have permission to query Agentic Graph RAG data.",
                    "score": 0.0,
                    "related": [],
                }
            ],
            "total_available": 0,
            "_access_level": ctx.role.value,
            "_user_id": ctx.user_id,
        }

    def _format_response(
        self,
        query: str,
        items: list[dict],
        user_context: Optional[UserContext] = None,
    ) -> dict[str, Any]:
        ctx = user_context or self.user_context
        _passthrough = ("pdf_page", "document_page", "region_kind", "visual_content")
        return {
            "query": query,
            "chunks": [
                {
                    "id": r.get("id", ""),
                    "title": r.get("title", ""),
                    "text": r.get("text", ""),
                    "score": round(float(r.get("score", 0.0)), 3),
                    "related": r.get("related", []),
                    **{k: r[k] for k in _passthrough if r.get(k) is not None},
                }
                for r in items
            ],
            "total_available": len(items),
            "_access_level": ctx.role.value,
            "_user_id": ctx.user_id,
        }


ESGComplianceRetriever = DocumentRAGRetriever
RAGDataRetriever = DocumentRAGRetriever
