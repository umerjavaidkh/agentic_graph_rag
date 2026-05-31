"""
Hybrid retrieval: gather related sections (vector + keywords + graph structure),
rerank, then pass the best chunks to the LLM.
"""
from __future__ import annotations

import json
import re
from typing import Dict, List, Optional

import numpy as np
from dotenv import load_dotenv
from neo4j import GraphDatabase

from ..auth.rbac_setup import GraphRBAC
from ..auth.roles import DEFAULT_PUBLIC_CONTEXT, UserContext
from ..config.settings import (
    MODEL_PROVIDER,
    OPENAI_API_KEY,
    RETRIEVAL_CANDIDATE_POOL,
    RETRIEVAL_FINAL_LIMIT,
    RETRIEVAL_MIN_RERANK_SCORE,
    NEO4J_URI,
    NEO4J_USER,
    NEO4J_PASSWORD,
)
from ..assets.page_images import resolve_image_url
from ..document.page_numbers import (
    is_valid_document_page_label,
    parse_page_number_from_query,
)
from ..document.page_vision import compact_visual_content
from ..document.patterns import TABLE_REF_PATTERN
from ..graph.constants import DOCUMENT_ROOT_CYPHER
from ..model_providers.factory import get_model_provider
from .visual_retrieval import (
    best_region_for_visual_focus,
    display_text_for_chunk,
    is_strict_page_lookup,
    parse_visual_intent,
    score_visual_candidate,
    wants_page_text,
)

load_dotenv()
provider = get_model_provider(MODEL_PROVIDER, OPENAI_API_KEY)

STOPWORDS = {
    "what", "are", "the", "that", "this", "with", "for", "and", "or",
    "in", "of", "to", "a", "an", "is", "it", "section", "document",
    "from", "do", "does", "did", "can", "could", "would", "should",
    "will", "have", "has", "had", "be", "been", "being", "give", "them",
    "their", "they", "we", "our", "us", "me", "my", "i", "about", "any",
    "all", "how", "when", "where", "which", "who", "why", "please",
}

# region_tags is stored as a Neo4j list — match tags with ANY(), not toString(list)
def _cypher_needle_match(var: str = "n") -> str:
    return f"""(
                toLower(coalesce({var}.title, '') + ' ' + coalesce({var}.text, '')
                  + ' ' + coalesce({var}.visual_content, '')) CONTAINS needle
                OR ANY(tag IN coalesce({var}.region_tags, []) WHERE toLower(tag) CONTAINS needle)
            )"""


class DocumentRAGRetriever:
    def __init__(
        self,
        uri=NEO4J_URI,
        user=NEO4J_USER,
        password=NEO4J_PASSWORD,
        user_context: Optional[UserContext] = None,
    ):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self._vector_index_ready = False
        self.user_context = user_context or DEFAULT_PUBLIC_CONTEXT
        self.rbac = GraphRBAC(uri, user, password)

    def semantic_retrieve(
        self,
        query: str,
        limit: int = RETRIEVAL_FINAL_LIMIT,
        user_context: Optional[UserContext] = None,
    ) -> Dict:
        """Broad hybrid retrieval + rerank (primary path for Q&A)."""
        return self.hybrid_retrieve(query, limit=limit, user_context=user_context)

    def hybrid_retrieve(
        self,
        query: str,
        limit: int = RETRIEVAL_FINAL_LIMIT,
        user_context: Optional[UserContext] = None,
    ) -> Dict:
        ctx = user_context or self.user_context
        denied = self._access_denied_response(query, ctx)
        if denied:
            return denied

        query_embedding = self._get_embedding(query)
        terms = self._extract_search_terms(query)
        pool_size = max(limit * 3, RETRIEVAL_CANDIDATE_POOL)

        if self._asks_for_subsections(query):
            return self._subsection_tree_retrieve(query, limit=limit, user_context=ctx)

        intent = parse_visual_intent(query, self._extract_search_terms)
        if self._should_use_unified_visual(intent, query):
            visual_result = self.unified_visual_retrieve(
                query, limit=max(limit, 5), user_context=ctx, intent=intent
            )
            if visual_result.get("chunks"):
                return visual_result

        if self._is_visual_element_query(query):
            with self.driver.session() as session:
                visual_rows = self._visual_page_search(
                    session, terms, query, limit=max(limit, 5)
                )
            table_result = self._table_reference_retrieve(
                query, limit=max(limit, 5), user_context=ctx
            )
            if visual_rows:
                pool = {c["id"]: c for c in table_result.get("chunks", [])}
                for row in visual_rows:
                    row["match_types"] = ["visual_page"]
                    pool[row["id"]] = {
                        "id": row["id"],
                        "title": row["title"],
                        "text": row["text"],
                        "cluster": row.get("cluster"),
                        "score": row.get("score", 1.0),
                        "related": [],
                        "match_types": ["visual_page"],
                    }
                chunks = sorted(pool.values(), key=lambda c: -c.get("score", 0))[:limit]
                return {
                    **table_result,
                    "chunks": chunks,
                    "total_available": len(chunks),
                    "mode": "table_reference+visual",
                }
            if table_result.get("chunks"):
                return table_result

        with self.driver.session() as session:
            pool: dict[str, dict] = {}

            def add_rows(rows: list, match_type: str, base_score: float = 0.0) -> None:
                for row in rows:
                    sid = row["id"]
                    if sid in pool:
                        pool[sid]["match_types"].add(match_type)
                        pool[sid]["score"] = max(pool[sid].get("score", 0), row.get("score", base_score))
                    else:
                        pool[sid] = {
                            **row,
                            "match_types": {match_type},
                            "score": row.get("score", base_score),
                        }

            vector_rows = self._vector_search(session, query_embedding, pool_size, ctx)
            if not vector_rows:
                vector_rows = self._legacy_similarity(session, query_embedding, pool_size, ctx)
            for row in vector_rows:
                row["match_type"] = "vector"
            add_rows(vector_rows, "vector")

            add_rows(self._keyword_section_search(session, terms, pool_size), "keyword", 0.45)
            add_rows(self._structural_section_search(session, terms, query), "structural", 0.55)
            add_rows(
                self._visual_page_search(session, terms, query, min(limit, 10)),
                "visual",
                0.58,
            )

            seed_ids = [r["id"] for r in sorted(vector_rows, key=lambda x: -x.get("score", 0))[:5]]
            add_rows(self._expand_structure(session, seed_ids), "graph_expand", 0.4)
            add_rows(self._semantic_neighbors(session, seed_ids, hops=1), "semantic_neighbor", 0.35)

            ranked = self._rerank_candidates(query, query_embedding, list(pool.values()), limit)
            ranked = self._enrich_context_batch(session, ranked)

        return self._format_response(
            query,
            ranked,
            user_context=ctx,
            meta={"candidates_pooled": len(pool), "candidates_returned": len(ranked)},
        )

    def structural_retrieve(
        self,
        query: str,
        limit: int = 15,
        user_context: Optional[UserContext] = None,
    ) -> Dict:
        """
        Title / hierarchy focused retrieval for section listing and subsection questions.
        """
        ctx = user_context or self.user_context
        denied = self._access_denied_response(query, ctx)
        if denied:
            return denied

        if self._asks_for_subsections(query):
            return self._subsection_tree_retrieve(query, limit=limit, user_context=ctx)

        focus, document_hint = self._section_query_focus(query)
        with self.driver.session() as session:
            parent = self._best_parent_section(session, focus, document_hint=document_hint)
            if parent and not self._asks_for_subsections(query):
                return self.section_content_retrieve(query, limit=limit, user_context=ctx)

        terms = self._extract_search_terms(query)
        with self.driver.session() as session:
            rows = self._structural_section_search(session, terms, query, child_limit=25)
            for row in rows:
                row.setdefault("score", 0.7)
                mt = row.get("match_type", "structural")
                row["match_types"] = {mt}
            ranked = self._rerank_candidates(query, self._get_embedding(query), rows, limit)
            ranked = self._enrich_context_batch(session, ranked)

        return self._format_response(
            query,
            ranked,
            user_context=ctx,
            meta={"mode": "structural"},
        )

    def section_content_retrieve(
        self,
        query: str,
        limit: int = 8,
        user_context: Optional[UserContext] = None,
    ) -> Dict:
        """
        User wants the body/content of a named section (not a list of child headings).
        e.g. "What can you tell me about 6 IMPLEMENTATION AT STRATEC?"
        """
        ctx = user_context or self.user_context
        denied = self._access_denied_response(query, ctx)
        if denied:
            return denied

        focus, document_hint = self._section_query_focus(query)
        with self.driver.session() as session:
            parent = self._best_parent_section(session, focus, document_hint=document_hint)
            if parent:
                rows = [parent]
                body = (parent.get("text") or "").strip()
                if len(body) < 400:
                    rows.extend(
                        self._fetch_section_subtree(
                            session,
                            parent["id"],
                            parent.get("doc_order", 0),
                            parent.get("title", ""),
                            max_sections=max(limit, 12),
                        )
                    )
                for row in rows:
                    row.setdefault("score", 1.0)
                    row["match_types"] = {row.get("match_type", "section_content")}
                ranked = sorted(rows, key=lambda r: r.get("doc_order", 0))[:limit]
                ranked = self._enrich_context_batch(session, ranked)
                return self._format_response(
                    query,
                    ranked,
                    user_context=ctx,
                    meta={
                        "mode": "section_content",
                        "focus_section_id": parent["id"],
                        "parent_title": parent.get("title"),
                        "document_hint": document_hint,
                    },
                )

        # Fallback: hybrid pool when title match fails
        hybrid = self.hybrid_retrieve(query, limit=limit, user_context=ctx)
        hybrid["mode"] = "section_content_fallback"
        return hybrid

    _INTRO_TITLE_HINTS = (
        "introduction",
        "executive",
        "summary",
        "overview",
        "background",
        "preface",
        "foreword",
        "about",
        "purpose",
        "scope",
        "annual report",
    )

    def document_overview_retrieve(
        self,
        query: str,
        limit: int = 12,
        user_context: Optional[UserContext] = None,
    ) -> Dict:
        """
        Whole-document questions: pull introduction / early sections / Go.Data front matter,
        not random mid-report subsections.
        """
        ctx = user_context or self.user_context
        denied = self._access_denied_response(query, ctx)
        if denied:
            return denied

        q_lower = query.lower()
        pool: dict[str, dict] = {}

        def add_rows(rows: list, match_type: str, base_score: float) -> None:
            for row in rows:
                sid = row["id"]
                sc = float(row.get("score", base_score))
                if sid in pool:
                    pool[sid]["match_types"].add(match_type)
                    pool[sid]["score"] = max(pool[sid].get("score", 0), sc)
                else:
                    pool[sid] = {
                        **row,
                        "match_types": {match_type},
                        "score": sc,
                    }

        terms = self._extract_search_terms(query)
        query_embedding = self._get_embedding(query)

        with self.driver.session() as session:
            intro = session.run(
                """
                MATCH (s:Section)
                WHERE ANY(h IN $hints WHERE toLower(s.title) CONTAINS h)
                  AND size(coalesce(s.text, '')) > 60
                RETURN s.id AS id, s.title AS title, s.text AS text,
                       s.cluster_id AS cluster, coalesce(s.order, 0) AS doc_order,
                       0.9 AS score
                ORDER BY s.order ASC
                LIMIT 10
                """,
                hints=list(self._INTRO_TITLE_HINTS),
            )
            add_rows([r.data() for r in intro], "intro_section", 0.9)

            early = session.run(
                """
                MATCH (s:Section)
                WHERE size(coalesce(s.text, '')) > 120
                RETURN s.id AS id, s.title AS title, s.text AS text,
                       s.cluster_id AS cluster, coalesce(s.order, 0) AS doc_order,
                       0.75 AS score
                ORDER BY coalesce(s.order, 9999) ASC
                LIMIT 10
                """,
            )
            add_rows([r.data() for r in early], "early_section", 0.75)

            if "go.data" in q_lower or "godata" in q_lower:
                gd = session.run(
                    """
                    MATCH (s:Section)
                    WHERE toLower(s.title) CONTAINS 'go.data'
                       OR toLower(s.text) CONTAINS 'go.data'
                    RETURN s.id AS id, s.title AS title, s.text AS text,
                           s.cluster_id AS cluster, coalesce(s.order, 0) AS doc_order,
                           0.82 AS score
                    ORDER BY s.order ASC
                    LIMIT 12
                    """,
                )
                add_rows([r.data() for r in gd], "godata_section", 0.82)

            add_rows(
                self._keyword_section_search(session, terms, max(limit, 15)),
                "keyword",
                0.5,
            )

            vector_rows = self._vector_search(
                session, query_embedding, max(limit, 12), ctx
            )
            if not vector_rows:
                vector_rows = self._legacy_similarity(
                    session, query_embedding, max(limit, 12), ctx
                )
            add_rows(vector_rows, "vector", 0.6)

            ranked = self._rerank_candidates(
                query, query_embedding, list(pool.values()), limit
            )
            ranked = self._enrich_context_batch(session, ranked)

        return self._format_response(
            query,
            ranked,
            user_context=ctx,
            meta={"mode": "document_overview", "candidates_pooled": len(pool)},
        )

    def section_detail_retrieve(
        self,
        section_id: str,
        user_context: Optional[UserContext] = None,
        parent_section_id: Optional[str] = None,
    ) -> Dict:
        """Fetch one Section node (and optional parent title) for subsection drill-down."""
        ctx = user_context or self.user_context
        denied = self._access_denied_response(query="section_detail", ctx=ctx)
        if denied:
            return denied

        with self.driver.session() as session:
            row = session.run(
                """
                MATCH (s:Section {id: $sid})
                RETURN s.id AS id, s.title AS title, s.text AS text,
                       s.cluster_id AS cluster, s.order AS doc_order,
                       s.page_start AS page_start, s.page_end AS page_end
                LIMIT 1
                """,
                sid=section_id,
            ).single()

            if not row:
                return self._format_response(
                    "section_detail",
                    [],
                    user_context=ctx,
                    meta={"mode": "section_detail", "found": False},
                )

            chunk = row.data()
            chunk["match_types"] = ["section_detail"]
            chunk["rerank_score"] = 1.0
            chunk["score"] = 1.0

            parent_title = None
            if parent_section_id:
                pr = session.run(
                    """
                    MATCH (p:Section {id: $pid})
                    RETURN p.title AS title LIMIT 1
                    """,
                    pid=parent_section_id,
                ).single()
                if pr:
                    parent_title = pr["title"]

            ranked = self._enrich_context_batch(session, [chunk])

        return self._format_response(
            "section_detail",
            ranked,
            user_context=ctx,
            meta={
                "mode": "section_detail",
                "found": True,
                "parent_id": parent_section_id,
                "parent_title": parent_title,
                "focus_section_id": section_id,
            },
        )

    def _subsection_tree_retrieve(
        self,
        query: str,
        limit: int = 15,
        user_context: Optional[UserContext] = None,
    ) -> Dict:
        """
        Find the best-matching parent section by title overlap, then return it plus
        all descendant sections (CONTAINS*), in document order.
        """
        ctx = user_context or self.user_context
        with self.driver.session() as session:
            parent = self._best_parent_section(session, query)
            if not parent:
                terms = self._extract_search_terms(query)
                rows = self._structural_section_search(session, terms, query, child_limit=25)
            else:
                rows = [parent]
                rows.extend(
                    self._fetch_descendant_sections(
                        session, parent["id"], max_depth=3, max_sections=max(limit - 1, 8)
                    )
                )

            for row in rows:
                row.setdefault("score", 1.0)
                mt = row.get("match_type", "subsection_tree")
                row["match_types"] = {mt}
                row["rerank_score"] = 1.0

            ranked = sorted(rows, key=lambda r: r.get("doc_order", 0))[:limit]
            ranked = self._enrich_context_batch(session, ranked)

        return self._format_response(
            query,
            ranked,
            user_context=ctx,
            meta={
                "mode": "subsection_tree",
                "parent_id": parent["id"] if parent else None,
                "parent_title": parent.get("title") if parent else None,
            },
        )

    def _extract_document_hint(self, query: str) -> Optional[str]:
        """Parse document name from TOC/section queries (tolerates 'form' typo for 'from')."""
        if not query:
            return None
        q = query.strip()
        doc_m = re.search(
            r"\b(?:from|form)\s+(?:the\s+)?(.+?)\s+document\b",
            q,
            re.I,
        )
        if doc_m:
            hint = doc_m.group(1).strip(" ?.")
            hint = re.sub(
                r"^(?:give\s+me\s+)?(?:all\s+)?(?:the\s+)?(?:full\s+)?(?:list\s+of\s+)?"
                r"(?:all\s+)?(?:table\s+of\s+contents?|toc)\s*",
                "",
                hint,
                flags=re.I,
            ).strip(" ?.")
            if hint and hint.lower() not in ("document", "pdf", "report"):
                return hint
        of_m = re.search(r"\bof\s+(?:the\s+)?(.+?)\s+document\b", q, re.I)
        if of_m:
            return of_m.group(1).strip(" ?.")
        q_compact = re.sub(r"[^a-z0-9]", "", q.lower())
        named = (
            ("setratec", "Stratec"),
            ("stratec", "Stratec"),
            ("godata", "Go.Data"),
            ("godataannual", "Go.Data"),
        )
        for key, label in named:
            if key in q_compact:
                return label
        if re.search(r"\bgo\.?\s*data\b", q, re.I):
            return "Go.Data"
        return None

    def _query_requests_named_document(self, query: str) -> bool:
        if self._extract_document_hint(query):
            return True
        q = query.lower()
        return bool(
            re.search(r"\b(?:from|form)\s+.+?\s+document\b", q)
            or re.search(r"\bgo\.?\s*data\b", q)
            or "stratec" in re.sub(r"[^a-z0-9]", "", q)
        )

    def _document_match_keys(self, hint: str) -> list[str]:
        """Normalize document name from query (typos: Setratec → stratec)."""
        compact = re.sub(r"[^a-z0-9]", "", (hint or "").lower())
        if not compact:
            return []
        keys = [compact]
        alias_map = {
            "setratec": ["stratec"],
            "stratec": ["stratec"],
            "godata": ["godata", "go"],
            "go": ["godata", "go"],
        }
        if compact in alias_map:
            keys = alias_map[compact] + keys
        if "godata" in compact or compact.startswith("go"):
            keys.extend(["godata", "go"])
        if "stratec" in compact:
            keys.append("stratec")
        return list(dict.fromkeys(k for k in keys if k))

    def _documents_with_sections(self, session) -> list[dict]:
        return [
            r.data()
            for r in session.run(
                """
                MATCH (b:Document|Book)-[:CONTAINS*1..20]->(s:Section)
                WITH b, count(s) AS n
                WHERE n > 0
                RETURN b.id AS id, coalesce(b.title, b.id) AS title, n AS sections
                ORDER BY title
                """
            )
        ]

    def _resolve_document_id(self, session, document_hint: str) -> Optional[str]:
        """Pick one Book/Document id for a user hint — no cross-document mixing."""
        keys = self._document_match_keys(document_hint)
        if not keys:
            return None

        for key in keys:
            row = session.run(
                """
                MATCH (b:Document|Book)
                WHERE toLower(replace(b.id, '_', ' ')) CONTAINS $key
                   OR toLower(coalesce(b.title, '')) CONTAINS $key
                RETURN b.id AS id
                ORDER BY size(coalesce(b.title, b.id)) ASC
                LIMIT 1
                """,
                key=key,
            ).single()
            if row:
                return row["id"]

        row = session.run(
            """
            MATCH (b:Document|Book)-[:CONTAINS*1..20]->(s:Section)
            WITH b,
                 count(s) AS total,
                 sum(
                   CASE
                     WHEN ANY(k IN $keys WHERE
                       toLower(s.title) CONTAINS k
                       OR toLower(coalesce(s.text, '')) CONTAINS k
                       OR replace(toLower(coalesce(s.text, '')), '.', '') CONTAINS k
                     ) THEN 1
                     ELSE 0
                   END
                 ) AS hits
            WHERE hits >= 2
            RETURN b.id AS id, hits, total
            ORDER BY hits DESC, total DESC
            LIMIT 1
            """,
            keys=keys,
        ).single()
        if row:
            return row["id"]
        return None

    def _filter_toc_rows_for_document_hint(
        self, rows: list[dict], document_hint: str
    ) -> list[dict]:
        """
        When multiple PDFs were ingested with legacy shared section ids (section_0_1),
        strip obvious cross-document headings from a single-doc TOC.
        """
        keys = self._document_match_keys(document_hint)
        if not keys:
            return rows

        godata_markers = (
            "go.data", "godata", "epiet", "openwho", "dhis2", "outbreak",
            "cox's bazar", "viet nam", "ukraine adaptation", "measles",
            "github", "kpi", "field epidemiology", "who regional",
        )
        stratec_markers = (
            "stratec", "compliance management", "whistle-blow", "corruption",
            "mission statement", "preamble", "antitrust", "money laundering",
            "ethics as an employer",
        )

        if any(k in keys for k in ("stratec", "setratec")):
            filtered = [
                r for r in rows
                if not any(m in (r.get("title") or "").lower() for m in godata_markers)
            ]
            return filtered if filtered else rows

        if any(k in keys for k in ("godata", "go")):
            filtered = [
                r for r in rows
                if not any(m in (r.get("title") or "").lower() for m in stratec_markers)
                or any(m in (r.get("title") or "").lower() for m in godata_markers)
            ]
            return filtered if filtered else rows

        return rows

    def _toc_sections_for_document(self, session, doc_id: str, document_hint: str = "") -> list[dict]:
        rows = [
            r.data()
            for r in session.run(
                """
                MATCH (b:Document|Book {id: $doc_id})-[:CONTAINS*1..20]->(s:Section)
                RETURN DISTINCT s.id AS id, s.title AS title, s.order AS order,
                       s.page_start AS page, s.cluster_id AS cluster
                ORDER BY s.order
                """,
                doc_id=doc_id,
            )
        ]
        if document_hint:
            rows = self._filter_toc_rows_for_document_hint(rows, document_hint)
        return rows

    def get_table_of_contents(
        self,
        query: str = "",
        user_context: Optional[UserContext] = None,
    ) -> Dict:
        ctx = user_context or self.user_context
        denied = self._access_denied_response(query="table_of_contents", ctx=ctx)
        if denied:
            return {**denied, "query": "table_of_contents", "chunks": []}

        _, document_hint = self._section_query_focus(query) if query else (query, None)
        document_hint = document_hint or self._extract_document_hint(query or "")
        if not document_hint and query:
            of_m = re.search(r"\bof\s+(?:the\s+)?(.+?)\s+document\b", query.strip(), re.I)
            if of_m:
                document_hint = of_m.group(1).strip()
        anchor_title = self._toc_anchor_title(query, document_hint=document_hint)
        wants_one_doc = bool(document_hint) or self._query_requests_named_document(query or "")

        with self.driver.session() as session:
            available = self._documents_with_sections(session)
            doc_id: Optional[str] = None
            resolved_title: Optional[str] = None

            if document_hint:
                doc_id = self._resolve_document_id(session, document_hint)
                if doc_id:
                    for d in available:
                        if d["id"] == doc_id:
                            resolved_title = d["title"]
                            break
                    if document_hint and (
                        not resolved_title
                        or resolved_title == doc_id
                        or re.search(r"^[a-f0-9]{32}", resolved_title or "", re.I)
                        or str(resolved_title).endswith("_rag_document")
                    ):
                        resolved_title = document_hint.strip().title()
                rows = (
                    self._toc_sections_for_document(session, doc_id, document_hint)
                    if doc_id
                    else []
                )
            elif wants_one_doc:
                rows = []
            else:
                rows = [
                    r.data()
                    for r in session.run(
                        """
                        MATCH (b:Document|Book)-[:CONTAINS*1..20]->(s:Section)
                        RETURN DISTINCT s.id AS id, s.title AS title, s.order AS order,
                               s.page_start AS page, s.cluster_id AS cluster
                        ORDER BY s.order
                        """
                    )
                ]

            if anchor_title and rows:
                anchor = self._best_title_match(rows, anchor_title)
                if anchor is not None:
                    rows = [r for r in rows if r["order"] >= anchor["order"]]

        meta: dict = {
            "anchor": anchor_title,
            "document_hint": document_hint,
        }
        if document_hint and not doc_id:
            meta["document_not_found"] = True
            meta["available_documents"] = [
                f"{d['title']} ({d['sections']} sections)" for d in available
            ]
        elif wants_one_doc and not doc_id and not document_hint:
            meta["document_not_found"] = True
            meta["document_hint"] = "named document"
            meta["available_documents"] = [
                f"{d['title']} ({d['sections']} sections)" for d in available
            ]

        return {
            "query": "table_of_contents",
            "mode": "table_of_contents",
            "document_id": doc_id,
            "document_title": resolved_title,
            "chunks": [
                {
                    "id": r["id"],
                    "title": r["title"],
                    "text": f"{r['title']} (Page {r['page']})" if r.get("page") else r["title"],
                    "page": r.get("page"),
                    "cluster": r["cluster"],
                    "related": [],
                    "score": 1.0,
                }
                for r in rows
            ],
            "total_available": len(rows),
            **meta,
        }

    def get_all_sections(self, user_context: Optional[UserContext] = None) -> Dict:
        """Backward-compatible wrapper — prefer get_table_of_contents(query=...)."""
        return self.get_table_of_contents(query="", user_context=user_context)

    def _toc_anchor_title(
        self, query: str, document_hint: Optional[str] = None
    ) -> Optional[str]:
        """
        Parse 'table of contents from MISSION STATEMENT' → section anchor title.
        'from Stratec document' is a document scope (handled elsewhere), not an anchor.
        """
        q = query.strip()
        if re.search(r"\b(?:from|form)\s+(?:the\s+)?.+?\s+document\b", q, re.I):
            return None
        if document_hint and re.search(
            rf"\bfrom\s+(?:the\s+)?{re.escape(document_hint.strip())}\s*$",
            q,
            re.I,
        ):
            return None
        for prefix in (r"\bfrom\s+", r"\bform\s+"):  # form = common typo
            m = re.search(
                prefix + r"(?:the\s+)?(.+?)(?:\s+of\s+[a-z0-9_\-\s]+)?\s*$",
                q,
                re.I,
            )
            if m:
                anchor = m.group(1).strip(" ?.")
                if anchor.lower() in ("document", "pdf", "report", "manual"):
                    return None
                return anchor or None
        return None

    def _best_title_match(self, rows: list, title: str) -> Optional[dict]:
        q = title.lower().strip()
        best: Optional[dict] = None
        best_score = 0.0
        for row in rows:
            t = (row.get("title") or "").lower()
            score = 0.0
            if t == q:
                score = 1.0
            elif q in t or t in q:
                score = 0.85
            else:
                q_words = [w for w in q.split() if len(w) > 2]
                if q_words:
                    score = sum(1 for w in q_words if w in t) / len(q_words)
            if score > best_score:
                best_score = score
                best = row
        return best if best_score >= 0.5 else None

    def multi_hop_retrieve(
        self,
        query: str,
        limit: int = RETRIEVAL_FINAL_LIMIT,
        hops: int = 2,
        user_context: Optional[UserContext] = None,
    ) -> Dict:
        """Hybrid pool first, then optional semantic graph expansion on top seeds."""
        result = self.hybrid_retrieve(query, limit=limit, user_context=user_context)
        seed_ids = [c["id"] for c in result["chunks"][:5]]
        if not seed_ids:
            return result

        ctx = user_context or self.user_context
        with self.driver.session() as session:
            expanded = self._semantic_neighbors(session, seed_ids, hops=hops)
            pool = {c["id"]: {**c, "match_types": {"hybrid"}} for c in result["chunks"]}
            for row in expanded:
                if row["id"] not in pool:
                    pool[row["id"]] = {**row, "match_types": {"semantic_neighbor"}}
            ranked = self._rerank_candidates(
                query, self._get_embedding(query), list(pool.values()), limit
            )
            ranked = self._enrich_context_batch(session, ranked)

        result["chunks"] = [
            {
                "id": r["id"],
                "title": r["title"],
                "text": r["text"],
                "cluster": r.get("cluster"),
                "score": round(r.get("rerank_score", r.get("score", 0)), 3),
                "related": r.get("related", []),
                "match_types": list(r.get("match_types", [])),
            }
            for r in ranked
        ]
        result["total_available"] = len(ranked)
        result["expanded"] = len(expanded)
        return result

    def close(self) -> None:
        self.driver.close()

    def _should_use_unified_visual(self, intent, query: str) -> bool:
        """Only when the user asked for a page #, figure/image, or visual scene — not long text Q&A."""
        if intent.pdf_page is not None or intent.document_page:
            return True
        if intent.list_all:
            return True
        if self._is_figure_caption_query(query):
            return True
        if self._is_visual_scene_query(query):
            return True
        return False

    def unified_visual_retrieve(
        self,
        query: str,
        limit: int = 5,
        user_context: Optional[UserContext] = None,
        intent=None,
    ) -> Dict:
        """
        Single path: page #, caption, scene, list-all figures.
        Scores Pages/Regions by vision text (photo) > printed text (caption).
        """
        ctx = user_context or self.user_context
        denied = self._access_denied_response(query, ctx)
        if denied:
            return denied

        intent = intent or parse_visual_intent(query, self._extract_search_terms)

        if wants_page_text(query):
            with self.driver.session() as session:
                chunks = self._page_text_chunks(session, intent)
            top = chunks[0] if chunks else {}
            return self._format_response(
                query,
                chunks,
                user_context=ctx,
                meta={
                    "mode": "page_text",
                    "found": bool(chunks),
                    "pdf_page": intent.pdf_page or top.get("pdf_page"),
                    "document_page": intent.document_page or top.get("document_page"),
                },
            )

        if is_strict_page_lookup(intent):
            with self.driver.session() as session:
                chunks = self._strict_page_visual_chunks(session, intent, query)
            top = chunks[0] if chunks else {}
            return self._format_response(
                query,
                chunks,
                user_context=ctx,
                meta={
                    "mode": "page_lookup",
                    "found": bool(chunks),
                    "pdf_page": intent.pdf_page or top.get("pdf_page"),
                    "document_page": intent.document_page or top.get("document_page"),
                    "single_visual": intent.single_visual,
                    "visual_focus": intent.visual_focus[:5],
                },
            )

        needles = list(dict.fromkeys(intent.terms + intent.phrases))[:14]
        if not needles and intent.pdf_page is None:
            return self._format_response(
                query, [], user_context=ctx, meta={"mode": "unified_visual", "found": False}
            )

        with self.driver.session() as session:
            page_row = self._resolve_page_row(session, intent)
            candidates: dict[str, dict] = {}

            if page_row:
                candidates[page_row["id"]] = {**page_row, "node_label": "Page"}
                for reg in self._fetch_regions_for_page(session, page_row["id"]):
                    reg["node_label"] = "Region"
                    candidates[reg["id"]] = reg

            if needles and intent.pdf_page is None and not intent.document_page:
                needle_match = _cypher_needle_match("n")
                result = session.run(
                    f"""
                    MATCH (n)
                    WHERE (n:Page OR n:Region)
                      AND ANY(needle IN $needles WHERE {needle_match})
                    RETURN n.id AS id, labels(n)[0] AS node_label,
                           coalesce(n.title, '') AS title, n.text AS text,
                           n.visual_content AS visual_content,
                           n.image_key AS image_key, n.region_kind AS region_kind,
                           n.region_tags AS region_tags, n.pdf_page AS pdf_page,
                           n.document_page AS document_page,
                           coalesce(n.order, 0) AS doc_order
                    LIMIT 50
                    """,
                    needles=needles,
                )
                for row in result:
                    d = row.data()
                    candidates.setdefault(d["id"], d)

            # Document-wide region scan only when NOT pinned to a single page number
            if intent.wants_image and intent.pdf_page is None and not intent.document_page:
                result = session.run(
                    """
                    MATCH (r:Region)
                    WHERE r.image_key IS NOT NULL
                    OPTIONAL MATCH (p:Page)
                    WHERE p.pdf_page = r.pdf_page OR p.id = 'page_' + toString(r.pdf_page)
                    RETURN r.id AS id, labels(r)[0] AS node_label,
                           r.title AS title, r.text AS text,
                           r.visual_content AS visual_content,
                           r.image_key AS image_key, r.region_kind AS region_kind,
                           r.region_tags AS region_tags, r.pdf_page AS pdf_page,
                           r.document_page AS document_page, coalesce(r.order, 0) AS doc_order,
                           p.visual_content AS page_visual, p.text AS page_text
                    LIMIT 80
                    """,
                )
                for row in result:
                    d = row.data()
                    if d.get("page_visual") and not d.get("visual_content"):
                        d["visual_content"] = d["page_visual"]
                    if d.get("page_text") and len((d.get("text") or "")) < 80:
                        d["text"] = (d.get("text") or "") + " " + d["page_text"]
                    candidates.setdefault(d["id"], d)

            scored_rows: list[tuple[float, dict]] = []
            for row in candidates.values():
                label = row.get("node_label", "Page")
                if intent.kind_filter and row.get("region_kind") != intent.kind_filter:
                    if label == "Region":
                        continue
                s = score_visual_candidate(intent, row, node_label=label)
                if s <= 0 and intent.pdf_page is None:
                    continue
                scored_rows.append((s, row))

            scored_rows.sort(key=lambda x: (-x[0], x[1].get("doc_order", 0)))

            if intent.list_all and page_row:
                chunks = self._visual_list_chunks(
                    session, page_row, intent, limit=max(limit, 10)
                )
                mode = "page_visual_list"
            elif not scored_rows:
                chunks = []
                mode = "unified_visual"
            else:
                chunks = self._visual_best_chunks(
                    session, scored_rows, intent, limit=limit
                )
                mode = "unified_visual"

            ranked = self._rerank_visual_by_embedding(
                query, chunks, limit, pin_pdf_page=intent.pdf_page if intent.pdf_page is not None else None
            )
            ranked = self._enrich_context_batch(session, ranked)

        top = ranked[0] if ranked else {}
        return self._format_response(
            query,
            ranked,
            user_context=ctx,
            meta={
                "mode": mode,
                "found": bool(ranked),
                "pdf_page": top.get("pdf_page"),
                "document_page": top.get("document_page"),
                "phrases": intent.phrases[:5],
                "list_kind": intent.kind_filter if mode == "page_visual_list" else None,
            },
        )

    def _page_text_chunks(self, session, intent) -> list[dict]:
        """Full page text for 'all text from page N' queries — no figure crop."""
        page_row = self._resolve_page_row(session, intent)
        if not page_row:
            return []

        pdf_p = intent.pdf_page or page_row.get("pdf_page")
        parts: list[str] = []
        title = page_row.get("title") or f"Page {pdf_p}"
        page_body = (page_row.get("text") or "").strip()
        if page_body:
            parts.append(page_body)

        if pdf_p is not None:
            result = session.run(
                """
                MATCH (s:Section)
                WHERE s.page_start IS NOT NULL AND s.page_end IS NOT NULL
                  AND s.page_start <= $p AND s.page_end >= $p
                RETURN s.title AS title, s.text AS text, s.order AS ord
                ORDER BY s.order ASC
                """,
                p=int(pdf_p),
            )
            seen = {page_body[:200]} if page_body else set()
            for row in result:
                data = row.data()
                sec_text = (data.get("text") or "").strip()
                if not sec_text or sec_text[:200] in seen:
                    continue
                seen.add(sec_text[:200])
                sec_title = (data.get("title") or "").strip()
                if sec_title and sec_title.lower() != title.lower():
                    parts.append(f"### {sec_title}\n\n{sec_text}")
                elif sec_text != page_body:
                    parts.append(sec_text)

        full_text = "\n\n".join(parts).strip()
        ch = self._attach_media_fields({
            "id": page_row["id"],
            "title": title,
            "text": full_text or page_body or "(No text extracted for this page.)",
            "pdf_page": pdf_p,
            "document_page": page_row.get("document_page"),
            "page_tags": page_row.get("page_tags"),
            "visual_content": page_row.get("visual_content"),
            "match_types": ["page_text"],
            "rerank_score": 1.0,
            "image_key": None,
            "image_url": None,
        })
        return [ch]

    def _strict_page_visual_chunks(
        self,
        session,
        intent,
        query: str,
    ) -> list[dict]:
        """One page only: no document-wide region scan or cross-page rerank."""
        page_row = self._resolve_page_row(session, intent)
        if not page_row:
            return []

        page_data = self._attach_media_fields(dict(page_row))
        regions = self._fetch_regions_for_page(session, page_data["id"])

        if intent.visual_focus:
            reg = best_region_for_visual_focus(
                regions, intent, page_data.get("visual_content")
            )
            if reg:
                ch = self._page_chunk_with_region(page_data, reg)
                ch["match_types"] = ["page_lookup", "visual_focus"]
                ch["rerank_score"] = 1.0
                ch["text"] = display_text_for_chunk(ch)
                return [ch]
            if intent.single_visual and page_data.get("image_key"):
                ch = dict(page_data)
                ch["image_source"] = "page"
                ch["text"] = display_text_for_chunk(ch)
                ch["match_types"] = ["page_lookup", "visual_focus"]
                ch["rerank_score"] = 0.5
                return [ch]

        if self._wants_whole_page_image(query) and page_data.get("image_key"):
            ch = dict(page_data)
            ch["image_source"] = "page"
            ch["text"] = display_text_for_chunk(ch)
            ch["match_types"] = ["page_lookup"]
            ch["rerank_score"] = 1.0
            return [ch]

        matched = self._match_regions_to_query(
            query, regions, page_data.get("visual_content")
        )
        if matched and matched[0].get("image_key"):
            ch = self._page_chunk_with_region(page_data, matched[0])
            ch["match_types"] = ["page_lookup"]
            ch["rerank_score"] = 1.0
            return [ch]

        pick_regions = regions
        if intent.kind_filter:
            pick_regions = self._filter_regions_by_kind(regions, intent.kind_filter)

        reg = self._first_region_with_image(pick_regions)
        if reg:
            ch = self._page_chunk_with_region(page_data, reg)
            ch["match_types"] = ["page_lookup"]
            ch["rerank_score"] = 1.0
            return [ch]

        if page_data.get("image_key"):
            ch = dict(page_data)
            ch["image_source"] = "page"
            ch["text"] = display_text_for_chunk(ch)
            ch["match_types"] = ["page_lookup"]
            ch["rerank_score"] = 1.0
            return [ch]

        return []

    def _resolve_page_row(self, session, intent) -> Optional[dict]:
        if intent.document_page:
            row = session.run(
                """
                MATCH (p:Page)
                WHERE p.document_page IS NOT NULL
                  AND toLower(trim(p.document_page)) = toLower(trim($label))
                RETURN p.id AS id, p.title AS title, p.text AS text,
                       p.visual_content AS visual_content,
                       p.pdf_page AS pdf_page, p.document_page AS document_page,
                       p.page_tags AS page_tags, p.image_key AS image_key,
                       p.order AS doc_order
                LIMIT 1
                """,
                label=intent.document_page,
            ).single()
            if row:
                return row.data()
            # Printed label missing in graph — try same number as PDF index
            if str(intent.document_page).strip().isdigit():
                intent.pdf_page = int(str(intent.document_page).strip())
                intent.document_page = None
        if intent.pdf_page is not None:
            row = session.run(
                """
                MATCH (p:Page)
                WHERE p.pdf_page = $pdf OR (p.pdf_page IS NULL AND p.order = $pdf)
                RETURN p.id AS id, p.title AS title, p.text AS text,
                       p.visual_content AS visual_content,
                       p.pdf_page AS pdf_page, p.document_page AS document_page,
                       p.page_tags AS page_tags, p.image_key AS image_key,
                       p.order AS doc_order
                LIMIT 1
                """,
                pdf=intent.pdf_page,
            ).single()
            if row:
                return row.data()
        return None

    def _dedupe_regions_for_list(self, regions: list[dict]) -> list[dict]:
        """Drop duplicate crops (same image_key); keep distinct figures on one page."""
        seen_keys: set[str] = set()
        out: list[dict] = []
        for reg in sorted(regions, key=lambda r: r.get("doc_order", 0)):
            key = reg.get("image_key")
            if key:
                if key in seen_keys:
                    continue
                seen_keys.add(key)
            out.append(reg)
        return out

    def _visual_list_chunks(
        self, session, page_row: dict, intent, limit: int
    ) -> list[dict]:
        regions = self._fetch_regions_for_page(session, page_row["id"])
        if intent.kind_filter:
            regions = self._filter_regions_by_kind(regions, intent.kind_filter)
        regions = self._dedupe_regions_for_list(regions)
        chunks = []
        for i, reg in enumerate(
            sorted(regions, key=lambda r: r.get("doc_order", 0))[:limit]
        ):
            if intent.wants_image and not reg.get("image_key"):
                continue
            ch = self._page_chunk_with_region(page_row, reg)
            ch["text"] = display_text_for_chunk(ch)
            ch["match_types"] = ["unified_visual", "page_visual_list"]
            ch["rerank_score"] = 1.0 - i * 0.02
            chunks.append(self._attach_media_fields(ch))
        return chunks

    def _visual_best_chunks(
        self,
        session,
        scored_rows: list[tuple[float, dict]],
        intent,
        limit: int,
    ) -> list[dict]:
        chunks: list[dict] = []
        seen_images: set[str] = set()

        for score, row in scored_rows:
            if len(chunks) >= limit:
                break
            if intent.pdf_page is not None:
                rp = row.get("pdf_page") or row.get("doc_order")
                if rp is not None and int(rp) != int(intent.pdf_page):
                    continue
            label = row.get("node_label", "Page")
            if label == "Region" and row.get("image_key"):
                key = row["image_key"]
                if key in seen_images and intent.pdf_page is None:
                    continue
                seen_images.add(key)
                page_data = self._page_data_for_region(session, row)
                ch = self._page_chunk_with_region(page_data, row)
            else:
                ch = self._attach_media_fields(dict(row))
                ch["image_source"] = "page"
            ch["text"] = display_text_for_chunk(ch)
            ch["_visual_score"] = score
            ch["match_types"] = ["unified_visual"]
            ch["rerank_score"] = float(score)
            chunks.append(ch)

        if not chunks and scored_rows:
            _, row = scored_rows[0]
            ch = self._attach_media_fields(dict(row))
            ch["text"] = display_text_for_chunk(ch)
            ch["match_types"] = ["unified_visual"]
            chunks.append(ch)
        return chunks

    def _page_data_for_region(self, session, region: dict) -> dict:
        rid = region["id"]
        page_match = session.run(
            """
            MATCH (p:Page)-[:CONTAINS]->(r {id: $rid})
            RETURN p.id AS id, p.title AS title, p.text AS text,
                   p.visual_content AS visual_content,
                   p.pdf_page AS pdf_page, p.document_page AS document_page,
                   p.image_key AS image_key, p.order AS doc_order
            LIMIT 1
            """,
            rid=rid,
        ).single()
        if page_match:
            return page_match.data()
        pdf_p = region.get("pdf_page") or region.get("doc_order")
        return {
            "id": f"page_{pdf_p}",
            "title": f"Page {pdf_p}",
            "pdf_page": pdf_p,
            "text": region.get("page_text") or "",
            "visual_content": region.get("page_visual") or region.get("visual_content"),
        }

    def _rerank_visual_by_embedding(
        self, query: str, chunks: list[dict], limit: int, *, pin_pdf_page: Optional[int] = None
    ) -> list[dict]:
        if pin_pdf_page is not None:
            chunks = [
                c for c in chunks
                if c.get("pdf_page") is None or int(c.get("pdf_page")) == int(pin_pdf_page)
            ]
        if len(chunks) <= 1:
            return chunks[:limit]
        try:
            q_emb = self._get_embedding(query)
        except Exception:
            return chunks[:limit]
        for ch in chunks:
            blob = display_text_for_chunk(ch) or ch.get("title", "")
            if len(blob) < 8:
                ch["rerank_score"] = ch.get("rerank_score", 0)
                continue
            try:
                c_emb = self._get_embedding(blob[:4000])
                sim = self._cosine_similarity(q_emb, c_emb)
            except Exception:
                sim = 0.0
            base = float(ch.get("_visual_score", ch.get("rerank_score", 0)))
            ch["rerank_score"] = base * 0.35 + sim * 0.65
        chunks.sort(key=lambda c: -c.get("rerank_score", 0))
        return chunks[:limit]

    def _is_visual_scene_query(self, query: str) -> bool:
        q = query.lower()
        pdf_p, doc_p = parse_page_number_from_query(query)
        if pdf_p is not None or (doc_p and is_valid_document_page_label(doc_p)):
            return False
        scene_hints = (
            "lady", "woman", "man", "person", "people", "holding", "phone",
            "screenshot", "photo", "picture", "image where", "image of",
            "showing", "illustration",
        )
        if not any(h in q for h in scene_hints):
            return False
        return bool(
            re.search(r"\b(image|picture|photo|screenshot|page|pdf)\b", q)
            or "holding" in q
        )

    def _wants_whole_page_image(self, query: str) -> bool:
        q = query.lower()
        return bool(
            re.search(r"\b(whole|full|entire)\s+(pdf\s+)?page\b", q)
            or re.search(r"\bpdf\s+page\s+data\b", q)
            or re.search(r"\bpage\s+data\b", q)
        )

    def page_lookup_retrieve(
        self,
        query: str,
        pdf_page: Optional[int] = None,
        document_page: Optional[str] = None,
        user_context: Optional[UserContext] = None,
    ) -> Dict:
        """Delegates to unified_visual_retrieve (page # + image + caption)."""
        intent = parse_visual_intent(query, self._extract_search_terms)
        if pdf_page is not None:
            intent.pdf_page = pdf_page
        if document_page:
            intent.document_page = document_page
        if intent.single_visual:
            lim = 1
        elif intent.list_all:
            lim = 12
        else:
            lim = 5
        return self.unified_visual_retrieve(
            query, limit=lim, user_context=user_context, intent=intent
        )

    def caption_figure_retrieve(
        self,
        query: str,
        limit: int = 3,
        user_context: Optional[UserContext] = None,
    ) -> Dict:
        return self.unified_visual_retrieve(
            query, limit=limit, user_context=user_context
        )

    def visual_scene_retrieve(
        self,
        query: str,
        limit: int = 3,
        user_context: Optional[UserContext] = None,
    ) -> Dict:
        return self.unified_visual_retrieve(
            query, limit=limit, user_context=user_context
        )

    def _is_visual_element_query(self, query: str) -> bool:
        """Tables, charts, diagrams, figures, maps, and other page visuals."""
        if TABLE_REF_PATTERN.search(query):
            return True
        q = query.lower()
        visual_terms = (
            "chart", "graph", "diagram", "flowchart", "flow chart",
            "figure", "fig.", "fig ", "map", "illustration", "shape",
            "plot", "visual", "screenshot", "infographic", "drawing",
        )
        if any(t in q for t in visual_terms):
            return True
        if "table" in q:
            return True
        return False

    def _asks_list_all_visuals(self, query: str) -> bool:
        """List every table/figure/etc. on a page (not a single numbered one)."""
        q = query.lower()
        if not re.search(r"\b(all|every|each|list|show\s+all)\b", q):
            return False
        return bool(
            re.search(
                r"\b(figures?|figs?\.?|tables?|charts?|diagrams?|visuals?|images?)\b",
                q,
            )
        )

    def _visual_kind_filter(self, query: str) -> Optional[str]:
        q = query.lower()
        if re.search(r"\b(figures?|figs?\.?)\b", q):
            return "figure"
        if re.search(r"\btables?\b", q):
            return "table"
        if re.search(r"\b(charts?|graphs?)\b", q):
            return "figure"
        if re.search(r"\b(diagrams?|flowcharts?)\b", q):
            return "figure"
        return None

    def _filter_regions_by_kind(
        self, regions: list[dict], kind_filter: Optional[str]
    ) -> list[dict]:
        if not kind_filter:
            return regions
        filtered = [
            r for r in regions
            if (r.get("region_kind") or "").lower() == kind_filter
        ]
        if filtered:
            return filtered
        return [
            r for r in regions
            if kind_filter in self._region_search_blob(r)
        ]

    def _region_to_list_chunk(self, page_data: dict, region: dict, index: int) -> dict:
        chunk = {
            "id": region.get("id") or f"{page_data['id']}_r{index}",
            "title": region.get("title") or f"Region {index}",
            "text": (region.get("text") or region.get("title") or "")[:800],
            "pdf_page": region.get("pdf_page") or page_data.get("pdf_page"),
            "document_page": region.get("document_page") or page_data.get("document_page"),
            "region_kind": region.get("region_kind"),
            "region_tags": region.get("region_tags"),
            "image_key": region.get("image_key"),
            "visual_content": None,
            "doc_order": region.get("doc_order", index),
            "score": 1.0 - index * 0.01,
            "rerank_score": 1.0 - index * 0.01,
            "match_types": ["page_visual_list"],
            "image_source": "region" if region.get("image_key") else None,
        }
        return self._attach_media_fields(chunk)

    def _figures_from_visual_content(self, page_data: dict) -> list[dict]:
        """Fallback when Region nodes are missing: parse Figure N lines from vision text."""
        visual = compact_visual_content((page_data.get("visual_content") or "").strip())
        if not visual:
            return []

        entries: list[dict] = []
        for m in re.finditer(
            r"(?:^|\n)\s*(?:[-*]\s*)?(Figure\s+(\d+(?:\.\d+)?)[^\n]*)",
            visual,
            re.IGNORECASE,
        ):
            line = m.group(1).strip()
            num = m.group(2)
            entries.append({
                "id": f"{page_data['id']}_vision_fig_{num}",
                "title": f"Figure {num}",
                "text": line,
                "region_kind": "figure",
                "region_tags": [f"figure:{num}", f"pdf:{page_data.get('pdf_page')}"],
                "image_key": page_data.get("image_key"),
                "pdf_page": page_data.get("pdf_page"),
                "document_page": page_data.get("document_page"),
                "doc_order": int(float(num)) if num.replace(".", "", 1).isdigit() else len(entries) + 1,
            })
        return entries

    def _build_visual_list_chunks(
        self,
        session,
        page_data: dict,
        regions: list[dict],
        kind_filter: Optional[str],
    ) -> list[dict]:
        filtered = self._filter_regions_by_kind(regions, kind_filter)
        if not filtered and kind_filter == "figure":
            filtered = self._figures_from_visual_content(page_data)

        chunks = [
            self._region_to_list_chunk(page_data, r, i)
            for i, r in enumerate(
                sorted(filtered, key=lambda x: x.get("doc_order", 0))
            )
        ]
        if not chunks:
            return []
        return self._enrich_context_batch(session, chunks)

    def _is_specific_visual_request(self, query: str) -> bool:
        """User asked for a particular table/figure/chart, not just 'page N'."""
        if self._asks_list_all_visuals(query):
            return False
        if re.search(r"\b(figure|fig\.?|table)\s+\d", query, re.I):
            return True
        if self._is_visual_element_query(query):
            return True
        return bool(
            re.search(
                r"\b(image|picture|photo|screenshot|scan|show|display|see)\b",
                query,
                re.I,
            )
            and re.search(
                r"\b(table|figure|fig|chart|diagram|graph|map|picture|image)\s+[a-z0-9]",
                query,
                re.I,
            )
        )

    def _fetch_regions_for_page(self, session, page_id: str) -> list[dict]:
        result = session.run(
            """
            MATCH (p:Page {id: $pid})-[:CONTAINS]->(r:Region)
            RETURN r.id AS id, r.title AS title, r.text AS text,
                   r.visual_content AS visual_content,
                   r.region_kind AS region_kind, r.region_tags AS region_tags,
                   r.image_key AS image_key, r.pdf_page AS pdf_page,
                   r.document_page AS document_page, r.order AS doc_order,
                   coalesce(r.order, 0) AS score
            ORDER BY r.order ASC
            """,
            pid=page_id,
        )
        rows = [r.data() for r in result]
        for row in rows:
            self._attach_media_fields(row)
        return rows

    def _first_region_with_image(self, regions: list[dict]) -> Optional[dict]:
        for row in sorted(regions, key=lambda r: r.get("doc_order", 0)):
            if row.get("image_key"):
                return row
        return None

    def _region_search_blob(self, region: dict) -> str:
        tags = region.get("region_tags") or []
        tag_text = " ".join(tags) if isinstance(tags, list) else str(tags)
        return " ".join([
            region.get("title") or "",
            region.get("text") or "",
            region.get("region_kind") or "",
            tag_text,
        ]).lower()

    def _visual_needles_from_query(
        self, query: str, visual_content: Optional[str] = None
    ) -> list[str]:
        needles: list[str] = []
        needles.extend(self._table_needles_from_query(query))
        for term in self._extract_search_terms(query):
            if len(term) > 2:
                needles.append(term.lower())

        q = query.lower()
        for pat in (
            (r"\bfigure\s+(\d+(?:\.\d+)?)\b", "figure"),
            (r"\bfig\.?\s+(\d+(?:\.\d+)?)\b", "figure"),
            (r"\bchart\s+(\d+(?:\.\d+)?)\b", "chart"),
            (r"\bdiagram\s+(\d+(?:\.\d+)?)\b", "diagram"),
        ):
            for m in re.finditer(pat[0], q):
                needles.append(m.group(0).strip())
                if pat[1] == "figure":
                    needles.append(f"figure:{m.group(1).lower()}")

        if visual_content:
            vc = visual_content.lower()
            for ref in TABLE_REF_PATTERN.findall(vc):
                needles.append(f"table {ref.lower()}")
                needles.append(f"table:{ref.lower()}")
            for m in re.finditer(r"\bfigure\s+(\d+(?:\.\d+)?)\b", vc):
                needles.append(m.group(0).strip())
                needles.append(f"figure:{m.group(1).lower()}")

        return list(dict.fromkeys(n for n in needles if n))[:12]

    def _match_regions_to_query(
        self,
        query: str,
        regions: list[dict],
        visual_content: Optional[str] = None,
    ) -> list[dict]:
        needles = self._visual_needles_from_query(query, visual_content)
        if not needles:
            return []

        scored: list[tuple[int, dict]] = []
        for region in regions:
            blob = self._region_search_blob(region)
            tags = region.get("region_tags") or []
            tag_blob = " ".join(tags).lower() if isinstance(tags, list) else str(tags).lower()
            hits = 0
            for needle in needles:
                if needle in blob or needle in tag_blob:
                    hits += 3 if needle.startswith("table:") or needle.startswith("figure:") else 1
            if hits:
                scored.append((hits, region))

        scored.sort(key=lambda x: (-x[0], x[1].get("doc_order", 0)))
        return [r for _, r in scored]

    def _page_chunk_with_region(self, page: dict, region: dict) -> dict:
        item = dict(page)
        item["title"] = region.get("title") or item.get("title")
        if region.get("image_key"):
            item["image_key"] = region["image_key"]
            item["image_source"] = "region"
        else:
            item["image_source"] = "page"
        item["region_kind"] = region.get("region_kind")
        item["region_tags"] = region.get("region_tags")
        item["matched_region_id"] = region.get("id")
        item["text"] = display_text_for_chunk(item)
        return self._attach_media_fields(item)

    def _caption_phrase_from_query(self, query: str) -> Optional[str]:
        """Long figure/table caption text before trailing 'search for figure…'."""
        q = re.sub(
            r"\s*(?:search|find|show|locate)\s+(?:for\s+)?(?:that\s+)?"
            r"(?:figure|fig\.?|table|image).*",
            "",
            query,
            flags=re.I,
        ).strip()
        if len(q) >= 30:
            return q.lower()[:200]
        return None

    def _is_figure_caption_query(self, query: str) -> bool:
        if not self._caption_phrase_from_query(query):
            return False
        return bool(re.search(r"\b(figure|fig\.?|image|photo|picture)\b", query, re.I))

    def _table_needles_from_query(self, query: str) -> list[str]:
        caption = self._caption_phrase_from_query(query)
        if caption:
            return [caption]
        needles = [f"table {ref}".lower() for ref in TABLE_REF_PATTERN.findall(query)]
        if needles:
            return needles
        if "table" in query.lower():
            return [query.lower().strip()[:100]]
        return []

    def _table_reference_retrieve(
        self,
        query: str,
        limit: int = 5,
        user_context: Optional[UserContext] = None,
    ) -> Dict:
        """Exact table caption search on Section + Page nodes."""
        ctx = user_context or self.user_context
        denied = self._access_denied_response(query, ctx)
        if denied:
            return denied

        needles = self._table_needles_from_query(query)
        extra = [
            t for t in self._extract_search_terms(query)
            if len(t) > 4 and t not in needles
        ]
        needles = list(dict.fromkeys(needles + extra[:6]))

        with self.driver.session() as session:
            rows = self._table_reference_search(session, needles, limit=max(limit, 5))
            for row in rows:
                row["match_types"] = {"table_match"}
                row["rerank_score"] = row.get("score", 1.0)
            ranked = [self._with_display_text(r) for r in rows]
            ranked = sorted(ranked, key=lambda r: (-r.get("score", 0), r.get("doc_order", 0)))[:limit]
            ranked = self._enrich_context_batch(session, ranked)

        return self._format_response(
            query,
            ranked,
            user_context=ctx,
            meta={"mode": "table_reference", "needles": needles},
        )

    def _table_reference_search(
        self, session, needles: list[str], limit: int
    ) -> list:
        if not needles:
            return []
        needle_match = _cypher_needle_match("n")
        result = session.run(
            f"""
            MATCH (n)
            WHERE (n:Section OR n:Page OR n:Region)
              AND ANY(needle IN $needles WHERE {needle_match})
            WITH n,
                 [needle IN $needles WHERE {needle_match}] AS hits
            WHERE size(hits) > 0
            RETURN n.id AS id,
                   coalesce(n.title, head(labels(n))) AS title,
                   n.text AS text,
                   n.visual_content AS visual_content,
                   n.image_key AS image_key,
                   n.region_kind AS region_kind,
                   n.region_tags AS region_tags,
                   n.pdf_page AS pdf_page,
                   n.document_page AS document_page,
                   n.cluster_id AS cluster,
                   coalesce(n.order, 0) AS doc_order,
                   (size(hits) * 3) AS score
            ORDER BY score DESC, size(n.text) ASC
            LIMIT $limit
            """,
            needles=needles,
            limit=limit,
        )
        return [
            self._attach_media_fields(self._with_display_text(r.data()))
            for r in result
        ]

    def _visual_page_search(
        self, session, terms: list[str], query: str, limit: int
    ) -> list:
        """Search Page.visual_content (vision-extracted tables/figures)."""
        needles = list(terms)
        needles.extend(self._table_needles_from_query(query))
        needles = list(dict.fromkeys(n for n in needles if n))[:10]
        if not needles:
            return []

        region_match = _cypher_needle_match("r")
        result = session.run(
            f"""
            MATCH (r:Region)
            WHERE r.image_key IS NOT NULL
              AND ANY(needle IN $needles WHERE {region_match})
            WITH r,
                 [needle IN $needles WHERE {region_match}] AS hits
            WHERE size(hits) > 0
            RETURN r.id AS id,
                   r.title AS title,
                   r.text AS text,
                   r.region_kind AS region_kind,
                   r.region_tags AS region_tags,
                   r.image_key AS image_key,
                   r.pdf_page AS pdf_page,
                   r.document_page AS document_page,
                   null AS cluster,
                   coalesce(r.order, 0) AS doc_order,
                   (size(hits) * 5) AS score
            ORDER BY score DESC
            LIMIT $limit
            """,
            needles=needles,
            limit=limit,
        )
        region_rows = [self._attach_media_fields(self._with_display_text(r.data())) for r in result]
        for row in region_rows:
            row["match_type"] = "visual_region"

        result = session.run(
            """
            MATCH (p:Page)
            WHERE p.visual_content IS NOT NULL
              AND ANY(needle IN $needles WHERE
                  toLower(p.visual_content) CONTAINS needle)
            WITH p,
                 [needle IN $needles WHERE
                  toLower(p.visual_content) CONTAINS needle] AS hits
            WHERE size(hits) > 0
            RETURN p.id AS id,
                   p.title AS title,
                   p.text AS text,
                   p.visual_content AS visual_content,
                   p.image_key AS image_key,
                   p.pdf_page AS pdf_page,
                   p.document_page AS document_page,
                   p.cluster_id AS cluster,
                   coalesce(p.order, 0) AS doc_order,
                   (size(hits) * 4) AS score
            ORDER BY score DESC, size(p.visual_content) DESC
            LIMIT $limit
            """,
            needles=needles,
            limit=limit,
        )
        page_rows = [self._with_display_text(r.data()) for r in result]
        for row in page_rows:
            row["match_type"] = "visual_page"
            self._attach_media_fields(row)

        merged: dict[str, dict] = {}
        for row in region_rows + page_rows:
            merged[row["id"]] = row
        return list(merged.values())

    def _attach_media_fields(self, row: dict) -> dict:
        key = row.get("image_key")
        if key:
            row["image_url"] = resolve_image_url(key)
        return row

    def _with_display_text(self, row: dict) -> dict:
        """Merge visual_content into chunk text shown to the LLM."""
        visual = compact_visual_content((row.get("visual_content") or "").strip())
        body = (row.get("text") or "").strip()
        if visual:
            page_no = row.get("doc_order", "")
            row["text"] = (
                f"[Visual page content — page {page_no}]\n"
                f"(tables, charts, diagrams, shapes)\n\n{visual}"
                + (f"\n\n[Extracted text]\n{body}" if body else "")
            )
        return row

    # ─────────────────────────────────────────
    # GATHER — broad related set
    # ─────────────────────────────────────────

    def _keyword_section_search(
        self, session, terms: list[str], limit: int
    ) -> list:
        if not terms:
            return []
        result = session.run(
            """
            MATCH (s:Section)
            WHERE ANY(term IN $terms WHERE
                toLower(s.title) CONTAINS term OR toLower(s.text) CONTAINS term)
            WITH s,
                 size([t IN $terms WHERE toLower(s.title) CONTAINS t]) AS title_hits,
                 size([t IN $terms WHERE toLower(s.text) CONTAINS t]) AS text_hits
            WITH s, (title_hits * 2 + text_hits) AS keyword_hits
            WHERE keyword_hits > 0
            RETURN s.id AS id,
                   s.title AS title,
                   s.text AS text,
                   s.cluster_id AS cluster,
                   s.order AS doc_order,
                   keyword_hits AS score
            ORDER BY keyword_hits DESC, s.order ASC
            LIMIT $limit
            """,
            terms=terms,
            limit=limit,
        )
        return [r.data() for r in result]

    def _structural_section_search(
        self,
        session,
        terms: list[str],
        query: str,
        child_limit: int = 15,
    ) -> list:
        if not terms:
            return []

        rows: list[dict] = []

        parents = session.run(
            """
            MATCH (p:Section)
            WHERE ANY(term IN $terms WHERE toLower(p.title) CONTAINS term)
            WITH p, size([t IN $terms WHERE toLower(p.title) CONTAINS t]) AS title_hits
            WHERE title_hits > 0
            RETURN p.id AS id, p.title AS title, p.text AS text,
                   p.cluster_id AS cluster, p.order AS doc_order, title_hits AS score
            ORDER BY title_hits DESC, size(p.title) ASC
            LIMIT 8
            """,
            terms=terms,
        )
        for p in parents:
            pdata = p.data()
            pdata["match_type"] = "title_match"
            rows.append(pdata)

            children = session.run(
                """
                MATCH (p:Section {id: $pid})-[:CONTAINS]->(c:Section)
                RETURN c.id AS id, c.title AS title, c.text AS text,
                       c.cluster_id AS cluster, c.order AS doc_order,
                       1.0 AS score
                ORDER BY c.order ASC
                LIMIT $limit
                """,
                pid=pdata["id"],
                limit=child_limit,
            )
            for c in children:
                cdata = c.data()
                cdata["match_type"] = "child_of_match"
                rows.append(cdata)

        return rows

    def _best_parent_section(
        self,
        session,
        query: str,
        document_hint: Optional[str] = None,
    ) -> Optional[dict]:
        """Pick the section whose title best matches the user question (not loose term hits)."""
        q = query.lower().strip()
        q_norm = re.sub(r"[^a-z0-9\s\.]", " ", q)
        document_key = re.sub(r"[^a-z0-9]", "", (document_hint or "").lower())

        if document_key:
            candidates = session.run(
                """
                MATCH (b:Document|Book)-[:CONTAINS*1..8]->(s:Section)
                WHERE size(s.title) >= 8
                  AND (
                    toLower(replace(b.id, '_', ' ')) CONTAINS $document_key
                    OR toLower(coalesce(b.title, '')) CONTAINS $document_key
                  )
                RETURN s.id AS id, s.title AS title, s.text AS text,
                       s.cluster_id AS cluster, s.order AS doc_order
                """,
                document_key=document_key,
            )
            candidate_rows = list(candidates)
            if not candidate_rows:
                # Document id may be a hash; match sections that mention the document name.
                candidates = session.run(
                    """
                    MATCH (s:Section)
                    WHERE size(s.title) >= 8
                      AND (
                        toLower(s.title) CONTAINS $document_key
                        OR toLower(coalesce(s.text, '')) CONTAINS $document_key
                      )
                    RETURN s.id AS id, s.title AS title, s.text AS text,
                           s.cluster_id AS cluster, s.order AS doc_order
                    """,
                    document_key=document_key,
                )
                candidate_rows = list(candidates)
        else:
            candidate_rows = list(
                session.run(
                    """
                    MATCH (s:Section)
                    WHERE size(s.title) >= 8
                    RETURN s.id AS id, s.title AS title, s.text AS text,
                           s.cluster_id AS cluster, s.order AS doc_order
                    """
                )
            )

        best: Optional[dict] = None
        best_score = 0.0

        for record in candidate_rows:
            row = record.data() if hasattr(record, "data") else record
            title = (row.get("title") or "").lower()
            title_norm = re.sub(r"[^a-z0-9\s\.]", " ", title)
            words = [
                w
                for w in title_norm.split()
                if w not in STOPWORDS and len(w) > 1
            ]
            if not words:
                continue

            q_tokens = q_norm.split()
            hits = sum(1 for w in words if w in q_tokens)
            score = hits / len(words)
            if title_norm in q_norm or q_norm in title_norm:
                score += 0.5
            # Boost when numeric prefix matches (e.g. query "6 implementation" → title "6 IMPLEMENTATION...")
            num_m = re.match(r"^(\d+)", title_norm.strip())
            if num_m and num_m.group(1) in q_norm:
                score += 0.25
            if "go.data" in title and "go.data" in q:
                score += 0.1

            if score > best_score:
                best_score = score
                best = {**row, "score": score, "match_type": "parent_match"}

        if best_score < 0.4:
            return None
        return best

    def _section_query_focus(self, query: str) -> tuple[str, Optional[str]]:
        """Strip document boilerplate; return (section focus text, optional book hint)."""
        q = query.strip()
        document_hint: Optional[str] = None
        doc_m = re.search(
            r"\b(?:from|form)\s+(?:the\s+)?(.+?)\s+document\b",
            q,
            re.I,
        )
        if doc_m:
            document_hint = doc_m.group(1).strip()
            q = q[: doc_m.start()].strip()
        q = re.sub(
            r"^(?:what\s+(?:can\s+you\s+)?tell\s+me\s+about|what\s+is|tell\s+me\s+about|"
            r"explain|describe|summarize|summary\s+of)\s+",
            "",
            q,
            flags=re.I,
        )
        q = re.sub(r"\s+", " ", q).strip(" ?.")
        return (q or query.strip(), document_hint)

    def _fetch_section_subtree(
        self,
        session,
        parent_id: str,
        parent_order: int,
        parent_title: str,
        max_sections: int = 12,
    ) -> list:
        """
        Descendants of a section, stopping before the next top-level numbered heading
        (e.g. don't pull in '7 SUMMARY' when querying section 6).
        """
        max_order = parent_order + 15
        parent_num_m = re.match(r"^(\d+)", (parent_title or "").strip())
        parent_num = int(parent_num_m.group(1)) if parent_num_m else None
        if parent_num is not None:
            for row in session.run(
                """
                MATCH (s:Section)
                WHERE s.order > $porder AND s.title =~ '^\\d+\\s+.*'
                RETURN s.order AS ord, s.title AS title
                ORDER BY s.order ASC
                LIMIT 30
                """,
                porder=parent_order,
            ):
                m = re.match(r"^(\d+)", (row["title"] or "").strip())
                if m and int(m.group(1)) > parent_num:
                    max_order = row["ord"]
                    break

        result = session.run(
            """
            MATCH (p:Section {id: $pid})-[:CONTAINS*1..6]->(c:Section)
            WHERE c.order > $porder AND c.order < $max_order
            RETURN c.id AS id, c.title AS title, c.text AS text,
                   c.cluster_id AS cluster, c.order AS doc_order,
                   1.0 AS score
            ORDER BY c.order ASC
            LIMIT $limit
            """,
            pid=parent_id,
            porder=parent_order,
            max_order=max_order,
            limit=max_sections,
        )
        rows = [r.data() for r in result]
        for row in rows:
            row["match_type"] = "descendant_of_match"
        return rows

    def _fetch_descendant_sections(
        self,
        session,
        parent_id: str,
        max_depth: int = 4,
        max_sections: int = 12,
    ) -> list:
        """All nested sections under parent (handles chained CONTAINS, not only direct children)."""
        result = session.run(
            f"""
            MATCH (p:Section {{id: $pid}})-[:CONTAINS*1..{max_depth}]->(c:Section)
            RETURN c.id AS id, c.title AS title, c.text AS text,
                   c.cluster_id AS cluster, c.order AS doc_order,
                   1.0 AS score
            ORDER BY c.order ASC
            LIMIT $limit
            """,
            pid=parent_id,
            limit=max_sections,
        )
        rows = [r.data() for r in result]
        for row in rows:
            row["match_type"] = "descendant_of_match"
        return rows

    def _expand_structure(self, session, seed_ids: list[str]) -> list:
        if not seed_ids:
            return []
        result = session.run(
            """
            MATCH (s:Section) WHERE s.id IN $ids
            OPTIONAL MATCH (s)-[:CONTAINS]->(child:Section)
            OPTIONAL MATCH (parent:Section)-[:CONTAINS]->(s)
            RETURN collect(DISTINCT child) AS children,
                   collect(DISTINCT parent) AS parents,
                   collect(DISTINCT s) AS seeds
            """,
            ids=seed_ids,
        )
        row = result.single()
        if not row:
            return []

        out: list[dict] = []
        for node in row["children"] or []:
            if node:
                out.append(self._section_row(node, 0.5, "child_expand"))
        for node in row["parents"] or []:
            if node:
                out.append(self._section_row(node, 0.35, "parent_expand"))
        return out

    def _semantic_neighbors(
        self, session, seed_ids: list[str], hops: int = 1
    ) -> list:
        if not seed_ids or hops < 1:
            return []
        cypher = f"""
            MATCH (seed:Section) WHERE seed.id IN $ids
            MATCH (seed)-[:SAME_CATEGORY|SHARES_ENTITY*1..{hops}]-(related:Section)
            WHERE NOT related.id IN $ids
            RETURN DISTINCT related.id AS id,
                   related.title AS title,
                   related.text AS text,
                   related.cluster_id AS cluster,
                   related.order AS doc_order,
                   0.4 AS score
            LIMIT $limit
        """
        result = session.run(
            cypher, ids=seed_ids, limit=RETRIEVAL_CANDIDATE_POOL
        )
        return [r.data() for r in result]

    def _section_row(self, node, score: float, match_type: str) -> dict:
        return {
            "id": node["id"],
            "title": node.get("title", ""),
            "text": node.get("text", ""),
            "cluster": node.get("cluster_id"),
            "doc_order": node.get("order", 0),
            "score": score,
            "match_type": match_type,
        }

    # ─────────────────────────────────────────
    # FILTER — rerank and trim for LLM context
    # ─────────────────────────────────────────

    def _rerank_candidates(
        self,
        query: str,
        query_embedding: np.ndarray,
        candidates: list[dict],
        limit: int,
    ) -> list:
        terms = self._extract_search_terms(query)
        scored: list[dict] = []

        for row in candidates:
            vec_score = float(row.get("score", 0))
            if vec_score <= 1.0 and "vector" not in row.get("match_types", set()):
                vec_score = vec_score * 0.5

            title = (row.get("title") or "").lower()
            text = (row.get("text") or "").lower()[:4000]
            bonus = 0.0
            for term in terms:
                if term in title:
                    bonus += 0.12
                if term in text:
                    bonus += 0.04

            match_types = row.get("match_types", set())
            if isinstance(match_types, list):
                match_types = set(match_types)
            if "child_of_match" in match_types or "child_expand" in match_types:
                bonus += 0.15
            if "table_match" in match_types:
                bonus += 0.35
            if "visual" in match_types or "visual_page" in match_types:
                bonus += 0.32
            if "title_match" in match_types:
                bonus += 0.18
            if "vector" in match_types:
                bonus += vec_score * 0.35
            elif vec_score > 0:
                bonus += min(vec_score, 1.0) * 0.25

            rerank_score = bonus
            if rerank_score < RETRIEVAL_MIN_RERANK_SCORE:
                continue

            row = {**row, "rerank_score": rerank_score}
            scored.append(row)

        scored.sort(key=lambda r: (-r["rerank_score"], r.get("doc_order", 0)))
        return scored[:limit]

    # ─────────────────────────────────────────
    # VECTOR + ENRICH (unchanged core)
    # ─────────────────────────────────────────

    def _get_embedding(self, text: str) -> np.ndarray:
        response = provider.embeddings(
            model="text-embedding-3-small",
            input=text[:8000],
        )
        return np.array(response.data[0].embedding)

    def _vector_search(
        self,
        session,
        embedding: np.ndarray,
        limit: int,
        user_context: Optional[UserContext] = None,
    ) -> list:
        try:
            result = session.run(
                """
                CALL db.index.vector.queryNodes('section_embedding', $limit, $embedding)
                YIELD node AS s, score
                RETURN s.id AS id,
                       s.title AS title,
                       s.text AS text,
                       s.cluster_id AS cluster,
                       s.order AS doc_order,
                       score
                """,
                embedding=embedding.tolist(),
                limit=limit,
            )
            rows = [r.data() for r in result]
            self._vector_index_ready = bool(rows)
            return rows
        except Exception:
            return []

    def _legacy_similarity(
        self,
        session,
        embedding: np.ndarray,
        limit: int,
        user_context: Optional[UserContext] = None,
    ) -> list:
        result = session.run(
            """
            MATCH (s:Section)
            WHERE s.embedding IS NOT NULL
            RETURN s.id AS id,
                   s.title AS title,
                   s.text AS text,
                   s.cluster_id AS cluster,
                   s.order AS doc_order,
                   s.embedding AS embedding
            """
        )
        rows = [r.data() for r in result]
        scored = []
        for row in rows:
            raw = row.pop("embedding")
            if isinstance(raw, str):
                raw = json.loads(raw)
            emb = np.array(raw, dtype=np.float32)
            score = self._cosine_similarity(embedding, emb)
            scored.append({**row, "score": score})
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:limit]

    def _enrich_context_batch(self, session, items: list) -> list:
        if not items:
            return items

        ids = [i["id"] for i in items]
        result = session.run(
            """
            MATCH (s:Section) WHERE s.id IN $ids
            OPTIONAL MATCH (s)-[:SAME_CATEGORY]-(cm:Section)
            OPTIONAL MATCH (s)-[:SHARES_ENTITY]-(em:Section)
            OPTIONAL MATCH (s)-[:PRECEDES|FOLLOWS]-(nb:Section)
            OPTIONAL MATCH (b:Document|Book)-[:CONTAINS*1..3]->(s)
            RETURN s.id AS id,
                   collect(DISTINCT cm.title)[0..3] AS cluster_context,
                   collect(DISTINCT em.title)[0..3] AS entity_context,
                   collect(DISTINCT nb.title)[0..2] AS sequence_context,
                   b.title AS source_doc
            """,
            ids=ids,
        )
        ctx_map = {r["id"]: r.data() for r in result}

        for item in items:
            ctx = ctx_map.get(item["id"], {})
            item["related"] = (ctx.get("cluster_context") or []) + (
                ctx.get("entity_context") or []
            )
            item["source_doc"] = ctx.get("source_doc", "")

        return items

    # ─────────────────────────────────────────
    # HELPERS
    # ─────────────────────────────────────────

    def _access_denied_response(self, query: str, ctx: UserContext) -> Optional[Dict]:
        if self.rbac.can_query_knowledge_area(ctx.user_id, "esg"):
            return None
        return {
            "query": query,
            "chunks": [{
                "id": "access_denied",
                "title": "Access Denied",
                "text": f"User {ctx.user_id} does not have permission to query Agentic Graph RAG data.",
            }],
            "total_available": 0,
            "_access_level": ctx.role.value,
        }

    def _extract_search_terms(self, query: str) -> list[str]:
        cleaned = re.sub(r"[^a-z0-9\s\.]", " ", query.lower())
        words = [w for w in cleaned.split() if w not in STOPWORDS and len(w) > 2]
        terms: list[str] = []
        for w in words:
            if w not in terms:
                terms.append(w)
        if "go.data" in query.lower() or "godata" in query.lower():
            for extra in ("go.data", "godata", "data"):
                if extra not in terms:
                    terms.append(extra)
        return terms[:8]

    def _asks_for_subsections(self, query: str) -> bool:
        q = query.lower()
        triggers = (
            "subsection",
            "sub-section",
            "sub section",
            "sub sections",
            "have sub",
            "headings under",
            "sections under",
            "under the section",
            "list the sections",
            "with headings",
            "give them with headings",
            "children of",
            "nested under",
        )
        return any(t in q for t in triggers)

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    def _format_response(
        self,
        query: str,
        items: list,
        user_context: Optional[UserContext] = None,
        meta: Optional[dict] = None,
    ) -> Dict:
        ctx = user_context or self.user_context
        out = {
            "query": query,
            "chunks": [
                {
                    "id": r["id"],
                    "title": r["title"],
                    "text": r["text"],
                    "cluster": r.get("cluster"),
                    "score": round(
                        r.get("rerank_score", r.get("score", 0.0)), 3
                    ),
                    "related": r.get("related", []),
                    "match_types": list(r.get("match_types", []))
                    if isinstance(r.get("match_types"), set)
                    else r.get("match_types", []),
                    "pdf_page": r.get("pdf_page"),
                    "document_page": r.get("document_page"),
                    "visual_content": r.get("visual_content"),
                    "page_tags": r.get("page_tags"),
                    "image_key": r.get("image_key"),
                    "image_url": r.get("image_url")
                    or resolve_image_url(r.get("image_key")),
                    "region_kind": r.get("region_kind"),
                    "region_tags": r.get("region_tags"),
                    "image_source": r.get("image_source"),
                    "matched_region_id": r.get("matched_region_id"),
                }
                for r in items
            ],
            "total_available": len(items),
            "_access_level": ctx.role.value,
            "_user_id": ctx.user_id,
        }
        if meta:
            out.update(meta)
        return out


# Backward-compatible aliases (deprecated)
ESGComplianceRetriever = DocumentRAGRetriever
RAGDataRetriever = DocumentRAGRetriever
