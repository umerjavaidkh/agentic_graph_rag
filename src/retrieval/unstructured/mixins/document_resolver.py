"""Document RAG retriever — document resolver."""
from __future__ import annotations

import re
from typing import Optional

from ....graph.constants import (
    DOC_REVISION_LABEL,
    DOCUMENT_LOGICAL_LABEL,
    DOCUMENT_ROOT_CYPHER,
)
from ....graph.versioning import lifecycle_active
from ..cypher_scope import _clean_doc_title
from ..query_intent import KEYWORD_STOP as _KEYWORD_STOP
from ..text_utils import _query_anchor_terms


class DocumentResolverMixin:
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

