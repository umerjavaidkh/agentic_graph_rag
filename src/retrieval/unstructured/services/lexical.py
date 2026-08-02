"""lexical.py — phrase/keyword CONTAINS retrieval, shared by retrieval strategies.

Extracted from mixins/lexical.py (LexicalRetrievalMixin). Depends on
RankingService (query keyword/phrase extraction) and DocumentResolver
(doc-scoped search) — constructed after both in the service bundle.
"""
from __future__ import annotations

import re
from typing import Optional

from ....graph.constants import DOCUMENT_ROOT_CYPHER
from ....graph.tenancy import tenant_filter
from ....storage.hydrator import get_hydrator
from ..constants import _TEXT_NODE_LABELS
from ..cypher_scope import _doc_scope_cypher
from ..text_utils import _extract_urls
from .document_resolver import DocumentResolver
from .ranking import RankingService


class LexicalService:
    def __init__(self, ranking: RankingService, document_resolver: DocumentResolver):
        self._ranking = ranking
        self._document_resolver = document_resolver

    def structural_keyword_retrieve(
        self, session, query: str, tenant_id: str = "", document_id: Optional[str] = None
    ) -> list[dict]:
        """
        Rank nodes by how many query keywords they match, weighted by each
        keyword's rarity (inverse document frequency) within the scoped
        document — not a flat count.

        A flat count over-rewards common English words that coincidentally
        appear in any long section of prose ("down", "across", "year") —
        an unrelated section matching 6 generic words can outscore the one
        page that actually answers the question but only matches 3 truly
        distinctive terms ("quarterly", "income"). Mirrors the same IDF
        weighting already used in
        document_resolver.resolve_document_for_query_strict's document
        scoring, applied here to node-level ranking instead of document
        selection.

        `document_id`: pass the caller's already-resolved document id (may
        be "" for "resolved to unscoped") to skip re-resolving it here —
        the caller (FullHybridStrategy) already pays for one resolution
        per query, and this ran a second, fully redundant one on every
        call before this parameter existed. Leave unset (None) to resolve
        internally, e.g. when called standalone.
        """
        keywords = self._ranking._content_keywords_from_query(query)
        if len(keywords) < 2:
            return []
        keywords = [k.lower() for k in keywords]

        # A fixed floor, not one that scales up with query length: a node
        # matching just 2-3 keywords should still be eligible if those few
        # matches are highly specific (weighted ranking below sorts that
        # correctly) — the old scaling-up floor (up to 4 for longer
        # queries) excluded exactly this case, filtering out a real answer
        # matching only its 3 most distinctive terms before the weighting
        # ever got a chance to rank it above a false match padded with
        # extra generic-word hits.
        min_hits = 2
        if document_id is None:
            doc_id, _ = self._document_resolver.resolve_document_for_query(session, query, tenant_id)
        else:
            # "" means the caller resolved and got no confident match —
            # normalize to None so _doc_scope_cypher's `$doc_id IS NULL`
            # branch degrades to unscoped, instead of comparing every
            # document's id against the literal empty string and matching
            # none of them.
            doc_id = document_id or None

        # Document frequency per keyword — how many scoped nodes contain it
        # at all, regardless of the min_hits threshold below.
        freq_rows = session.run(
            f"""
            MATCH (d:{DOCUMENT_ROOT_CYPHER})
            WHERE {_doc_scope_cypher("d")}
              AND {tenant_filter("d")}
            MATCH (n)
            WHERE any(l IN labels(n) WHERE l IN $labels)
              AND coalesce(n.search_text, '') <> ''
              AND (
                EXISTS {{ MATCH (d)-[:CONTAINS*0..6]->(n) }}
                OR n.id STARTS WITH d.id + '_'
              )
            UNWIND $keywords AS k
            WITH k, n WHERE toLower(n.search_text) CONTAINS k
            RETURN k AS keyword, count(DISTINCT n) AS df
            """,
            doc_id=doc_id,
            keywords=keywords,
            labels=list(_TEXT_NODE_LABELS),
            tenant_id=tenant_id,
        )
        doc_freq = {r["keyword"]: int(r["df"]) for r in freq_rows}
        if not doc_freq:
            return []
        max_df = max(doc_freq.values())
        weight = {k: max_df / df for k, df in doc_freq.items() if df > 0}

        # The weighted score is computed here in Cypher (not after fetching
        # candidates into Python) because the candidate pool can be large
        # (e.g. 200+ nodes matching >= min_hits keywords in a long filing) —
        # truncating with LIMIT before ordering by relevance would silently
        # drop the best match in Neo4j's arbitrary internal row order. Only
        # ORDER BY ... LIMIT keeps the truncation itself relevance-informed.
        rows = session.run(
            f"""
            MATCH (d:{DOCUMENT_ROOT_CYPHER})
            WHERE {_doc_scope_cypher("d")}
              AND {tenant_filter("d")}
            MATCH (n)
            WHERE any(l IN labels(n) WHERE l IN $labels)
              AND coalesce(n.search_text, '') <> ''
              AND (
                EXISTS {{ MATCH (d)-[:CONTAINS*0..6]->(n) }}
                OR n.id STARTS WITH d.id + '_'
              )
            WITH n,
              [k IN $keywords WHERE toLower(n.search_text) CONTAINS k] AS matched
            WHERE size(matched) >= $min_hits
            WITH n, matched,
              reduce(s = 0.0, k IN matched | s + coalesce($weight[k], 0.0)) AS w
            RETURN
              coalesce(n.id, '') AS id,
              coalesce(n.title, '') AS title,
              n.blob_key_text AS blob_key_text,
              coalesce(n.search_text, '') AS search_text,
              n.page_start AS page_start,
              matched,
              w
            ORDER BY w DESC, size(coalesce(n.search_text, '')) ASC
            LIMIT 6
            """,
            doc_id=doc_id,
            keywords=keywords,
            min_hits=min_hits,
            labels=list(_TEXT_NODE_LABELS),
            tenant_id=tenant_id,
            weight=weight,
        )

        hydrator = get_hydrator()
        items: list[dict] = []
        for r in rows:
            if not r.get("id"):
                continue
            matched = r.get("matched") or []
            w = float(r.get("w") or 0.0)
            title = r.get("title") or r["id"]
            full_text = hydrator.hydrate(r.get("blob_key_text"), r.get("search_text") or "")
            items.append({
                "id": r["id"],
                "title": title,
                "text": self.enrich_chunk_text_for_facts(title, full_text),
                "page_start": r.get("page_start"),
                "score": 0.88 + 0.06 * min(len(matched), 4) + 0.01 * min(w, 20),
                "related": ["via:keyword_search"],
            })
        return items

    def enrich_chunk_text_for_facts(self, title: str, text: str) -> str:
        body = (text or "").strip()
        urls = _extract_urls(body)
        if not urls:
            return body
        url_block = "\n".join(f"- {u}" for u in urls)
        return f"{body}\n\n[Extracted URLs]\n{url_block}".strip()

    def structural_phrase_retrieve(
        self, session, query: str, tenant_id: str = "", document_id: Optional[str] = None
    ) -> list[dict]:
        """
        Direct phrase CONTAINS search for fact/URL questions vector search often misses.

        `document_id`: see structural_keyword_retrieve — same
        skip-re-resolution contract.
        """
        phrases = self._ranking._search_phrases_from_query(query)
        if not phrases:
            return []

        if document_id is None:
            doc_id, _ = self._document_resolver.resolve_document_for_query(session, query, tenant_id)
        else:
            doc_id = document_id or None
        rows = session.run(
            f"""
            MATCH (d:{DOCUMENT_ROOT_CYPHER})
            WHERE {_doc_scope_cypher("d")}
              AND {tenant_filter("d")}
            MATCH (n)
            WHERE any(l IN labels(n) WHERE l IN $labels)
              AND coalesce(n.search_text, '') <> ''
              AND (
                EXISTS {{ MATCH (d)-[:CONTAINS*0..6]->(n) }}
                OR n.id STARTS WITH d.id + '_'
              )
              AND any(phrase IN $phrases WHERE toLower(n.search_text) CONTAINS phrase)
            WITH n, d,
              size([p IN $phrases WHERE toLower(n.search_text) CONTAINS p]) AS phrase_hits
            RETURN
              coalesce(n.id, '') AS id,
              coalesce(n.title, '') AS title,
              n.blob_key_text AS blob_key_text,
              coalesce(n.search_text, '') AS search_text,
              n.page_start AS page_start,
              phrase_hits,
              coalesce(d.title, d.id) AS doc_title
            ORDER BY phrase_hits DESC, size(coalesce(n.search_text, '')) ASC
            LIMIT 6
            """,
            doc_id=doc_id,
            phrases=[p.lower() for p in phrases],
            labels=list(_TEXT_NODE_LABELS),
            tenant_id=tenant_id,
        )

        hydrator = get_hydrator()
        items: list[dict] = []
        for r in rows:
            if not r.get("id"):
                continue
            title = r.get("title") or r["id"]
            full_text = hydrator.hydrate(r.get("blob_key_text"), r.get("search_text") or "")
            text = self.enrich_chunk_text_for_facts(title, full_text)
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
                "page_start": r.get("page_start"),
                "score": score,
                "related": ["via:phrase_search"],
            })
        return items
