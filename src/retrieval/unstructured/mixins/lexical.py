"""Document RAG retriever — lexical."""
from __future__ import annotations

import re

from ....graph.constants import DOCUMENT_ROOT_CYPHER
from ..constants import _TEXT_NODE_LABELS
from ..cypher_scope import _doc_scope_cypher
from ..text_utils import _extract_urls


class LexicalRetrievalMixin:
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

