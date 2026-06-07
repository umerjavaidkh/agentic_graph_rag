"""Document RAG retriever — box strategy."""
from __future__ import annotations

import re
from typing import Optional

from ....graph.constants import DOCUMENT_ROOT_CYPHER
from ..constants import _TEXT_NODE_LABELS
from ..cypher_scope import _doc_scope_cypher


class BoxStrategyMixin:
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

