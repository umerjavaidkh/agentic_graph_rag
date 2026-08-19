"""box.py — "Box N" heading enumeration and content-fetch strategy.

Extracted from mixins/box_strategy.py (BoxStrategyMixin). Handles both box
list ("list all box headings") and box content ("Box 5") — the two were
already one mixin/one concern, kept as one strategy here too. Preserves the
original mutual-exclusion behavior exactly: if the query looks like a list
request, only the list branch is tried (never falls through to content,
even when the list comes back empty) — matching mixins/hybrid.py's original
`box_n is not None and not is_box_list_request(query)` guard.
"""
from __future__ import annotations

import re
from typing import Any, Optional

from ....shared.auth.roles import UserContext
from ....graph.constants import DOCUMENT_ROOT_CYPHER
from ....shared.neo4j.tenancy import tenant_filter
from ....shared.storage.hydrator import get_hydrator
from ..constants import _TEXT_NODE_LABELS
from ..cypher_scope import _doc_scope_cypher
from ..executor import DocumentQueryExecutor
from ..services.document_resolver import DocumentResolver
from ..services.formatter import ResponseFormatter


class BoxStrategy:
    name = "structural_box_list"

    def __init__(
        self,
        document_resolver: DocumentResolver,
        formatter: ResponseFormatter,
        exec_: DocumentQueryExecutor,
    ):
        self._document_resolver = document_resolver
        self._formatter = formatter
        self._exec = exec_

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
        if self._exec.is_box_list_request(query):
            items = self._structural_box_headings(session, query, tenant_id, document_id_hint)
            if not items:
                return None
            response = self._formatter.format(query, items, ctx=ctx)
            response["mode"] = "structural_box_list"
            response["strategy"] = "graph_rag"
            response["vector_seeds"] = 0
            response["fulltext_hits"] = 0
            response["graph_expanded"] = len(items)
            doc_id, doc_title = self._document_resolver.resolve_document_for_query(
                session, query, tenant_id, document_id_hint=document_id_hint
            )
            response["document_id"] = doc_id
            response["document_title"] = doc_title
            return response

        box_n = self._exec.parse_box_number(query)
        if box_n is not None:
            items = self._structural_box_content(session, query, box_n, tenant_id, document_id_hint)
            if items:
                response = self._formatter.format(query, items, ctx=ctx)
                response["mode"] = "structural_box_content"
                response["strategy"] = "graph_rag"
                response["vector_seeds"] = 0
                response["fulltext_hits"] = 0
                response["graph_expanded"] = len(items)
                doc_id, doc_title = self._document_resolver.resolve_document_for_query(
                    session, query, tenant_id, document_id_hint=document_id_hint
                )
                response["document_id"] = doc_id
                response["document_title"] = doc_title
                return response
        return None

    def _structural_box_headings(
        self, session: Any, query: str, tenant_id: str = "", document_id_hint: str = ""
    ) -> list[dict]:
        """
        Enumerate Box headings (e.g. "Box 10") inside a document.
        Generic: works for any document that contains "Box <number>" in titles or text.
        """
        doc_id, doc_title = self._document_resolver.resolve_document_for_query(
            session, query, tenant_id, document_id_hint=document_id_hint
        )
        rows = session.run(
            f"""
            MATCH (d:{DOCUMENT_ROOT_CYPHER})
            WHERE {_doc_scope_cypher("d")}
              AND {tenant_filter("d")}
            MATCH (n)
            WHERE any(l IN labels(n) WHERE l IN $labels)
              AND (
                EXISTS {{ MATCH (d)-[:CONTAINS*0..6]->(n) }}
                OR n.id STARTS WITH d.id + '_'
              )
              AND (
                (n.title IS NOT NULL AND toLower(n.title) CONTAINS 'box')
                OR (n.search_text IS NOT NULL AND toLower(n.search_text) CONTAINS 'box')
              )
            RETURN
              coalesce(n.id,'') AS id,
              coalesce(n.title,'') AS title,
              n.blob_key_text AS blob_key_text,
              coalesce(n.search_text,'') AS search_text,
              n.page_start AS page_start
            LIMIT 250
            """,
            doc_id=doc_id,
            labels=list(_TEXT_NODE_LABELS),
            tenant_id=tenant_id,
        )

        hydrator = get_hydrator()
        found: dict[int, dict] = {}
        for r in rows:
            rid = r.get("id") or ""
            title = (r.get("title") or "").strip()
            text = hydrator.hydrate(r.get("blob_key_text"), r.get("search_text") or "").strip()
            hay = f"{title}\n{text}"
            for num in self._exec.extract_box_numbers(hay):
                if num in found:
                    continue
                # Prefer title if it contains Box, else synthesize a heading.
                heading = title if re.search(rf"(?i)\bbox\s+{num}\b", title) else f"Box {num}"
                snippet = ""
                if text:
                    # keep a compact preview
                    snippet = text[:800]
                found[num] = {
                    "id": rid or f"box_{num}",
                    "title": heading,
                    "text": snippet,
                    "page_start": r.get("page_start"),
                    "score": 1.0,
                    "related": [f"doc:{doc_title}" if doc_title else "doc:unknown", "via:box_scan"],
                }

        return [found[k] for k in sorted(found.keys())]

    def _structural_box_content(
        self, session: Any, query: str, box_n: int, tenant_id: str = "", document_id_hint: str = ""
    ) -> list[dict]:
        """
        Retrieve content for a specific Box N (e.g. Box 5).
        Looks for nodes whose title/text mention the box, then returns the best matches.
        """
        doc_id, doc_title = self._document_resolver.resolve_document_for_query(
            session, query, tenant_id, document_id_hint=document_id_hint
        )
        box_phrase = f"box {int(box_n)}"
        rows = session.run(
            f"""
            MATCH (d:{DOCUMENT_ROOT_CYPHER})
            WHERE {_doc_scope_cypher("d")}
              AND {tenant_filter("d")}
            MATCH (n)
            WHERE any(l IN labels(n) WHERE l IN $labels)
              AND (
                EXISTS {{ MATCH (d)-[:CONTAINS*0..6]->(n) }}
                OR n.id STARTS WITH d.id + '_'
              )
              AND (
                (n.title IS NOT NULL AND toLower(n.title) CONTAINS $box_phrase)
                OR (n.search_text IS NOT NULL AND toLower(n.search_text) CONTAINS $box_phrase)
              )
            RETURN
              coalesce(n.id,'') AS id,
              coalesce(n.title,'') AS title,
              n.blob_key_text AS blob_key_text,
              coalesce(n.search_text,'') AS search_text,
              n.page_start AS page_start
            LIMIT 20
            """,
            doc_id=doc_id,
            box_phrase=box_phrase,
            labels=list(_TEXT_NODE_LABELS),
            tenant_id=tenant_id,
        )

        hydrator = get_hydrator()
        items: list[dict] = []
        for r in rows:
            rid = r.get("id") or ""
            title = (r.get("title") or "").strip()
            text = hydrator.hydrate(r.get("blob_key_text"), r.get("search_text") or "").strip()
            if not rid or not (title or text):
                continue
            # Prefer chunks whose title explicitly contains Box N. (Regex
            # bug fixed: rf"...\\bbox..." with a doubled backslash matches a
            # literal backslash character, which never appears in real
            # text — this check silently always failed, so every candidate
            # fell through to the default score=1.0 and ranking degraded to
            # "longest text wins" regardless of which Box it actually was.)
            score = 1.0
            if re.search(rf"(?i)\bbox\s+{box_n}\b", title):
                score = 1.08
            elif re.search(rf"(?i)\bbox\s+{box_n}\b", text[:200]):
                score = 1.02
            # Keep a larger snippet since user asked "all the data".
            snippet = text[:2500] if text else ""
            items.append({
                "id": rid,
                "title": title or f"Box {box_n}",
                "text": snippet,
                "page_start": r.get("page_start"),
                "score": score,
                "related": [f"doc:{doc_title}" if doc_title else "doc:unknown", "via:box_content"],
            })

        items.sort(
            key=lambda x: (float(x.get("score", 0.0)), len(x.get("text") or "")),
            reverse=True,
        )
        if items and len((items[0].get("text") or "")) < 200:
            page_items = self._box_content_from_page_text(
                session, query, box_n, doc_id, tenant_id, document_id_hint
            )
            if page_items:
                return page_items
        return items[:6]

    def _box_content_from_page_text(
        self,
        session: Any,
        query: str,
        box_n: int,
        doc_id: Optional[str],
        tenant_id: str = "",
        document_id_hint: str = "",
    ) -> list[dict]:
        """Fallback when Box sections in Neo4j only store the label (pre-fix ingest)."""
        _, doc_title = self._document_resolver.resolve_document_for_query(
            session, query, tenant_id, document_id_hint=document_id_hint
        )
        rows = session.run(
            f"""
            MATCH (d:{DOCUMENT_ROOT_CYPHER})
            WHERE {_doc_scope_cypher("d")}
              AND {tenant_filter("d")}
            MATCH (p:Page)
            WHERE p.id STARTS WITH d.id + '_page_'
              AND toLower(coalesce(p.search_text, '')) CONTAINS $box_phrase
              AND {tenant_filter("p")}
            RETURN p.id AS id, p.title AS title, p.search_text AS text, p.pdf_page AS pdf_page, p.document_page AS document_page
            ORDER BY size(coalesce(p.search_text, '')) DESC
            LIMIT 3
            """,
            doc_id=doc_id,
            box_phrase=f"box {int(box_n)}",
            tenant_id=tenant_id,
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
                # Both numbers travel with the chunk: pdf_page is what
                # navigation opens, document_page is what the reader
                # sees printed on the page. Hand-built chunk dicts like
                # this one bypass the formatter's passthrough, so the
                # printed label was lost on every box answer.
                "document_page": r.get("document_page"),
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
