"""subsection.py — numbered-section child/parent lookup strategy.

Extracted from mixins/subsection.py (SubsectionMixin) plus the doc-choice
clarification logic that previously lived inline in mixins/hybrid.py's
ladder (the two were split across files for the same "subsection request"
concern — consolidated here into one self-contained strategy).
"""
from __future__ import annotations

from typing import Any, Optional

from ....auth.roles import UserContext
from ....graph.constants import DOCUMENT_ROOT_CYPHER
from ....graph.tenancy import tenant_filter
from ....storage.hydrator import get_hydrator
from ..cypher_scope import _doc_scope_cypher
from ..executor import DocumentQueryExecutor
from ..services.document_resolver import DocumentResolver
from ..services.formatter import ResponseFormatter


# Per-reference excerpt cap for multi-reference comparison queries (see
# _structural_reference_excerpt) -- keeps several named references'
# combined content well within the synthesis context budget even though
# any single one of them (a whole chapter) could be 100k+ characters alone.
_MULTI_REF_EXCERPT_CHARS = 6000

# A plain `title STARTS WITH sec_num` match has no defense against a longer
# number/letter-suffixed sibling sharing the same prefix -- "item 7"
# STARTS WITH-matches "Item 7A. Quantitative and Qualitative..." just as
# readily as the real "Item 7. Management's discussion...", with no
# ORDER BY to break the tie deterministically. Same failure class for
# "chapter 1" matching "Chapter 10"-"Chapter 19", "note 3" matching a
# hypothetical "Note 30", etc. Verified live: "What does Item 7 discuss
# regarding results of operations?" (JNJ 10-K, which has both a real
# "Item 7" and "Item 7A" section) non-deterministically returned "Item 7A"
# content instead. Fixed by additionally requiring the character right
# after the matched prefix to be non-alphanumeric (or the title to be
# exactly that length) -- "item 7." and "item 7 —" still match "item 7";
# "item 7a" no longer does. Doesn't need sec_num escaped into a regex (only
# a fixed, safe `[a-z0-9]` pattern is used) since STARTS WITH still does
# the literal prefix comparison.
_PREFIX_BOUNDARY_CYPHER = (
    "(size(s.title) = size($sec_num) "
    "OR NOT substring(toLower(s.title), size($sec_num), 1) =~ '[a-z0-9]')"
)


class SubsectionStrategy:
    name = "subsection_tree"

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
        if not self._exec.is_subsection_request(query):
            return None
        # A query naming two+ distinct references ("How does Chapter 9
        # relate to Chapter 11?") must NOT be narrowed to whichever one
        # parse_section_number happens to match first -- that silently
        # drops the other reference's content entirely, and the LLM then
        # (correctly, given only what it was shown) reports it as "not
        # covered." Route to a dedicated lookup that fetches EACH named
        # reference directly (falling through to full_hybrid's generic
        # vector/keyword seeding turned out to be too imprecise on its own
        # -- verified live, it surfaced neither named chapter, just
        # whatever scored similarly on embeddings).
        sec_nums = self._exec.parse_all_section_numbers(query)
        if len(sec_nums) > 1:
            return self._retrieve_multi_reference(session, query, sec_nums, tenant_id, ctx, document_id_hint)
        sec_num = self._exec.parse_section_number(query)
        if not sec_num:
            return None

        # If the user didn't name a document and multiple documents exist,
        # ask them to pick rather than silently guessing the biggest one.
        if not self._document_resolver.document_match_terms(query):
            docs = self._document_resolver.list_documents(session, limit=5, tenant_id=tenant_id)
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

        items, parent, doc_id, doc_title = self._structural_subsections(
            session, query, sec_num, tenant_id, document_id_hint
        )

        # A chapter reference ("Chapter 15") is a broader scope than Note/
        # Item/Example's direct-lookup use case -- "list every Check Your
        # Understanding question in Chapter 15" needs the LLM to extract/
        # filter from the chapter's content, not a raw dump of it. Note/
        # Item/Example matches are tagged with the existing fast-path modes
        # (dumping the matched text verbatim IS the correct answer for
        # "tell me about Note 3"); chapter matches get distinct mode names
        # deliberately excluded from _STRUCTURAL_FAST_MODES in graph.py, so
        # they fall through to normal LLM synthesis over the (correctly
        # scoped, budget-capped) retrieved content instead.
        is_chapter_ref = sec_num.startswith("chapter ")

        if items:
            response = self._formatter.format(query, items, ctx=ctx)
            response["mode"] = "chapter_children" if is_chapter_ref else "subsection_tree"
            response["strategy"] = "graph_rag"
            response["parent_id"] = parent.get("id")
            response["parent_title"] = parent.get("title")
            response["vector_seeds"] = 0
            response["fulltext_hits"] = 0
            response["graph_expanded"] = len(items)
            response["document_id"] = doc_id
            response["document_title"] = doc_title
            return response

        if parent and parent.get("text"):
            response = self._formatter.format(query, [parent], ctx=ctx)
            response["mode"] = "chapter_detail" if is_chapter_ref else "section_detail"
            response["strategy"] = "graph_rag"
            response["parent_id"] = parent.get("id")
            response["parent_title"] = parent.get("title")
            response["vector_seeds"] = 0
            response["fulltext_hits"] = 0
            response["graph_expanded"] = 1
            response["document_id"] = doc_id
            response["document_title"] = doc_title
            return response

        return None

    def _retrieve_multi_reference(
        self,
        session: Any,
        query: str,
        sec_nums: list[str],
        tenant_id: str,
        ctx: UserContext,
        document_id_hint: str = "",
    ) -> Optional[dict[str, Any]]:
        doc_id, doc_title = self._document_resolver.resolve_document_for_query(
            session, query, tenant_id, document_id_hint=document_id_hint
        )
        items: list[dict] = []
        for sec_num in sec_nums:
            item = self._structural_reference_excerpt(session, sec_num, tenant_id, doc_id)
            if item:
                items.append(item)
        if not items:
            return None

        response = self._formatter.format(query, items, ctx=ctx)
        response["mode"] = "multi_reference_detail"
        response["strategy"] = "graph_rag"
        response["vector_seeds"] = 0
        response["fulltext_hits"] = 0
        response["graph_expanded"] = len(items)
        response["document_id"] = doc_id
        response["document_title"] = doc_title
        return response

    def _structural_reference_excerpt(
        self, session: Any, sec_num: str, tenant_id: str, doc_id: Optional[str]
    ) -> Optional[dict]:
        """Fetch one named reference's own content for a multi-reference
        comparison query -- a short chapter-level summary when available
        (populated at ingestion for every Chapter), else a bounded excerpt
        of its raw text. Deliberately NOT the full chapter text: comparison
        questions can name several references, and full chapters would
        blow the synthesis context budget combined (a single chapter alone
        can be 100k+ characters)."""
        row = session.run(
            f"""
            MATCH (d:{DOCUMENT_ROOT_CYPHER})
            WHERE {_doc_scope_cypher("d")}
              AND {tenant_filter("d")}
            MATCH (s)
            WHERE (s:Section OR s:Chapter)
              AND (s.id STARTS WITH d.id + '_' OR EXISTS {{ MATCH (d)-[:CONTAINS*1..6]->(s) }})
              AND s.title IS NOT NULL
              AND trim(s.title) <> ''
              AND toLower(s.title) STARTS WITH toLower($sec_num)
              AND {_PREFIX_BOUNDARY_CYPHER}
              AND {tenant_filter("s")}
            RETURN
              s.id AS sid,
              s.title AS stitle,
              coalesce(s.summary, '') AS ssummary,
              s.blob_key_text AS sblob_key,
              coalesce(s.search_text, '') AS stext,
              s.page_start AS spage_start
            LIMIT 1
            """,
            doc_id=doc_id,
            sec_num=sec_num,
            tenant_id=tenant_id,
        ).single()
        if not row:
            return None
        summary = (row.get("ssummary") or "").strip()
        full_text = get_hydrator().hydrate(row.get("sblob_key"), row.get("stext") or "")
        text = summary if summary else full_text.strip()[:_MULTI_REF_EXCERPT_CHARS]
        if not text:
            return None
        return {
            "id": row.get("sid") or "",
            "title": (row.get("stitle") or "").strip(),
            "text": text,
            "page_start": row.get("spage_start"),
            "score": 1.0,
            "related": ["via:multi_reference_lookup"],
        }

    def _structural_subsections(
        self,
        session: Any,
        query: str,
        sec_num: str,
        tenant_id: str = "",
        document_id_hint: str = "",
    ) -> tuple[list[dict], dict, Optional[str], Optional[str]]:
        """Return (children items, parent item, doc_id, doc_title) for a numbered section like 2.5."""
        doc_id, doc_title = self._document_resolver.resolve_document_for_query(
            session, query, tenant_id, document_id_hint=document_id_hint
        )
        row = session.run(
            f"""
            MATCH (d:{DOCUMENT_ROOT_CYPHER})
            WHERE {_doc_scope_cypher("d")}
              AND {tenant_filter("d")}
            MATCH (s)
            WHERE (s:Section OR s:Chapter)
              AND (s.id STARTS WITH d.id + '_' OR EXISTS {{ MATCH (d)-[:CONTAINS*1..6]->(s) }})
              AND s.title IS NOT NULL
              AND trim(s.title) <> ''
              AND toLower(s.title) STARTS WITH toLower($sec_num)
              AND {_PREFIX_BOUNDARY_CYPHER}
              AND {tenant_filter("s")}
            WITH s
            OPTIONAL MATCH (s)-[:CONTAINS]->(c:Section)
            WHERE c.title IS NOT NULL AND trim(c.title) <> ''
              AND (c IS NULL OR {tenant_filter("c")})
            RETURN
              s.id AS sid,
              s.title AS stitle,
              s.blob_key_text AS sblob_key,
              coalesce(s.search_text,'') AS stext,
              s.page_start AS spage_start,
              collect({{id: c.id, title: c.title, blob_key_text: c.blob_key_text,
                        text: coalesce(c.search_text,''), page_start: c.page_start}}) AS children
            LIMIT 1
            """,
            doc_id=doc_id,
            sec_num=sec_num,
            tenant_id=tenant_id,
        ).single()
        if not row:
            return [], {}, doc_id, doc_title

        hydrator = get_hydrator()
        parent = {
            "id": row.get("sid") or "",
            "title": (row.get("stitle") or "").strip(),
            "text": hydrator.hydrate(row.get("sblob_key"), row.get("stext") or "").strip(),
            "page_start": row.get("spage_start"),
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
                "text": hydrator.hydrate(c.get("blob_key_text"), c.get("text") or "").strip(),
                "page_start": c.get("page_start"),
                "score": 1.0,
                "related": ["via:subsections"],
            })
        return items, parent, doc_id, doc_title
