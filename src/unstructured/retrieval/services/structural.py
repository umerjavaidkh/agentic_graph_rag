"""structural.py — answer a structural question from the hierarchy, not by similarity.

A structural question names a graph address: "section 4.2", "Box 9",
"Figure 1", "the table of contents". Those are addresses, not concepts, and
the nearest neighbours of an address are other addresses.

That is the general argument. On this corpus it is sharper than general:
Page nodes are 1.0% embedded and Region nodes (8,068 tables, 4,750 figures)
are 0% embedded, so the vector index cannot see the units these questions
name at all. An answer about "Figure 1" today comes from Section prose that
mentions the figure, never from the figure.

The exhaustive rule lives here too. Asked for the table of contents of a
9-chapter, 21-section document, the hybrid path returned 8 headings out of
30, out of order, phrased with full confidence -- `LIMIT 6` on the lexical
queries. A hierarchy read has no LIMIT, and orders by position rather than
by relevance, because a table of contents that is missing two thirds of its
entries is not a partial answer, it is a wrong one.
"""
from __future__ import annotations

import re
from typing import Any, Optional

from ....shared.neo4j.tenancy import tenant_filter
from ....shared.storage.hydrator import get_hydrator
from ..cypher_scope import content_scope_where_multi

# "Box 9" / "Figure 1" / "Table 3.2" / "page 12" / "section 4.2" -> (kind, number)
_ADDRESS_PARTS = re.compile(
    r"\b(section|clause|article|chapter|part|annex|appendix|box|figure|fig\.?|table|page)\s*"
    r"(?:no\.?\s*)?(\d+(?:\.\d+)*)\b",
    re.I,
)

# Which node label answers which address word. Tables and figures are Region
# nodes discriminated by `region_kind`, which is why no new labels were
# needed to support them -- the substrate already existed, unembedded.
_KIND_TO_LABEL = {
    "page": "Page",
    "figure": "Region", "fig": "Region", "fig.": "Region", "box": "Region", "table": "Region",
    "section": "Section", "clause": "Section", "article": "Section",
    "part": "Section", "annex": "Section", "appendix": "Section",
    "chapter": "Chapter",
}
_KIND_TO_REGION = {"figure": "figure", "fig": "figure", "fig.": "figure",
                   "table": "table", "box": "table"}


class StructuralService:
    """Reads of the document hierarchy, by address."""

    def outline(
        self, session: Any, doc_ids: list[str], tenant_id: str = ""
    ) -> list[dict]:
        """The document's full heading hierarchy, in reading order.

        Returned as ONE item rather than many. A table of contents is a
        single coherent passage; splitting it into thirty items invites
        every downstream stage that takes a top-k to truncate it again,
        which is the bug this exists to fix.
        """
        rows = session.run(
            f"""
            MATCH (n)
            WHERE (n:Chapter OR n:Section)
              AND {content_scope_where_multi("n", scoped=bool(doc_ids))}
              AND {tenant_filter("n")}
              AND trim(coalesce(n.title, '')) <> ''
            RETURN trim(n.title) AS title,
                   coalesce(n.order, 0) AS ord,
                   coalesce(n.depth, 99) AS depth,
                   labels(n)[0] AS label,
                   n.page_start AS page_start
            ORDER BY ord, depth, title
            """,
            doc_ids=doc_ids or None,
            tenant_id=tenant_id,
        )
        seen: set[str] = set()
        lines: list[str] = []
        for r in rows:
            title = (r.get("title") or "").strip()
            key = title.casefold()
            if not title or key in seen:
                continue
            seen.add(key)
            indent = "  " * max(0, min(int(r.get("depth") or 0), 4))
            page = r.get("page_start")
            lines.append(f"{indent}{title}" + (f"  (p.{page})" if page else ""))
        if not lines:
            return []
        return [{
            "id": f"{(doc_ids or ['?'])[0]}::outline",
            "title": "Table of contents",
            "text": "\n".join(lines),
            "score": 1.0,
            "related": [],
            "entry_count": len(lines),
        }]

    def by_address(
        self, session: Any, address: str, doc_ids: list[str], tenant_id: str = ""
    ) -> list[dict]:
        """The node a query addressed by number, e.g. "Box 9" or "section 4.2"."""
        m = _ADDRESS_PARTS.search(address or "")
        if not m:
            return []
        kind = m.group(1).lower().rstrip(".")
        number = m.group(2)
        label = _KIND_TO_LABEL.get(kind, "Section")
        region_kind = _KIND_TO_REGION.get(kind)

        # Page is addressed by its printed number, everything else by the
        # number appearing in its title/caption at a word boundary -- so
        # "Box 9" does not also match "Box 19" or "Box 9.1".
        if label == "Page":
            where = ("(toString(coalesce(n.document_page, n.page_start)) = $number "
                     "OR toString(n.pdf_page) = $number)")
        else:
            where = ("(toLower(coalesce(n.title, '')) =~ $pat "
                     "OR toLower(coalesce(n.search_text, '')) =~ $pat)")

        region_clause = " AND n.region_kind = $region_kind" if region_kind else ""
        rows = session.run(
            f"""
            MATCH (n:{label})
            WHERE {content_scope_where_multi("n", scoped=bool(doc_ids))}
              AND {tenant_filter("n")}
              AND {where}{region_clause}
            RETURN coalesce(n.id, '') AS id,
                   coalesce(n.title, '') AS title,
                   n.blob_key_text AS blob_key_text,
                   coalesce(n.search_text, '') AS search_text,
                   n.page_start AS page_start,
                   n.document_page AS document_page,
                   n.region_kind AS region_kind,
                   n.visual_content AS visual_content,
                   coalesce(n.order, 0) AS ord
            ORDER BY ord
            LIMIT 8
            """,
            doc_ids=doc_ids or None,
            tenant_id=tenant_id,
            number=number,
            pat=rf"(?s).*\b{re.escape(kind)}\s*{re.escape(number)}\b.*",
            region_kind=region_kind,
        )
        hydrator = get_hydrator()
        out: list[dict] = []
        for r in rows:
            if not r.get("id"):
                continue
            text = hydrator.hydrate(r.get("blob_key_text"), r.get("search_text") or "")
            out.append({
                "id": r["id"],
                "title": r.get("title") or r["id"],
                "text": text,
                "score": 1.0,
                "related": [],
                "page_start": r.get("page_start"),
                "document_page": r.get("document_page"),
                "region_kind": r.get("region_kind"),
                "visual_content": r.get("visual_content"),
            })
        return out
