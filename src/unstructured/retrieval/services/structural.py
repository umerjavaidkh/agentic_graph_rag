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

from ....shared.config.settings import DEFAULT_LANGUAGE
from ....shared.neo4j.tenancy import language_filter, tenant_filter
from ....shared.storage.hydrator import get_hydrator
from ..cypher_scope import content_scope_where_multi, match_key_cypher

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

# "how many chapters", "number of tables", "how many figures are there"
_COUNT_ASK_UNIT = re.compile(
    r"\b(?:how\s+many|number\s+of|count\s+of|total\s+(?:number\s+of\s+)?)\s+"
    r"(chapters?|sections?|figures?|tables?|boxes|box|pages?|appendi(?:x|ces)|annexes?)\b",
    re.I,
)
_UNIT_TO_LABEL = {
    "chapter": "Chapter", "section": "Section", "appendix": "Section",
    "appendices": "Section", "annex": "Section",
    "figure": "Region", "table": "Region", "box": "Region", "boxes": "Region",
    "page": "Page",
}
_UNIT_TO_REGION = {"figure": "figure", "table": "table", "box": "table",
                   "boxes": "table"}


def _singular(unit: str) -> str:
    u = unit.lower()
    if u == "appendices":
        return "appendix"
    if u == "boxes":
        return "box"
    return u[:-1] if u.endswith("s") and not u.endswith("ss") else u


class StructuralService:
    """Reads of the document hierarchy, by address."""

    def outline(
        self,
        session: Any,
        doc_ids: list[str],
        tenant_id: str = "",
        language: str = DEFAULT_LANGUAGE,
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
              AND {tenant_filter("n")} AND {language_filter("n")}
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
            language=language,
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

    def count_units(
        self,
        session: Any,
        query: str,
        doc_ids: list[str],
        tenant_id: str = "",
        language: str = DEFAULT_LANGUAGE,
    ) -> list[dict]:
        """How many chapters/tables/figures/pages a document has, by counting them.

        The count is a fact about the graph, and the graph is the only place
        it exists: no sentence in NIST SP 800-161 says how many tables it
        contains. Asked to count from prose, the model read a "List of
        Tables" fragment and answered 23 against an actual 88, and answered
        "3 main chapters" against 15 -- confidently both times, because a
        plausible-looking list was in front of it.

        Counting the nodes cannot be confidently wrong in that way. Titled
        units are counted by distinct title, matching `outline()` above, so a
        heading split across two nodes is one chapter in both answers rather
        than one here and two there.
        """
        m = _COUNT_ASK_UNIT.search(query or "")
        if not m or not doc_ids:
            return []
        unit = _singular(m.group(1))
        label = _UNIT_TO_LABEL.get(unit)
        if not label:
            return []
        region_kind = _UNIT_TO_REGION.get(unit)

        titled = label in ("Chapter", "Section")
        counted = (
            "count(DISTINCT toLower(trim(n.title)))" if titled else "count(n)"
        )
        title_clause = (
            " AND trim(coalesce(n.title, '')) <> ''" if titled else ""
        )
        region_clause = " AND n.region_kind = $region_kind" if region_kind else ""
        row = session.run(
            f"""
            MATCH (n:{label})
            WHERE {content_scope_where_multi("n", scoped=True)}
              AND {tenant_filter("n")} AND {language_filter("n")}{title_clause}{region_clause}
            RETURN {counted} AS total
            """,
            doc_ids=doc_ids,
            tenant_id=tenant_id,
            language=language,
            region_kind=region_kind,
        ).single()
        total = int((row or {}).get("total") or 0)
        if not total:
            return []

        plural = unit + ("es" if unit in ("box",) else "s")
        text = f"This document contains {total} {plural if total != 1 else unit}."
        if titled:
            names = self.outline(session, doc_ids, tenant_id)
            if names:
                text += "\n\n" + names[0]["text"]
        return [{
            "id": "graph_count",
            "title": f"Number of {plural}",
            "text": text,
            "score": 1.0,
            "related": [],
            "count": total,
            "unit": plural,
        }]

    def by_address(
        self,
        session: Any,
        address: str,
        doc_ids: list[str],
        tenant_id: str = "",
        language: str = DEFAULT_LANGUAGE,
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
            # Two shapes, because documents write addresses two ways. A
            # heading carries the bare number -- "1.1. Purpose",
            # "2. INTEGRATION OF C-SCRM" -- and never the word "section" or
            # "chapter", so requiring the kind word matched nothing and both
            # "what does Section 1.1 say" and "summarize Chapter 2" fell
            # through to a hybrid search that reported the section missing.
            # A box or figure caption does carry its kind word ("Box 9"), so
            # that form is kept alongside.
            where = ("(toLower(coalesce(n.title, '')) =~ $numpat "
                     "OR toLower(coalesce(n.title, '')) =~ $pat "
                     f"OR {match_key_cypher('n')} =~ $pat)")

        region_clause = " AND n.region_kind = $region_kind" if region_kind else ""
        rows = session.run(
            f"""
            MATCH (n:{label})
            WHERE {content_scope_where_multi("n", scoped=bool(doc_ids))}
              AND {tenant_filter("n")} AND {language_filter("n")}
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
            language=language,
            number=number,
            pat=rf"(?s).*\b{re.escape(kind)}\s*{re.escape(number)}\b.*",
            # Anchored, and requiring a separator after the number, so "1.1"
            # does not also match the "1.1.1" beneath it.
            numpat=rf"(?s)\s*{re.escape(number)}\.?[\s).:-].*",
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
