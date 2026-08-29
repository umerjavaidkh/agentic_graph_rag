"""lexical.py — phrase/keyword CONTAINS retrieval, shared by retrieval strategies.

Extracted from mixins/lexical.py (LexicalRetrievalMixin). Depends on
RankingService (query keyword/phrase extraction) and DocumentResolver
(doc-scoped search) — constructed after both in the service bundle.
"""
from __future__ import annotations

import math
import re
from typing import Optional

from ...graph.constants import DOCUMENT_ROOT_CYPHER, INDEXED_NODE_CYPHER
from ....shared.config.settings import DEFAULT_LANGUAGE
from ....shared.neo4j.tenancy import language_filter, tenant_filter
from ....shared.storage.hydrator import get_hydrator
from ..constants import _TEXT_NODE_LABELS
from ..cypher_scope import (
    _doc_scope_cypher,
    as_doc_id_list,
    content_match_cypher,
    content_scope_where_multi,
)
from ..text_utils import _extract_urls
from .document_resolver import DocumentResolver
from .ranking import RankingService

# A query phrase may only scope retrieval when it is discriminating WITHIN
# the document -- present in some nodes, absent from most. Above this share
# of the document's text nodes a phrase is the document's own boilerplate
# (its subject's name, "Annual Report", a running header), which would scope
# retrieval to nearly everything and therefore scope it to nothing. Set well
# below the 0.40 genericity cutoff semantic/axis2.py uses for entity
# anchoring: that threshold answers "may this term justify a link between two
# sections", a far weaker claim than "may this term decide which section
# answers the question", so the bar here is deliberately stricter.
_SCOPE_PHRASE_MAX_DF_RATIO = 0.25
# Small on purpose -- these are pinned into a context window alongside the
# vector/graph candidates, so this is a precision instrument, not a
# recall net.
_SCOPE_PHRASE_LIMIT = 6

# Questions asking for a count/total, where the answer is a numeral rather
# than a passage. Phrasing only -- no domain vocabulary.
_QUANTITY_QUESTION_RE = re.compile(
    r"\b(how\s+many|how\s+much|number\s+of|total\s+number|count\s+of)\b", re.I
)
# Each noun costs one regex scan across the document's chunks, so cap it.
_QUANTITY_MAX_NOUNS = 6
_QUANTITY_LIMIT = 4


class LexicalService:
    def __init__(self, ranking: RankingService, document_resolver: DocumentResolver):
        self._ranking = ranking
        self._document_resolver = document_resolver

    def structural_keyword_retrieve(
        self,
        session,
        query: str,
        tenant_id: str = "",
        language: str = DEFAULT_LANGUAGE,
        document_id: Optional[str] = None,
        row_limit: int = 6,
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
            resolved, _ = self._document_resolver.resolve_document_for_query(
                session, query, tenant_id
            )
            doc_ids = as_doc_id_list(resolved)
        else:
            # "" (or []) means the caller resolved and got no confident
            # match — as_doc_id_list normalises it to None so the scope
            # predicate's empty branch degrades to unscoped, instead of
            # comparing every document's id against the literal empty
            # string and matching none of them.
            doc_ids = as_doc_id_list(document_id)

        # Document frequency per keyword — how many scoped nodes contain it
        # at all, regardless of the min_hits threshold below.
        freq_rows = session.run(
            f"""
            MATCH {content_match_cypher("n")}
            WHERE {content_scope_where_multi("n", scoped=bool(doc_ids))}
              AND n.search_text IS NOT NULL AND n.search_text <> ''
              AND {tenant_filter("n")} AND {language_filter("n")}
            UNWIND $keywords AS k
            WITH k, n WHERE toLower(n.search_text) CONTAINS k
            RETURN k AS keyword, count(DISTINCT n) AS df
            """,
            doc_ids=doc_ids,
            keywords=keywords,
            labels=list(_TEXT_NODE_LABELS),
            tenant_id=tenant_id,
            language=language,
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
            MATCH {content_match_cypher("n")}
            WHERE {content_scope_where_multi("n", scoped=bool(doc_ids))}
              AND n.search_text IS NOT NULL AND n.search_text <> ''
              AND {tenant_filter("n")} AND {language_filter("n")}
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
              n.document_page AS document_page,
              matched,
              w
            ORDER BY w DESC, size(coalesce(n.search_text, '')) ASC
            LIMIT $row_limit
            """,
            doc_ids=doc_ids,
            keywords=keywords,
            min_hits=min_hits,
            labels=list(_TEXT_NODE_LABELS),
            tenant_id=tenant_id,
            language=language,
            weight=weight,
            row_limit=int(row_limit),
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
                "document_page": r.get("document_page"),
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
        self,
        session,
        query: str,
        tenant_id: str = "",
        language: str = DEFAULT_LANGUAGE,
        document_id: Optional[str] = None,
        row_limit: int = 6,
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
            resolved, _ = self._document_resolver.resolve_document_for_query(
                session, query, tenant_id
            )
            doc_ids = as_doc_id_list(resolved)
        else:
            doc_ids = as_doc_id_list(document_id)
        rows = session.run(
            f"""
            MATCH {content_match_cypher("n")}
            WHERE {content_scope_where_multi("n", scoped=bool(doc_ids))}
              AND n.search_text IS NOT NULL AND n.search_text <> ''
              AND {tenant_filter("n")} AND {language_filter("n")}
              AND any(phrase IN $phrases WHERE toLower(n.search_text) CONTAINS phrase)
            OPTIONAL MATCH (d:Document)
              WHERE d.logical_doc_id = n.logical_doc_id
                AND d.lifecycle_status = '""" + "ACTIVE" + """'
            WITH n, d,
              size([p IN $phrases WHERE toLower(n.search_text) CONTAINS p]) AS phrase_hits
            RETURN
              coalesce(n.id, '') AS id,
              coalesce(n.title, '') AS title,
              n.blob_key_text AS blob_key_text,
              coalesce(n.search_text, '') AS search_text,
              n.page_start AS page_start,
              n.document_page AS document_page,
              phrase_hits,
              coalesce(d.title, d.id) AS doc_title
            ORDER BY phrase_hits DESC, size(coalesce(n.search_text, '')) ASC
            LIMIT $row_limit
            """,
            doc_ids=doc_ids,
            phrases=[p.lower() for p in phrases],
            row_limit=int(row_limit),
            labels=list(_TEXT_NODE_LABELS),
            tenant_id=tenant_id,
            language=language,
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
                "document_page": r.get("document_page"),
                "score": score,
                "related": ["via:phrase_search"],
            })
        return items

    def expand_unit_siblings(
        self, session, item_ids: list[str], tenant_id: str = "", language: str = DEFAULT_LANGUAGE
    ) -> list[dict]:
        """The other parts of any multi-chunk unit the hits belong to.

        A table continued across pages is recorded at ingestion as one unit
        (see Axis1StructuralBuilder._link_continuations). Retrieval still
        matched a single page of it, so a question about the whole table was
        answered from whichever part happened to rank -- a count over a
        three-page table came back from one page, with nothing to say the
        rest existed.

        Looked up BY HIT ID rather than by reading unit_id off the hits, so
        no retrieval query has to select the column and every strategy gets
        this without changing. Returns only the siblings; the hit itself is
        already in the candidate set.

        The labels are in the MATCH PATTERN, not a WHERE predicate, and the
        difference is 155 seconds. `WHERE any(l IN labels(n) WHERE l IN $x)`
        reads as a label filter but is an ordinary predicate: the planner
        still starts from AllNodesScan and filters afterwards, so it cannot
        use the id index either. Measured on a graph sharing 550k structured
        nodes, this one call took 155.3s of a 165.2s query -- and an earlier
        attempt that "added label filters" as WHERE predicates changed
        nothing at all, which is why the pattern form is spelled out here.
        """
        ids = [i for i in item_ids if i]
        if not ids:
            return []
        rows = session.run(
            f"""
            MATCH (hit:{INDEXED_NODE_CYPHER})
            WHERE hit.id IN $ids
              AND coalesce(hit.unit_id, '') <> ''
            MATCH (part:{INDEXED_NODE_CYPHER})
            WHERE part.unit_id = hit.unit_id
              AND part.revision_id = hit.revision_id
              AND part.id <> hit.id
              AND {tenant_filter("part")} AND {language_filter("part")}
            RETURN DISTINCT
              coalesce(part.id, '') AS id,
              coalesce(part.title, '') AS title,
              part.blob_key_text AS blob_key_text,
              coalesce(part.search_text, '') AS search_text,
              part.page_start AS page_start,
              part.document_page AS document_page,
              coalesce(part.unit_part, 0) AS unit_part
            ORDER BY unit_part
            LIMIT 12
            """,
            ids=ids,
            tenant_id=tenant_id,
            language=language,
        )
        hydrator = get_hydrator()
        return [
            {
                "id": r["id"],
                "title": r.get("title") or r["id"],
                "text": hydrator.hydrate(r.get("blob_key_text"), r.get("search_text") or ""),
                "page_start": r.get("page_start"),
                "document_page": r.get("document_page"),
                # Ranked with the hit that pulled it in: a continuation is
                # only here because its sibling earned a place.
                "score": 9.0,
                "related": ["via:unit_continuation"],
            }
            for r in rows
            if r.get("id")
        ]

    def quantity_evidence_retrieve(
        self,
        session,
        query: str,
        tenant_id: str = "",
        language: str = DEFAULT_LANGUAGE,
        document_id: Optional[str] = None,
    ) -> list[dict]:
        """
        For a counting question ("how many X"), retrieve chunks that actually
        contain a NUMBER counting X -- "115 institutions", "65 countries".

        Keyword retrieval cannot answer these, and not because of a bug: in a
        document about country implementations, "countries" and
        "institutions" appear nearly everywhere, so idf correctly judges them
        uninformative and every chunk ties. Verified live: "How many
        countries and institutions used Go.Data?" returned "this document
        does not cover", while the answering chunk (holding both "65
        countries" and "115 institutions") did not even place in the top ten
        keyword hits -- it matched the same terms as everything else and then
        lost the length tie-break for being the longest.

        What separates the answer from every other mention is not WHICH words
        appear but that a numeral sits immediately before the counted noun.
        That is a fact about how quantities are written in prose, not about
        this document or this domain, so the pattern generalizes: "12
        patients", "40 municipalities", "7 exhibits".
        """
        if not _QUANTITY_QUESTION_RE.search(query or ""):
            return []
        nouns = [
            k for k in self._ranking._content_keywords_from_query(query)
            if k.isalpha() and len(k) >= 4
        ][:_QUANTITY_MAX_NOUNS]
        if not nouns:
            return []

        if document_id is None:
            resolved, _ = self._document_resolver.resolve_document_for_query(
                session, query, tenant_id
            )
            doc_ids = as_doc_id_list(resolved)
        else:
            doc_ids = as_doc_id_list(document_id)

        # "<number> <noun>", allowing thousands separators and an optional
        # qualifier between them ("over 65 countries", "115 total institutions").
        patterns = [rf"(?s).*\d[\d,.]*\s+(?:\w+\s+){{0,2}}{re.escape(n)}.*" for n in nouns]
        rows = session.run(
            f"""
            MATCH {content_match_cypher("n")}
            WHERE {content_scope_where_multi("n", scoped=bool(doc_ids))}
              AND n.search_text IS NOT NULL AND n.search_text <> ''
              AND {tenant_filter("n")} AND {language_filter("n")}
              AND any(p IN $patterns WHERE toLower(n.search_text) =~ p)
            OPTIONAL MATCH (d:Document)
              WHERE d.logical_doc_id = n.logical_doc_id
                AND d.lifecycle_status = 'ACTIVE'
            WITH n, d,
              size([p IN $patterns WHERE toLower(n.search_text) =~ p]) AS matched
            RETURN
              coalesce(n.id, '') AS id,
              coalesce(n.title, '') AS title,
              n.blob_key_text AS blob_key_text,
              coalesce(n.search_text, '') AS search_text,
              n.page_start AS page_start,
              n.document_page AS document_page,
              matched,
              coalesce(d.title, d.id) AS doc_title
            ORDER BY matched DESC, size(coalesce(n.search_text, '')) DESC
            LIMIT $limit
            """,
            doc_ids=doc_ids,
            patterns=patterns,
            labels=list(_TEXT_NODE_LABELS),
            tenant_id=tenant_id,
            language=language,
            limit=_QUANTITY_LIMIT,
        )

        hydrator = get_hydrator()
        items: list[dict] = []
        for r in rows:
            if not r.get("id"):
                continue
            title = r.get("title") or r["id"]
            full_text = hydrator.hydrate(r.get("blob_key_text"), r.get("search_text") or "")
            items.append({
                "id": r["id"],
                "title": title,
                "text": self.enrich_chunk_text_for_facts(title, full_text),
                "page_start": r.get("page_start"),
                "document_page": r.get("document_page"),
                # A chunk quantifying BOTH nouns the question asked about
                # ("65 countries" and "115 institutions") answers it outright,
                # so it must outrank one quantifying only half of it.
                "score": 1.0 + float(r.get("matched") or 0),
                "related": ["via:quantity_evidence"],
                "doc_title": r.get("doc_title"),
            })
        return items

    def scope_phrase_retrieve(
        self,
        session,
        query: str,
        tenant_id: str = "",
        language: str = DEFAULT_LANGUAGE,
        document_id: Optional[str] = None,
    ) -> list[dict]:
        """
        Retrieve chunks by a SHORT, discriminating scope phrase from the query
        ("International Upstream"), rather than the long whole-question
        n-grams structural_phrase_retrieve uses.

        Why this exists: a filing repeats identical row labels under every
        segment -- verified live on a real 10-K, 15 nodes contain "net
        oil-equivalent production", 18 contain "liquids production". The only
        token distinguishing one segment's table from another's is the scope
        heading, and it is a few characters inside a ~2,000-character chunk,
        so vector cosine cannot separate them: asked for International
        Upstream's liquids production (962 MBD) the pipeline answered with a
        sibling segment's figure instead. Long phrases could not rescue it
        either -- they match nothing, because the document states facts in its
        own words, not the question's.

        The document-frequency guard is what keeps this general rather than a
        list of segment names: a phrase is only allowed to scope retrieval
        when it is DISCRIMINATING within this document (present in some nodes,
        absent from most). "Chevron" and "Annual Report" appear nearly
        everywhere and are dropped automatically; "International Upstream"
        appears in a handful and survives. Nothing here knows what a segment
        is, so it works the same way for regions, notes, exhibits or chapters
        in any other document.

        `document_id`: see structural_keyword_retrieve — same
        skip-re-resolution contract.
        """
        phrases = self._ranking.scope_phrases_from_query(query)
        if not phrases:
            return []

        if document_id is None:
            resolved, _ = self._document_resolver.resolve_document_for_query(
                session, query, tenant_id
            )
            doc_ids = as_doc_id_list(resolved)
        else:
            doc_ids = as_doc_id_list(document_id)

        lowered = [p.lower() for p in phrases]
        # One pass to measure each phrase's document frequency, so a phrase
        # that is really boilerplate never gets to scope anything.
        stats = session.run(
            f"""
            MATCH {content_match_cypher("n")}
            WHERE {content_scope_where_multi("n", scoped=bool(doc_ids))}
              AND n.search_text IS NOT NULL AND n.search_text <> ''
              AND {tenant_filter("n")} AND {language_filter("n")}
            WITH collect(toLower(n.search_text)) AS texts
            UNWIND $phrases AS phrase
            RETURN phrase,
                   size([t IN texts WHERE t CONTAINS phrase]) AS df,
                   size(texts) AS total
            """,
            doc_ids=doc_ids,
            phrases=lowered,
            labels=list(_TEXT_NODE_LABELS),
            tenant_id=tenant_id,
            language=language,
        )
        # Weight each surviving phrase by RARITY, don't treat them as equal.
        # A frequency CUTOFF alone is not enough: verified live, a question
        # mentioning "the Chevron 2025 Annual Report" yields both "Annual
        # Report" (27/632 chunks) and "International Upstream" (12/632).
        # Both clear any sane cutoff, but only one names the scope -- and
        # weighted equally, short chunks matching merely "Annual Report"
        # outranked the segment's own table, which is the exact failure this
        # whole path exists to fix. idf makes the rarer, more specific phrase
        # dominate, the same way axis2 weights entity anchors.
        scoping: list[str] = []
        weights: list[float] = []
        for row in stats:
            df, total = int(row.get("df") or 0), int(row.get("total") or 0)
            if total <= 0 or df <= 0:
                continue
            if df / total <= _SCOPE_PHRASE_MAX_DF_RATIO:
                scoping.append(row["phrase"])
                weights.append(math.log(total / df) + 1.0)
        if not scoping:
            return []

        rows = session.run(
            f"""
            MATCH {content_match_cypher("n")}
            WHERE {content_scope_where_multi("n", scoped=bool(doc_ids))}
              AND n.search_text IS NOT NULL AND n.search_text <> ''
              AND {tenant_filter("n")} AND {language_filter("n")}
              AND any(phrase IN $phrases WHERE toLower(n.search_text) CONTAINS phrase)
            OPTIONAL MATCH (d:Document)
              WHERE d.logical_doc_id = n.logical_doc_id
                AND d.lifecycle_status = 'ACTIVE'
            WITH n, d,
              reduce(w = 0.0, i IN range(0, size($phrases) - 1) |
                w + CASE WHEN toLower(n.search_text) CONTAINS $phrases[i]
                         THEN $weights[i] ELSE 0.0 END) AS phrase_weight
            RETURN
              coalesce(n.id, '') AS id,
              coalesce(n.title, '') AS title,
              n.blob_key_text AS blob_key_text,
              coalesce(n.search_text, '') AS search_text,
              n.page_start AS page_start,
              n.document_page AS document_page,
              phrase_weight,
              coalesce(d.title, d.id) AS doc_title
            ORDER BY phrase_weight DESC, size(coalesce(n.search_text, '')) DESC
            LIMIT $limit
            """,
            doc_ids=doc_ids,
            phrases=scoping,
            weights=weights,
            labels=list(_TEXT_NODE_LABELS),
            tenant_id=tenant_id,
            language=language,
            limit=_SCOPE_PHRASE_LIMIT,
        )

        hydrator = get_hydrator()
        items: list[dict] = []
        for r in rows:
            if not r.get("id"):
                continue
            title = r.get("title") or r["id"]
            full_text = hydrator.hydrate(r.get("blob_key_text"), r.get("search_text") or "")
            items.append({
                "id": r["id"],
                "title": title,
                "text": self.enrich_chunk_text_for_facts(title, full_text),
                "page_start": r.get("page_start"),
                "document_page": r.get("document_page"),
                # Scaled by how many of the query's scope phrases the chunk
                # satisfies -- a chunk matching both "International" and
                # "Upstream" scoping is a better scope match than one
                # matching either alone.
                "score": 1.0 + float(r.get("phrase_weight") or 0.0),
                "related": ["via:scope_phrase"],
                "doc_title": r.get("doc_title"),
            })
        return items
