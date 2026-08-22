"""document_resolver.py — logical-document resolution, shared by retrieval strategies.

Extracted from mixins/document_resolver.py (DocumentResolverMixin). Depends
on GraphSeedService (embedding + vector-seed, used by the vector-majority
resolution path) — constructed after GraphSeedService in the service bundle.
Depended on by nearly every strategy (Toc, Page, Box, Subsection, Lexical,
FullHybrid all need "which document is this question about").
"""
from __future__ import annotations

import math
import re
from typing import Optional

from ...graph.constants import (
    DOC_REVISION_LABEL,
    DOCUMENT_LOGICAL_LABEL,
    DOCUMENT_ROOT_CYPHER,
)
from ....shared.neo4j.tenancy import tenant_filter
from ....shared.neo4j.versioning import lifecycle_active
from ..cypher_scope import _clean_doc_title
from ..query_intent import KEYWORD_STOP as _KEYWORD_STOP
from ..text_utils import _query_anchor_terms
from .graph_seeds import GraphSeedService

# A structural reference inside a question ("Note 3 (Commitments and
# Contingencies)", "Box 9", "Item 7") names a location WITHIN whichever
# document the conversation is already about, not a different document to
# search for. Standard footnote/item titles like "Commitments and
# Contingencies" are boilerplate shared across most filings in a corpus, so
# left unstripped they get picked up by the mid-sentence-capitalization scan
# below and wrongly treated as document-identifying anchors -- verified
# live: this single-handedly overrode a correct conversation document hint,
# resolving "What does Note 3 (Commitments and Contingencies) discuss?" to
# an unrelated document that merely had more occurrences of those generic
# legal/financial terms.
_STRUCTURAL_REF_RE = re.compile(
    r"\b(?:note|box|item|figure|fig\.?|section)\s+(?:no\.?\s*)?\d+[a-z]?(?:\.\d+)*\b", re.I
)
_PAREN_RE = re.compile(r"\([^)]*\)")

# Cypher regex fragment for word-boundary term matching, used in place of a
# raw `CONTAINS term` substring check anywhere a query term is matched
# against node title/text or a document's own title/logical_id. Plain
# substring matching lets a short term spuriously match INSIDE unrelated
# longer words -- verified live: "EPS" (picked up as a doc-name anchor from
# "diluted EPS", an all-caps mid-sentence token) substring-matched "st-EPS-",
# "sw-EPS-" throughout an unrelated physics textbook via raw CONTAINS,
# outnumbering a financial filing's genuine "EPS" mentions and winning
# strict document resolution -- which runs BEFORE the conversation's
# thread-continuity hint is even consulted, so the wrong document then got
# saved as the new hint and silently corrupted every subsequent generic
# follow-up in that thread until a strongly distinctive term happened to
# reset it. `\b` is a safe general fix (never introduces a new match CONTAINS
# didn't already have, only removes accidental substring collisions) and
# works correctly across hyphenated/structured ids too (hyphens are
# non-word characters, so "10k" still \b-matches inside "gs-10k-2026-02-25").
# A term must not match inside a longer word ("559" must not hit "8559"),
# but \b is the wrong tool for document identifiers: there is no word
# boundary between "p" and "559" in "doc_irs_p559", so a publication number
# -- the single most distinctive thing a user can name -- matched nothing at
# all. Digit runs are bounded by digits instead, which is what actually
# distinguishes "559" from "8559" while still matching "p559" and "ir8286".
_WORD_BOUNDARY_PATTERN = "('(?s).*\\\\b' + term + '\\\\b.*')"
_DIGIT_BOUNDARY_PATTERN = "('(?s).*(?<![0-9])' + term + '(?![0-9]).*')"
_POSSESSIVE_RE = re.compile(r"'s?$")

#: "Publication 559", "Pub. 502", "NIST IR 8286", "SP 800-53", "Form 1040".
#: Anchored on the document-type word so a dollar amount or a year in the
#: question is never taken for an identifier.
_DOC_NUMBER_RE = re.compile(
    r"\b(?:publication|pub\.?|form|report|bulletin|circular|ir|nistir|sp|no\.?|#)\s*"
    r"([0-9]{2,5}(?:[-–][0-9]{1,4})?[a-z]?)\b",
    re.I,
)


class DocumentResolver:
    def __init__(self, graph_seeds: GraphSeedService):
        self._graph_seeds = graph_seeds

    def score_documents_for_query(
        self, session, query: str, tenant_id: str = ""
    ) -> list[tuple[float, str, str]]:
        """Every candidate document for `query`, ranked, as (score, id, title).

        The ranking the strict resolver already computed and then discarded
        whenever it could not pick a winner. Keeping it is what lets a
        caller offer the choice to the user instead of guessing or giving
        up: on a 50-document corpus an unscoped question lands on the wrong
        document often enough that guessing is the worse option.
        """
        return self._scored_documents(session, query, tenant_id)

    def _scored_documents(
        self, session, query: str, tenant_id: str = ""
    ) -> list[tuple[float, str, str]]:
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
        terms = self.doc_name_terms(query)
        if not terms:
            return []

        lc = lifecycle_active("d")
        lc_n = lifecycle_active("n")
        rows = session.run(
            f"""
            MATCH (dl:{DOCUMENT_LOGICAL_LABEL})-[:ACTIVE_REVISION]
                  ->(:{DOC_REVISION_LABEL})-[:ROOT]->(d:{DOCUMENT_ROOT_CYPHER})
            WHERE {lc}
              AND {tenant_filter("dl")}
              AND {tenant_filter("d")}
            WITH dl, d
            UNWIND $terms AS term
            OPTIONAL MATCH (d)-[:CONTAINS*1..6]->(n)
            WHERE {lc_n}
              AND {tenant_filter("n")}
              AND (toLower(coalesce(n.title, '')) =~ {_WORD_BOUNDARY_PATTERN}
                   OR toLower(coalesce(n.search_text, '')) =~ {_WORD_BOUNDARY_PATTERN}
                   OR (term =~ '\\d+' AND (
                        toLower(coalesce(n.title, '')) =~ {_DIGIT_BOUNDARY_PATTERN}
                        OR toLower(coalesce(n.search_text, '')) =~ {_DIGIT_BOUNDARY_PATTERN})))
            WITH dl, term, count(DISTINCT n) AS cnt,
                 // Bare 4-digit years never count as a title match: our own
                 // logical_ids are systematically date-suffixed (ticker_form_
                 // YYYY-MM-DD), so a query mentioning any year ("as of
                 // December 31, 2025") spuriously "title-matches" every
                 // filing from that year via id substring alone, handing it
                 // the 1000x identity bonus meant for real name matches
                 // (verified live: this let an unrelated Costco 10-K outrank
                 // Chevron's own 10-K, whose id doesn't happen to contain
                 // the query's year).
                 (NOT term =~ '\\d{{4}}'
                  AND (toLower(coalesce(dl.title, '')) =~ {_WORD_BOUNDARY_PATTERN}
                       OR toLower(dl.logical_id) =~ {_WORD_BOUNDARY_PATTERN}
                       OR (term =~ '\\d+' AND (
                            toLower(coalesce(dl.title, '')) =~ {_DIGIT_BOUNDARY_PATTERN}
                            OR toLower(dl.logical_id) =~ {_DIGIT_BOUNDARY_PATTERN})))) AS title_match
            RETURN dl.logical_id AS id,
                   coalesce(dl.title, dl.logical_id) AS title,
                   collect({{term: term, cnt: cnt, title_match: title_match}}) AS term_hits
            """,
            terms=terms,
            tenant_id=tenant_id,
        )

        docs: list[dict] = [dict(r) for r in rows]
        if not docs:
            return []

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
            # log-IDF, not linear: a term present in most of the corpus
            # should contribute only a token amount, not a merely-reduced
            # multiple of a raw count that can still run into the hundreds
            # in a large document (verified live: linear IDF's ~1.7x
            # downweight on a term shared by 3 of 5 documents wasn't enough
            # to stop a 130+ raw count from nearly canceling out a
            # genuinely distinctive term's contribution). The +0.1 floor
            # keeps a term appearing in EVERY current document from going
            # to exactly zero, preserving this tier's documented fallback
            # of guessing by size when nothing in the query is distinctive.
            return math.log(float(total_docs) / float(df)) + 0.1

        scored: list[tuple[float, str, str]] = []
        for d in docs:
            score = 0.0
            for h in d["term_hits"]:
                w = term_weight(h["term"])
                if h["title_match"]:
                    score += 1000.0 * w  # title/id match dominates
                # log1p, not a raw count: a document that's simply much
                # larger has proportionally more nodes containing ANY given
                # term, so raw counts let sheer size dominate even a
                # correctly-IDF-weighted score (verified live: "note"/
                # "commitments"/"contingencies" occurring 500+/130+/14+
                # times in one large filing outweighed a genuinely
                # distinctive term occurring 34 times in the right, smaller
                # document). Saturating the count lets distinctiveness
                # (the weight) decide close calls instead of magnitude.
                score += math.log1p(float(h["cnt"])) * w
            scored.append((score, str(d["id"]), _clean_doc_title(str(d["title"] or d["id"]))))

        scored.sort(key=lambda x: x[0], reverse=True)
        return scored

    #: A winner must lead the runner-up by this much to be taken without
    #: asking. Below it the two documents are, on the evidence in the query,
    #: equally plausible.
    AMBIGUITY_LEAD = 1.5

    def resolve_document_for_query_strict(
        self, session, query: str, tenant_id: str = ""
    ) -> tuple[Optional[str], Optional[str]]:
        """The single document this query names, or (None, None).

        Unchanged behaviour: returns nothing when there is no real match or
        when the top two are too close to separate, so the caller can ask
        rather than guess.
        """
        scored = self._scored_documents(session, query, tenant_id)
        if not scored or scored[0][0] <= 0.0:
            return None, None

        if len(scored) > 1:
            runner = scored[1][0]
            if runner > 0.0 and scored[0][0] < runner * self.AMBIGUITY_LEAD:
                return None, None

        return scored[0][1], scored[0][2]

    def names_an_unresolvable_document(
        self, session, query: str, tenant_id: str = ""
    ) -> bool:
        """Did this query name a document that cannot be pinned down?

        Distinct from naming none at all, and the two want opposite
        handling. A question with no document vocabulary ("what's on page
        6?") should fall through to the thread, the vectors, and finally
        the largest document -- guessing is the only thing left, and it is
        usually right.

        A near-tie is not that. The user named something; two documents
        simply match it about equally, and picking one is wrong about as
        often as it is right. Verified: "the arXiv paper Attention Is All
        You Need" scored arxiv_attention at 1.000 and arxiv_t5 at 0.972,
        and the lower tiers returned t5.
        """
        scored = [x for x in self._scored_documents(session, query, tenant_id) if x[0] > 0.0]
        return len(scored) > 1 and scored[0][0] < scored[1][0] * self.AMBIGUITY_LEAD

    def candidates_for_query(
        self, session, query: str, tenant_id: str = "", limit: int = 10
    ) -> list[dict]:
        """Plausible documents for a query that named none clearly.

        Empty when one document wins outright -- there is nothing to ask
        about -- and empty when nothing matches at all, since a list of
        documents that match nothing is worse than saying so.
        """
        scored = [s for s in self._scored_documents(session, query, tenant_id) if s[0] > 0.0]
        if not scored:
            return []
        if len(scored) == 1 or scored[0][0] >= scored[1][0] * self.AMBIGUITY_LEAD:
            return []          # unambiguous; the resolver will just use it

        top = scored[0][0]
        return [
            {"document_id": doc_id, "title": title, "score": round(score, 2),
             "relative": round(score / top, 3)}
            for score, doc_id, title in scored[:limit]
        ]

    @staticmethod
    def _logical_id_from_node_id(node_id: str) -> Optional[str]:
        """Extract the logical document id prefix from a content node id."""
        if not node_id:
            return None
        # Revision-scoped ids look like "<logical_id>:<rev>::<...>".
        return node_id.split(":", 1)[0] or None

    def resolve_document_by_vector(
        self, session, query: str, tenant_id: str = ""
    ) -> tuple[Optional[str], Optional[str]]:
        """
        Resolve the target document via semantic similarity (corpus-agnostic):
        embed the query, take the top vector seeds, and pick the logical document
        that owns a clear majority of them. No per-document or per-topic terms.
        """
        try:
            embedding = self._graph_seeds.get_embedding(query)
        except Exception:
            return None, None
        if not embedding:
            return None, None

        seeds = self._graph_seeds.vector_seed(session, embedding, 12, tenant_id)
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

        title = self.document_title_for_logical_id(session, top_id, tenant_id)
        return top_id, title

    def document_title_for_logical_id(
        self, session, logical_id: str, tenant_id: str = ""
    ) -> Optional[str]:
        if not logical_id:
            return None
        row = session.run(
            f"""
            MATCH (dl:{DOCUMENT_LOGICAL_LABEL})
            WHERE dl.logical_id = $lid
              AND {tenant_filter("dl")}
            RETURN coalesce(dl.title, dl.logical_id) AS title
            LIMIT 1
            """,
            lid=logical_id,
            tenant_id=tenant_id,
        ).single()
        if row and row.get("title"):
            return _clean_doc_title(str(row["title"]))
        return _clean_doc_title(logical_id)

    @staticmethod
    def _pick_best_by_term_weight(
        rows, terms: list[str], *, require_distinctive: bool = False
    ) -> Optional[tuple[str, str]]:
        """Shared IDF-weighted scoring for a `collect({term, cnt, title_match})
        per document` result set — same approach as
        resolve_document_for_query_strict, factored out so the generic
        (document_match_terms-driven) fallback tier can use it too instead
        of a flat count() that favors whichever document has the most
        content overall regardless of relevance. Unlike the strict
        resolver, does not require a clear lead over the runner-up — this
        tier is already the lower-confidence fallback, so ties are broken
        by whatever Neo4j returns first rather than declining to guess.

        require_distinctive: decline (return None) unless at least one
        matched term is non-universal (0 < df < total docs) — i.e. actually
        excludes some document in the corpus, not just present-in-all
        boilerplate. Used by resolve_document_for_query to gate whether
        this tier is trustworthy enough to run BEFORE vector-majority,
        which has no defense against corpus-size skew of its own.
        """
        docs: list[dict] = [dict(r) for r in rows]
        if not docs:
            return None

        term_doc_freq: dict[str, int] = {t: 0 for t in terms}
        for d in docs:
            for h in d["term_hits"]:
                if h["cnt"] > 0 or h["title_match"]:
                    term_doc_freq[h["term"]] = term_doc_freq.get(h["term"], 0) + 1

        total_docs = len(docs)
        if require_distinctive and not any(
            0 < df < total_docs for df in term_doc_freq.values()
        ):
            return None

        def term_weight(term: str) -> float:
            df = term_doc_freq.get(term, 0)
            if df <= 0:
                return 0.0
            # See resolve_document_for_query_strict's identical formula for
            # why this is log-IDF with a +0.1 floor, not linear IDF.
            return math.log(float(total_docs) / float(df)) + 0.1

        scored: list[tuple[float, str, str]] = []
        for d in docs:
            score = 0.0
            for h in d["term_hits"]:
                w = term_weight(h["term"])
                if h["title_match"]:
                    score += 1000.0 * w
                # log1p, not a raw count -- see resolve_document_for_query_strict's
                # identical comment; the same size-bias reaches this tier too.
                score += math.log1p(float(h["cnt"])) * w
            scored.append((score, str(d["id"]), _clean_doc_title(str(d["title"] or d["id"]))))

        scored.sort(key=lambda x: x[0], reverse=True)
        top = scored[0]
        if top[0] <= 0.0:
            return None
        return top[1], top[2]

    def _validate_document_id(
        self, session, logical_id: str, tenant_id: str = ""
    ) -> Optional[str]:
        """Unlike document_title_for_logical_id (which always returns
        *something*, falling back to the id itself), this returns None when
        the id doesn't correspond to a real, active document — used to
        confirm a conversation-carried document_id hint is still valid
        before trusting it (e.g. the document could have been deleted or
        expired since the prior turn)."""
        if not logical_id:
            return None
        row = session.run(
            f"""
            MATCH (dl:{DOCUMENT_LOGICAL_LABEL})
            WHERE dl.logical_id = $lid
              AND {tenant_filter("dl")}
            RETURN coalesce(dl.title, dl.logical_id) AS title
            LIMIT 1
            """,
            lid=logical_id,
            tenant_id=tenant_id,
        ).single()
        if row and row.get("title"):
            return _clean_doc_title(str(row["title"]))
        return None

    @staticmethod
    def _normalise_reference(text: str) -> str:
        """Lowercase, with every run of non-alphanumerics collapsed to one space.

        Lets a question written as "rag_document_2" match a title stored as
        "rag document 2" or "rag-document-2". Document ids and titles are
        punctuated inconsistently -- filenames use underscores, display titles
        use spaces, filing ids use hyphens -- and none of that changes which
        document is meant.
        """
        return re.sub(r"[^a-z0-9]+", " ", (text or "").lower()).strip()

    def exact_document_reference(
        self, session, query: str, tenant_id: str = ""
    ) -> Optional[tuple[str, str]]:
        """The document whose id or title the question states outright.

        The strongest signal there is, and the one the heuristics miss: a
        title like "rag_document_2" is lowercase with underscores and a digit,
        so it is neither an anchor term nor a mid-sentence proper noun, and
        strict resolution declined on it. The thread's own document then won
        by default -- so a question naming one document was answered from
        another, and labelled as the one that was asked for.

        Widening the heuristics was not an option: this method's own comments
        record that treating ordinary vocabulary as document names made a
        638-section 10-K beat an unrelated policy document on the strength of
        "employees" and "benefits". Matching against ids and titles that
        actually exist cannot do that, because nothing generic is in the list.

        Ambiguity declines rather than guesses: if two documents share a title
        -- which happens, the same filing ingested twice -- neither is a safe
        answer, and the ordinary resolution path is better placed to choose.
        """
        haystack = self._normalise_reference(query)
        # No session means no catalogue to compare against -- callers that
        # resolve without one (and tests that stub the tiers around this)
        # must fall through to the next tier rather than fail.
        if not haystack or session is None:
            return None
        rows = session.run(
            f"""
            MATCH (dl:{DOCUMENT_LOGICAL_LABEL})-[:ACTIVE_REVISION]->(:{DOC_REVISION_LABEL})
            WHERE {tenant_filter("dl")}
            RETURN dl.logical_id AS logical_id, dl.title AS title
            """,
            tenant_id=tenant_id,
        )
        matches: dict[str, str] = {}
        for row in rows:
            # Tolerant read: this tier runs inside a chain whose other steps
            # issue different queries, so a row that carries neither field is
            # simply not a document to match against.
            try:
                logical_id, title = row.get("logical_id"), row.get("title")
            except Exception:
                continue
            if not logical_id:
                continue
            for candidate in (logical_id, title):
                needle = self._normalise_reference(candidate or "")
                # Guard against a one-word title ("Policy") matching any
                # sentence containing that word: an outright reference is a
                # multi-token identifier, not a single common noun.
                if len(needle) < 6 or " " not in needle:
                    continue
                if needle in haystack:
                    matches[logical_id] = title or logical_id
        if len(matches) != 1:
            return None
        logical_id, title = next(iter(matches.items()))
        return logical_id, title

    def resolve_document_for_query(
        self, session, query: str, tenant_id: str = "", document_id_hint: str = ""
    ) -> tuple[Optional[str], Optional[str]]:
        """Return logical document id (preferred) and display title for doc-scoped retrieval.

        `document_id_hint`: the document the current conversation thread was
        already discussing (see conversation/thread_memory.py), checked
        after strict name-matching but before vector-majority resolution —
        an explicitly named document always wins (a real topic switch), but
        for a question with no distinguishing vocabulary of its own ("what's
        on page 6 of this document"), staying on the document the thread was
        already about is a better default than vector-majority's fallback
        behavior of favoring whichever document has the most content
        overall, regardless of relevance.

        Priority: strict name > hint > confident distinctive term (see
        require_distinctive on _pick_best_by_term_weight) > vector-majority
        > any term match (lower bar) > largest-document fallback. The
        confident-term tier runs before vector-majority specifically
        because vector-majority has no defense against corpus-size skew —
        a genuinely distinctive keyword in the query is more trustworthy
        than embedding-space nearest-neighbor voting, which naturally
        favors whichever document is simply larger.
        """
        strict_id, strict_title = self.resolve_document_for_query_strict(session, query, tenant_id)
        if strict_id:
            return strict_id, strict_title

        # An outright reference to a document's id or title outranks the thread.
        # Staying on the thread's document is the right default for a question
        # with no vocabulary of its own ("what's on page 6?"), but it must not
        # override a question that names a different document.
        named = self.exact_document_reference(session, query, tenant_id)
        if named:
            return named

        if document_id_hint:
            hint_title = self._validate_document_id(session, document_id_hint, tenant_id)
            if hint_title is not None:
                return document_id_hint, hint_title

        # Below this point every tier guesses -- by broad term overlap, by
        # vector majority, and finally by document size. That is the right
        # behaviour for a question carrying no document name, which is what
        # those tiers were built for.
        #
        # It is the wrong behaviour when the query named a document clearly
        # enough to produce two close candidates: the guess then lands on
        # the runner-up about as often as the winner, and the user is given
        # a confident answer from the wrong document. Declining here lets
        # the caller offer the candidates instead, which is the one thing
        # that reliably resolves a tie between documents that share a
        # vocabulary. The thread hint above still wins, so follow-ups in a
        # grounded conversation are unaffected.
        if self.names_an_unresolvable_document(session, query, tenant_id):
            return None, None

        terms = self.document_match_terms(query)
        lc = lifecycle_active("d")
        lc_n = lifecycle_active("n")
        term_rows: list[dict] = []
        if terms:
            # IDF-weighted, not a flat count() — document_match_terms is
            # deliberately broader/lower-confidence than doc_name_terms (it
            # includes ordinary content words, not just anchors/proper
            # nouns), so without weighting, a merely-common word ("annual",
            # "report") reliably favors whichever document has the most
            # content overall, regardless of relevance — the same failure
            # mode already fixed in resolve_document_for_query_strict,
            # reachable here too since this tier runs after that one
            # declines. Mirrors that method's exact scoring approach.
            term_rows = [
                dict(r)
                for r in session.run(
                    f"""
                    MATCH (dl:{DOCUMENT_LOGICAL_LABEL})-[:ACTIVE_REVISION]
                          ->(:{DOC_REVISION_LABEL})-[:ROOT]->(d:{DOCUMENT_ROOT_CYPHER})
                    WHERE {lc}
                      AND {tenant_filter("dl")}
                      AND {tenant_filter("d")}
                    WITH dl, d
                    UNWIND $terms AS term
                    OPTIONAL MATCH (d)-[:CONTAINS*1..5]->(n)
                    WHERE {lc_n}
                      AND {tenant_filter("n")}
                      AND (toLower(coalesce(n.title, '')) =~ {_WORD_BOUNDARY_PATTERN}
                           OR toLower(coalesce(n.search_text, '')) =~ {_WORD_BOUNDARY_PATTERN})
                    WITH dl, term, count(DISTINCT n) AS cnt,
                         (NOT term =~ '\\d{{4}}'
                          AND (toLower(coalesce(dl.title, '')) =~ {_WORD_BOUNDARY_PATTERN}
                               OR toLower(dl.logical_id) =~ {_WORD_BOUNDARY_PATTERN})) AS title_match
                    RETURN dl.logical_id AS id, coalesce(dl.title, dl.logical_id) AS title,
                           collect({{term: term, cnt: cnt, title_match: title_match}}) AS term_hits
                    """,
                    terms=terms,
                    tenant_id=tenant_id,
                )
            ]
            # A literal, distinctive keyword match ("amazon") is a more
            # reliable signal than vector-majority's embedding-space
            # voting, which has no defense against corpus-size skew (a
            # bigger document naturally owns more of any top-K nearest-
            # neighbor result, even for queries dominated by generic shared
            # vocabulary with only one truly distinctive word) — verified
            # live: a query mentioning "amazon" alongside standard
            # financial/legal boilerplate resolved to an unrelated, much
            # larger filing via vector-majority, despite this same
            # term-weighted scoring correctly favoring the right document
            # once tried. require_distinctive=True keeps this from firing
            # on pure boilerplate overlap with no real signal at all.
            confident = self._pick_best_by_term_weight(term_rows, terms, require_distinctive=True)
            if confident:
                return confident

        vector_id, vector_title = self.resolve_document_by_vector(session, query, tenant_id)
        if vector_id:
            return vector_id, vector_title

        if terms:
            match = self._pick_best_by_term_weight(term_rows, terms)
            if match:
                return match

            row = session.run(
                f"""
                UNWIND $terms AS term
                MATCH (d:{DOCUMENT_ROOT_CYPHER})
                WHERE {lc}
                  AND {tenant_filter("d")}
                  AND (toLower(coalesce(d.title, '')) =~ {_WORD_BOUNDARY_PATTERN}
                   OR EXISTS {{
                     MATCH (d)-[:CONTAINS*1..5]->(n)
                     WHERE {lc_n}
                       AND (toLower(coalesce(n.title, '')) =~ {_WORD_BOUNDARY_PATTERN}
                            OR toLower(coalesce(n.search_text, '')) =~ {_WORD_BOUNDARY_PATTERN})
                   }})
                RETURN coalesce(d.logical_doc_id, d.id) AS id,
                       coalesce(d.title, d.id) AS title,
                       count(*) AS hits
                ORDER BY hits DESC
                LIMIT 1
                """,
                terms=terms,
                tenant_id=tenant_id,
            ).single()
            if row and row.get("id"):
                return str(row["id"]), _clean_doc_title(str(row.get("title") or row["id"]))

        row = session.run(
            f"""
            MATCH (dl:{DOCUMENT_LOGICAL_LABEL})-[:ACTIVE_REVISION]->(:{DOC_REVISION_LABEL})
                  -[:ROOT]->(d:{DOCUMENT_ROOT_CYPHER})-[:CONTAINS*1..4]->(s:Section)
            WHERE {lc} AND {lifecycle_active("s")}
              AND {tenant_filter("dl")} AND {tenant_filter("s")}
            WITH dl, count(s) AS n
            ORDER BY n DESC
            LIMIT 1
            RETURN dl.logical_id AS id, coalesce(dl.title, dl.logical_id) AS title
            """,
            tenant_id=tenant_id,
        ).single()
        if row and row.get("id"):
            return str(row["id"]), _clean_doc_title(str(row.get("title") or row["id"]))

        row = session.run(
            f"""
            MATCH (d:{DOCUMENT_ROOT_CYPHER})-[:CONTAINS*1..4]->(s:Section)
            WHERE {lc} AND {lifecycle_active("s")}
              AND {tenant_filter("d")} AND {tenant_filter("s")}
            WITH d, count(s) AS n
            ORDER BY n DESC
            LIMIT 1
            RETURN coalesce(d.logical_doc_id, d.id) AS id, coalesce(d.title, d.id) AS title
            """,
            tenant_id=tenant_id,
        ).single()
        if row and row.get("id"):
            return str(row["id"]), _clean_doc_title(str(row.get("title") or row["id"]))
        return None, None

    def document_match_terms(self, query: str) -> list[str]:
        terms: list[str] = list(_query_anchor_terms(query))
        for raw in re.findall(r"[\w'-]{3,}", (query or "").lower()):
            # Strip a trailing possessive ("jpmorgan's" -> "jpmorgan") before
            # it becomes a CONTAINS search term -- the possessive form almost
            # never appears verbatim in the document's own prose (which says
            # "JPMorgan Chase & Co." or "the Firm", not "JPMorgan's"), so an
            # unstripped possessive scores zero everywhere and silently drops
            # what should have been the query's strongest anchor.
            t = _POSSESSIVE_RE.sub("", raw)
            if len(t) < 3:
                continue
            if t in _KEYWORD_STOP:
                continue
            if t in {"table", "contents", "content", "provide", "list", "show", "give", "from", "form", "page", "fetch", "document"}:
                continue
            if t not in terms:
                terms.append(t)
        return terms[:6]

    def doc_name_terms(self, query: str) -> list[str]:
        """
        Return only the high-confidence document-name tokens from a query.

        Unlike document_match_terms (which adds generic keywords for broad matching),
        this returns anchor tokens and proper nouns from the question only.

        Used by the strict document resolver to avoid matching the wrong document
        via common words like "all", "toc", etc.
        """
        cleaned = _PAREN_RE.sub(" ", _STRUCTURAL_REF_RE.sub(" ", query or ""))
        terms: list[str] = list(_query_anchor_terms(cleaned))

        # A document number is the most distinctive thing a query can carry,
        # and the token scan below cannot see it: it requires a leading
        # letter, so "IRS Publication 559" yielded ['irs', 'publication'] --
        # terms every IRS publication shares. Only numbers introduced by a
        # document-type word count, so an amount or a year in the question
        # is not mistaken for an identifier.
        for m in _DOC_NUMBER_RE.finditer(cleaned):
            token = m.group(1).lower()
            if token not in terms:
                terms.append(token)

        # Tokens that are capitalised mid-sentence are likely proper nouns / doc names
        words = re.findall(r"[A-Za-z][\w'-]*", cleaned)
        for i, w in enumerate(words):
            if i == 0:
                continue  # skip sentence-start capitalisation
            if not w[0].isupper():
                continue
            # See document_match_terms's identical strip -- a possessive
            # ("JPMorgan's") almost never appears verbatim in the document's
            # own prose, so leaving it attached silently zeroes out the term.
            t = _POSSESSIVE_RE.sub("", w.lower())
            if len(t) >= 3 and t not in _KEYWORD_STOP and t not in terms:
                terms.append(t)

        # Deliberately NOT falling back to "any word >= 6 chars that isn't a
        # stopword" here (a prior version of this method did, guarded only
        # by a small hardcoded exclusion list). That treated ordinary
        # content vocabulary — "employees", "conflicts", "benefits",
        # "discussed" — as document-identifying anchor terms, and this
        # method's own per-document scoring counts RAW occurrences: in a
        # corpus with one much larger document, generic words that are
        # merely common (not distinctive) reliably favor whichever document
        # has the most content, regardless of which document the question
        # is actually about (verified: a 638-section 10-K wrongly won over
        # an 83-section unrelated policy document on the strength of
        # "employees"/"conflicts"/"benefits" alone). document_match_terms()
        # already exists as the deliberately broader, lower-confidence
        # sibling for exactly this kind of generic-keyword matching —
        # resolve_document_for_query() falls back to it (and to vector
        # similarity) after this strict resolver declines to guess.
        return terms[:6]

    def resolve_document_id(self, session, name: str, tenant_id: str = "") -> Optional[str]:
        if not name:
            return None
        row = session.run(
            f"""
            MATCH (dl:{DOCUMENT_LOGICAL_LABEL})
            WHERE {tenant_filter("dl")}
              AND (toLower(coalesce(dl.title, '')) CONTAINS toLower($name)
               OR toLower(dl.logical_id) CONTAINS toLower($name))
            RETURN dl.logical_id AS id
            LIMIT 1
            """,
            name=name.strip(),
            tenant_id=tenant_id,
        ).single()
        if row and row.get("id"):
            return str(row["id"])
        row = session.run(
            f"""
            MATCH (d:{DOCUMENT_ROOT_CYPHER})
            WHERE {lifecycle_active("d")}
              AND {tenant_filter("d")}
              AND d.title IS NOT NULL
              AND toLower(d.title) CONTAINS toLower($name)
            RETURN coalesce(d.logical_doc_id, d.id) AS id
            LIMIT 1
            """,
            name=name.strip(),
            tenant_id=tenant_id,
        ).single()
        return str(row["id"]) if row and row.get("id") else None

    def list_documents(self, session, limit: int = 5, tenant_id: str = "") -> list[dict[str, str]]:
        rows = session.run(
            f"""
            MATCH (dl:{DOCUMENT_LOGICAL_LABEL})
            WHERE {tenant_filter("dl")}
            RETURN dl.logical_id AS id, coalesce(dl.title, dl.logical_id) AS title
            ORDER BY title
            LIMIT $limit
            """,
            limit=max(1, int(limit)),
            tenant_id=tenant_id,
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
              AND {tenant_filter("d")}
            RETURN coalesce(d.logical_doc_id, d.id) AS id, coalesce(d.title, d.id) AS title
            ORDER BY title
            LIMIT $limit
            """,
            limit=max(1, int(limit)),
            tenant_id=tenant_id,
        )
        out: list[dict[str, str]] = []
        for r in rows:
            if not r.get("id"):
                continue
            out.append({"id": str(r["id"]), "title": str(r.get("title") or r["id"])})
        return out
