"""vector_first_hybrid.py — hybrid retrieval that scopes by vector evidence
when the name-matching resolver cannot decide.

Registered separately from `graph_rag_hybrid` and reports its own
`mode`/`strategy` label, so the two can run side by side and be compared on
real traffic rather than on one hand-picked query. Nothing routes to it
automatically; it is opt-in per request.

Why it exists
-------------
`DocumentResolver.resolve_document_for_query` walks every document's subtree
applying a regex per content node -- 15.61s across 998 documents, measured,
paid before retrieval starts. Worse, when the top two documents land within
`AMBIGUITY_LEAD` of each other it returns nothing at all, and the caller
falls back to searching the entire corpus unscoped.

Measured on "What is the table of contents of Go.Data annual report?"
against the 998-document corpus: the resolver ranked the right document
first (39.61) but the runner-up (32.60) held the lead to 1.215, under the
1.5 threshold, so it declined and the user got a clarification listing
arbitrary documents instead of an answer. The vector pass put the same
document first outright, `best_rank=0` -- the nearest chunk in the whole
corpus belonged to it.

There is already a vector tier inside the resolver
(`resolve_document_by_vector`), but it sits *below* the point where an
ambiguous name causes the resolver to decline, so for this class of query it
never runs. It also re-embeds the query, and requires the winner to own a
majority of 12 seeds -- the correct document here held 22 of the top 200,
decisive but nowhere near a majority.

Scope shape
-----------
The candidate documents become the lexical passes' search set, via
`logical_doc_id IN $doc_ids` -- one indexed query with a seek per document,
not one query per document. `document_id` stays the single best candidate,
so vector/fulltext/graph-expand behave exactly as before.

Follow-ups do not re-run the ANN lookup: a grounded follow-up carries the
thread's document as `document_id_hint`, the resolver returns it, and this
returns before touching the vector store. A bounded per-query cache covers
re-phrasings that arrive without a hint.
"""
from __future__ import annotations

import re
import threading
from dataclasses import replace
from typing import Any, Optional

from ..cypher_scope import as_doc_id_list

# Above this share of the winner's score, the runner-up is not meaningfully
# behind and the choice should be offered rather than made.
AMBIGUOUS_LEAD_RATIO = 0.80

# Shortest squashed document name that may be trusted as a reference. Below
# this, a match is coincidence -- "sp" or "irs" appears in many ids.
MIN_NAME_MATCH = 8

# Below this many content words, a question with no document signal is not a
# question about a document -- it is a fragment. "What is the deadline?"
# names nothing, carries no topic, and was answered from an arbitrary IRS
# publication with total fluency. A close runner-up is a different problem
# with a different gate; this one is the absence of any signal at all.
# Asking for the document's own structural units -- "list every chapter
# heading", "what are the sections", "give me the appendices". The unit
# word is what makes it a hierarchy read; the listing verb alone is not,
# because "list the datasets used" is a content question.
_ASKS_FOR_STRUCTURE = re.compile(
    r"\b(chapters?|sections?|headings?|subsections?|appendi(?:x|ces)|annexes?)\b",
    re.I,
)

MIN_KEYWORDS_UNSCOPED = 2

# A question whose rarest content word still appears in this share of the
# corpus has not named anything. Measured, not chosen: real questions reach
# 0.042 at most and generic ones start at 0.093, so this is the midpoint of
# an empty gap rather than a bound fitted to either side.
MAX_GENERIC_DOC_FREQUENCY = 0.07


def _tokens(text: str) -> tuple[set, set]:
    """Alphabetic and numeric runs of a name, separately.

    Squashing to a single string compares "IRS Publication 225" against
    "irs_p225" as `irspublication225` vs `irsp225`, and containment fails --
    the abbreviation sits between the letters and the number. Comparing runs
    instead lets "irs" and "225" match across "p" vs "publication", which is
    the only difference that matters.

    Measured before this existed: 12 of 28 questions that each named "IRS
    Publication 225" outright were answered from a different IRS
    publication, because the corpus holds dozens of near-identical ones.
    """
    low = (text or "").lower()
    alpha = {w for w in re.findall(r"[a-z]+", low) if len(w) > 1}
    nums = {w for w in re.findall(r"\d+", low) if len(w) > 1}
    return alpha, nums


def _squash(text: str) -> str:
    """Letters and digits only, lowercased.

    Document names survive contact with prose in many shapes: "NIST SP
    800-161r1", "SP800-161", "nist_sp800_161". Comparing them needs the
    punctuation and spacing gone, because that is the only thing that
    differs -- the resolver's word-boundary matching failed on exactly this
    and answered a NIST question from an IRS publication.
    """
    return "".join(c for c in (text or "").lower() if c.isalnum())
from ..query_plan import Shape, classify
from ..services.candidate_docs import CandidateDocService
from ..services.term_stats import CorpusTermStats
from ..services.structural import StructuralService
from .full_hybrid import FullHybridStrategy


class VectorFirstHybridStrategy(FullHybridStrategy):
    name = "graph_rag_vector_first"

    def __init__(self, *args, candidate_docs: Optional[CandidateDocService] = None,
                 term_stats: Optional[CorpusTermStats] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self._candidate_docs = candidate_docs or CandidateDocService()
        self._term_stats = term_stats or CorpusTermStats()
        self._structural = StructuralService()
        # Follow-ups inside a thread must not re-run the ANN lookup. The
        # primary mechanism is `document_id_hint`, which DocumentResolver
        # already honours: a grounded follow-up arrives with the thread's
        # document, the resolver returns it, and _refine_scope below returns
        # before touching the vector store at all.
        #
        # This cache covers the rest -- a repeated or re-phrased question in
        # the same session, where no hint was carried. Bounded and keyed by
        # the exact query, so a genuinely new question misses it and pays for
        # a fresh lookup, which is the correct behaviour for one "outside the
        # scope".
        self._scope_cache: dict[tuple[str, str], list] = {}
        self._scope_lock = threading.Lock()
        # Set on each retrieve() so a caller reading the response can see
        # which path actually decided the scope, rather than inferring it.
        self._names = None
        # Per-request, not per-instance. The registry hands every request the
        # same strategy object, so a plain attribute is shared across
        # concurrent queries -- one request's "low confidence" verdict
        # appeared on another's answer, which is worse than not reporting it.
        self._local = threading.local()

    def _flag(self, name: str, value: Any = None) -> Any:
        if value is None:
            return getattr(self._local, name, None)
        setattr(self._local, name, value)
        return value

    @property
    def last_scope_source(self) -> str:
        return getattr(self._local, "scope_source", "unset")

    @property
    def last_scope_ambiguous(self) -> bool:
        return bool(getattr(self._local, "ambiguous", False))

    @property
    def last_underspecified(self) -> bool:
        return bool(getattr(self._local, "underspecified", False))

    @property
    def last_candidates(self) -> list:
        return getattr(self._local, "candidates", []) or []

    def _named_document(self, tenant_id: str, query: str) -> Optional[str]:
        """The document this query names outright, however it is written."""
        q = _squash(query)
        q_alpha, q_nums = _tokens(query)

        best, best_score = None, 0
        for doc_id, key, raw in self._name_index(tenant_id):
            # Exact-ish: the whole name appears verbatim once punctuation is
            # gone. Strongest evidence, so it outranks token agreement.
            if len(key) >= MIN_NAME_MATCH and key in q:
                score = 1000 + len(key)
            else:
                d_alpha, d_nums = self._name_tokens(raw)
                # A document identified by a number must have that number in
                # the query -- "Publication 225" must not match p17. Without
                # a number to anchor on, fall back to the verbatim test
                # above rather than matching on words alone, or "document"
                # would match anything.
                if not d_nums or not d_nums <= q_nums:
                    continue
                shared = d_alpha & q_alpha
                if not shared:
                    continue
                score = len(d_nums) * 10 + len(shared)
            if score > best_score:
                best, best_score = doc_id, score
        return best

    def _name_tokens(self, key: str):
        cache = getattr(self, "_tokcache", None)
        if cache is None:
            cache = self._tokcache = {}
        if key not in cache:
            cache[key] = _tokens(key)
        return cache[key]

    def _name_index(self, tenant_id: str) -> list:
        """(logical_id, squashed name) for every document, built once.

        998 rows, and the corpus only changes on ingest, so this is read
        once per process rather than per query.
        """
        if self._names is None:
            rows = self._neo4j_session_call(
                lambda sess: list(sess.run(
                    "MATCH (dl:DocumentLogical) "
                    "RETURN dl.logical_id AS id, coalesce(dl.title, '') AS title"
                ))
            )
            idx = []
            for r in rows:
                lid = r["id"]
                if not lid:
                    continue
                # The "doc_" prefix is ours, not part of the name the user
                # would type.
                raw = lid[4:] if lid.startswith("doc_") else lid
                idx.append((lid, _squash(raw), raw))
                if r["title"]:
                    idx.append((lid, _squash(r["title"]), r["title"]))
            self._names = idx
        return self._names

    def _scope_for_query(
        self,
        query: str,
        tenant_id: str,
        document_id_hint: str,
        embed_future,
    ) -> tuple[Optional[str], Optional[str], list[str]]:
        """Cheap and certain first, similarity next, corpus scan last.

        The parent asks `resolve_document_for_query`, whose first tier is a
        15.61s scan of every document's subtree. That tier is both the most
        expensive and the least reliable -- it is the one that produced the
        1.215 near-tie and declined to answer. Here it runs only when
        nothing cheaper decided, so the common case never pays it.

        Order:
          1. thread hint -- free, and the user already told us
          2. an outright reference to a document's id or title -- one
             indexed lookup
          3. vector candidates -- the embedding is already in flight, so
             this costs one ANN lookup (420ms measured)
          4. the full resolver -- correct but expensive, so it is what we
             fall back TO rather than start from
        """
        # A document named in THIS question outranks the one the thread was
        # discussing. DocumentResolver already orders it that way -- "an
        # outright reference to a document's id or title outranks the thread"
        # -- and checking the hint first inverted it: in a UI session, every
        # question after the first inherited the first question's document
        # and answered "this document does not cover it" about a document
        # that was never asked for.
        named = self._neo4j_session_call(
            self._document_resolver.exact_document_reference, query, tenant_id
        )
        if named and named[0]:
            self._local.scope_source = "exact_reference"
            return named[0], named[1], [named[0]]

        # The same question asked without punctuation, so a name is not lost
        # to a hyphen -- that is how a question about SP 800-161r1 came back
        # answered from a different NIST report.
        squashed = self._named_document(tenant_id, query)
        if squashed:
            self._local.scope_source = "named"
            return squashed, None, [squashed]

        # Only now the thread. A follow-up that names nothing ("what about
        # page 7?") should stay on the document under discussion.
        if document_id_hint:
            validated = self._neo4j_session_call(
                self._document_resolver._validate_document_id, document_id_hint, tenant_id
            )
            if validated:
                self._local.scope_source = "hint"
                return document_id_hint, None, [document_id_hint]

        cache_key = (tenant_id or "", (query or "").strip().lower())
        with self._scope_lock:
            cached = self._scope_cache.get(cache_key)
        if cached:
            self._local.scope_source = "cache"
            return cached[0], None, list(cached)

        embedding = None
        if embed_future is not None:
            try:
                embedding = embed_future.result()
            except Exception:
                embedding = None

        candidates = []
        try:
            candidates = self._candidate_docs.candidates(query, tenant_id, embedding=embedding)
        except Exception:
            # A vector store that is down must not fail the query; fall
            # through to the resolver, which is what the parent would do.
            candidates = []

        if candidates:
            top = candidates[0]
            # Nothing named the document, nothing hinted it, and the query
            # carries too little to have meant one. Similarity will always
            # return something; that is not the same as the user having
            # asked about it.
            keywords = self._ranking._content_keywords_from_query(query)
            self._local.underspecified = len(keywords) < MIN_KEYWORDS_UNSCOPED
            # A thin lead is not a decision. Two documents that match about
            # equally often means the query named neither clearly, and
            # guessing is wrong about as often as it is right -- the failure
            # that answered a NIST question from an IRS publication and
            # attributed the formula to the wrong standard. Recorded so the
            # caller can offer the choice instead of presenting one.
            runner = candidates[1].relative if len(candidates) > 1 else 0.0
            self._local.ambiguous = runner >= AMBIGUOUS_LEAD_RATIO
            self._local.candidates = [
                {"document_id": c.document_id, "relative": round(c.relative, 3)}
                for c in candidates[:5]
            ]
            doc_ids = [c.document_id for c in candidates]
            with self._scope_lock:
                if len(self._scope_cache) > 256:   # bounded; a cache, not a store
                    self._scope_cache.clear()
                self._scope_cache[cache_key] = doc_ids
            self._local.scope_source = (
                f"vector(n={len(doc_ids)},rel={top.relative:.2f},"
                f"hits={top.hits},rank={top.best_rank})"
            )
            return top.document_id, None, doc_ids

        # Nothing cheap decided, and the vector pass had no opinion (it
        # returns none when the nearest chunk in the corpus is not close).
        # Only now is the scan worth its cost.
        document_id, document_title = self._neo4j_session_call(
            self._document_resolver.resolve_document_for_query,
            query,
            tenant_id=tenant_id,
            document_id_hint=document_id_hint,
        )
        self._local.scope_source = "resolver_fallback"
        return document_id, document_title, as_doc_id_list(document_id) or []

    def _cannot_be_placed(self, query: str) -> bool:
        """Whether the question's own words can single out a document.

        Two ways to fail, and both are about the question rather than about
        what came back:

        Too few content words -- "What is the value?" carries one.

        Every content word is corpus-wide vocabulary -- "What are the key
        findings?" carries two, clearing the count, and is answerable in
        every document. Document frequency is what separates these:
        measured over 110 generated questions and 12 generic ones, real
        questions top out at 0.042 and generic ones bottom out at 0.093.
        MAX_GENERIC_DOC_FREQUENCY sits in that gap rather than on either
        edge.

        The corpus supplies the figure, so no list of "generic words" is
        hardcoded -- on a corpus about research methods, "methodology"
        would be common and correctly stop identifying a document, while
        here a term is generic only because the ingested documents made it
        so.
        """
        keywords = self._ranking._content_keywords_from_query(query)
        if len(keywords) < MIN_KEYWORDS_UNSCOPED:
            return True
        try:
            # Opens its own session rather than using the one passed in. The
            # universal path calls this strategy with session=None (it opens
            # per-task sessions internally, since Neo4j sessions are not
            # thread-safe), so reading the corpus table off the argument
            # raised on None and the except below turned every generic
            # question back into a confident answer -- silently, which is
            # how it survived a full round of end-to-end checks.
            rarest = self._neo4j_session_call(
                self._term_stats.min_term_frequency, keywords
            )
        except Exception:
            # A statistic this gate cannot compute must not turn into a
            # refusal: answering is the behaviour that was there before.
            return False
        return rarest is not None and rarest >= MAX_GENERIC_DOC_FREQUENCY

    def _underspecified_response(self, query, ctx, candidates):
        """Ask which document, instead of answering from whichever one ranked first.

        "What is the value?" names no document, carries no topic, and matches
        something in every one of 998 documents. Similarity always returns a
        nearest neighbour, so the system answered from an arbitrary paper with
        no indication that the question had not actually been placed --
        measured 0/5 on the ambiguous shape, every one a confident answer.

        The candidates come from the vector pass that already ran, so offering
        the choice costs nothing beyond what answering would have cost.
        """
        names = [c["document_id"] for c in (candidates or [])][:5]
        listed = "\n".join(f"- {n}" for n in names)
        text = (
            "That question does not say which document to look in, and it "
            "matches several. Ask again naming one of these, or add enough "
            "detail to identify it:\n\n" + listed
            if names else
            "That question does not say which document to look in. Ask again "
            "naming the document, or add enough detail to identify it."
        )
        response = self._formatter.format(query, [{
            "id": "underspecified", "title": "Which document?",
            "text": text, "score": 0.0, "related": [],
        }], ctx=ctx)
        response["strategy"] = self.name
        response["low_confidence"] = True
        response["underspecified"] = True
        response["document_candidates"] = candidates or []
        response["document_id"] = None
        return response

    def retrieve(self, session, query, *, tenant_id, limit, ctx, document_id_hint=""):
        """Structural questions are answered from the hierarchy; everything
        else falls through to the inherited hybrid path.

        The short-circuit is here rather than inside the hybrid because the
        hybrid's whole shape -- recall, rank, truncate -- is wrong for an
        address. There is nothing to rank: the document either has a Box 9
        or it does not.
        """
        plan = classify(query, default_limit=limit)

        # Before anything else: if the question names no document, the thread
        # has none, and its own words cannot single one out, asking is the
        # correct answer. Guessing here is not a ranking problem -- there is
        # nothing to rank against.
        if (
            not document_id_hint
            and plan.shape is not Shape.STRUCTURAL
            and self._cannot_be_placed(query)
            and not self._named_document(tenant_id, query)
        ):
            try:
                cands = self._candidate_docs.candidates(query, tenant_id)
            except Exception:
                cands = []
            return self._underspecified_response(
                query, ctx,
                [{"document_id": c.document_id, "relative": round(c.relative, 3)}
                 for c in cands[:5]],
            )

        if plan.shape is Shape.AGGREGATION:
            # Answer from passages as usual, but put the document's own
            # outline in front of them. A count of "critical success factors"
            # is six because sections 3.1 to 3.6 exist, and no sentence in
            # the document says so -- the structure is the evidence, and
            # without it the honest answer is "no count is given".
            response = super().retrieve(
                session, query, tenant_id=tenant_id, limit=limit, ctx=ctx,
                document_id_hint=document_id_hint,
            )
            doc = (response or {}).get("document_id")
            if response and doc:
                # A count of units the graph holds is answered by counting
                # them. Reading it out of prose produced 23 tables against an
                # actual 88 and "3 main chapters" against 15 -- the document
                # never states either figure, so there was nothing to read.
                counted = self._neo4j_session_call(
                    self._structural.count_units, query, [doc], tenant_id
                )
                if counted:
                    response["chunks"] = counted
                    response["query_shape"] = plan.shape.value
                    return response
                outline = self._neo4j_session_call(
                    self._structural.outline, [doc], tenant_id
                )
                if outline:
                    response["chunks"] = outline + (response.get("chunks") or [])
                    response["query_shape"] = plan.shape.value
            return response

        # "List every chapter heading in X" classifies as ENUMERATIVE, and
        # the enumerative shape never read the hierarchy -- it fell through
        # to prose search and answered "this document does not cover chapter
        # headings" while outline() returned all 31 of them. One document in
        # five passed, and only because the hybrid search happened to land on
        # a chunk that listed the sections; that is luck, not retrieval.
        #
        # A request to list the document's own structural units is a
        # hierarchy read whichever shape it classifies as, so it joins the
        # structural path below rather than getting a branch of its own.
        if plan.shape is Shape.ENUMERATIVE and _ASKS_FOR_STRUCTURE.search(query or ""):
            plan = replace(plan, shape=Shape.STRUCTURAL)

        if plan.shape is not Shape.STRUCTURAL:
            return super().retrieve(
                session, query, tenant_id=tenant_id, limit=limit, ctx=ctx,
                document_id_hint=document_id_hint,
            )

        embed_future = self._pool.submit(self._graph_seeds.get_embedding, query)
        document_id, document_title, doc_ids = self._scope_for_query(
            query, tenant_id, document_id_hint, embed_future
        )
        if not doc_ids:
            # No document to read a hierarchy from; the hybrid path at
            # least searches, which beats answering nothing.
            return super().retrieve(
                session, query, tenant_id=tenant_id, limit=limit, ctx=ctx,
                document_id_hint=document_id_hint,
            )

        # ONE document, not the candidate set. The candidate list exists to
        # widen lexical recall; a hierarchy belongs to a single document, and
        # scoping an outline to eight of them merges eight tables of contents
        # into one answer -- verified live, with NIST and arXiv headings
        # appearing under a question about the Go.Data report.
        target = [document_id] if document_id else doc_ids[:1]
        items = self._neo4j_session_call(
            self._structural.outline, target, tenant_id
        ) if plan.exhaustive else self._neo4j_session_call(
            self._structural.by_address, plan.address or query, target, tenant_id
        )
        if not items:
            return super().retrieve(
                session, query, tenant_id=tenant_id, limit=limit, ctx=ctx,
                document_id_hint=document_id_hint,
            )

        # An outline is returned verbatim, not handed to the model to
        # re-render. Routing "list every chapter heading" here first made
        # two documents right and two others worse: given a 20-chapter
        # outline the model rewrote it and dropped most of it, so 12 of 14
        # headings became 9. outline() is already exhaustive and ordered --
        # there is nothing for generation to add, and one thing it reliably
        # takes away. Same reason the count in _generate_document_answer is
        # passed through untouched.
        if plan.exhaustive and items and _ASKS_FOR_STRUCTURE.search(query or ""):
            items = [{**items[0], "id": "graph_outline"}] + items[1:]

        response = self._formatter.format(query, items, ctx=ctx)
        response["mode"] = "structural"
        response["strategy"] = self.name
        response["document_id"] = document_id or (doc_ids[0] if doc_ids else None)
        response["document_title"] = document_title
        response["query_shape"] = plan.shape.value
        return response
