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

import threading
from typing import Optional

from ..services.candidate_docs import CandidateDocService
from .full_hybrid import FullHybridStrategy


class VectorFirstHybridStrategy(FullHybridStrategy):
    name = "graph_rag_vector_first"

    def __init__(self, *args, candidate_docs: Optional[CandidateDocService] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self._candidate_docs = candidate_docs or CandidateDocService()
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
        self.last_scope_source: str = "unset"

    def _refine_scope(
        self,
        query: str,
        tenant_id: str,
        document_id: Optional[str],
        document_title: Optional[str],
        embed_future,
    ) -> tuple[Optional[str], Optional[str], list[str]]:
        """Fall back to vector evidence only where the resolver gave up.

        Deliberately does not override a resolved document. When the
        resolver names one it had a strong reason -- an explicit reference,
        a thread hint, a distinctive term -- and that evidence is better
        than similarity. This only fills the gap that previously became an
        unscoped corpus-wide search.
        """
        if document_id:
            # Includes every grounded follow-up: the thread's document
            # arrives as document_id_hint and the resolver returns it, so
            # no vector search happens for the rest of the conversation
            # until a question the resolver cannot place.
            self.last_scope_source = "resolver"
            return document_id, document_title, [document_id]

        cache_key = (tenant_id or "", (query or "").strip().lower())
        with self._scope_lock:
            cached = self._scope_cache.get(cache_key)
        if cached:
            self.last_scope_source = "cache"
            return cached[0], document_title, list(cached)

        embedding = None
        if embed_future is not None:
            try:
                # Already in flight; result() is cached, so awaiting it here
                # costs nothing the query was not paying anyway.
                embedding = embed_future.result()
            except Exception:
                embedding = None

        try:
            candidates = self._candidate_docs.candidates(
                query, tenant_id, embedding=embedding
            )
        except Exception:
            # Vector store unreachable is not a reason to fail the query --
            # degrade to exactly what this strategy's parent would have done.
            self.last_scope_source = "vector_error"
            return document_id, document_title, []

        if not candidates:
            self.last_scope_source = "no_candidates"
            return document_id, document_title, []

        top = candidates[0]
        doc_ids = [c.document_id for c in candidates]
        with self._scope_lock:
            if len(self._scope_cache) > 256:      # bounded; this is a cache, not a store
                self._scope_cache.clear()
            self._scope_cache[cache_key] = doc_ids
        self.last_scope_source = (
            f"vector(n={len(doc_ids)},rel={top.relative:.2f},"
            f"hits={top.hits},rank={top.best_rank})"
        )
        # document_id stays the single best candidate -- vector/fulltext/
        # graph-expand still scope to one document. The list is what the
        # lexical passes search, so "fetch the related docs, then search
        # each of them completely" happens in one indexed query via
        # `logical_doc_id IN $doc_ids` rather than one query per document.
        return top.document_id, document_title, doc_ids
