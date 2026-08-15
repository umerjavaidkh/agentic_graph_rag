"""full_hybrid.py — the terminal vector+lexical+graph fetch/merge/pin strategy.

Extracted from mixins/hybrid.py's fallthrough body (the branch every
non-structural query hits when no structural fast-path matches). This is
one execution path with a data-driven mode label, not three separate
strategies — `response["mode"]` ends up as one of `graph_rag`,
`graph_rag_lexical`, or `graph_rag_hybrid` depending on which sources
actually contributed, exactly as before.

Important: unlike the other strategies, this one does NOT use the `session`
parameter passed into `retrieve()` — Neo4j sessions aren't thread-safe, so
each concurrently-submitted fetch (phrase/keyword/vector/fulltext/graph-hop)
needs its own session, opened fresh per task via `self._driver`, mirroring
the original `_neo4j_session_call` helper exactly. `retrieve()` still takes
`session` for Protocol conformance with the other strategies, it's just
unused here.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Optional, TypeVar

from neo4j import Driver

from ....auth.roles import UserContext
from ....config.settings import (
    RETRIEVAL_CANDIDATE_POOL,
)
from ....feedback_loop import get_feedback_routing
from ....telemetry.pipeline import record_pipeline_step
from ..constants import (
    _FULLTEXT_LIMIT,
    _GRAPH_1HOP_LIMIT,
    _GRAPH_2HOP_LIMIT,
    _VECTOR_SEED_LIMIT,
)
from ..query_intent import (
    is_enumeration_question,
    is_firmwide_financial_metric_question,
    is_overview_question,
    is_quarterly_breakdown_question,
    is_synthesis_question,
)
from ..services.chapter_summary import ChapterSummaryService
from ..services.financial_summary import FinancialSummaryService
from ..services.document_resolver import DocumentResolver
from ..services.formatter import ResponseFormatter
from ..services.graph_seeds import GraphSeedService
from ..services.lexical import LexicalService
from ..services.ranking import RankingService
from ..services.reranker import RerankerService

_T = TypeVar("_T")


class FullHybridStrategy:
    name = "graph_rag_hybrid"

    def __init__(
        self,
        driver: Driver,
        ranking: RankingService,
        graph_seeds: GraphSeedService,
        lexical: LexicalService,
        formatter: ResponseFormatter,
        document_resolver: DocumentResolver,
        chapter_summaries: Optional[ChapterSummaryService] = None,
        financial_summaries: Optional[FinancialSummaryService] = None,
        reranker: Optional[RerankerService] = None,
    ):
        self._driver = driver
        self._ranking = ranking
        self._graph_seeds = graph_seeds
        self._lexical = lexical
        self._formatter = formatter
        self._document_resolver = document_resolver
        self._chapter_summaries = chapter_summaries or ChapterSummaryService()
        self._financial_summaries = financial_summaries or FinancialSummaryService()
        self._reranker = reranker or RerankerService()
        # One process-wide pool, not a fresh ThreadPoolExecutor spun up and
        # torn down on every retrieve() call — this strategy is itself a
        # process-wide singleton (see strategies/registration.py), so the
        # pool can live exactly as long as it does, same lifecycle as the
        # shared Neo4j driver. Sized for one query's own peak fan-out (up to
        # 4 concurrent tasks: resolve+embed, then phrase/keyword/vector/
        # fulltext) with headroom for more than one query overlapping.
        self._pool = ThreadPoolExecutor(max_workers=8, thread_name_prefix="hybrid")

    def _neo4j_session_call(self, fn: Callable[..., _T], /, *args: Any, **kwargs: Any) -> _T:
        """Run a Neo4j read in its own session (safe for thread-pool parallelism)."""
        with self._driver.session() as session:
            return fn(session, *args, **kwargs)

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
        synthesis = is_synthesis_question(query)
        enumeration = is_enumeration_question(query)
        # Deliberately broader than `synthesis` and kept separate from it —
        # is_overview_question also matches plain "what does this document/
        # chapter discuss" phrasing that _SYNTHESIS_RE doesn't, without
        # pulling that phrasing into `synthesis`'s other effects below
        # (vector/graph/lexical weighting, fetch_limit sizing).
        wants_overview = synthesis or is_overview_question(query)
        wants_firmwide = is_firmwide_financial_metric_question(query)
        wants_quarterly = is_quarterly_breakdown_question(query)
        fetch_limit = limit
        if synthesis:
            fetch_limit = max(limit, 16)
        if enumeration:
            fetch_limit = max(fetch_limit, 18)
        vector_limit = min(RETRIEVAL_CANDIDATE_POOL, 16) if synthesis else _VECTOR_SEED_LIMIT
        graph_1hop = 32 if synthesis else _GRAPH_1HOP_LIMIT
        graph_2hop = 24 if synthesis else _GRAPH_2HOP_LIMIT

        mode_hint = get_feedback_routing().document_mode(query)
        skip_vector = mode_hint == "graph_rag_lexical"

        # Scope vector/fulltext/graph-expand to a single document when one
        # can be confidently resolved (explicit naming, or the query's own
        # vocabulary is distinctive enough) — this is the terminal fallback
        # every non-structural question reaches, and previously the only
        # unscoped retrieval path (TOC/page/box/subsection strategies
        # already resolve via the same DocumentResolver). A generic
        # question ("what does Part II discuss") has no distinguishing
        # vocabulary to bias vector similarity toward the right document on
        # its own once more than one document is ingested, so without this
        # it silently searches the whole tenant corpus. Ambiguous queries
        # still degrade to unscoped search — resolve_document_for_query
        # only returns a document when it has a clear-enough signal.
        #
        # Resolution (a Neo4j round trip) and embedding (an external API
        # call) are independent of each other, so they run as concurrent
        # futures on the shared pool instead of one blocking the other's
        # start — the same pool is reused below for the phrase/keyword/
        # vector/fulltext fetches once both are in hand.
        pool = self._pool
        doc_id_future = pool.submit(
            self._neo4j_session_call,
            self._document_resolver.resolve_document_for_query,
            query,
            tenant_id=tenant_id,
            document_id_hint=document_id_hint,
        )
        embed_future = None if skip_vector else pool.submit(self._graph_seeds.get_embedding, query)

        document_id, document_title = doc_id_future.result()
        document_id = document_id or ""
        embedding = None if embed_future is None else embed_future.result()

        # document_id is already resolved here, so the lexical calls take
        # it directly instead of each re-resolving it themselves — that
        # used to cost 2 extra, fully redundant Neo4j round trips (with
        # their own sessions) on every single query.
        phrase_future = pool.submit(
            self._neo4j_session_call,
            self._lexical.structural_phrase_retrieve,
            query,
            tenant_id=tenant_id,
            document_id=document_id,
        )
        keyword_future = pool.submit(
            self._neo4j_session_call,
            self._lexical.structural_keyword_retrieve,
            query,
            tenant_id=tenant_id,
            document_id=document_id,
        )
        if skip_vector:
            vector_future = None
            vector_hits: list[dict] = []
        else:
            vector_future = pool.submit(
                self._neo4j_session_call,
                self._graph_seeds.vector_seed,
                embedding,
                vector_limit,
                tenant_id=tenant_id,
                document_id=document_id,
            )
        fulltext_future = pool.submit(
            self._neo4j_session_call,
            self._graph_seeds.fulltext_seed,
            query,
            _FULLTEXT_LIMIT,
            tenant_id=tenant_id,
            document_id=document_id,
        )
        # Scope-phrase retrieval: finds the chunk belonging to the SEGMENT the
        # question names, which neither vector nor the other lexical passes
        # can isolate when a document repeats identical row labels under every
        # segment (see LexicalService.scope_phrase_retrieve). Runs on the same
        # pool as the other fetches, so it costs no extra wall-clock time.
        scope_future = pool.submit(
            self._neo4j_session_call,
            self._lexical.scope_phrase_retrieve,
            query,
            tenant_id=tenant_id,
            document_id=document_id,
        )
        # A counting question is answered by a numeral next to the counted
        # noun, which keyword/vector retrieval cannot isolate when that noun
        # is common in the document (see quantity_evidence_retrieve). Returns
        # [] immediately for every non-counting question.
        quantity_future = pool.submit(
            self._neo4j_session_call,
            self._lexical.quantity_evidence_retrieve,
            query,
            tenant_id=tenant_id,
            document_id=document_id,
        )
        # Only fetched for overview-shaped questions ("what does this
        # document/chapter discuss") — chapter_summary_weight in
        # _merge_and_rank already de-prioritizes these for everything else,
        # so skipping the round trip entirely when it can't matter avoids
        # paying for it on the common case.
        chapter_summary_future = (
            pool.submit(
                self._neo4j_session_call,
                self._chapter_summaries.fetch_for_document,
                document_id,
                tenant_id=tenant_id,
            )
            if wants_overview
            else None
        )
        # Only for firmwide financial-metric questions, and only once a single
        # document is resolved — otherwise the authoritative summary section
        # can't be attributed to the right issuer. Pinned below so the firm
        # total wins over segment tables (see _pin_firmwide_summary_chunks).
        financial_summary_future = (
            pool.submit(
                self._neo4j_session_call,
                self._financial_summaries.fetch_for_document,
                document_id,
                tenant_id=tenant_id,
            )
            if wants_firmwide and document_id
            else None
        )
        # Same pin-worthy-authoritative-section idea as financial_summary_future,
        # for quarter-by-quarter questions instead of firmwide-vs-segment ones —
        # see fetch_quarterly_for_document's docstring for why this table needs
        # its own fetch rather than reusing the title-matched one above.
        quarterly_summary_future = (
            pool.submit(
                self._neo4j_session_call,
                self._financial_summaries.fetch_quarterly_for_document,
                document_id,
                tenant_id=tenant_id,
            )
            if wants_quarterly and document_id
            else None
        )
        phrase_hits = phrase_future.result()
        keyword_hits = keyword_future.result()
        if vector_future is not None:
            vector_hits = vector_future.result()
        fulltext_hits = fulltext_future.result()
        chapter_summary_hits = chapter_summary_future.result() if chapter_summary_future is not None else []
        financial_summary_hits = (
            financial_summary_future.result() if financial_summary_future is not None else []
        )
        quarterly_summary_hits = (
            quarterly_summary_future.result() if quarterly_summary_future is not None else []
        )

        scope_hits = scope_future.result()
        quantity_hits = quantity_future.result()
        lexical_hits = self._ranking._merge_retrieval_chunks(phrase_hits, keyword_hits)
        # Merged into the lexical pool so scope hits reach _merge_and_rank and
        # the reranker as ordinary candidates; they are additionally pinned
        # below, because ranking alone is exactly what failed here -- the
        # right chunk was reachable all along and still never made the cut.
        lexical_hits = self._ranking._merge_retrieval_chunks(lexical_hits, scope_hits)
        lexical_hits = self._ranking._merge_retrieval_chunks(lexical_hits, quantity_hits)
        seed_ids = [h["id"] for h in vector_hits if h.get("id")]
        seed_scores = {h["id"]: float(h["score"]) for h in vector_hits if h.get("id")}

        graph_hits: list[dict] = []
        if seed_ids:
            hop1_future = pool.submit(
                self._neo4j_session_call,
                self._graph_seeds.graph_expand,
                seed_ids,
                hops=1,
                limit=graph_1hop,
                tenant_id=tenant_id,
                document_id=document_id,
            )
            hop2_future = pool.submit(
                self._neo4j_session_call,
                self._graph_seeds.graph_expand,
                seed_ids,
                hops=2,
                limit=graph_2hop,
                document_id=document_id,
                tenant_id=tenant_id,
            )
            graph_hits = hop1_future.result() + hop2_future.result()

        items = self._ranking._merge_and_rank(
            query,
            vector_hits,
            fulltext_hits,
            graph_hits,
            seed_scores,
            lexical_hits=lexical_hits,
            chapter_summary_hits=chapter_summary_hits,
            synthesis=synthesis,
            chapter_summary_boost=wants_overview,
            limit=max(1, int(fetch_limit)),
        )
        items = self._reranker.rerank(
            query,
            items,
            sources={
                "vector": vector_hits,
                "fulltext": fulltext_hits,
                "graph": graph_hits,
                "lexical": lexical_hits,
                "chapter_summary": chapter_summary_hits,
            },
        )
        if lexical_hits:
            items = self._ranking._pin_precision_lexical_chunks(
                query, items, lexical_hits, limit=max(1, int(fetch_limit))
            )
        if synthesis and lexical_hits:
            items = self._ranking._pin_contrast_lexical_chunks(
                query, items, lexical_hits, limit=max(1, int(fetch_limit))
            )
        pin_summary_hits = financial_summary_hits + quarterly_summary_hits
        if pin_summary_hits:
            items = self._ranking._pin_firmwide_summary_chunks(
                items, pin_summary_hits, limit=max(1, int(fetch_limit))
            )
        # Pinned AFTER the firmwide pin so the two never fight: a question
        # naming a specific segment is by definition not a firmwide one
        # (is_firmwide_financial_metric_question excludes segment-scoped
        # queries), so in practice only one of these ever has hits. When a
        # question somehow produces both, the more specific scope should win
        # the top of the context window.
        if scope_hits:
            items = self._ranking._pin_scope_chunks(
                items, scope_hits, limit=max(1, int(fetch_limit))
            )
        # Last, and only when no more specific scope was identified: a scope
        # phrase names the answer's location explicitly, so it should not be
        # displaced by a mere keyword leader. With no scope to go on, the
        # best idf-ranked keyword match is the strongest remaining signal
        # for a narrow factual question.
        # Numeric evidence is the most specific signal available for a
        # counting question -- pinned above scope, which only narrows WHERE
        # to look, not what actually answers it.
        if quantity_hits:
            items = self._ranking._pin_scope_chunks(
                items, quantity_hits, limit=max(1, int(fetch_limit))
            )
        if not scope_hits and not quantity_hits and keyword_hits:
            items = self._ranking._pin_keyword_leader(
                items, keyword_hits, limit=max(1, int(fetch_limit))
            )

        # Last: a hit that is one part of a multi-page unit brings the rest
        # with it. Done here, after every pin, so it applies to whatever
        # actually survived rather than to one strategy's candidates -- and
        # so no retrieval query has to carry unit metadata itself.
        siblings = self._neo4j_session_call(
            self._lexical.expand_unit_siblings,
            [i.get("id") for i in items if i.get("id")],
            tenant_id=tenant_id,
        )
        if siblings:
            items = self._ranking._merge_retrieval_chunks(items, siblings)[
                : max(1, int(fetch_limit)) + len(siblings)
            ]

        response = self._formatter.format(query, items, ctx=ctx)
        if mode_hint == "graph_rag_lexical":
            response["mode"] = "graph_rag_lexical"
        elif mode_hint == "graph_rag":
            response["mode"] = "graph_rag"
        else:
            response["mode"] = "graph_rag"
            if lexical_hits and vector_hits:
                response["mode"] = "graph_rag_hybrid"
            elif lexical_hits:
                response["mode"] = "graph_rag_lexical"
        response["strategy"] = "graph_rag"
        response["vector_seeds"] = len(vector_hits)
        response["fulltext_hits"] = len(fulltext_hits)
        response["graph_expanded"] = len(graph_hits)
        response["document_id"] = document_id or None
        response["document_title"] = document_title
        record_pipeline_step(
            "document.hybrid.merge",
            meta={
                "mode": response.get("mode"),
                "vector_seeds": len(vector_hits),
                "fulltext_hits": len(fulltext_hits),
                "graph_expanded": len(graph_hits),
                "returned": len(items),
            },
        )
        return response
