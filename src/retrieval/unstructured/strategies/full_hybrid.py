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
from ..query_intent import is_enumeration_question, is_synthesis_question
from ..services.formatter import ResponseFormatter
from ..services.graph_seeds import GraphSeedService
from ..services.lexical import LexicalService
from ..services.ranking import RankingService

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
    ):
        self._driver = driver
        self._ranking = ranking
        self._graph_seeds = graph_seeds
        self._lexical = lexical
        self._formatter = formatter

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
    ) -> Optional[dict[str, Any]]:
        synthesis = is_synthesis_question(query)
        enumeration = is_enumeration_question(query)
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

        embedding = None if skip_vector else self._graph_seeds.get_embedding(query)

        with ThreadPoolExecutor(max_workers=4, thread_name_prefix="hybrid_seed") as pool:
            phrase_future = pool.submit(
                self._neo4j_session_call,
                self._lexical.structural_phrase_retrieve,
                query,
                tenant_id=tenant_id,
            )
            keyword_future = pool.submit(
                self._neo4j_session_call,
                self._lexical.structural_keyword_retrieve,
                query,
                tenant_id=tenant_id,
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
                )
            fulltext_future = pool.submit(
                self._neo4j_session_call,
                self._graph_seeds.fulltext_seed,
                query,
                _FULLTEXT_LIMIT,
                tenant_id=tenant_id,
            )
            phrase_hits = phrase_future.result()
            keyword_hits = keyword_future.result()
            if vector_future is not None:
                vector_hits = vector_future.result()
            fulltext_hits = fulltext_future.result()

        lexical_hits = self._ranking._merge_retrieval_chunks(phrase_hits, keyword_hits)
        seed_ids = [h["id"] for h in vector_hits if h.get("id")]
        seed_scores = {h["id"]: float(h["score"]) for h in vector_hits if h.get("id")}

        graph_hits: list[dict] = []
        if seed_ids:
            with ThreadPoolExecutor(max_workers=2, thread_name_prefix="hybrid_graph") as pool:
                hop1_future = pool.submit(
                    self._neo4j_session_call,
                    self._graph_seeds.graph_expand,
                    seed_ids,
                    hops=1,
                    limit=graph_1hop,
                    tenant_id=tenant_id,
                )
                hop2_future = pool.submit(
                    self._neo4j_session_call,
                    self._graph_seeds.graph_expand,
                    seed_ids,
                    hops=2,
                    limit=graph_2hop,
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
            synthesis=synthesis,
            limit=max(1, int(fetch_limit)),
        )
        if lexical_hits:
            items = self._ranking._pin_precision_lexical_chunks(
                query, items, lexical_hits, limit=max(1, int(fetch_limit))
            )
        if synthesis and lexical_hits:
            items = self._ranking._pin_contrast_lexical_chunks(
                query, items, lexical_hits, limit=max(1, int(fetch_limit))
            )

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
