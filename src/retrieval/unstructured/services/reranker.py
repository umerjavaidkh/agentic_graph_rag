"""reranker.py — pluggable rerank backends for the merged retrieval candidate pool.

`_merge_and_rank` combines vector/fulltext/graph/lexical signals with fixed
heuristic weights; none of those signals directly judge whether a chunk
answers the query, so a chunk that merely shares generic terms/entities can
score competitively with the chunk that actually answers it. A reranker
re-scores each (query, chunk) pair directly and reorders on that.

Backends are registered by name in ../../reranker_registry.py (same
key -> factory shape as strategy_registry.py) so a different technique can
be swapped in, A/B'd, or added as a custom backend by registering it under a
new key and pointing RERANK_BACKEND at it — nothing else in the retrieval
pipeline needs to change. `RerankerService` is the single thing strategies
depend on; it resolves the configured backend lazily and degrades to a
no-op on any load/inference failure rather than breaking retrieval.
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Optional

from ....shared.config.settings import RERANK_BACKEND, RERANK_ENABLED, RERANK_MODEL
from ...reranker_registry import get_reranker_backend, register_reranker

logger = logging.getLogger(__name__)

_TEXT_TRUNCATE_CHARS = 2000


class RerankerBackend(ABC):
    @abstractmethod
    def rerank(
        self,
        query: str,
        items: list[dict],
        *,
        text_key: str = "text",
        sources: Optional[dict[str, list[dict]]] = None,
    ) -> list[dict]:
        """Reorder `items` by relevance to `query`. Order-only: never drops or adds items.

        `sources` (optional) is the pre-merge per-signal hit lists
        (vector/fulltext/graph/lexical/chapter_summary), each already sorted
        by that signal's own score — rank-based backends (RRF) need this;
        pairwise backends (cross-encoder) ignore it.
        """


class NoopRerankerBackend(RerankerBackend):
    """Passthrough — the disabled/unavailable default and a baseline for A/B comparison."""

    def rerank(
        self,
        query: str,
        items: list[dict],
        *,
        text_key: str = "text",
        sources: Optional[dict[str, list[dict]]] = None,
    ) -> list[dict]:
        return items


class CrossEncoderRerankerBackend(RerankerBackend):
    """Standard retrieve-then-rerank cross-encoder (query, chunk) scoring.

    Most precise (judges each chunk against the query text directly), but
    pulls in torch/sentence-transformers — a slow first install and a real
    image-size cost. Kept registered so it's a one-line RERANK_BACKEND swap
    once that's acceptable; ReciprocalRankFusionBackend is the default.
    """

    def __init__(self, model_name: str = RERANK_MODEL):
        self._model_name = model_name
        self._model = None
        self._load_failed = False

    def _get_model(self):
        if self._model is not None or self._load_failed:
            return self._model
        try:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self._model_name)
        except Exception:
            logger.warning("Reranker model %s unavailable; skipping rerank.", self._model_name, exc_info=True)
            self._load_failed = True
        return self._model

    def rerank(
        self,
        query: str,
        items: list[dict],
        *,
        text_key: str = "text",
        sources: Optional[dict[str, list[dict]]] = None,
    ) -> list[dict]:
        if len(items) < 2:
            return items
        model = self._get_model()
        if model is None:
            return items

        pairs = [(query, (item.get(text_key) or "")[:_TEXT_TRUNCATE_CHARS]) for item in items]
        try:
            scores = model.predict(pairs)
        except Exception:
            logger.warning("Reranker inference failed; keeping original order.", exc_info=True)
            return items

        ranked = sorted(zip(items, scores), key=lambda pair: float(pair[1]), reverse=True)
        return [item for item, _ in ranked]


# Constant from Cormack et al. 2009 ("Reciprocal Rank Fusion outperforms
# Condorcet and individual Rank Learning Methods") — the value the paper and
# most production hybrid-search implementations (Elasticsearch, Azure AI
# Search) use; dampens the gap between rank 1 and rank 2 so one signal's top
# hit doesn't automatically dominate.
_RRF_K = 60


class ReciprocalRankFusionBackend(RerankerBackend):
    """Fuses the per-signal rankings by rank position instead of raw score.

    No model, no dependency: `_merge_and_rank`'s weighted-score combination
    mixes scores from different distributions (vector cosine, BM25-ish
    fulltext, graph hop-decay, lexical) that aren't on a shared scale, which
    is exactly how an off-topic chunk that scores well on two signals for
    unrelated reasons ties with the chunk that actually answers the query.
    RRF sidesteps the scale problem entirely by using each list's rank
    instead of its score: score(d) = sum over lists containing d of
    1/(_RRF_K + rank_in_that_list(d)).
    """

    def rerank(
        self,
        query: str,
        items: list[dict],
        *,
        text_key: str = "text",
        sources: Optional[dict[str, list[dict]]] = None,
    ) -> list[dict]:
        if not sources or len(items) < 2:
            return items

        rrf_scores: dict[str, float] = {}
        for hit_list in sources.values():
            if not hit_list:
                continue
            ranked_hits = sorted(hit_list, key=lambda h: float(h.get("score", 0.0)), reverse=True)
            for rank, hit in enumerate(ranked_hits, start=1):
                cid = hit.get("id")
                if not cid:
                    continue
                rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (_RRF_K + rank)

        if not rrf_scores:
            return items
        return sorted(items, key=lambda item: rrf_scores.get(item.get("id") or "", 0.0), reverse=True)


register_reranker("noop", NoopRerankerBackend)
register_reranker("cross_encoder", CrossEncoderRerankerBackend)
register_reranker("rrf", ReciprocalRankFusionBackend)


class RerankerService:
    """Resolves the configured backend once and delegates to it.

    Pass `backend` directly to pin a specific implementation (tests, A/B
    comparison); otherwise resolves RERANK_BACKEND from the registry on
    first use.
    """

    def __init__(self, enabled: bool = RERANK_ENABLED, backend: Optional[RerankerBackend] = None):
        self._enabled = enabled
        self._backend = backend

    def _get_backend(self) -> RerankerBackend:
        if self._backend is None:
            try:
                self._backend = get_reranker_backend(RERANK_BACKEND)
            except ValueError:
                logger.warning("Unknown RERANK_BACKEND %r; falling back to noop.", RERANK_BACKEND)
                self._backend = NoopRerankerBackend()
        return self._backend

    def rerank(
        self,
        query: str,
        items: list[dict],
        *,
        text_key: str = "text",
        sources: Optional[dict[str, list[dict]]] = None,
    ) -> list[dict]:
        if not self._enabled or not query or len(items) < 2:
            return items
        return self._get_backend().rerank(query, items, text_key=text_key, sources=sources)
