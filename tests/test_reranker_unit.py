"""Unit tests for the pluggable reranker (registry + service + backends)."""
from __future__ import annotations

from src.retrieval.reranker_registry import (
    get_reranker_backend,
    list_rerankers,
    register_reranker,
)
from src.retrieval.unstructured.services.reranker import (
    CrossEncoderRerankerBackend,
    NoopRerankerBackend,
    ReciprocalRankFusionBackend,
    RerankerService,
)


def _items(*texts: str) -> list[dict]:
    return [{"id": f"c{i}", "text": t, "score": 1.0} for i, t in enumerate(texts)]


class _FakeCrossEncoder:
    """Stands in for sentence_transformers.CrossEncoder: scores by keyword match."""

    def __init__(self, *_args, **_kwargs):
        pass

    def predict(self, pairs):
        scores = []
        for query, text in pairs:
            q_words = set(query.lower().split())
            t_words = set(text.lower().split())
            scores.append(float(len(q_words & t_words)))
        return scores


def test_registry_has_default_backends():
    assert {"noop", "cross_encoder", "rrf"}.issubset(list_rerankers())


def test_registry_get_unknown_key_raises():
    try:
        get_reranker_backend("does_not_exist")
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_registry_custom_backend_can_be_registered():
    register_reranker("custom_test_backend", NoopRerankerBackend)
    backend = get_reranker_backend("custom_test_backend")
    assert isinstance(backend, NoopRerankerBackend)


def test_noop_backend_preserves_order():
    items = _items("alpha", "beta", "gamma")
    out = NoopRerankerBackend().rerank("alpha", items)
    assert [i["text"] for i in out] == ["alpha", "beta", "gamma"]


def test_service_disabled_returns_items_unchanged():
    items = _items("irrelevant well count table", "international upstream production 2025")
    service = RerankerService(enabled=False, backend=NoopRerankerBackend())
    out = service.rerank("international upstream production", items)
    assert out == items


def test_service_empty_query_returns_items_unchanged():
    items = _items("a", "b")
    service = RerankerService(enabled=True, backend=NoopRerankerBackend())
    out = service.rerank("", items)
    assert out == items


def test_service_single_item_returns_unchanged():
    items = _items("only one")
    service = RerankerService(enabled=True, backend=NoopRerankerBackend())
    out = service.rerank("query", items)
    assert out == items


def test_cross_encoder_backend_reorders_by_relevance(monkeypatch):
    backend = CrossEncoderRerankerBackend(model_name="fake-model")
    monkeypatch.setattr(
        "src.retrieval.unstructured.services.reranker.CrossEncoderRerankerBackend._get_model",
        lambda self: _FakeCrossEncoder(),
    )
    items = _items(
        "productive wells gross net acreage table",
        "international upstream liquids production 2025",
    )
    out = backend.rerank("international upstream liquids production 2025", items)
    assert out[0]["text"] == "international upstream liquids production 2025"


def test_cross_encoder_backend_falls_back_when_model_unavailable(monkeypatch):
    backend = CrossEncoderRerankerBackend(model_name="does-not-exist/definitely-not-a-real-model")
    monkeypatch.setattr(
        "src.retrieval.unstructured.services.reranker.CrossEncoderRerankerBackend._get_model",
        lambda self: None,
    )
    items = _items("a", "b")
    out = backend.rerank("query", items)
    assert out == items


def test_cross_encoder_backend_falls_back_on_inference_error(monkeypatch):
    backend = CrossEncoderRerankerBackend(model_name="fake-model")

    class _RaisingModel:
        def predict(self, pairs):
            raise RuntimeError("boom")

    monkeypatch.setattr(
        "src.retrieval.unstructured.services.reranker.CrossEncoderRerankerBackend._get_model",
        lambda self: _RaisingModel(),
    )
    items = _items("a", "b")
    out = backend.rerank("query", items)
    assert out == items


def test_service_resolves_backend_from_registry_by_default(monkeypatch):
    register_reranker("service_default_test", lambda: NoopRerankerBackend())
    monkeypatch.setattr(
        "src.retrieval.unstructured.services.reranker.RERANK_BACKEND", "service_default_test"
    )
    service = RerankerService(enabled=True)
    items = _items("a", "b")
    out = service.rerank("query", items)
    assert out == items


def test_rrf_backend_no_sources_returns_items_unchanged():
    items = _items("a", "b")
    out = ReciprocalRankFusionBackend().rerank("query", items, sources=None)
    assert out == items


def test_rrf_backend_promotes_item_ranked_high_across_multiple_signals():
    # c1 is the true answer: mediocre in vector, but top of fulltext+graph
    # (the exact "irrelevant table ties with the real answer" failure mode
    # this backend exists to fix). c0 is a generic-term false positive that
    # only shows up in vector.
    items = [
        {"id": "c0", "text": "well count table", "score": 5.0},
        {"id": "c1", "text": "international upstream production", "score": 3.0},
    ]
    sources = {
        "vector": [
            {"id": "c0", "score": 0.9},
            {"id": "c1", "score": 0.5},
        ],
        "fulltext": [
            {"id": "c1", "score": 10.0},
        ],
        "graph": [
            {"id": "c1", "score": 0.8},
        ],
    }
    out = ReciprocalRankFusionBackend().rerank("international upstream production", items, sources=sources)
    assert out[0]["id"] == "c1"


def test_rrf_backend_ignores_hits_missing_id():
    items = _items("a", "b")
    sources = {"vector": [{"score": 1.0}, {"id": "c0", "score": 0.5}]}
    out = ReciprocalRankFusionBackend().rerank("query", items, sources=sources)
    assert len(out) == 2


def test_service_uses_rrf_by_default_backend_key():
    service = RerankerService(enabled=True)
    backend = service._get_backend()
    assert isinstance(backend, ReciprocalRankFusionBackend)


def test_service_falls_back_to_noop_on_unknown_backend(monkeypatch):
    monkeypatch.setattr(
        "src.retrieval.unstructured.services.reranker.RERANK_BACKEND", "not_registered_anywhere"
    )
    service = RerankerService(enabled=True)
    items = _items("a", "b")
    out = service.rerank("query", items)
    assert out == items
