"""
tests/test_storage_vector_unit.py — VectorStore interface correctness tests.

Run with:
    python -m pytest tests/test_storage_vector_unit.py -v
"""
from __future__ import annotations

import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

_root = Path(__file__).resolve().parents[1]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from src.storage.vector.memory_store import InMemoryVectorStore
from src.storage.vector.factory import get_vector_store


def test_query_ranks_by_cosine_similarity():
    store = InMemoryVectorStore()
    store.upsert("a", [1.0, 0.0, 0.0])
    store.upsert("b", [0.9, 0.1, 0.0])
    store.upsert("c", [0.0, 1.0, 0.0])  # orthogonal, least similar

    results = store.query([1.0, 0.0, 0.0], top_k=3)
    ids_in_order = [id for id, _ in results]
    assert ids_in_order == ["a", "b", "c"]
    assert results[0][1] > results[1][1] > results[2][1]


def test_top_k_limits_result_count():
    store = InMemoryVectorStore()
    for i in range(5):
        store.upsert(f"id{i}", [float(i), 1.0, 0.0])
    results = store.query([1.0, 1.0, 0.0], top_k=2)
    assert len(results) == 2


def test_upsert_batch_matches_individual_upserts():
    store = InMemoryVectorStore()
    store.upsert_batch(
        [
            ("x", [1.0, 0.0], {"doc": "1"}),
            ("y", [0.0, 1.0], {"doc": "1"}),
        ]
    )
    assert store.query([1.0, 0.0], top_k=1)[0][0] == "x"


def test_delete_removes_vector_from_results():
    store = InMemoryVectorStore()
    store.upsert("a", [1.0, 0.0])
    store.upsert("b", [1.0, 0.0])
    store.delete("a")
    ids = [id for id, _ in store.query([1.0, 0.0], top_k=10)]
    assert ids == ["b"]


def test_query_empty_store_returns_empty_list():
    store = InMemoryVectorStore()
    assert store.query([1.0, 0.0], top_k=5) == []


def test_query_respects_metadata_filters():
    store = InMemoryVectorStore()
    store.upsert("a", [1.0, 0.0], metadata={"revision": "r1"})
    store.upsert("b", [1.0, 0.0], metadata={"revision": "r2"})
    results = store.query([1.0, 0.0], top_k=10, filters={"revision": "r1"})
    assert [id for id, _ in results] == ["a"]


def test_delete_by_filter_removes_matching_entries():
    store = InMemoryVectorStore()
    store.upsert("a", [1.0, 0.0], metadata={"revision": "old"})
    store.upsert("b", [1.0, 0.0], metadata={"revision": "new"})
    store.delete_by_filter({"revision": "old"})
    ids = [id for id, _ in store.query([1.0, 0.0], top_k=10)]
    assert ids == ["b"]


def test_factory_defaults_to_memory_backend():
    # VECTOR_STORE_BACKEND defaults to "memory" (settings.py) with no env override.
    import src.storage.vector.factory as factory_mod

    factory_mod._store_singleton = None
    try:
        store = get_vector_store()
        assert isinstance(store, InMemoryVectorStore)
    finally:
        factory_mod._store_singleton = None


# ── QdrantVectorStore, with `qdrant_client` mocked (no real Qdrant required) ─


def _install_fake_qdrant_sdk() -> MagicMock:
    """Stub the `qdrant_client` package in sys.modules; returns the fake QdrantClient class."""
    qc_mod = types.ModuleType("qdrant_client")
    qc_models_mod = types.ModuleType("qdrant_client.models")

    fake_client_cls = MagicMock(name="QdrantClient")
    qc_mod.QdrantClient = fake_client_cls

    # Model classes only need to be constructible; MagicMock stands in fine
    # since the store never inspects their internals, only passes them through.
    for name in (
        "Distance",
        "VectorParams",
        "PointStruct",
        "FieldCondition",
        "Filter",
        "MatchValue",
        "FilterSelector",
    ):
        setattr(qc_models_mod, name, MagicMock(name=name))

    qc_mod.models = qc_models_mod
    sys.modules["qdrant_client"] = qc_mod
    sys.modules["qdrant_client.models"] = qc_models_mod
    return fake_client_cls


def _reload_qdrant_store():
    import importlib

    if "src.storage.vector.qdrant_store" in sys.modules:
        importlib.reload(sys.modules["src.storage.vector.qdrant_store"])
    import src.storage.vector.qdrant_store as mod

    return mod


def test_qdrant_store_creates_collection_if_missing():
    fake_cls = _install_fake_qdrant_sdk()
    fake_client = MagicMock()
    fake_client.collection_exists.return_value = False
    fake_cls.return_value = fake_client
    mod = _reload_qdrant_store()

    mod.QdrantVectorStore("http://localhost:6333", "sections", dim=3)

    fake_client.create_collection.assert_called_once()
    assert fake_client.create_collection.call_args.kwargs["collection_name"] == "sections"


def test_qdrant_store_skips_create_when_collection_exists():
    fake_cls = _install_fake_qdrant_sdk()
    fake_client = MagicMock()
    fake_client.collection_exists.return_value = True
    fake_cls.return_value = fake_client
    mod = _reload_qdrant_store()

    mod.QdrantVectorStore("http://localhost:6333", "sections", dim=3)

    fake_client.create_collection.assert_not_called()


def test_qdrant_store_upsert_calls_client_upsert():
    fake_cls = _install_fake_qdrant_sdk()
    fake_client = MagicMock()
    fake_client.collection_exists.return_value = True
    fake_cls.return_value = fake_client
    mod = _reload_qdrant_store()

    store = mod.QdrantVectorStore("http://localhost:6333", "sections", dim=3)
    store.upsert("node1", [0.1, 0.2, 0.3], metadata={"doc": "d1"})

    assert fake_client.upsert.called
    assert fake_client.upsert.call_args.kwargs["collection_name"] == "sections"


def test_qdrant_store_query_maps_hits_back_to_source_id():
    fake_cls = _install_fake_qdrant_sdk()
    fake_client = MagicMock()
    fake_client.collection_exists.return_value = True
    fake_hit = MagicMock()
    fake_hit.payload = {"_source_id": "node1"}
    fake_hit.score = 0.87
    fake_hit.id = "some-uuid"
    fake_client.search.return_value = [fake_hit]
    fake_cls.return_value = fake_client
    mod = _reload_qdrant_store()

    store = mod.QdrantVectorStore("http://localhost:6333", "sections", dim=3)
    results = store.query([0.1, 0.2, 0.3], top_k=5)

    assert results == [("node1", 0.87)]


def test_qdrant_store_point_id_is_stable_uuid5():
    fake_cls = _install_fake_qdrant_sdk()
    fake_client = MagicMock()
    fake_client.collection_exists.return_value = True
    fake_cls.return_value = fake_client
    mod = _reload_qdrant_store()

    assert mod._point_id("node1") == mod._point_id("node1")
    assert mod._point_id("node1") != mod._point_id("node2")
