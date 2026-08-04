"""
tests/test_hydrator_unit.py — BlobHydrator (docs/DESIGN_unstructured_graph_v2.md
phase 3): resolves a blob_key pointer back to full text.

Run with:
    python -m pytest tests/test_hydrator_unit.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_root = Path(__file__).resolve().parents[1]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from src.storage.hydrator import BlobHydrator, CachingHydrator, get_hydrator


class _FakeBlobStore:
    def __init__(self, data: dict[str, str]):
        self._data = data

    def get(self, key):
        return self._data.get(key)


class _RaisingBlobStore:
    def get(self, key):
        raise RuntimeError("blob store unavailable")


def test_hydrate_returns_stored_text_on_hit():
    hydrator = BlobHydrator(_FakeBlobStore({"k1": "full text"}))
    assert hydrator.hydrate("k1") == "full text"


def test_hydrate_missing_key_returns_fallback():
    hydrator = BlobHydrator(_FakeBlobStore({}))
    assert hydrator.hydrate(None) == ""
    assert hydrator.hydrate(None, fallback="default") == "default"


def test_hydrate_key_not_in_store_returns_fallback():
    hydrator = BlobHydrator(_FakeBlobStore({}))
    assert hydrator.hydrate("missing_key", fallback="x") == "x"


def test_hydrate_store_error_degrades_to_fallback_not_raise():
    hydrator = BlobHydrator(_RaisingBlobStore())
    assert hydrator.hydrate("k1", fallback="safe") == "safe"


def test_hydrate_batch_maps_each_key_independently():
    hydrator = BlobHydrator(_FakeBlobStore({"k1": "one", "k2": "two"}))
    result = hydrator.hydrate_batch({"a": "k1", "b": "k2", "c": None, "d": "missing"})
    assert result == {"a": "one", "b": "two", "c": "", "d": ""}


def test_default_constructor_resolves_real_blob_store(monkeypatch):
    """No-arg construction must resolve via get_blob_store(), same DI
    pattern as GraphConstructionService's own defaults."""
    import src.storage.hydrator as hydrator_mod

    sentinel = _FakeBlobStore({"k": "v"})
    monkeypatch.setattr(hydrator_mod, "get_blob_store", lambda: sentinel)

    hydrator = BlobHydrator()
    assert hydrator.blob_store is sentinel
    assert hydrator.hydrate("k") == "v"


class _CountingHydrator:
    """Records call count so caching can be proven by call count, not just
    return value equality."""

    def __init__(self, data: dict[str, str]):
        self._data = data
        self.calls = 0

    def hydrate(self, blob_key, fallback=""):
        self.calls += 1
        return self._data.get(blob_key, fallback)

    def hydrate_batch(self, blob_keys):
        return {k: self.hydrate(v) for k, v in blob_keys.items()}


def test_caching_hydrator_only_calls_inner_once_per_key():
    inner = _CountingHydrator({"k1": "cached text"})
    hydrator = CachingHydrator(inner)
    assert hydrator.hydrate("k1") == "cached text"
    assert hydrator.hydrate("k1") == "cached text"
    assert hydrator.hydrate("k1") == "cached text"
    assert inner.calls == 1


def test_caching_hydrator_does_not_cache_misses():
    inner = _CountingHydrator({})
    hydrator = CachingHydrator(inner)
    hydrator.hydrate("missing", fallback="x")
    hydrator.hydrate("missing", fallback="x")
    assert inner.calls == 2


def test_caching_hydrator_evicts_lru_beyond_max_entries():
    inner = _CountingHydrator({"a": "A", "b": "B", "c": "C"})
    hydrator = CachingHydrator(inner, max_entries=2)
    hydrator.hydrate("a")
    hydrator.hydrate("b")
    hydrator.hydrate("c")  # evicts "a" (least recently used)
    hydrator.hydrate("a")
    assert inner.calls == 4  # a, b, c, then a again (evicted)


def test_get_hydrator_returns_caching_variant_when_enabled(monkeypatch):
    import src.storage.hydrator as hydrator_mod

    monkeypatch.setattr(hydrator_mod, "_hydrator_singleton", None)
    monkeypatch.setattr("src.config.settings.HYDRATOR_CACHE", True)
    assert isinstance(get_hydrator(), CachingHydrator)
    monkeypatch.setattr(hydrator_mod, "_hydrator_singleton", None)


def test_get_hydrator_returns_plain_variant_when_disabled(monkeypatch):
    import src.storage.hydrator as hydrator_mod

    monkeypatch.setattr(hydrator_mod, "_hydrator_singleton", None)
    monkeypatch.setattr("src.config.settings.HYDRATOR_CACHE", False)
    assert type(get_hydrator()) is BlobHydrator
    monkeypatch.setattr(hydrator_mod, "_hydrator_singleton", None)


def test_get_hydrator_is_a_singleton(monkeypatch):
    import src.storage.hydrator as hydrator_mod

    monkeypatch.setattr(hydrator_mod, "_hydrator_singleton", None)
    assert get_hydrator() is get_hydrator()
    monkeypatch.setattr(hydrator_mod, "_hydrator_singleton", None)
