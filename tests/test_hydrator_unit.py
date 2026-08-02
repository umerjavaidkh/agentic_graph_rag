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

from src.storage.hydrator import BlobHydrator


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
