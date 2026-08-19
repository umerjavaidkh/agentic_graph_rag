"""tests/test_memory_vector_store_unit.py — InMemoryVectorStore size guard.

Covers the fix for a silent-scaling-cliff risk: this backend is a
brute-force O(n) cosine scan meant for tests/single-process dev (per its own
docstring), but nothing surfaced it if a misconfigured deployment ended up
relying on it at real scale. It now logs one warning (not per-query spam)
once the vector count crosses a threshold.

Run with:
    python -m pytest tests/test_memory_vector_store_unit.py -v
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path


from src.storage.vector import memory_store as memory_store_mod
from src.storage.vector.memory_store import InMemoryVectorStore


def test_no_warning_below_threshold(caplog):
    store = InMemoryVectorStore()
    with caplog.at_level(logging.WARNING, logger=memory_store_mod.__name__):
        for i in range(5):
            store.upsert(f"id{i}", [0.1, 0.2, 0.3])
    assert caplog.records == []


def test_warns_once_past_threshold(monkeypatch, caplog):
    monkeypatch.setattr(memory_store_mod, "_SIZE_WARNING_THRESHOLD", 3)
    store = InMemoryVectorStore()
    with caplog.at_level(logging.WARNING, logger=memory_store_mod.__name__):
        for i in range(10):
            store.upsert(f"id{i}", [0.1, 0.2, 0.3])
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1


def test_query_behavior_unaffected_by_size_guard():
    store = InMemoryVectorStore()
    store.upsert("a", [1.0, 0.0])
    store.upsert("b", [0.0, 1.0])
    results = store.query([1.0, 0.0], top_k=1)
    assert results and results[0][0] == "a"
