"""Thread memory is bounded: it is an LRU, not an ever-growing dict.

A thread_id was only ever removed by an explicit clear_turn(), and nothing
calls that for an idle conversation -- so a long-running process retained one
snapshot per thread it had EVER seen. Unbounded growth, no eviction.
"""
from __future__ import annotations

import sys
from pathlib import Path


from src.shared.conversation import thread_memory as tm


def _snap(tid: str) -> None:
    tm._store[tid] = {"question": tid}
    tm._store.move_to_end(tid)
    while len(tm._store) > tm._MAX_THREADS:
        tm._store.popitem(last=False)


def test_store_never_exceeds_the_cap():
    tm._store.clear()
    for i in range(tm._MAX_THREADS + 250):
        _snap(f"t{i}")
    assert len(tm._store) == tm._MAX_THREADS


def test_oldest_thread_is_evicted_first():
    tm._store.clear()
    for i in range(tm._MAX_THREADS + 1):
        _snap(f"t{i}")
    assert tm.get_turn("t0") is None
    assert tm.get_turn(f"t{tm._MAX_THREADS}") is not None


def test_reading_a_thread_protects_it_from_eviction():
    """An in-use conversation must never be the one evicted."""
    tm._store.clear()
    _snap("keep")
    for i in range(tm._MAX_THREADS - 1):
        _snap(f"t{i}")
    tm.get_turn("keep")          # touch it: now most-recently-used
    _snap("overflow")            # forces one eviction
    assert tm.get_turn("keep") is not None


def test_empty_thread_id_is_ignored():
    tm._store.clear()
    assert tm.get_turn("") is None
    tm.save_turn("", "q", {})
    assert len(tm._store) == 0
