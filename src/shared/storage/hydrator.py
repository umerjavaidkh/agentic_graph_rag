"""storage/hydrator.py — Hydrator: resolves a blob_key pointer back to full
text (docs/DESIGN_unstructured_graph_v2.md phase 3/4).

`src/storage` is the one place already imported by every layer that needs
this (exporter.py, ontology_report.py, graph_seeds.py, retrieval
strategies) without introducing a new cross-package import edge.

get_hydrator() is the swappable-module seam (same shape as
storage/vector/factory.py's get_vector_store()): default is a plain
BlobHydrator; HYDRATOR_CACHE=true wraps it in CachingHydrator, a bounded
in-process LRU -- the same node's text is often hydrated multiple times
within one query (e.g. subsection parent+children, box list scan), so a
short-lived per-process cache avoids repeat blob-store round-trips without
needing a distributed cache for what's still a per-request-scoped win.
"""
from __future__ import annotations

from collections import OrderedDict
from typing import Optional, Protocol

from .blob.base import BlobStore
from .blob.factory import get_blob_store


_MISS = object()  # private sentinel, see CachingHydrator.hydrate


class Hydrator(Protocol):
    def hydrate(self, blob_key: Optional[str], fallback: str = "") -> str: ...

    def hydrate_batch(self, blob_keys: dict[str, Optional[str]]) -> dict[str, str]: ...


class BlobHydrator:
    """Default Hydrator: fetches full text from a BlobStore via blob_key.
    Missing key, missing blob, or a store error all degrade to `fallback`
    rather than raising -- same swallow-and-log posture already used for
    the blob/vector dual-write in Neo4jExporter._dual_write_chunk."""

    def __init__(self, blob_store: Optional[BlobStore] = None):
        self.blob_store = blob_store or get_blob_store()

    def hydrate(self, blob_key: Optional[str], fallback: str = "") -> str:
        if not blob_key:
            return fallback
        try:
            return self.blob_store.get(blob_key) or fallback
        except Exception:
            return fallback

    def hydrate_batch(self, blob_keys: dict[str, Optional[str]]) -> dict[str, str]:
        return {k: self.hydrate(v) for k, v in blob_keys.items()}


class CachingHydrator:
    """Wraps another Hydrator with a bounded in-process LRU keyed by
    blob_key. Only successful (non-fallback) hydrations are cached -- a
    miss/error is retried next time rather than pinned as a permanent
    empty result."""

    def __init__(self, inner: Optional["Hydrator"] = None, max_entries: int = 512):
        self.inner = inner or BlobHydrator()
        self.max_entries = max_entries
        self._cache: OrderedDict[str, str] = OrderedDict()

    def hydrate(self, blob_key: Optional[str], fallback: str = "") -> str:
        if not blob_key:
            return fallback
        if blob_key in self._cache:
            self._cache.move_to_end(blob_key)
            return self._cache[blob_key]
        # A private sentinel (not the caller's own fallback) distinguishes
        # a genuine hit from a miss, so a miss is never mistaken for a hit
        # just because the caller's fallback happens to be non-empty/truthy.
        value = self.inner.hydrate(blob_key, _MISS)
        if value is _MISS:
            return fallback
        self._cache[blob_key] = value
        self._cache.move_to_end(blob_key)
        if len(self._cache) > self.max_entries:
            self._cache.popitem(last=False)
        return value

    def hydrate_batch(self, blob_keys: dict[str, Optional[str]]) -> dict[str, str]:
        return {k: self.hydrate(v) for k, v in blob_keys.items()}


_hydrator_singleton: Optional["Hydrator"] = None


def get_hydrator() -> "Hydrator":
    """Process-wide Hydrator singleton, resolved from settings.HYDRATOR_CACHE
    -- the swappable-module seam for this stage, same pattern as
    storage/vector/factory.py's get_vector_store(). A singleton (not a
    fresh instance per call) so CachingHydrator's cache actually persists
    across the many separate strategy call sites that each hydrate the
    same handful of nodes within one query."""
    global _hydrator_singleton
    if _hydrator_singleton is not None:
        return _hydrator_singleton

    from ..config.settings import HYDRATOR_CACHE

    _hydrator_singleton = CachingHydrator() if HYDRATOR_CACHE else BlobHydrator()
    return _hydrator_singleton
