"""storage/hydrator.py — Hydrator: resolves a blob_key pointer back to full
text (docs/DESIGN_unstructured_graph_v2.md phase 3/4).

`src/storage` is the one place already imported by every layer that needs
this (exporter.py, ontology_report.py, graph_seeds.py) without introducing a
new cross-package import edge -- see phase-3 plan's reasoning for why this
isn't in src/graph or src/document instead.

Deliberately minimal: no registry/swap-technique machinery yet (a
cache-backed Hydrator variant is real Phase 4 scope, once more call sites
depend on this).
"""
from __future__ import annotations

from typing import Optional, Protocol

from .blob.base import BlobStore
from .blob.factory import get_blob_store


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
