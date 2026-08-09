"""reranker_registry.py — flat name-keyed dispatch for reranker backends.

Mirrors strategy_registry.py's shape exactly (key -> factory in a
module-level dict, resolved via a lookup function) so swapping or A/B-ing a
reranking technique is the same motion as swapping a retrieval strategy:
register a new factory under a new key, point RERANK_BACKEND at it.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from .unstructured.services.reranker import RerankerBackend

_RERANKER_REGISTRY: dict[str, Callable[[], "RerankerBackend"]] = {}


def register_reranker(key: str, factory: Callable[[], "RerankerBackend"]) -> None:
    """Register a reranker backend factory under `key`."""
    _RERANKER_REGISTRY[key] = factory


def get_reranker_backend(key: str) -> "RerankerBackend":
    """Resolve and construct the reranker backend registered under `key`."""
    factory = _RERANKER_REGISTRY.get(key)
    if factory is None:
        raise ValueError(f"No reranker backend registered for key {key!r}")
    return factory()


def list_rerankers() -> set[str]:
    return set(_RERANKER_REGISTRY)
