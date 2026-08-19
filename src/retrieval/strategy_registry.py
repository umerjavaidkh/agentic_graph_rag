"""strategy_registry.py — flat name-keyed dispatch for retrieval strategies.

Mirrors the shape of src/document/parser_registry.py (key -> factory in a
module-level dict, resolved via a lookup function) but for retrieval
strategies rather than document parsers. Two separate registries — one for
structured, one for unstructured — since the two Protocols take different
constructor dependencies (an unstructured strategy needs the shared-service
bundle; a structured strategy needs the schema/cypher collaborators), but
both are exposed through this one module so callers on either side use the
same lookup convention.

One deliberate deviation from parser_registry.py's template: parser
factories are genuinely zero-arg (parsers are stateless). Strategies are
not — they need their dependencies (shared services, driver, etc.) at
construction time, so factories here are `Callable[..., Strategy]` invoked
with whatever positional/keyword args the caller passes through, not
zero-arg callables.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from ..structured.retrieval.strategies.base import StructuredStrategy
    from ..unstructured.retrieval.strategies.base import UnstructuredStrategy

_UNSTRUCTURED_REGISTRY: dict[str, Callable[..., "UnstructuredStrategy"]] = {}
_STRUCTURED_REGISTRY: dict[str, Callable[..., "StructuredStrategy"]] = {}


def register_unstructured(key: str, factory: Callable[..., "UnstructuredStrategy"]) -> None:
    """Register an unstructured strategy factory under `key` (the mode string)."""
    _UNSTRUCTURED_REGISTRY[key] = factory


def register_structured(key: str, factory: Callable[..., "StructuredStrategy"]) -> None:
    """Register a structured strategy factory under `key` (e.g. 'text2cypher')."""
    _STRUCTURED_REGISTRY[key] = factory


def get_unstructured(key: str, *args: Any, **kwargs: Any) -> "UnstructuredStrategy":
    """Resolve and construct the unstructured strategy registered under `key`."""
    factory = _UNSTRUCTURED_REGISTRY.get(key)
    if factory is None:
        raise ValueError(f"No unstructured strategy registered for key {key!r}")
    return factory(*args, **kwargs)


def get_structured(key: str, *args: Any, **kwargs: Any) -> "StructuredStrategy":
    """Resolve and construct the structured strategy registered under `key`."""
    factory = _STRUCTURED_REGISTRY.get(key)
    if factory is None:
        raise ValueError(f"No structured strategy registered for key {key!r}")
    return factory(*args, **kwargs)


def list_unstructured() -> set[str]:
    return set(_UNSTRUCTURED_REGISTRY)


def list_structured() -> set[str]:
    return set(_STRUCTURED_REGISTRY)
