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
    from ...structured.retrieval.strategies.base import StructuredStrategy
    from ...unstructured.retrieval.strategies.base import UnstructuredStrategy

_UNSTRUCTURED_REGISTRY: dict[str, Callable[..., "UnstructuredStrategy"]] = {}
_STRUCTURED_REGISTRY: dict[str, Callable[..., "StructuredStrategy"]] = {}

# Strategies built with no arguments are cached and reused.
#
# Without this, every `get_unstructured("graph_rag_hybrid")` built a new
# strategy -- and FullHybridStrategy.__init__ creates a
# ThreadPoolExecutor(max_workers=8) whose own comment states it is safe
# because "this strategy is itself a process-wide singleton". It was not.
# Measured: 20 concurrent strategies held 78 live threads, and releasing
# them left 66 still alive, since a pool that is garbage collected rather
# than shut down reclaims its workers late.
#
# Only the zero-argument form is cached. A caller passing dependencies
# explicitly wants its own instance (tests do this), so that path still
# constructs, exactly as before.
_UNSTRUCTURED_SINGLETONS: dict[str, "UnstructuredStrategy"] = {}
_STRUCTURED_SINGLETONS: dict[str, "StructuredStrategy"] = {}


def register_unstructured(key: str, factory: Callable[..., "UnstructuredStrategy"]) -> None:
    """Register an unstructured strategy factory under `key` (the mode string)."""
    _UNSTRUCTURED_REGISTRY[key] = factory
    _UNSTRUCTURED_SINGLETONS.pop(key, None)


def register_structured(key: str, factory: Callable[..., "StructuredStrategy"]) -> None:
    """Register a structured strategy factory under `key` (e.g. 'text2cypher')."""
    _STRUCTURED_REGISTRY[key] = factory
    _STRUCTURED_SINGLETONS.pop(key, None)


def get_unstructured(key: str, *args: Any, **kwargs: Any) -> "UnstructuredStrategy":
    """Resolve and construct the unstructured strategy registered under `key`."""
    factory = _UNSTRUCTURED_REGISTRY.get(key)
    if factory is None:
        raise ValueError(f"No unstructured strategy registered for key {key!r}")
    if args or kwargs:
        return factory(*args, **kwargs)
    cached = _UNSTRUCTURED_SINGLETONS.get(key)
    if cached is None:
        cached = _UNSTRUCTURED_SINGLETONS[key] = factory()
    return cached


def get_structured(key: str, *args: Any, **kwargs: Any) -> "StructuredStrategy":
    """Resolve and construct the structured strategy registered under `key`."""
    factory = _STRUCTURED_REGISTRY.get(key)
    if factory is None:
        raise ValueError(f"No structured strategy registered for key {key!r}")
    if args or kwargs:
        return factory(*args, **kwargs)
    cached = _STRUCTURED_SINGLETONS.get(key)
    if cached is None:
        cached = _STRUCTURED_SINGLETONS[key] = factory()
    return cached


def list_unstructured() -> set[str]:
    return set(_UNSTRUCTURED_REGISTRY)


def list_structured() -> set[str]:
    return set(_STRUCTURED_REGISTRY)
