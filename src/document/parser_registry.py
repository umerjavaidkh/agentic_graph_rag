"""parser_registry.py — extension-keyed dispatch for DocumentParser implementations.

Mirrors the shape of model_providers/factory.py (name -> implementation)
but keyed by file extension (+ optional backend qualifier), since parsers
are scoped to a file type rather than a single global provider choice.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable

from ..config.settings import PDF_PARSER_BACKEND
from .light.parser import LightPdfParser
from .parser_base import DocumentParser
from .rtldoc_backend.parser import RtldocPdfParser
from .table_aware.parser import TableAwarePdfParser

_PARSER_REGISTRY: dict[str, Callable[[], DocumentParser]] = {}


def register_parser(key: str, factory: Callable[[], DocumentParser]) -> None:
    """Register a parser factory under a key of the form ".ext" or ".ext:backend"."""
    _PARSER_REGISTRY[key.lower()] = factory


def get_parser(source: str | Path, *, backend: str | None = None) -> DocumentParser:
    """Resolve the parser registered for `source`'s extension (and backend, if given)."""
    ext = Path(source).suffix.lower()
    backend = (backend or PDF_PARSER_BACKEND or "").lower()

    qualified_key = f"{ext}:{backend}" if backend else ""
    key = qualified_key if qualified_key in _PARSER_REGISTRY else ext
    factory = _PARSER_REGISTRY.get(key)
    if factory is None:
        raise ValueError(
            f"No parser registered for extension {ext!r} (backend={backend!r})"
        )
    return factory()


def supported_extensions() -> set[str]:
    """Bare extensions (e.g. '.pdf') with at least one registered parser."""
    return {key.split(":", 1)[0] for key in _PARSER_REGISTRY}


register_parser(".pdf", RtldocPdfParser)
register_parser(".pdf:light", LightPdfParser)
register_parser(".pdf:table-aware", TableAwarePdfParser)
register_parser(".pdf:rtldoc", RtldocPdfParser)
