"""parser_base.py — structural interface all document parsers conform to."""
from __future__ import annotations

from pathlib import Path
from typing import Protocol

from ..models import DKGEdge, DKGNode
from .ir import DocumentIR


class DocumentParser(Protocol):
    """Any parser registered in parser_registry must implement this shape."""

    def parse(self, source: str | Path) -> tuple[list[DKGNode], list[DKGEdge]]:
        ...

    def parse_ir(self, source: str | Path) -> DocumentIR:
        ...
