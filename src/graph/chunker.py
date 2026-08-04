"""graph/chunker.py — Chunker: splits a DocumentIR into retrieval-sized
units before graph construction (docs/DESIGN_unstructured_graph_v2.md §3/§4).

Chunk granularity/overlap for the default impl is an explicit open
question in the design doc (§10), not resolved here. `StructuralChunker`
is the minimal default that satisfies the interface without pre-judging
that question: one `Chunk` per page, no sub-splitting, no cross-page
merging -- informationally identical to iterating `ir.pages` directly.
It's still wired load-bearingly, not decoratively: Axis1StructuralBuilder
builds Page nodes from `chunks` grouped by page rather than from
`ir.pages` directly, so a future non-default `Chunker` (fixed-window,
semantic, cross-page) changes Page-node granularity without
Axis1StructuralBuilder itself changing.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from ..document.ir import DocumentIR


@dataclass
class Chunk:
    id: str
    text: str
    page_start: int
    page_end: int
    source_pages: list[int] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)


class Chunker(Protocol):
    def chunk(self, ir: DocumentIR) -> list[Chunk]:
        ...


class StructuralChunker:
    """Default Chunker: one Chunk per page, verbatim."""

    def chunk(self, ir: DocumentIR) -> list[Chunk]:
        return [
            Chunk(
                id=f"page_{page.page}",
                text=page.text,
                page_start=page.page,
                page_end=page.page,
                source_pages=[page.page],
            )
            for page in ir.pages
        ]
