"""graph/axis_builder.py — GraphAxisBuilder: the interface Axis1StructuralBuilder
and Axis2IdeaBuilder both implement (docs/DESIGN_unstructured_graph_v2.md §4).

Deliberately loose (`*args, **kwargs`) rather than one shared parameter
shape: axis1 builds a graph from scratch (`DocumentIR` + `Chunk`s in,
`(nodes, edges)` out), axis2 enriches an already-built node list
(`nodes` in, `(nodes, edges)` out) -- genuinely different-shaped
operations, and neither axis has a second implementation to swap in
today (the design doc's own "Swap to try…" column says "—" for axis1).
Forcing a single rigid context object now would be speculative
generality with no current payoff; tighten this once a second impl of
either axis actually exists.
"""
from __future__ import annotations

from typing import Protocol

from ..models import DKGEdge, DKGNode


class GraphAxisBuilder(Protocol):
    def build(self, *args, **kwargs) -> tuple[list[DKGNode], list[DKGEdge]]:
        ...
