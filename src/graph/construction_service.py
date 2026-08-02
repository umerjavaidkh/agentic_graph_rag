"""graph/construction_service.py — GraphConstructionService: the thin
orchestrator over Chunker + Axis1StructuralBuilder + Axis2IdeaBuilder
(docs/DESIGN_unstructured_graph_v2.md §3-4, phase 2 step 5).

Two methods, not one opaque `.build()`, because src/ingestion/service.py
interleaves apply_revision_to_graph, X1 snapshot writes, vision enrichment,
and the chat-provider-availability gate between structure and ideas — it
needs to stop and resume between the two axes, not get them atomically.
"""
from __future__ import annotations

from typing import Optional

from ..document.ir import DocumentIR
from ..models import DKGEdge, DKGNode
from ..semantic.axis2 import Axis2IdeaBuilder
from .axis1_structural import Axis1StructuralBuilder
from .axis_builder import GraphAxisBuilder
from .chunker import Chunk, Chunker, StructuralChunker


class GraphConstructionService:
    def __init__(
        self,
        chunker: Optional[Chunker] = None,
        axis1: Optional[GraphAxisBuilder] = None,
        axis2: Optional[GraphAxisBuilder] = None,
    ):
        self.chunker = chunker or StructuralChunker()
        self.axis1 = axis1 or Axis1StructuralBuilder()
        self.axis2 = axis2 or Axis2IdeaBuilder()

    def build_structure(
        self, ir: DocumentIR
    ) -> tuple[list[DKGNode], list[DKGEdge], list[Chunk]]:
        chunks = self.chunker.chunk(ir)
        nodes, edges = self.axis1.build(ir, chunks)
        return nodes, edges, chunks

    def build_ideas(
        self, nodes: list[DKGNode], run_llm_pass: bool = False
    ) -> tuple[list[DKGNode], list[DKGEdge]]:
        return self.axis2.build(nodes, run_llm_pass=run_llm_pass)
