"""
tests/test_graph_construction_service_unit.py — GraphConstructionService
(docs/DESIGN_unstructured_graph_v2.md phase 2, step 5).

Proves the thin orchestrator delegates to whichever chunker/axis1/axis2 it
was given rather than hardcoding concrete classes — same DI-provability
goal as test_ingestion_manager_di_unit.py, one level down.

Run with:
    python -m pytest tests/test_graph_construction_service_unit.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

_root = Path(__file__).resolve().parents[1]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from src.document.ir import DocumentIR
from src.graph.chunker import Chunk
from src.graph.construction_service import GraphConstructionService


class FakeChunker:
    def __init__(self, chunks):
        self._chunks = chunks
        self.chunked_irs = []

    def chunk(self, ir):
        self.chunked_irs.append(ir)
        return self._chunks


class FakeAxis1:
    def __init__(self, nodes, edges):
        self._nodes, self._edges = nodes, edges
        self.build_calls = []

    def build(self, ir, chunks):
        self.build_calls.append((ir, chunks))
        return self._nodes, self._edges


class FakeAxis2:
    def __init__(self, nodes, edges):
        self._nodes, self._edges = nodes, edges
        self.build_calls = []

    def build(self, nodes, run_llm_pass=False):
        self.build_calls.append((nodes, run_llm_pass))
        return self._nodes, self._edges


def test_build_structure_delegates_to_chunker_and_axis1():
    ir = DocumentIR(source_name="doc", page_count=0)
    chunks = [Chunk(id="c1", text="hi", page_start=1, page_end=1, source_pages=[1])]
    fake_chunker = FakeChunker(chunks)
    fake_axis1 = FakeAxis1(nodes=["node1"], edges=["edge1"])

    service = GraphConstructionService(chunker=fake_chunker, axis1=fake_axis1, axis2=FakeAxis2([], []))
    nodes, edges, returned_chunks = service.build_structure(ir)

    assert fake_chunker.chunked_irs == [ir]
    assert fake_axis1.build_calls == [(ir, chunks)]
    assert nodes == ["node1"]
    assert edges == ["edge1"]
    assert returned_chunks == chunks


def test_build_ideas_delegates_to_axis2_with_run_llm_pass_flag():
    fake_axis2 = FakeAxis2(nodes=["enriched"], edges=["semantic_edge"])
    service = GraphConstructionService(
        chunker=FakeChunker([]), axis1=FakeAxis1([], []), axis2=fake_axis2
    )

    nodes, edges = service.build_ideas(["node1"], run_llm_pass=True)

    assert fake_axis2.build_calls == [(["node1"], True)]
    assert nodes == ["enriched"]
    assert edges == ["semantic_edge"]


def test_build_ideas_defaults_run_llm_pass_to_false():
    fake_axis2 = FakeAxis2(nodes=[], edges=[])
    service = GraphConstructionService(
        chunker=FakeChunker([]), axis1=FakeAxis1([], []), axis2=fake_axis2
    )

    service.build_ideas(["node1"])

    assert fake_axis2.build_calls == [(["node1"], False)]


def test_defaults_wire_up_real_classes():
    """No-arg construction must resolve to the real StructuralChunker /
    Axis1StructuralBuilder / Axis2IdeaBuilder implementations -- this only
    checks wiring (construction doesn't touch the network), not behavior,
    which the golden-output and axis2 tests already cover elsewhere."""
    from src.graph.axis1_structural import Axis1StructuralBuilder
    from src.graph.chunker import StructuralChunker
    from src.semantic.axis2 import Axis2Builder, Axis2IdeaBuilder

    assert Axis2IdeaBuilder is Axis2Builder

    service = GraphConstructionService()
    assert isinstance(service.chunker, StructuralChunker)
    assert isinstance(service.axis1, Axis1StructuralBuilder)
    assert isinstance(service.axis2, Axis2IdeaBuilder)
