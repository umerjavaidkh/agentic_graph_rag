"""
tests/test_axis2_edge_grounding_unit.py — independent LLM-verifier grounding
for CONTRADICTS/ELABORATES/PREREQUISITE_OF edges (axis2.py's _ground_edge,
wired into _build_llm_edges's _llm_pair via AXIS2_GROUND_LLM_EDGES).

The relationship-detection call and the grounding call are two separate
chat_completion invocations -- tests here drive them via side_effect in
call order, not return_value (which would answer both calls identically).

Run with:
    python -m pytest tests/test_axis2_edge_grounding_unit.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

_root = Path(__file__).resolve().parents[1]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from src.models import DKGNode, NodeType
from src.semantic import axis2 as axis2_module
from src.semantic.axis2 import Axis2Builder


def _embedded_node(node_id: str, embedding: list[float]) -> DKGNode:
    node = DKGNode(id=node_id, type=NodeType.SECTION, title=node_id, text="text", order=0)
    node.embedding = embedding
    return node


def _builder() -> Axis2Builder:
    builder = Axis2Builder.__new__(Axis2Builder)
    builder.client = MagicMock()
    return builder


def _relationship_response(rel: str, confidence: float, reason: str = "reason") -> MagicMock:
    return MagicMock(
        choices=[MagicMock(message=MagicMock(
            content=f'{{"relationship": "{rel}", "direction": "A_TO_B", '
                    f'"confidence": {confidence}, "reason": "{reason}"}}'
        ))]
    )


def _grounding_response(grounded: bool, confidence: float = 0.5) -> MagicMock:
    return MagicMock(
        choices=[MagicMock(message=MagicMock(
            content=f'{{"grounded": {str(grounded).lower()}, "confidence": {confidence}}}'
        ))]
    )


def test_ungrounded_edge_is_dropped_despite_high_relationship_confidence(monkeypatch):
    monkeypatch.setattr(axis2_module, "AXIS2_GROUND_LLM_EDGES", True)
    a = _embedded_node("a", [1.0, 0.0])
    b = _embedded_node("b", [0.99, 0.02])

    builder = _builder()
    builder.client.chat_completion.side_effect = [
        _relationship_response("ELABORATES", 0.95),
        _grounding_response(grounded=False, confidence=0.1),
    ]

    edges = builder._build_llm_edges([a, b])

    assert edges == []
    assert builder.client.chat_completion.call_count == 2


def test_grounded_edge_is_kept_with_grounding_properties(monkeypatch):
    monkeypatch.setattr(axis2_module, "AXIS2_GROUND_LLM_EDGES", True)
    a = _embedded_node("a", [1.0, 0.0])
    b = _embedded_node("b", [0.99, 0.02])

    builder = _builder()
    builder.client.chat_completion.side_effect = [
        _relationship_response("CONTRADICTS", 0.9),
        _grounding_response(grounded=True, confidence=0.77),
    ]

    edges = builder._build_llm_edges([a, b])

    assert len(edges) == 1
    assert edges[0].properties["grounding_checked"] is True
    assert edges[0].properties["grounding_confidence"] == 0.77


def test_grounding_disabled_keeps_edge_with_a_single_call(monkeypatch):
    monkeypatch.setattr(axis2_module, "AXIS2_GROUND_LLM_EDGES", False)
    a = _embedded_node("a", [1.0, 0.0])
    b = _embedded_node("b", [0.99, 0.02])

    builder = _builder()
    builder.client.chat_completion.return_value = _relationship_response("ELABORATES", 0.9)

    edges = builder._build_llm_edges([a, b])

    assert len(edges) == 1
    assert "grounding_checked" not in edges[0].properties
    assert builder.client.chat_completion.call_count == 1


def test_grounding_call_provider_error_fails_closed(monkeypatch):
    """A grounding call that raises (provider error, malformed JSON, etc.)
    must drop the edge, not keep it -- an ungrounded-by-default edge is the
    safe failure mode, same posture as ontology_validation's judge."""
    monkeypatch.setattr(axis2_module, "AXIS2_GROUND_LLM_EDGES", True)
    a = _embedded_node("a", [1.0, 0.0])
    b = _embedded_node("b", [0.99, 0.02])

    builder = _builder()
    builder.client.chat_completion.side_effect = [
        _relationship_response("PREREQUISITE_OF", 0.9),
        RuntimeError("provider timeout"),
    ]

    edges = builder._build_llm_edges([a, b])

    assert edges == []
