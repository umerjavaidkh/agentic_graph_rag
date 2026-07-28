"""
tests/test_axis2_ner_batching_unit.py — Axis2's batched NER extraction.

Regression: _extract_entities made one LLM call per Section/Page node, so a
single 7,165-node document (a physics textbook, with thousands of such
nodes) burned an entire 10,000-request daily OpenAI quota by itself —
reproduced live twice, each time exhausting the quota mid-ingestion with no
error surfaced (the per-call `except Exception: return node.id, []` silently
swallows 429s, so the job just crawls forward instead of failing loudly).

Fix: batch AXIS2_NER_BATCH_SIZE nodes' text into a single call, prompting for
a JSON object keyed by each excerpt's local index. These tests guard: call
count actually drops by ~batch size, entities map back to the correct node
(not misaligned), a batch that fails entirely degrades to empty entities for
just that batch (not a crash), and a response missing some indices still
assigns empty (not KeyError) for those.

Run with:
    python -m pytest tests/test_axis2_ner_batching_unit.py -v
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

_root = Path(__file__).resolve().parents[1]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from src.config.settings import AXIS2_NER_BATCH_SIZE
from src.models import DKGNode, NodeType
from src.semantic.axis2 import Axis2Builder


def _section_node(node_id: str, text: str = "some physics text") -> DKGNode:
    node = DKGNode(id=node_id, type=NodeType.SECTION, title=node_id, text=text, order=0)
    node.embedding = None
    node.entities = []
    return node


def _builder() -> Axis2Builder:
    builder = Axis2Builder.__new__(Axis2Builder)
    builder.client = MagicMock()
    return builder


def _batch_response(entity_map: dict) -> MagicMock:
    return MagicMock(
        choices=[MagicMock(message=MagicMock(content=json.dumps(entity_map)))]
    )


def test_call_count_drops_by_roughly_batch_size():
    """The whole point of batching: N nodes should cost ceil(N/batch) calls,
    not N calls. This is what actually fixes the daily-quota exhaustion."""
    n = AXIS2_NER_BATCH_SIZE * 5  # exactly 5 full batches
    nodes = [_section_node(f"n{i}") for i in range(n)]

    builder = _builder()
    builder.client.chat_completion.return_value = _batch_response(
        {str(i): ["entity"] for i in range(AXIS2_NER_BATCH_SIZE)}
    )

    builder._extract_entities(nodes)

    assert builder.client.chat_completion.call_count == 5
    assert builder.client.chat_completion.call_count < n


def test_entities_map_back_to_the_correct_node_not_misaligned():
    nodes = [_section_node(f"n{i}") for i in range(3)]
    builder = _builder()
    builder.client.chat_completion.return_value = _batch_response({
        "0": ["alpha"],
        "1": ["beta", "gamma"],
        "2": [],
    })

    result = builder._extract_entities(nodes)

    by_id = {n.id: n.entities for n in result}
    assert by_id["n0"] == ["alpha"]
    assert by_id["n1"] == ["beta", "gamma"]
    assert by_id["n2"] == []


def test_batch_call_failure_yields_empty_entities_not_a_crash():
    nodes = [_section_node(f"n{i}") for i in range(3)]
    builder = _builder()
    builder.client.chat_completion.side_effect = RuntimeError("429 rate limited")

    result = builder._extract_entities(nodes)  # must not raise

    assert all(n.entities == [] for n in result)


def test_response_missing_an_index_assigns_empty_not_keyerror():
    nodes = [_section_node(f"n{i}") for i in range(3)]
    builder = _builder()
    # Model only returned entries for indices 0 and 2, skipping 1.
    builder.client.chat_completion.return_value = _batch_response({
        "0": ["alpha"],
        "2": ["gamma"],
    })

    result = builder._extract_entities(nodes)

    by_id = {n.id: n.entities for n in result}
    assert by_id["n0"] == ["alpha"]
    assert by_id["n1"] == []  # missing index -> empty, not an error
    assert by_id["n2"] == ["gamma"]


def test_malformed_json_response_yields_empty_entities():
    nodes = [_section_node("n0")]
    builder = _builder()
    builder.client.chat_completion.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content="not valid json"))]
    )

    result = builder._extract_entities(nodes)

    assert result[0].entities == []
