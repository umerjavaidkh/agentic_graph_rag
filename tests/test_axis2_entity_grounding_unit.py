"""
tests/test_axis2_entity_grounding_unit.py — deterministic entity grounding
and typed NER (axis2._is_entity_grounded, axis2.Axis2Builder._extract_entities).

Covers: a hallucinated entity (not present in the source text) is dropped
before it can anchor a SHARES_ENTITY/SAME_CATEGORY edge; a genuine entity
survives; typed entities populate node.entity_types; an untyped fallback
(model ignores the typed-format instruction) still grounds and keeps text.

Run with:
    python -m pytest tests/test_axis2_entity_grounding_unit.py -v
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock


from src.models import DKGNode, NodeType
from src.semantic.axis2 import Axis2Builder, _is_entity_grounded


def _section_node(node_id: str, text: str) -> DKGNode:
    node = DKGNode(id=node_id, type=NodeType.SECTION, title=node_id, text=text, order=0)
    node.entities = []
    return node


def _builder() -> Axis2Builder:
    builder = Axis2Builder.__new__(Axis2Builder)
    builder.client = MagicMock()
    return builder


def _batch_response(entity_map: dict) -> MagicMock:
    return MagicMock(choices=[MagicMock(message=MagicMock(content=json.dumps(entity_map)))])


# ── _is_entity_grounded (pure) ───────────────────────────────────────────


def test_grounded_entity_found_verbatim():
    assert _is_entity_grounded("Isaac Newton", "A biography of Isaac Newton follows.")


def test_ungrounded_entity_not_in_text():
    assert not _is_entity_grounded("NeuralCorp Acquisition", "This page discusses Newton's laws of motion.")


def test_grounded_entity_case_insensitive():
    assert _is_entity_grounded("newton", "NEWTON discovered gravity.")


def test_grounded_entity_corp_suffix_fallback():
    """Corp-suffix-stripped fallback mirrors _canonicalize_entities's own
    normalization, so a legitimate variant isn't punished here for a
    difference canonicalization would merge away anyway."""
    assert _is_entity_grounded("Pfizer Inc.", "Pfizer reported record earnings this quarter.")


def test_empty_entity_never_grounded():
    assert not _is_entity_grounded("", "some text")
    assert not _is_entity_grounded("   ", "some text")


# ── _extract_entities integration: grounding + typing ───────────────────


def test_hallucinated_entity_is_dropped():
    node = _section_node("n0", text="Newton discovered the laws of motion.")
    builder = _builder()
    builder.client.chat_completion.return_value = _batch_response({
        "0": [
            {"text": "Newton", "type": "PERSON"},
            {"text": "NeuralCorp", "type": "ORG"},  # not in the text -- hallucinated
        ],
    })

    result = builder._extract_entities([node])

    assert result[0].entities == ["Newton"]


def test_typed_entities_populate_entity_types():
    node = _section_node("n0", text="Apple released a new iPhone model.")
    builder = _builder()
    builder.client.chat_completion.return_value = _batch_response({
        "0": [{"text": "Apple", "type": "ORG"}, {"text": "iPhone", "type": "PRODUCT"}],
    })

    result = builder._extract_entities([node])

    assert set(result[0].entities) == {"Apple", "iPhone"}
    assert result[0].entity_types == {"apple": "ORG", "iphone": "PRODUCT"}


def test_untyped_flat_string_response_still_grounds():
    """A model that ignores the typed-format instruction and returns plain
    strings must still work -- grounded and kept, just without a type."""
    node = _section_node("n0", text="Newton discovered gravity.")
    builder = _builder()
    builder.client.chat_completion.return_value = _batch_response({"0": ["Newton"]})

    result = builder._extract_entities([node])

    assert result[0].entities == ["Newton"]
    assert result[0].entity_types == {}


def test_all_entities_ungrounded_yields_empty_list():
    node = _section_node("n0", text="This page is about something else entirely.")
    builder = _builder()
    builder.client.chat_completion.return_value = _batch_response({
        "0": [{"text": "Fabricated Entity", "type": "ORG"}],
    })

    result = builder._extract_entities([node])

    assert result[0].entities == []
    assert result[0].entity_types == {}
