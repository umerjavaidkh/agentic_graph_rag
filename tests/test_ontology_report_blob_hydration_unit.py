"""
tests/test_ontology_report_blob_hydration_unit.py — _sample_edges/
_sample_entities hydrate from blob_key_text (docs/DESIGN_unstructured_
graph_v2.md phase 3), not from `.text` read directly off Neo4j.

DI-style fake session + fake BlobHydrator, mirrors
tests/test_ingestion_manager_di_unit.py's FakeGraphService pattern: proves
the Cypher asks for blob_key_text (not text) and that hydration -- not a
raw Cypher column -- is what ends up in source_text/target_text, so this
scoring pass reads identical content whether Neo4j still carries `.text`
or not (the axis-2 "score must not move" gate depends on this).

Run with:
    python -m pytest tests/test_ontology_report_blob_hydration_unit.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest


from src.unstructured.document import ontology_report as ontology_report_mod
from src.unstructured.document.ontology_report import _sample_edges, _sample_entities


class _FakeSession:
    def __init__(self, rows):
        self.rows = rows
        self.last_cypher = None

    def run(self, cypher, **kwargs):
        self.last_cypher = cypher
        return self.rows


class _FakeHydrator:
    def __init__(self, data: dict[str, str] | None = None):
        self._data = data or {}

    def hydrate(self, blob_key, fallback=""):
        if not blob_key:
            return fallback
        return self._data.get(blob_key, fallback)


def test_sample_edges_hydrates_source_and_target_from_blob_key(monkeypatch):
    monkeypatch.setattr(
        ontology_report_mod,
        "get_hydrator",
        lambda: _FakeHydrator({"blob/a": "full source text", "blob/b": "full target text"}),
    )
    session = _FakeSession(
        [
            {
                "source_blob_key": "blob/a",
                "target_blob_key": "blob/b",
                "rel_type": "SHARES_ENTITY",
                "shared": '{"shared_entities": ["x"]}',
                "source_entities": ["x", "y"],
                "target_entities": ["x"],
            }
        ]
    )

    result = _sample_edges(session, "doc1", "rev1", 5)

    assert result == [
        {
            "source_text": "full source text",
            "target_text": "full target text",
            "rel_type": "SHARES_ENTITY",
            "shared": '{"shared_entities": ["x"]}',
            "source_entities": ["x", "y"],
            "target_entities": ["x"],
        }
    ]
    # Cypher must ask for blob_key_text, not .text directly -- the whole
    # point of this rewire.
    assert "blob_key_text" in session.last_cypher
    assert "a.text" not in session.last_cypher
    assert "b.text" not in session.last_cypher


def test_sample_edges_missing_blob_key_degrades_to_empty_string(monkeypatch):
    monkeypatch.setattr(ontology_report_mod, "get_hydrator", lambda: _FakeHydrator({}))
    session = _FakeSession(
        [{
            "source_blob_key": None, "target_blob_key": "blob/missing", "rel_type": "SAME_CATEGORY", "shared": "",
            "source_entities": [], "target_entities": [],
        }]
    )

    result = _sample_edges(session, "doc1", "rev1", 5)

    assert result[0]["source_text"] == ""
    assert result[0]["target_text"] == ""


def test_sample_entities_hydrates_source_text_from_blob_key(monkeypatch):
    monkeypatch.setattr(
        ontology_report_mod,
        "get_hydrator",
        lambda: _FakeHydrator({"blob/node1": "node one full text"}),
    )
    session = _FakeSession(
        [{"source_blob_key": "blob/node1", "entities": ["Newton", "Force"]}]
    )

    result = _sample_entities(session, "doc1", "rev1", 5)

    assert {"source_text": "node one full text", "entity": "Newton"} in result
    assert {"source_text": "node one full text", "entity": "Force"} in result
    assert "node.text" not in session.last_cypher
    assert "blob_key_text" in session.last_cypher


def test_sample_entities_respects_n_cap_after_flattening(monkeypatch):
    monkeypatch.setattr(
        ontology_report_mod, "get_hydrator", lambda: _FakeHydrator({"blob/node1": "text"})
    )
    session = _FakeSession(
        [{"source_blob_key": "blob/node1", "entities": ["a", "b", "c", "d", "e"]}]
    )

    result = _sample_entities(session, "doc1", "rev1", 2)

    assert len(result) == 2
