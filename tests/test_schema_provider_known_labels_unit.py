"""
tests/test_schema_provider_known_labels_unit.py — SchemaProvider.known_labels()
parses real node labels from live schema introspection, cached alongside the
existing schema string (no extra DB round trip).

Built to support unknown_label_issue's fix for a real bug: a generated Cypher
query MATCHed a node label ("Employee") that doesn't exist in the graph at
all -- employeeID is just a property on Order in this Northwind dataset, not
its own node label the classic Northwind schema would have. Neo4j doesn't
error on an unknown label, so this went undetected until the whole pipeline
fell through to an unrelated document-search fallback.

Run with:
    python -m pytest tests/test_schema_provider_known_labels_unit.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock


from src.structured.retrieval.schema.provider import SchemaProvider, _parse_node_type_labels


def test_parse_node_type_labels_single():
    assert _parse_node_type_labels(":`Order`") == ["Order"]


def test_parse_node_type_labels_multi():
    assert _parse_node_type_labels(":`Employee`:`Person`") == ["Employee", "Person"]


def test_parse_node_type_labels_empty():
    assert _parse_node_type_labels("") == []


class _FakeResult:
    def __init__(self, rows):
        self._rows = rows

    def __iter__(self):
        return iter(self._rows)


class _FakeSession:
    def __init__(self, node_type_rows, pattern_rows=None, rel_prop_rows=None):
        self._node_type_rows = node_type_rows
        self._pattern_rows = pattern_rows or []
        self._rel_prop_rows = rel_prop_rows or []
        self.run_count = 0

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def run(self, cypher, **kwargs):
        self.run_count += 1
        if "nodeTypeProperties" in cypher:
            return _FakeResult(self._node_type_rows)
        if "MATCH (a)-[r]->(b)" in cypher:
            return _FakeResult(self._pattern_rows)
        if "relTypeProperties" in cypher:
            return _FakeResult(self._rel_prop_rows)
        return _FakeResult([])


def _fake_driver(session):
    driver = MagicMock()
    driver.session.return_value = session
    return driver


def test_known_labels_parses_from_live_schema():
    session = _FakeSession([
        {"nodeType": ":`Order`", "properties": ["orderID: STRING"]},
        {"nodeType": ":`Product`", "properties": ["productName: STRING"]},
    ])
    provider = SchemaProvider(_fake_driver(session))
    assert provider.known_labels() == {"Order", "Product"}


def test_known_labels_does_not_cost_a_second_db_round_trip_after_fetch():
    session = _FakeSession([{"nodeType": ":`Order`", "properties": ["orderID: STRING"]}])
    provider = SchemaProvider(_fake_driver(session))
    provider.fetch()
    calls_after_fetch = session.run_count
    provider.known_labels()
    assert session.run_count == calls_after_fetch  # no new query


def test_known_labels_works_when_called_before_fetch():
    session = _FakeSession([{"nodeType": ":`Customer`", "properties": ["customerID: STRING"]}])
    provider = SchemaProvider(_fake_driver(session))
    assert provider.known_labels() == {"Customer"}


def test_clear_cache_clears_both_caches():
    session = _FakeSession([{"nodeType": ":`Order`", "properties": ["orderID: STRING"]}])
    provider = SchemaProvider(_fake_driver(session))
    provider.fetch()
    provider.clear_cache()
    assert provider._cache is None
    assert provider._labels_cache is None
