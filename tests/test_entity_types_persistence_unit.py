"""
tests/test_entity_types_persistence_unit.py — DKGNode.entity_types survives
the trip into Neo4j.

Why this matters beyond "another field should be saved": entity_types was
defined on the model and populated by NER, but never written by ANY exporter
path, so it existed only in memory during a single ingestion run. That made
Axis-2 edges impossible to rebuild from the stored graph -- entity TYPE is
what drives the DATE exclusion (_NON_TOPICAL_ENTITY_TYPES), the same-type
enumeration cap (_dedupe_enumeration_types) and homonym separation
("Apple" ORG vs "apple" CONCEPT) in src/semantic/axis2.py. A rebuild without
it would silently produce WORSE edges than the ingestion that created them
while looking like it succeeded, so every iteration on edge quality required
re-running NER: the expensive, daily-quota-limited step.

Stored as a JSON string rather than a native map because Neo4j properties
cannot hold nested maps (same treatment `entities` already gets in the CSV
paths).

Run with:
    python -m pytest tests/test_entity_types_persistence_unit.py -v
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

_root = Path(__file__).resolve().parents[1]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from src.models import DKGNode, NodeType


def _node(entity_types: dict | None = None) -> DKGNode:
    node = DKGNode(id="n1", type=NodeType.SECTION, title="T", text="body", order=0)
    node.entities = ["u.s.", "chevron"]
    if entity_types is not None:
        node.entity_types = entity_types
    node.logical_doc_id = "doc_test"
    node.revision_id = "rev_001"
    node.lifecycle_status = "ACTIVE"
    node.content_hash = "abc"
    node.version_number = 1
    node.ingested_at = 0
    node.source_filename = "test.pdf"
    return node


def test_param_dict_includes_entity_types():
    from src.exporter.exporter import Neo4jExporter

    d = Neo4jExporter._node_to_param_dict(_node({"u.s.": "LOCATION", "chevron": "ORG"}))
    assert "entity_types" in d


def test_param_dict_entity_types_round_trips_as_json():
    from src.exporter.exporter import Neo4jExporter

    types = {"u.s.": "LOCATION", "chevron": "ORG"}
    d = Neo4jExporter._node_to_param_dict(_node(types))
    assert json.loads(d["entity_types"]) == types


def test_param_dict_entity_types_is_a_string_not_a_map():
    """Neo4j rejects a nested map as a property value -- writing the dict
    directly would fail at ingestion time, not here, so assert the encoding
    explicitly."""
    from src.exporter.exporter import Neo4jExporter

    d = Neo4jExporter._node_to_param_dict(_node({"u.s.": "LOCATION"}))
    assert isinstance(d["entity_types"], str)


def test_node_without_entity_types_serializes_to_empty_object():
    """Nodes from a path that never ran typed NER must still produce a
    valid, decodable value rather than null -- readers can then treat
    "no types" uniformly instead of special-casing a missing property."""
    from src.exporter.exporter import Neo4jExporter

    d = Neo4jExporter._node_to_param_dict(_node())
    assert json.loads(d["entity_types"]) == {}


def test_entity_types_round_trip_preserves_axis2_type_behavior():
    """The point of persisting this: an Axis-2 rebuild that reads types back
    out of storage must reach the same typed-canonicalization result as the
    original in-memory ingestion. Decoding the stored JSON and re-attaching
    it must keep "apple" the ORG distinct from "apple" the CONCEPT."""
    from src.exporter.exporter import Neo4jExporter
    from src.semantic.axis2 import _resolve_canonical_entities

    a = DKGNode(id="a", type=NodeType.SECTION, title="a", text="x", order=0)
    a.entities = ["apple"]
    a.entity_types = {"apple": "ORG"}
    b = DKGNode(id="b", type=NodeType.SECTION, title="b", text="x", order=1)
    b.entities = ["apple"]
    b.entity_types = {"apple": "CONCEPT"}

    before = _resolve_canonical_entities([a, b])

    # simulate the storage round trip
    for node in (a, b):
        stored = Neo4jExporter._node_to_param_dict(node)["entity_types"]
        node.entity_types = json.loads(stored)

    after = _resolve_canonical_entities([a, b])

    assert before == after
    assert after[("a", "apple")] != after[("b", "apple")]


def _csv_exporter(tmp_path):
    """Exporter with only the CSV-writing state initialized -- the real
    constructor also wires up blob/vector stores, which these path-level
    writer tests neither use nor should require."""
    from src.exporter.exporter import Neo4jExporter

    exporter = Neo4jExporter.__new__(Neo4jExporter)
    exporter.out = tmp_path
    (tmp_path / "nodes").mkdir(parents=True, exist_ok=True)
    (tmp_path / "edges").mkdir(parents=True, exist_ok=True)
    return exporter


def test_csv_export_includes_entity_types_column(tmp_path):
    """The CSV/LOAD CSV path is a separate writer from the batch param dict
    -- a field added to one and not the other silently drops on that path."""
    import csv as _csv

    exporter = _csv_exporter(tmp_path)
    exporter._write_node_csvs([_node({"u.s.": "LOCATION"})])

    csv_files = list((tmp_path / "nodes").glob("*.csv"))
    assert csv_files, "no node CSV written"
    with open(csv_files[0], newline="", encoding="utf-8") as f:
        rows = list(_csv.DictReader(f))
    assert "entity_types" in rows[0]
    assert json.loads(rows[0]["entity_types"]) == {"u.s.": "LOCATION"}


def test_load_csv_cypher_sets_entity_types(tmp_path):
    """A column in the CSV that the generated LOAD CSV statement never SETs
    would be written to disk and then silently dropped on import."""
    exporter = _csv_exporter(tmp_path)
    exporter._write_load_csv_cypher([_node({"u.s.": "LOCATION"})], [])

    cypher = (tmp_path / "import.cypher").read_text()
    assert "n.entity_types = row.entity_types" in cypher


def test_full_cypher_export_includes_entity_types(tmp_path):
    """Third independent writer (the single-file MERGE script)."""
    exporter = _csv_exporter(tmp_path)
    exporter._write_full_cypher([_node({"u.s.": "LOCATION"})], [])

    cypher = (tmp_path / "full_import.cypher").read_text()
    assert "n.entity_types=" in cypher
