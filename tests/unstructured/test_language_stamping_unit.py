"""
tests/unstructured/test_language_stamping_unit.py — one language per document,
on every node of it.

The design puts `language` where `tenant_id` already is: decided once per
document, stamped by apply_revision_to_graph, and carried on the live
Neo4j write path. Both halves are tested here because a property stamped
onto the model but missing from the writer is written nowhere, and the
exporter has four writers -- the failure mode
test_entity_types_persistence_unit.py already exists to catch.

Run with:
    python -m pytest tests/unstructured/test_language_stamping_unit.py -v
"""
from __future__ import annotations

from pathlib import Path

from src.unstructured.document.versioning import (
    DocumentRevisionPlan,
    apply_revision_to_graph,
    build_revision_plan,
    revision_metadata_nodes,
)
from src.unstructured.exporter.exporter import Neo4jExporter
from src.unstructured.models import DKGEdge, DKGNode, RelType


def _plan(language: str = "en") -> DocumentRevisionPlan:
    return DocumentRevisionPlan(
        logical_id="doc_annual_report",
        revision_id="doc_annual_report:r1",
        version_number=1,
        content_hash="abc123",
        content_root_id="doc_annual_report:r1::doc_annual_report",
        title="Annual Report",
        source_filename="annual_report.pdf",
        tenant_id="default",
        language=language,
    )


def _node(node_id: str = "sec_1") -> DKGNode:
    return DKGNode(
        id=node_id,
        type="Section",
        title="Results",
        text="Revenue increased.",
        search_text="Revenue increased.",
        order=1,
    )


def test_every_node_of_a_document_carries_the_document_language():
    """Per document, not per node.

    An earlier draft labelled each node by the language of its own text so
    that English sections of a bilingual document stayed reachable. That
    was rejected: it puts one document in two corpora, and every count and
    every "which document" answer then has to say which half it means.
    """
    nodes = [_node("a"), _node("b"), _node("c")]
    stamped, _ = apply_revision_to_graph(nodes, [], _plan("ar"))
    assert [n.language for n in stamped] == ["ar", "ar", "ar"]


def test_edges_are_stamped_alongside_nodes():
    edge = DKGEdge("a", "b", RelType.CONTAINS)
    _, edges = apply_revision_to_graph([_node("a"), _node("b")], [edge], _plan("ar"))
    assert [e.language for e in edges] == ["ar"]


def test_the_document_metadata_nodes_carry_the_language():
    """DocumentLogical is what the :Language edge attaches to, so it of all
    nodes must not be the one that misses the property."""
    nodes, _ = revision_metadata_nodes(_plan("ar"))
    assert {n.type for n in nodes} == {"DocumentLogical", "DocRevision"}
    assert all(n.language == "ar" for n in nodes)


def test_a_plan_built_without_a_language_is_the_default(tmp_path):
    """Every existing caller keeps working and lands in the default language,
    which is exactly the backfill rule applied going forward."""
    pdf = tmp_path / "annual_report.pdf"
    pdf.write_bytes(b"%PDF-1.4\n")
    plan = build_revision_plan(pdf, tenant_id="default")
    assert plan.language == "en"


def test_the_live_writer_carries_the_language():
    """A field on the model that the writer never sends is written nowhere.

    This is the batch param dict -- the path a real ingest takes. The CSV
    and full-cypher exporters carry neither tenant_id nor lifecycle_status
    either; scoping properties live on the live path only, and language
    follows that precedent rather than inventing a second one.
    """
    node = _node()
    node.language = "ar"
    row = Neo4jExporter._node_to_param_dict(node)
    assert row["language"] == "ar"


def test_language_sits_beside_tenant_id_in_the_written_row():
    """Both are scoping properties on the same node; if one is present and
    the other is not, that asymmetry is a bug rather than a decision."""
    node = _node()
    row = Neo4jExporter._node_to_param_dict(node)
    assert "tenant_id" in row and "language" in row
