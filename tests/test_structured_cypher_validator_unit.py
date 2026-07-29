"""
tests/test_structured_cypher_validator_unit.py — static Cypher pre-execution checks.

Regression guard for the eval-discovered nw_05 bug ("Show monthly order count in
1997."): the LLM generated `apoc.date.format(datetime(...), ...)`, which always
throws a Neo4j type-mismatch error (apoc.date.format expects epoch-millis, not a
temporal value). This pattern is now caught statically before ever reaching Neo4j.

Run with:
    python -m pytest tests/test_structured_cypher_validator_unit.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_root = Path(__file__).resolve().parents[1]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from src.retrieval.structured.cypher.validator import sql_cypher_issue, unknown_label_issue


@pytest.mark.parametrize(
    "cypher",
    [
        "WITH apoc.date.format(datetime(o.orderDate).epochMillis, 'ms', 'yyyy-MM') AS month, count(o) AS c RETURN month, c",
        "WITH apoc.date.format(datetime(toString(o.orderDate)), 'ms', 'yyyy-MM') AS month, count(o) AS c RETURN month, c",
        "RETURN apoc.date.format(date(o.orderDate), 'ms', 'yyyy-MM-dd')",
    ],
)
def test_apoc_date_format_on_temporal_value_flagged(cypher):
    issue = sql_cypher_issue(cypher)
    assert issue is not None
    assert "apoc.date.format" in issue


def test_apoc_date_format_on_integer_millis_not_flagged():
    cypher = "RETURN apoc.date.format(o.orderDateMillis, 'ms', 'yyyy-MM-dd')"
    assert sql_cypher_issue(cypher) is None


def test_substring_based_month_bucketing_not_flagged():
    cypher = "WITH substring(toString(o.orderDate), 0, 7) AS month, count(o) AS c RETURN month, c ORDER BY month"
    assert sql_cypher_issue(cypher) is None


# ── MULTI_TENANCY_ENABLED gating ─────────────────────────────────────────────


def test_tenant_filter_not_checked_when_multi_tenancy_disabled(monkeypatch):
    import src.retrieval.structured.cypher.validator as v

    monkeypatch.setattr(v, "MULTI_TENANCY_ENABLED", False)
    cypher = "MATCH (p:Product) RETURN p.name"  # no tenant filter at all
    assert sql_cypher_issue(cypher) is None


def test_tenant_filter_checked_when_multi_tenancy_enabled(monkeypatch):
    import src.retrieval.structured.cypher.validator as v

    monkeypatch.setattr(v, "MULTI_TENANCY_ENABLED", True)
    cypher = "MATCH (p:Product) RETURN p.name"  # no tenant filter at all
    issue = sql_cypher_issue(cypher)
    assert issue is not None
    assert "tenant" in issue.lower()


def test_tenant_filter_satisfied_when_multi_tenancy_enabled(monkeypatch):
    import src.retrieval.structured.cypher.validator as v

    monkeypatch.setattr(v, "MULTI_TENANCY_ENABLED", True)
    cypher = "MATCH (p:Product) WHERE p.tenant_id = $tenant_id RETURN p.name"
    assert sql_cypher_issue(cypher) is None


# ── unknown_label_issue ──────────────────────────────────────────────────────
#
# Regression: "Which employee has the most sales?" generated
# `MATCH (e:Employee)<-[:ORDERED]-(c:Customer)-...`, but this graph has no
# `Employee` node label at all -- employeeID is just a property on Order.
# Neo4j doesn't error on an unknown label, the MATCH just silently returns 0
# rows, so this went undetected until the whole pipeline fell all the way
# through to the unstructured-document fallback and answered from an
# unrelated SEC filing. Verified live against the real Northwind graph.

_NORTHWIND_LABELS = {"Product", "Category", "Supplier", "Customer", "Order", "Address"}


def test_hallucinated_employee_label_flagged():
    cypher = (
        "MATCH (e:Employee)<-[:ORDERED]-(c:Customer)-[:ORDERED]->(o:Order) "
        "RETURN e.employeeID"
    )
    issue = unknown_label_issue(cypher, _NORTHWIND_LABELS)
    assert issue is not None
    assert "Employee" in issue


def test_real_labels_not_flagged():
    cypher = "MATCH (o:Order)-[:ORDER_CONTAINS]->(p:Product) RETURN o, p"
    assert unknown_label_issue(cypher, _NORTHWIND_LABELS) is None


def test_relationship_types_are_not_mistaken_for_labels():
    """[r:ORDERED] is a relationship TYPE, a completely different namespace
    from node labels -- must never be flagged just because "ORDERED" isn't
    in the label set."""
    cypher = "MATCH (c:Customer)-[r:ORDERED]->(o:Order) RETURN c, o"
    assert unknown_label_issue(cypher, _NORTHWIND_LABELS) is None


def test_multi_label_node_each_checked():
    cypher = "MATCH (e:Employee:Person) RETURN e"
    issue = unknown_label_issue(cypher, _NORTHWIND_LABELS)
    assert issue is not None
    assert "Employee" in issue and "Person" in issue


def test_no_known_labels_declines_rather_than_false_positive():
    """If schema introspection ever comes back empty, decline rather than
    flag everything as unknown -- that would just make every query fail."""
    cypher = "MATCH (o:Order) RETURN o"
    assert unknown_label_issue(cypher, set()) is None


def test_unlabeled_node_pattern_not_flagged():
    cypher = "MATCH (n)-[:ORDERED]->(o:Order) RETURN n, o"
    assert unknown_label_issue(cypher, _NORTHWIND_LABELS) is None
