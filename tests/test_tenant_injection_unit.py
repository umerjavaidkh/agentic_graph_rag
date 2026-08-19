"""
tests/test_tenant_injection_unit.py — deterministic tenant_id filter injection.

The highest-risk new code in the multi-tenancy pass: LLM-generated Cypher is
mechanically rewritten (not trusted) to require a tenant_id predicate on every
labeled node variable, mirroring how repair.py already mechanically fixes
relationship directions today. Covers multi-hop patterns, OPTIONAL MATCH
null-safety, WITH-chained multi-stage queries, and anonymous nodes.

Run with:
    python -m pytest tests/test_tenant_injection_unit.py -v
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest


from src.retrieval.structured.cypher.tenant_injection import (
    inject_tenant_filters,
    missing_tenant_filter_issue,
)


def _norm(s: str) -> str:
    return " ".join(s.split())


# ── Simple single-clause cases ───────────────────────────────────────────────


def test_simple_match_gets_tenant_filter():
    cypher = "MATCH (p:Product) RETURN p.name"
    out = inject_tenant_filters(cypher)
    assert "p.tenant_id = $tenant_id" in out
    assert missing_tenant_filter_issue(out) is None


def test_existing_where_gets_predicate_appended():
    cypher = "MATCH (p:Product) WHERE p.price > 10 RETURN p.name"
    out = inject_tenant_filters(cypher)
    assert "p.price > 10" in out
    assert "p.tenant_id = $tenant_id" in out
    assert missing_tenant_filter_issue(out) is None


def test_multi_hop_path_filters_every_labeled_node():
    cypher = (
        "MATCH (o:Order)-[li:ORDER_CONTAINS]->(p:Product)-[:SUPPLIED_BY]->(s:Supplier) "
        "RETURN o, p, s"
    )
    out = inject_tenant_filters(cypher)
    assert "o.tenant_id = $tenant_id" in out
    assert "p.tenant_id = $tenant_id" in out
    assert "s.tenant_id = $tenant_id" in out
    assert missing_tenant_filter_issue(out) is None


def test_comma_separated_patterns_in_one_match():
    cypher = "MATCH (a:Customer), (b:Order) WHERE a.id = b.customerId RETURN a, b"
    out = inject_tenant_filters(cypher)
    assert "a.tenant_id = $tenant_id" in out
    assert "b.tenant_id = $tenant_id" in out
    assert missing_tenant_filter_issue(out) is None


# ── OPTIONAL MATCH null-safety ───────────────────────────────────────────────


def test_optional_match_gets_null_safe_predicate():
    cypher = "MATCH (o:Order) OPTIONAL MATCH (o)-[:SHIPPED_TO]->(a:Address) RETURN o, a"
    out = inject_tenant_filters(cypher)
    # The optional node's predicate must not exclude legitimately-null matches.
    assert re.search(r"\(a IS NULL OR a\.tenant_id = \$tenant_id\)", out)
    assert missing_tenant_filter_issue(out) is None


def test_optional_match_without_existing_where():
    cypher = "OPTIONAL MATCH (p:Product)-[:SUPPLIED_BY]->(s:Supplier) RETURN p, s"
    out = inject_tenant_filters(cypher)
    assert "(s IS NULL OR s.tenant_id = $tenant_id)" in out
    assert "(p IS NULL OR p.tenant_id = $tenant_id)" in out


# ── WITH-chained multi-stage queries ─────────────────────────────────────────


def test_with_chained_stages_each_get_own_filter():
    cypher = (
        "MATCH (c:Customer) WITH c "
        "MATCH (c)-[:ORDERED]->(o:Order) "
        "RETURN c, o"
    )
    out = inject_tenant_filters(cypher)
    assert "c.tenant_id = $tenant_id" in out
    assert "o.tenant_id = $tenant_id" in out
    assert missing_tenant_filter_issue(out) is None


# ── Anonymous nodes ───────────────────────────────────────────────────────────


def test_anonymous_node_gets_synthetic_variable_and_filter():
    cypher = "MATCH (:Customer)-[:ORDERED]->(o:Order) RETURN o"
    out = inject_tenant_filters(cypher)
    # The anonymous Customer node must get a synthetic var and its own filter.
    assert re.search(r"_tf\d+\.tenant_id = \$tenant_id", out)
    assert "o.tenant_id = $tenant_id" in out
    assert missing_tenant_filter_issue(out) is None


def test_fully_anonymous_unlabeled_node_still_filtered():
    cypher = "MATCH (a:Customer)-[:ORDERED]->(o:Order)-[:ORDER_CONTAINS]->(x) RETURN a, o, x"
    out = inject_tenant_filters(cypher)
    # x has no label at all — fail closed, still requires a filter.
    assert re.search(r"\bx\.tenant_id = \$tenant_id", out)


# ── Exempt RBAC/control-plane labels ─────────────────────────────────────────


def test_exempt_labels_are_not_filtered():
    cypher = "MATCH (u:User {user_id: $uid})-[:HAS_ROLE]->(r:Role) RETURN r"
    out = inject_tenant_filters(cypher)
    assert "u.tenant_id" not in out
    assert "r.tenant_id" not in out
    assert missing_tenant_filter_issue(out) is None


def test_mixed_exempt_and_content_labels():
    cypher = "MATCH (u:User)-[:HAS_ROLE]->(r:Role)-[:CAN_VIEW]->(d:Document) RETURN d"
    out = inject_tenant_filters(cypher)
    assert "u.tenant_id" not in out
    assert "r.tenant_id" not in out
    assert "d.tenant_id = $tenant_id" in out


# ── Idempotence (repair_fn is called multiple times across retry loops) ─────


def test_injection_is_idempotent():
    cypher = "MATCH (p:Product) RETURN p.name"
    once = inject_tenant_filters(cypher)
    twice = inject_tenant_filters(once)
    assert once == twice
    assert once.count("tenant_id") == twice.count("tenant_id")


def test_idempotent_with_existing_where():
    cypher = "MATCH (p:Product) WHERE p.price > 10 RETURN p.name"
    once = inject_tenant_filters(cypher)
    twice = inject_tenant_filters(once)
    assert once == twice


# ── missing_tenant_filter_issue fail-closed behavior ─────────────────────────


def test_missing_filter_detected_on_raw_unfiltered_query():
    cypher = "MATCH (p:Product) RETURN p.name"
    assert missing_tenant_filter_issue(cypher) is not None


def test_no_issue_when_fully_filtered():
    cypher = "MATCH (p:Product) WHERE p.tenant_id = $tenant_id RETURN p.name"
    assert missing_tenant_filter_issue(cypher) is None


def test_no_issue_for_exempt_only_query():
    cypher = "MATCH (u:User)-[:HAS_ROLE]->(r:Role) RETURN r.name"
    assert missing_tenant_filter_issue(cypher) is None


def test_empty_cypher_no_issue():
    assert missing_tenant_filter_issue("") is None
    assert inject_tenant_filters("") == ""


# ── Golden fixtures: real Cypher captured from the eval suite this session ──


def test_golden_multi_stage_supplier_revenue_query():
    """Real generated Cypher for eval case adv_02 (5-hop path + 2 WITH stages)."""
    cypher = (
        "MATCH (c:Customer)-[:ORDERED]->(o:Order)-[li:ORDER_CONTAINS]->(p:Product)"
        "-[:SUPPLIED_BY]->(s:Supplier) "
        "WHERE o.orderDate >= date('1997-01-01') AND o.orderDate < date('1998-01-01') "
        "WITH s.supplierID AS supplierID, s.companyName AS companyName, "
        "c.customerID AS customerID, li.unitPrice * li.quantity * (1 - li.discount) AS revenueLine "
        "WITH supplierID, companyName, SUM(revenueLine) AS totalRevenue, "
        "COLLECT(DISTINCT customerID) AS customers "
        "RETURN supplierID AS supplierID, companyName AS companyName, "
        "totalRevenue AS totalRevenue, SIZE(customers) AS distinctCustomerCount"
    )
    out = inject_tenant_filters(cypher)
    for var in ("c", "o", "p", "s"):
        assert f"{var}.tenant_id = $tenant_id" in out
    assert missing_tenant_filter_issue(out) is None
    # Original filter and aggregation logic must survive untouched.
    assert "o.orderDate >= date('1997-01-01')" in out
    assert "SUM(revenueLine) AS totalRevenue" in out


def test_golden_not_exists_subquery_with_nested_match():
    """
    Real generated Cypher for eval case nw_09 ("categories never in an order"):
    NOT EXISTS {...} embeds its own MATCH, which naive clause segmentation
    mistakes for a new top-level clause, corrupting the query (confirmed via
    live testing against the real app before this fix).
    """
    cypher = (
        "MATCH (cat:Category) WHERE NOT EXISTS {\n"
        "  MATCH (cat)<-[:BELONGS_TO]-(p:Product)<-[:ORDER_CONTAINS]-(o:Order)\n"
        "} RETURN cat.categoryID AS categoryID, cat.categoryName AS categoryName "
        "ORDER BY cat.categoryName ASC"
    )
    out = inject_tenant_filters(cypher)
    # The outer clause's predicate must land AFTER the closing brace, not inside it.
    assert re.search(r"\}\s*AND\s+cat\.tenant_id = \$tenant_id", out)
    # The nested MATCH's own variables must also get filtered.
    assert "p.tenant_id = $tenant_id" in out
    assert "o.tenant_id = $tenant_id" in out
    assert missing_tenant_filter_issue(out) is None
    # The braces must stay balanced — this is what makes the query valid Cypher.
    assert out.count("{") == out.count("}")


def test_exists_subquery_without_not():
    cypher = (
        "MATCH (c:Customer) WHERE EXISTS {\n"
        "  MATCH (c)-[:ORDERED]->(o:Order)\n"
        "} RETURN c.customerID"
    )
    out = inject_tenant_filters(cypher)
    assert "c.tenant_id = $tenant_id" in out
    assert "o.tenant_id = $tenant_id" in out
    assert missing_tenant_filter_issue(out) is None
    assert out.count("{") == out.count("}")


def test_count_subquery_with_nested_match():
    cypher = (
        "MATCH (c:Customer) WHERE COUNT {\n"
        "  MATCH (c)-[:ORDERED]->(o:Order)\n"
        "} > 5 RETURN c.customerID"
    )
    out = inject_tenant_filters(cypher)
    assert "c.tenant_id = $tenant_id" in out
    assert "o.tenant_id = $tenant_id" in out
    assert missing_tenant_filter_issue(out) is None
    assert out.count("{") == out.count("}")


def test_golden_month_bucketing_query():
    """Real generated Cypher for eval case nw_05 after the apoc.date.format fix."""
    cypher = (
        'MATCH (o:Order) WHERE o.orderDate >= date("1997-01-01") '
        'AND o.orderDate < date("1998-01-01") '
        "WITH substring(toString(o.orderDate), 0, 7) AS month, count(o) AS orderCount "
        "RETURN month, orderCount ORDER BY month"
    )
    out = inject_tenant_filters(cypher)
    assert "o.tenant_id = $tenant_id" in out
    assert missing_tenant_filter_issue(out) is None
    assert "ORDER BY month" in out
