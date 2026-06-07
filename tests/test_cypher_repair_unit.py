"""Unit tests for deterministic Cypher repair helpers."""
from src.retrieval.structured.cypher.repair import (
    fix_extra_paren_as_alias,
    fix_with_missing_aliases,
    normalize_generated_cypher,
)
from src.retrieval.structured.cypher.validator import sql_cypher_issue

_SCHEMA = """
RELATIONSHIP TYPES:
(:Product)-[:SUPPLIED_BY]->(:Supplier)
(:Order)-[:ORDER_CONTAINS]->(:Product)
"""


def test_fix_with_missing_aliases():
    cypher = "MATCH (p:Product) WITH p.productName RETURN p.productName"
    fixed = fix_with_missing_aliases(cypher)
    assert "p.productName AS productName" in fixed


def test_fix_extra_paren_as_alias():
    cypher = "WITH sum(x) AS total) AS t RETURN t"
    fixed = fix_extra_paren_as_alias(cypher)
    assert ") AS" not in fixed or "total) AS" not in fixed


def test_count_distinct_in_with_not_flagged():
    cypher = (
        "MATCH (s:Supplier)<-[:SUPPLIED_BY]-(p:Product)<-[li:ORDER_CONTAINS]-(o:Order) "
        "WITH s.supplierID AS supplierID, COUNT(DISTINCT o.orderID) AS orderCount "
        "RETURN supplierID, orderCount"
    )
    assert sql_cypher_issue(cypher) is None


def test_multihop_path_preserved_by_normalize():
    cypher = (
        "MATCH (s:Supplier)<-[:SUPPLIED_BY]-(p:Product)<-[li:ORDER_CONTAINS]-(o:Order)\n"
        "WITH s.supplierID AS supplierID, COUNT(DISTINCT o.orderID) AS orderCount\n"
        "RETURN supplierID, orderCount"
    )
    fixed = normalize_generated_cypher(cypher, _SCHEMA)
    assert "<-[:SUPPLIED_BY]-" in fixed
    assert "Product" in fixed


def test_normalize_generated_cypher_idempotent():
    cypher = "MATCH (p:Product)-[:SUPPLIED_BY]->(s:Supplier) WITH p.productName RETURN p.productName"
    once = normalize_generated_cypher(cypher, _SCHEMA)
    twice = normalize_generated_cypher(once, _SCHEMA)
    assert once == twice
