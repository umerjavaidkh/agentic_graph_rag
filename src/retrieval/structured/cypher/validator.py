"""Cypher static validation (SQL idioms, temporal filters)."""
from __future__ import annotations

import re
from typing import Optional

# Schema-agnostic hints when a query executes but returns no rows.
EMPTY_RESULT_HINTS = (
    (
        "Query executed successfully but returned 0 rows. "
        "Verify every relationship direction matches RELATIONSHIP TYPES exactly "
        "(if schema shows (:A)-[:R]->(:B), traverse A-[:R]->B, never B-[:R]->A). "
        "Remove unnecessary WHERE filters."
    ),
    (
        "Query still returned 0 rows. "
        "Rebuild the MATCH path by chaining RELATIONSHIP TYPES from source to target. "
        "When counting unique orders/customers/entities across joins, use COUNT(DISTINCT node)."
    ),
)

# SQL idioms that are invalid or fragile in Cypher — trigger regeneration before execute.
SQL_CYPHER_ISSUES: list[tuple[str, str]] = [
    (r"\bGROUP\s+BY\b", "Neo4j Cypher does not support GROUP BY. Use WITH to group/aggregate."),
    (
        r"\bROW_NUMBER\s*\(\s*\)\s+OVER\b|\bOVER\s*\(\s*PARTITION\s+BY\b|\bPARTITION\s+BY\b",
        "Cypher does not support ROW_NUMBER() OVER / PARTITION BY. "
        "Use WITH ... ORDER BY groupKey, metric DESC ... collect({...}) AS rows then rows[0..N-1].",
    ),
    (
        r"\bRANK\s*\(\s*\)\s+OVER\b|\bDENSE_RANK\s*\(\s*\)\s+OVER\b",
        "Cypher does not support RANK() OVER. Use ordered collect + slice for top-N per group.",
    ),
    (
        r"RETURN\b[\s\S]*\bMATCH\b",
        "Never nest MATCH inside RETURN. Use WITH and a separate MATCH stage.",
    ),
    (
        r"\.\s*ORDER_CONTAINS\s*\.",
        "Bind ORDER_CONTAINS as a variable: (o)-[li:ORDER_CONTAINS]->(p) and use li.quantity, li.unitPrice, li.discount.",
    ),
    (
        r"\bWITH\b[^\n]*[, ]\s*\w+\.\w+\s*(?!\s+AS\b)",
        "Cypher syntax: every expression in WITH must be aliased using AS (e.g. `WITH p.productName AS productName`).",
    ),
    (
        r"\bAS\s+\w+\)\s+AS\s+\w+",
        "Cypher syntax: you have an extra ')' before an AS alias (e.g. 'AS x) AS y'). Remove the extra parenthesis.",
    ),
    (
        r"\bapoc\.coll\.sortNodes\b",
        "Do not use apoc.coll.sortNodes: it only sorts a LIST<NODE> and takes exactly 2 args, so it fails on lists of maps. "
        "To get top-K within a group, ORDER BY the metric DESC BEFORE collect(...), then slice the collected list: "
        "WITH groupKey, item ORDER BY metric DESC WITH groupKey, collect(item)[0..K] AS topItems.",
    ),
]

_QUESTION_YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")


def sql_cypher_issue(cypher: str) -> Optional[str]:
    for pattern, msg in SQL_CYPHER_ISSUES:
        if re.search(pattern, cypher, re.I | re.S):
            return msg
    return None


def dropped_year_filter_issue(cypher: str, query: str) -> Optional[str]:
    """
    Corpus-agnostic guard against multistep steps that silently drop a temporal
    filter the question requires.
    """
    years = set(_QUESTION_YEAR_RE.findall(query or ""))
    if not years:
        return None
    c = cypher or ""
    traverses = bool(re.search(r"\bMATCH\b", c, re.I)) and bool(re.search(r"-\s*\[", c))
    if not traverses:
        return None
    if any(y in c for y in years):
        return None
    yrs = ", ".join(sorted(years))
    first = sorted(years)[0]
    return (
        f"This step re-matches the graph but is missing the date filter for {yrs} "
        f"that the question requires. Every step that traverses the graph MUST repeat "
        f"the {yrs} filter (e.g. WHERE o.orderDate STARTS WITH '{first}'), or instead "
        f"UNWIND the prior step's already-filtered rows. Add the missing {yrs} filter."
    )
