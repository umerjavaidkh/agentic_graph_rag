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

from src.retrieval.structured.cypher.validator import sql_cypher_issue


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
