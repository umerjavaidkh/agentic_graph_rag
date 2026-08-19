"""
tests/test_structured_cypher_executor_unit.py — StructuredCypherExecutor retry loop.

Regression guard for a real bug found via the eval suite (nw_05 — "Show monthly
order count in 1997."): when the LAST allowed attempt's execute_once() raised and
regenerate() produced a genuinely different (and in that case correct) cypher, the
old loop set `cypher = fixed` then `continue`'d straight past the end of
`range(1, max_attempts + 1)` — exiting without ever executing the fix. The final
ExecuteResult then paired the STALE error from the failed attempt with the NEW
(never-tried) cypher string.

Run with:
    python -m pytest tests/test_structured_cypher_executor_unit.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest


from src.structured.retrieval.executor import StructuredCypherExecutor


def _no_issue(_cypher: str):
    return None


def _identity_repair(cypher: str) -> str:
    return cypher


def test_last_attempt_regeneration_is_actually_executed():
    """The exact nw_05 bug: attempt 1 fails, regen fails too, second regen is correct."""
    calls = {"execute": [], "regenerate": []}

    def execute_once(cypher: str):
        calls["execute"].append(cypher)
        if cypher == "GOOD":
            return [{"month": "1997-01", "orderCount": 10}]
        raise RuntimeError(f"boom: {cypher}")

    def regenerate(prev: str, err: str):
        calls["regenerate"].append((prev, err))
        if prev == "BAD_1":
            return "BAD_2"
        if prev == "BAD_2":
            return "GOOD"
        return None

    executor = StructuredCypherExecutor(max_attempts=2)
    result = executor.run(
        initial_cypher="BAD_1",
        question="Show monthly order count in 1997.",
        schema="",
        limit=10,
        execute_once=execute_once,
        regenerate=regenerate,
        sql_issue=_no_issue,
        repair=_identity_repair,
    )

    assert result.error is None
    assert result.rows == [{"month": "1997-01", "orderCount": 10}]
    assert result.cypher == "GOOD"
    # BAD_1 executed, failed; BAD_2 executed, failed; GOOD executed, succeeded.
    assert calls["execute"] == ["BAD_1", "BAD_2", "GOOD"]


def test_first_attempt_success_short_circuits():
    executor = StructuredCypherExecutor(max_attempts=3)
    result = executor.run(
        initial_cypher="GOOD",
        question="q",
        schema="",
        limit=10,
        execute_once=lambda c: [{"ok": 1}],
        regenerate=lambda c, e: pytest.fail("regenerate should not be called"),
        sql_issue=_no_issue,
        repair=_identity_repair,
    )
    assert result.error is None
    assert result.rows == [{"ok": 1}]
    assert result.attempts == 1


def test_exhausts_budget_and_returns_last_error_when_regeneration_keeps_failing():
    """
    regenerate() is capped at max_attempts LLM calls. Once that budget is spent, the
    executor still spends exactly ONE grace execution on whatever candidate the final
    regenerate() call produced (that LLM cost is already sunk) — then gives up for
    good without calling regenerate() again. So total executions = max_attempts + 1,
    but regenerate() itself is called at most max_attempts times.
    """
    regenerate_calls = []

    def execute_once(cypher: str):
        raise RuntimeError(f"fail: {cypher}")

    def regenerate(prev: str, err: str):
        regenerate_calls.append(prev)
        return prev + "_v2"  # always produces a new distinct (still-broken) cypher

    executor = StructuredCypherExecutor(max_attempts=2)
    result = executor.run(
        initial_cypher="BAD",
        question="q",
        schema="",
        limit=10,
        execute_once=execute_once,
        regenerate=regenerate,
        sql_issue=_no_issue,
        repair=_identity_repair,
    )
    assert result.rows == []
    assert result.error is not None
    assert result.attempts == 3  # max_attempts executions + 1 grace execution
    assert len(regenerate_calls) == 2  # regenerate() itself stays capped at max_attempts


def test_empty_cypher_short_circuits_without_executing():
    executor = StructuredCypherExecutor(max_attempts=3)
    result = executor.run(
        initial_cypher="   ",
        question="q",
        schema="",
        limit=10,
        execute_once=lambda c: pytest.fail("execute_once should not be called"),
        regenerate=lambda c, e: pytest.fail("regenerate should not be called"),
        sql_issue=_no_issue,
        repair=_identity_repair,
    )
    assert result.rows == []
    assert result.error == "Empty Cypher."
    assert result.attempts == 0


def test_free_static_repair_before_execution_does_not_consume_an_attempt():
    """A pre-check `sql_issue` repaired for free (no LLM regen) shouldn't burn an execution slot."""
    execute_calls = []

    def sql_issue(cypher: str):
        return "bad syntax" if cypher == "RAW" else None

    def repair(cypher: str) -> str:
        return "FIXED" if cypher == "RAW" else cypher

    def execute_once(cypher: str):
        execute_calls.append(cypher)
        return [{"ok": cypher}]

    executor = StructuredCypherExecutor(max_attempts=1)
    result = executor.run(
        initial_cypher="RAW",
        question="q",
        schema="",
        limit=10,
        execute_once=execute_once,
        regenerate=lambda c, e: pytest.fail("regenerate should not be called"),
        sql_issue=sql_issue,
        repair=repair,
    )
    assert result.error is None
    assert result.rows == [{"ok": "FIXED"}]
    assert execute_calls == ["FIXED"]
    assert result.attempts == 1
