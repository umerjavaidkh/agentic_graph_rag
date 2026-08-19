from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional


@dataclass
class ExecuteResult:
    rows: list[dict]
    cypher: str
    error: Optional[str] = None
    attempts: int = 0


class StructuredCypherExecutor:
    """
    Goal-oriented Cypher execution with bounded retries.

    This is schema-driven and generic: it retries based on error classes, not dataset hacks.
    """

    def __init__(
        self,
        *,
        max_attempts: int = 5,
    ):
        self.max_attempts = max(1, int(max_attempts))

    def run(
        self,
        *,
        initial_cypher: str,
        question: str,
        schema: str,
        limit: int,
        execute_once: Callable[[str], list[dict]],
        regenerate: Callable[[str, str], Optional[str]],
        sql_issue: Callable[[str], Optional[str]],
        repair: Optional[Callable[[str], str]] = None,
    ) -> ExecuteResult:
        """
        Parameters are injected to keep this executor independent of Neo4j driver and LLM provider.

        - execute_once(cypher) -> rows or raises
        - regenerate(previous_cypher, execution_error) -> new cypher or None
        - sql_issue(cypher) -> message if known-invalid pattern, else None

        `attempts` counts actual execute_once() calls. Regeneration (an LLM call) is
        capped at `max_attempts` calls, matching the budget's original cost intent.
        But once that budget is spent, if the last regenerate() call already produced
        a genuinely different candidate, it is always given ONE final execution before
        giving up — that LLM call already happened (cost sunk), so discarding its
        result without even trying it against Neo4j (a cheap, local call) is pure
        waste. This grace execution never triggers another regenerate() call.
        """
        cypher = (initial_cypher or "").strip()
        last_err: Optional[str] = None
        executions = 0
        regenerations = 0
        grace_used = False

        while True:
            if not cypher:
                return ExecuteResult(rows=[], cypher="", error="Empty Cypher.", attempts=executions)

            issue = sql_issue(cypher)
            if issue:
                if repair:
                    repaired = repair(cypher)
                    if repaired.strip() != cypher.strip() and not sql_issue(repaired):
                        cypher = repaired.strip()
                        continue
                if regenerations >= self.max_attempts:
                    return ExecuteResult(rows=[], cypher=cypher, error=last_err or issue, attempts=executions)
                fixed = regenerate(cypher, issue)
                regenerations += 1
                if fixed and fixed.strip() != cypher.strip():
                    cypher = fixed.strip()
                    continue
                return ExecuteResult(rows=[], cypher=cypher, error=last_err or issue, attempts=executions)

            budget_spent = executions >= self.max_attempts
            if budget_spent and grace_used:
                return ExecuteResult(rows=[], cypher=cypher, error=last_err, attempts=executions)

            executions += 1
            if budget_spent:
                grace_used = True
            try:
                rows = execute_once(cypher)
                return ExecuteResult(rows=rows, cypher=cypher, error=None, attempts=executions)
            except Exception as exc:
                last_err = str(exc)
                if grace_used and budget_spent:
                    return ExecuteResult(rows=[], cypher=cypher, error=last_err, attempts=executions)
                if repair:
                    repaired = repair(cypher)
                    if repaired.strip() != cypher.strip():
                        cypher = repaired.strip()
                        continue
                if regenerations >= self.max_attempts:
                    return ExecuteResult(rows=[], cypher=cypher, error=last_err, attempts=executions)
                fixed = regenerate(cypher, last_err)
                regenerations += 1
                if fixed and fixed.strip() != cypher.strip():
                    cypher = fixed.strip()
                    continue
                return ExecuteResult(rows=[], cypher=cypher, error=last_err, attempts=executions)

