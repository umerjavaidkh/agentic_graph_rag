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
    ) -> ExecuteResult:
        """
        Parameters are injected to keep this executor independent of Neo4j driver and LLM provider.

        - execute_once(cypher) -> rows or raises
        - regenerate(previous_cypher, execution_error) -> new cypher or None
        - sql_issue(cypher) -> message if known-invalid pattern, else None
        """
        cypher = (initial_cypher or "").strip()
        last_err: Optional[str] = None

        # Attempt loop: pre-check issues, then execute, then repair/regenerate.
        for attempt in range(1, self.max_attempts + 1):
            if not cypher:
                return ExecuteResult(rows=[], cypher="", error="Empty Cypher.", attempts=attempt)

            issue = sql_issue(cypher)
            if issue:
                fixed = regenerate(cypher, issue)
                if fixed and fixed.strip() != cypher.strip():
                    cypher = fixed.strip()
                    continue

            try:
                rows = execute_once(cypher)
                return ExecuteResult(rows=rows, cypher=cypher, error=None, attempts=attempt)
            except Exception as exc:
                last_err = str(exc)
                fixed = regenerate(cypher, last_err)
                if fixed and fixed.strip() != cypher.strip():
                    cypher = fixed.strip()
                    continue
                break

        return ExecuteResult(rows=[], cypher=cypher, error=last_err, attempts=self.max_attempts)

