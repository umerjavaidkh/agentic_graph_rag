"""Single-query Text-to-Cypher execution with repair and retry."""
from __future__ import annotations

from typing import Callable, Optional

from neo4j import Driver

from ....auth.roles import UserContext
from ....config.settings import (
    STRUCTURED_TEXT2CYPHER_LONG_MAX_TOKENS,
    STRUCTURED_TEXT2CYPHER_LONG_QUERY_CHARS,
    STRUCTURED_TEXT2CYPHER_MAX_TOKENS,
)
from ....telemetry.context import TelemetryEvent, get_telemetry
from ..executor import StructuredCypherExecutor
from ..formatting.chunks import rows_to_chunks
from ..neo4j_sanitize import sanitize_row
from ..schema.provider import SchemaProvider
from .generator import CypherGenerator, regenerate_for_issue
from .repair import fix_relationship_directions, repair_schema_paths
from .validator import EMPTY_RESULT_HINTS, sql_cypher_issue


class Text2CypherPipeline:
    def __init__(
        self,
        driver: Driver,
        schema: SchemaProvider,
        cypher: CypherGenerator,
        *,
        can_query: Callable[[str], bool],
        executor: Optional[StructuredCypherExecutor] = None,
    ):
        self._driver = driver
        self._schema = schema
        self._cypher = cypher
        self._can_query = can_query
        self._executor = executor or StructuredCypherExecutor(max_attempts=5)

    def run(self, query: str, limit: int, user_context: UserContext) -> list[dict]:
        if not self._can_query(user_context.user_id):
            return [{
                "id": "access_denied",
                "title": "Access Denied",
                "text": f"User {user_context.user_id} does not have permission to query structured data.",
                "score": 0.0,
                "related": [],
            }]

        schema = self._schema.fetch()
        max_tokens = (
            STRUCTURED_TEXT2CYPHER_LONG_MAX_TOKENS
            if len(query) > STRUCTURED_TEXT2CYPHER_LONG_QUERY_CHARS
            else STRUCTURED_TEXT2CYPHER_MAX_TOKENS
        )
        cypher = self._cypher.generate(query, schema, limit, max_tokens=max_tokens)
        if not cypher:
            return []

        for _ in range(4):
            issue = sql_cypher_issue(cypher)
            if not issue:
                break
            cypher = regenerate_for_issue(self._cypher, query, schema, limit, cypher, issue)
        cypher = fix_relationship_directions(cypher, schema)
        cypher = repair_schema_paths(cypher, schema)

        def _execute_once(c: str) -> list[dict]:
            with self._driver.session() as session:
                result = session.run(c)
                return [sanitize_row(r.data()) for r in result]

        def _regenerate(prev: str, err: str) -> Optional[str]:
            return self._cypher.generate(
                query,
                schema,
                limit,
                previous_cypher=prev,
                execution_error=err,
            )

        exec_res = self._executor.run(
            initial_cypher=cypher,
            question=query,
            schema=schema,
            limit=limit,
            execute_once=_execute_once,
            regenerate=_regenerate,
            sql_issue=sql_cypher_issue,
        )
        tel = get_telemetry()
        if tel is not None:
            tel.add(TelemetryEvent(kind="structured_execute", meta={"attempts": exec_res.attempts}))
        rows, cypher, err = exec_res.rows, exec_res.cypher, exec_res.error
        if err:
            return [{
                "id": "error",
                "title": "Query Error",
                "text": f"Generated Cypher failed: {err}\nCypher: {cypher}",
                "score": 0.0,
                "related": [],
                "cypher": cypher,
            }]
        if not rows:
            corrected = fix_relationship_directions(cypher, schema)
            if corrected.strip() != cypher.strip():
                rows2, cypher2, err2 = self._execute_cypher_rows(
                    corrected, query, schema=schema, limit=limit
                )
                if not err2 and rows2:
                    rows = rows2
                    cypher = cypher2
        if not rows:
            for _attempt, retry_msg in enumerate(EMPTY_RESULT_HINTS, start=1):
                fixed = self._cypher.generate(
                    query,
                    schema,
                    limit,
                    previous_cypher=cypher,
                    execution_error=retry_msg,
                )
                if not fixed or fixed.strip() == cypher.strip():
                    continue
                fixed = fix_relationship_directions(fixed, schema)
                fixed = repair_schema_paths(fixed, schema)
                rows2, cypher2, err2 = self._execute_cypher_rows(
                    fixed, query, schema=schema, limit=limit
                )
                if err2:
                    continue
                cypher = cypher2
                if rows2:
                    rows = rows2
                    break
        return rows_to_chunks(rows, cypher)

    def _execute_cypher_rows(
        self,
        cypher: str,
        query: str,
        *,
        schema: Optional[str],
        limit: int,
    ) -> tuple[list[dict], str, Optional[str]]:
        last_err: Optional[str] = None
        for attempt in range(2):
            try:
                with self._driver.session() as session:
                    result = session.run(cypher)
                    return [sanitize_row(r.data()) for r in result], cypher, None
            except Exception as e:
                last_err = str(e)
                if attempt == 0 and schema:
                    fixed = self._cypher.generate(
                        query,
                        schema,
                        limit,
                        previous_cypher=cypher,
                        execution_error=last_err,
                    )
                    if fixed and fixed.strip() != cypher.strip():
                        cypher = fixed
                        continue
                break
        return [], cypher, last_err
