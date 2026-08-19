"""Single-query Text-to-Cypher execution with repair and retry."""
from __future__ import annotations

from typing import Callable, Optional

from neo4j import Driver

from ....shared.auth.roles import UserContext
from ....shared.config.settings import (
    STRUCTURED_FALLBACK_MODEL,
    STRUCTURED_CYPHER_MAX_ATTEMPTS,
    STRUCTURED_CYPHER_SQL_LLM_RETRIES,
    STRUCTURED_EMPTY_RESULT_LLM_RETRIES,
    STRUCTURED_TEXT2CYPHER_LONG_MAX_TOKENS,
    STRUCTURED_TEXT2CYPHER_LONG_QUERY_CHARS,
    STRUCTURED_TEXT2CYPHER_MAX_TOKENS,
)
from ....shared.telemetry.context import TelemetryEvent, get_telemetry
from ..executor import StructuredCypherExecutor
from ..formatting.chunks import rows_to_chunks
from ..neo4j_sanitize import sanitize_row
from ..schema.provider import SchemaProvider
from .generator import CypherGenerator, regenerate_for_issue
from .repair import fix_relationship_directions, normalize_generated_cypher
VERIFY_PROMPT = """A question was asked of a graph database, and this Cypher was run to answer it.

SCHEMA:
{schema}

QUESTION: {question}

CYPHER: {cypher}

Does the property being aggregated actually measure what the question asked
for? Check the SCHEMA above before deciding anything is missing -- if the
property exists and holds the right kind of quantity, the answer is OK. A weight is not a cost. A delivery duration is not a supplier lead time.
An id is not a name. A timestamp is not a salary. Revenue IS legitimately the
sum of a price, and a duration IS legitimately the gap between two timestamps
-- derived metrics are fine when the underlying quantity is the right kind of
thing.

Answer with one line and nothing else:
  OK
or
  MEASURES_SOMETHING_ELSE: <the thing the question asked for that this data does not hold>
"""

from .validator import (
    EMPTY_RESULT_HINTS,
    no_such_data_subject,
    sql_cypher_issue,
    unknown_label_issue,
    unknown_property_issue,
)


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
        self._executor = executor or StructuredCypherExecutor(
            max_attempts=max(1, STRUCTURED_CYPHER_MAX_ATTEMPTS)
        )

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

        # The generator's one way to say "this graph does not hold that".
        # Without it every path ends in "produce some Cypher that runs", and a
        # question about absent data gets answered with whatever property does
        # return numbers -- observed as an average of created_at timestamps
        # reported to the user as an average salary.
        absent = no_such_data_subject(cypher)
        if absent is not None:
            return [{
                "id": "no_such_data",
                "title": "Not in this dataset",
                "text": (
                    f"The connected data does not contain {absent}. "
                    f"State that it is not available; do not estimate it or "
                    f"substitute a different measure."
                ),
                "score": 0.0,
                "related": [],
            }]

        repair_fn = lambda c: normalize_generated_cypher(c, schema)  # noqa: E731
        cypher = repair_fn(cypher)

        known_labels = self._schema.known_labels()
        known_props = self._schema.known_properties()
        def _issue(c: str) -> Optional[str]:  # noqa: E306
            return (
                sql_cypher_issue(c)
                or unknown_label_issue(c, known_labels)
                or unknown_property_issue(c, known_props)
            )

        llm_sql_retries = 0
        for _ in range(max(1, STRUCTURED_CYPHER_SQL_LLM_RETRIES) + 1):
            issue = _issue(cypher)
            if not issue:
                break
            repaired = repair_fn(cypher)
            if repaired.strip() != cypher.strip() and not _issue(repaired):
                cypher = repaired
                continue
            if llm_sql_retries >= STRUCTURED_CYPHER_SQL_LLM_RETRIES:
                break
            regen = regenerate_for_issue(self._cypher, query, schema, limit, cypher, issue)
            llm_sql_retries += 1
            if regen.strip() == cypher.strip():
                break
            cypher = repair_fn(regen)

        def _execute_once(c: str) -> list[dict]:
            with self._driver.session() as session:
                result = session.run(c, tenant_id=user_context.tenant_id)
                return [sanitize_row(r.data()) for r in result]

        def _regenerate(prev: str, err: str) -> Optional[str]:
            repaired = repair_fn(prev)
            if repaired.strip() != prev.strip() and not _issue(repaired):
                return repaired
            # Escalate. The first attempt already failed with this model, and
            # the errors that survive a repair are reasoning mistakes -- a
            # join that leaves the entity it was meant to filter, an average
            # over rows already collapsed -- which the same model reproduces.
            return self._cypher.generate(
                query,
                schema,
                limit,
                previous_cypher=prev,
                execution_error=err,
                model=STRUCTURED_FALLBACK_MODEL,
            )

        exec_res = self._executor.run(
            initial_cypher=cypher,
            question=query,
            schema=schema,
            limit=limit,
            execute_once=_execute_once,
            regenerate=_regenerate,
            sql_issue=_issue,
            repair=repair_fn,
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
                    corrected, query, schema=schema, limit=limit, repair_fn=repair_fn,
                    user_context=user_context,
                )
                if not err2 and rows2:
                    rows = rows2
                    cypher = cypher2
        if not rows:
            for _attempt, retry_msg in enumerate(
                EMPTY_RESULT_HINTS[: max(0, STRUCTURED_EMPTY_RESULT_LLM_RETRIES)],
                start=1,
            ):
                fixed = self._cypher.generate(
                    query,
                    schema,
                    limit,
                    previous_cypher=cypher,
                    execution_error=retry_msg,
                    model=STRUCTURED_FALLBACK_MODEL,
                )
                if not fixed or fixed.strip() == cypher.strip():
                    continue
                fixed = repair_fn(fixed)
                rows2, cypher2, err2 = self._execute_cypher_rows(
                    fixed, query, schema=schema, limit=limit, repair_fn=repair_fn,
                    user_context=user_context,
                )
                if err2:
                    continue
                cypher = cypher2
                if rows2:
                    rows = rows2
                    break
        substitution = self._substitution_check(query, cypher, rows, schema)
        if substitution is not None:
            return [{
                "id": "no_such_data",
                "title": "Not in this dataset",
                "text": (
                    f"The connected data does not contain {substitution}. "
                    f"State that it is not available; do not estimate it or "
                    f"substitute a different measure."
                ),
                "score": 0.0,
                "related": [],
            }]
        return rows_to_chunks(rows, cypher)

    def _substitution_check(
        self, query: str, cypher: str, rows: list[dict], schema: Optional[str]
    ) -> Optional[str]:
        """Catch a query that answers a DIFFERENT question than the one asked.

        Every fabrication observed took the same shape: one row, one number,
        aggregated from whatever numeric property was to hand. Cost of goods
        sold came back as the sum of product weight in grams; supplier lead
        time as the average purchase-to-delivery gap; an average salary as
        avg(created_at). All of them execute cleanly and return a plausible
        figure, so no error path and no empty-result retry can see them --
        and a larger model makes the same mistake, just with a different
        column, so this is not something better generation fixes.

        Checked here rather than in the prompt because three rounds of prompt
        instruction did not hold. Scoped to single-row aggregates so ordinary
        listing and ranking queries never pay for it.
        """
        if len(rows) != 1 or not schema:
            return None
        values = list(rows[0].values())
        if len(values) != 1 or not isinstance(values[0], (int, float)) or isinstance(values[0], bool):
            return None
        try:
            verdict = self._cypher.ask_raw(
                VERIFY_PROMPT.format(schema=schema, question=query, cypher=cypher),
                model=STRUCTURED_FALLBACK_MODEL,
            )
        except Exception:
            return None  # a failed check must never block a real answer
        text = (verdict or "").strip()
        if text.upper().startswith("MEASURES_SOMETHING_ELSE"):
            subject = text.split(":", 1)[-1].strip() if ":" in text else "that figure"
            return subject or "that figure"
        return None

    def _execute_cypher_rows(
        self,
        cypher: str,
        query: str,
        *,
        schema: Optional[str],
        limit: int,
        repair_fn: Callable[[str], str],
        user_context: UserContext,
    ) -> tuple[list[dict], str, Optional[str]]:
        last_err: Optional[str] = None
        for attempt in range(2):
            try:
                with self._driver.session() as session:
                    result = session.run(cypher, tenant_id=user_context.tenant_id)
                    return [sanitize_row(r.data()) for r in result], cypher, None
            except Exception as e:
                last_err = str(e)
                repaired = repair_fn(cypher)
                if attempt == 0 and repaired.strip() != cypher.strip():
                    cypher = repaired
                    continue
                if attempt == 0 and schema:
                    fixed = self._cypher.generate(
                        query,
                        schema,
                        limit,
                        previous_cypher=cypher,
                        execution_error=last_err,
                    )
                    if fixed and fixed.strip() != cypher.strip():
                        cypher = repair_fn(fixed)
                        continue
                break
        return [], cypher, last_err
