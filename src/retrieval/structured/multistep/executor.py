"""Execute a multistep Cypher plan sequentially."""
from __future__ import annotations

import json
import re
from typing import Any, Callable, Optional

from neo4j import Driver

from ....auth.roles import UserContext
from ....config.settings import STRUCTURED_MULTISTEP_STEP_ATTEMPTS
from ..cypher.generator import CypherGenerator, regenerate_for_issue
from ..cypher.repair import normalize_generated_cypher
from ..cypher.validator import dropped_year_filter_issue, sql_cypher_issue
from ..neo4j_sanitize import sanitize_row
from ..schema.provider import SchemaProvider
from .context import collect_values_from_ctx, find_param_names, normalize_row_keys
from .models import MultiStepPlan


class MultiStepExecutor:
    def __init__(
        self,
        driver: Driver,
        schema: SchemaProvider,
        cypher: CypherGenerator,
        *,
        can_query: Callable[[str], bool],
    ):
        self._driver = driver
        self._schema = schema
        self._cypher = cypher
        self._can_query = can_query

    def execute(
        self,
        plan: MultiStepPlan,
        user_context: UserContext,
        query: str = "",
    ) -> list[dict[str, Any]]:
        """
        Execute each step sequentially and return chunks for synthesis.

        No hardcoded business logic: each step is a Cypher query produced by the planner.
        """
        out: list[dict[str, Any]] = []
        if not self._can_query(user_context.user_id):
            return [{
                "id": "access_denied",
                "title": "Access Denied",
                "text": f"User {user_context.user_id} does not have permission to query structured data.",
                "score": 0.0,
                "related": [],
            }]

        ctx: dict[str, Any] = {"plan_reason": plan.reason, "final_hint": plan.final_answer_hint}
        schema = self._schema.fetch()
        repair_fn = lambda c: normalize_generated_cypher(c, schema)  # noqa: E731
        max_step_attempts = max(1, STRUCTURED_MULTISTEP_STEP_ATTEMPTS)

        with self._driver.session() as session:
            for _idx, step in enumerate(plan.steps, 1):
                cypher = repair_fn((step.cypher or "").strip())
                issue = sql_cypher_issue(cypher)
                if issue:
                    repaired = repair_fn(cypher)
                    if repaired.strip() != cypher.strip() and not sql_cypher_issue(repaired):
                        cypher = repaired
                    else:
                        cypher = repair_fn(
                            regenerate_for_issue(
                                self._cypher, step.purpose, schema, 10, cypher, issue
                            )
                        )
                filt_issue = dropped_year_filter_issue(cypher, query)
                if filt_issue:
                    regenerated = self._cypher.generate(
                        step.purpose, schema, 10, previous_cypher=cypher, execution_error=filt_issue
                    )
                    if regenerated and regenerated.strip():
                        cypher = repair_fn(regenerated.strip())
                rows: list[dict[str, Any]] = []

                def _build_params_and_run(c: str) -> list[dict[str, Any]]:
                    params: dict[str, Any] = {}
                    for k, v in ctx.items():
                        if isinstance(v, dict) and isinstance(v.get("rows"), list):
                            params[f"{k}_rows"] = v["rows"]
                    param_names = find_param_names(c)
                    if param_names:
                        for pn in param_names:
                            if pn in params:
                                continue
                            candidates = [pn, pn.rstrip("s")]
                            vals: list[Any] = []
                            for cand in candidates:
                                vals = collect_values_from_ctx(ctx, cand)
                                if vals:
                                    break
                            if vals:
                                if re.search(rf"\bUNWIND\s+\${re.escape(pn)}\b", c, re.I):
                                    params[pn] = vals
                                else:
                                    params[pn] = vals[0] if len(vals) == 1 else vals
                    if param_names and not params:
                        raise RuntimeError(
                            f"Step requires parameters {param_names} but no values were found from prior steps."
                        )

                    scalar_like = any(re.search(rf"\${re.escape(pn)}\b", c) for pn in params)
                    list_param_names = [k for k, v in params.items() if isinstance(v, list)]
                    if list_param_names and scalar_like and not re.search(r"\bUNWIND\b", c, re.I):
                        merged: list[dict] = []
                        first = list_param_names[0]
                        for v in params[first]:
                            p2 = {**params, first: v}
                            merged.extend([sanitize_row(r.data()) for r in session.run(c, p2)])
                        return merged
                    return [sanitize_row(r.data()) for r in session.run(c, params)]

                last_err: Optional[str] = None
                for _attempt in range(max_step_attempts):
                    try:
                        rows = _build_params_and_run(cypher)
                        last_err = None
                        break
                    except Exception as exc:
                        last_err = str(exc)
                        repaired = repair_fn(cypher)
                        if repaired.strip() != cypher.strip():
                            cypher = repaired
                            continue
                        regen = self._cypher.generate(
                            step.purpose, schema, 10, previous_cypher=cypher, execution_error=last_err
                        )
                        if not regen or regen.strip() == cypher.strip():
                            break
                        cypher = repair_fn(regen.strip())
                if last_err is not None:
                    out.append({
                        "id": f"{step.id}_error",
                        "title": f"{step.id} error",
                        "text": f"Step failed: {last_err}\nCypher: {cypher}",
                        "score": 0.0,
                        "related": [],
                        "cypher": cypher,
                    })
                    return out

                norm_rows = [normalize_row_keys(r) for r in rows if isinstance(r, dict)]
                ctx[step.id] = {"rows": norm_rows}

                preview = norm_rows[:15]
                out.append({
                    "id": step.id,
                    "title": f"{step.id}: {step.purpose}",
                    "text": json.dumps(preview, indent=2, ensure_ascii=False),
                    "raw": norm_rows,
                    "score": 1.0,
                    "related": [],
                    "cypher": cypher,
                })
        return out
