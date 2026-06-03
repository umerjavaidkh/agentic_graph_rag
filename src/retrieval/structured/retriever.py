"""
retrieval/structured/retriever.py — Structured Neo4j retriever.

Schema introspection + LLM Text-to-Cypher + execute/repair.
"""

import json
import re
from typing import Any, Optional

from pydantic import BaseModel, Field, ValidationError, field_validator

from ...auth.rbac_setup import GraphRBAC
from ...graph.driver import get_neo4j_driver
from ...auth.roles import DEFAULT_PUBLIC_CONTEXT, UserContext
from ...config.settings import (
    CHAT_MODEL,
    STRUCTURED_ALWAYS_MULTISTEP_PLAN,
    STRUCTURED_MODEL,
    STRUCTURED_PLAN_MAX_TOKENS,
    STRUCTURED_PLAN_QUERY_LARGE_CHARS,
    STRUCTURED_PLAN_QUERY_MEDIUM_CHARS,
    STRUCTURED_PLAN_SCHEMA_LARGE_CHARS,
    STRUCTURED_PLAN_SCHEMA_MEDIUM_CHARS,
    STRUCTURED_PLAN_TOKENS_LARGE,
    STRUCTURED_PLAN_TOKENS_MEDIUM,
    STRUCTURED_PLAN_TOKENS_SMALL,
    STRUCTURED_TEXT2CYPHER_LONG_MAX_TOKENS,
    STRUCTURED_TEXT2CYPHER_LONG_QUERY_CHARS,
    STRUCTURED_TEXT2CYPHER_MAX_TOKENS,
    MODEL_PROVIDER,
    NEO4J_PASSWORD,
    NEO4J_URI,
    NEO4J_USER,
    OPENAI_API_KEY,
)
from ...config.prompts import load_prompt
from ...model_providers.factory import get_model_provider
from .neo4j_sanitize import sanitize_row
from .query_intent import analytics_result_limit, likely_needs_multistep_plan
from ...conversation.clarification import format_clarification_answer
from .executor import StructuredCypherExecutor
from ...telemetry.context import TelemetryEvent, get_telemetry

provider = get_model_provider(MODEL_PROVIDER, OPENAI_API_KEY)
# Use stronger model for structured planning + Text-to-Cypher if configured.
LLM_MODEL = STRUCTURED_MODEL or CHAT_MODEL

# Schema-agnostic hints when a query executes but returns no rows.
_EMPTY_RESULT_HINTS = (
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
_SQL_CYPHER_ISSUES: list[tuple[str, str]] = [
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
]


def _sql_cypher_issue(cypher: str) -> Optional[str]:
    for pattern, msg in _SQL_CYPHER_ISSUES:
        if re.search(pattern, cypher, re.I | re.S):
            return msg
    return None


def _regenerate_cypher_for_issue(
    retriever: "StructuredRetriever",
    query: str,
    schema: str,
    limit: int,
    cypher: str,
    issue: str,
) -> str:
    fixed = retriever._generate_cypher(
        query,
        schema,
        limit,
        previous_cypher=cypher,
        execution_error=issue,
    )
    return fixed.strip() if fixed else cypher


class _MultiStepStep(BaseModel):
    id: str
    purpose: str
    cypher: str
    expects: str = Field(default="rows")  # rows | scalar

    @field_validator("expects")
    @classmethod
    def _expects_allowed(cls, v: str) -> str:
        if v not in ("rows", "scalar"):
            return "rows"
        return v


class _MultiStepPlan(BaseModel):
    needs_multistep: bool = False
    reason: str = ""
    steps: list[_MultiStepStep] = Field(default_factory=list)
    final_answer_hint: str = ""


def _extract_json(text: str) -> str:
    """Best-effort extraction of a JSON object from an LLM response."""
    t = (text or "").strip()
    if not t:
        return "{}"
    start = t.find("{")
    end = t.rfind("}")
    if start >= 0 and end > start:
        return t[start : end + 1]
    return t


def _multistep_plan_token_budget(query: str, schema: str) -> int:
    """
    Token budget for the multistep planner.

    We keep this dynamic because long questions require longer JSON plans.
    Allows override via STRUCTURED_PLAN_MAX_TOKENS.
    """
    if STRUCTURED_PLAN_MAX_TOKENS.isdigit():
        return max(300, min(int(STRUCTURED_PLAN_MAX_TOKENS), 4000))

    q_len = len((query or "").strip())
    s_len = len((schema or "").strip())
    # Heuristic: longer schema/question → more room for valid JSON + Cypher strings.
    if q_len > STRUCTURED_PLAN_QUERY_LARGE_CHARS or s_len > STRUCTURED_PLAN_SCHEMA_LARGE_CHARS:
        return STRUCTURED_PLAN_TOKENS_LARGE
    if q_len > STRUCTURED_PLAN_QUERY_MEDIUM_CHARS or s_len > STRUCTURED_PLAN_SCHEMA_MEDIUM_CHARS:
        return STRUCTURED_PLAN_TOKENS_MEDIUM
    return STRUCTURED_PLAN_TOKENS_SMALL


_PARAM_RE = re.compile(r"\$(\w+)")


def _find_param_names(cypher: str) -> list[str]:
    return sorted(set(_PARAM_RE.findall(cypher or "")))


def _collect_values_from_ctx(ctx: dict[str, Any], key: str) -> list[Any]:
    """
    Collect values for a key from previous step rows.

    ctx holds {step_id: {"rows": [...]}} entries.
    """
    values: list[Any] = []
    for v in ctx.values():
        if not isinstance(v, dict):
            continue
        rows = v.get("rows")
        if not isinstance(rows, list):
            continue
        for r in rows:
            if isinstance(r, dict) and key in r and r[key] is not None:
                values.append(r[key])
    # de-dupe while preserving order
    return list(dict.fromkeys(values))


def _normalize_row_keys(row: dict[str, Any]) -> dict[str, Any]:
    """Normalize keys so they can be referenced as row.<key> in later UNWIND steps."""
    out: dict[str, Any] = {}
    for k, v in (row or {}).items():
        nk = str(k).replace(".", "_").strip()
        out[nk] = v
    return out


def _parse_schema_relationships(schema: str) -> list[tuple[str, str, str]]:
    """Extract (from_label, rel_type, to_label) triples from schema text."""
    patterns: list[tuple[str, str, str]] = []
    for line in schema.splitlines():
        m = re.search(r"\(:?(\w+)\)-\[:(\w+)\]->\(:?(\w+)\)", line)
        if m:
            patterns.append((m.group(1), m.group(2), m.group(3)))
    return patterns


def _fix_relationship_directions(cypher: str, schema: str) -> str:
    """
    Flip relationship arrows that contradict schema directions.

    Schema (:A)-[:R]->(:B) allows (a:A)-[:R]->(b:B) or (b:B)<-[:R]-(a:A).
    Only flip explicit wrong traversals: (b:B)-[:R]->(a:A) or (a:A)<-[:R]-(b:B).
    """
    fixed = cypher
    for from_label, rel, to_label in _parse_schema_relationships(schema):
        wrong_fwd = re.compile(
            rf"\((\w+)\s*:\s*{re.escape(to_label)}\)\s*-\[:{re.escape(rel)}\]\s*->\s*\((\w+)\s*:\s*{re.escape(from_label)}\)",
            re.IGNORECASE,
        )
        fixed = wrong_fwd.sub(rf"(\2:{from_label})-[:{rel}]->(\1:{to_label})", fixed)

        # Wrong reverse: (From)<-[:REL]-(To) means To->From, opposite of schema.
        wrong_rev = re.compile(
            rf"\((\w+)\s*:\s*{re.escape(from_label)}\)\s*<-\[:{re.escape(rel)}\]-\s*\((\w+)\s*:\s*{re.escape(to_label)}\)",
            re.IGNORECASE,
        )
        fixed = wrong_rev.sub(rf"(\1:{from_label})-[:{rel}]->(\2:{to_label})", fixed)
    return fixed


def _shortest_schema_path(
    from_label: str,
    to_label: str,
    edges: list[tuple[str, str, str]],
    max_depth: int = 4,
) -> list[tuple[str, str, str]] | None:
    """BFS over schema edges (forward only). Returns hops as (from, rel, to)."""
    if from_label == to_label:
        return []
    from collections import deque

    adj: dict[str, list[tuple[str, str]]] = {}
    for a, r, b in edges:
        adj.setdefault(a, []).append((r, b))

    queue: deque[tuple[str, list[tuple[str, str, str]]]] = deque([(from_label, [])])
    visited = {from_label}
    while queue:
        node, path = queue.popleft()
        if len(path) >= max_depth:
            continue
        for rel, nxt in adj.get(node, []):
            step = (node, rel, nxt)
            new_path = path + [step]
            if nxt == to_label:
                return new_path
            if nxt not in visited:
                visited.add(nxt)
                queue.append((nxt, new_path))
    return None


def _parse_labeled_path(path: str) -> tuple[list[tuple[str, str]], list[tuple[str, str, str, str, str]]] | None:
    """Parse a single path pattern into ordered nodes and directed edges."""
    path = path.strip()
    m0 = re.match(r"\((\w+)\s*:\s*(\w+)\)", path)
    if not m0:
        return None

    nodes: list[tuple[str, str]] = [(m0.group(1), m0.group(2))]
    edges: list[tuple[str, str, str, str, str]] = []
    pos = m0.end()

    while pos < len(path):
        fwd = re.match(r"-\[:(\w+)\]->\((\w+)\s*:\s*(\w+)\)", path[pos:])
        if fwd:
            fv, fl = nodes[-1]
            tv, tl = fwd.group(2), fwd.group(3)
            edges.append((fv, fl, fwd.group(1), tv, tl))
            nodes.append((tv, tl))
            pos += fwd.end()
            continue

        rev = re.match(r"<-?\[:(\w+)\]-\((\w+)\s*:\s*(\w+)\)", path[pos:])
        if rev:
            lv, ll = nodes[-1]
            sv, sl = rev.group(2), rev.group(3)
            edges.append((sv, sl, rev.group(1), lv, ll))
            nodes.append((sv, sl))
            pos += rev.end()
            continue
        break

    return nodes, edges


def _repair_hanging_reverse_paths(path: str, schema_set: set[tuple[str, str, str]]) -> str:
    """
    Fix (mid)-[:R1]->(end)<-[:R2]-(start) when R2 should connect start->mid, not start->end.
    """
    pattern = re.compile(
        r"\((\w+)\s*:\s*(\w+)\)\s*-\[:(\w+)\]->\((\w+)\s*:\s*(\w+)\)\s*"
        r"<-?\[:(\w+)\]-\((\w+)\s*:\s*(\w+)\)",
        re.IGNORECASE,
    )

    def repl(m: re.Match[str]) -> str:
        mid_v, mid_l, r1, end_v, end_l, r2, start_v, start_l = m.groups()
        if (start_l, r2, end_l) in schema_set:
            return m.group(0)
        if (start_l, r2, mid_l) in schema_set:
            return f"({start_v}:{start_l})-[:{r2}]->({mid_v}:{mid_l})-[:{r1}]->({end_v}:{end_l})"
        return m.group(0)

    prev = None
    fixed = path
    while prev != fixed:
        prev = fixed
        fixed = pattern.sub(repl, fixed)
    return fixed


def _repair_invalid_edges(path: str, schema_edges: list[tuple[str, str, str]]) -> str:
    """Replace schema-invalid hops with shortest valid forward paths."""
    schema_set = set(schema_edges)
    parsed = _parse_labeled_path(path)
    if not parsed:
        return path

    nodes, edge_list = parsed
    if not edge_list:
        return path

    label_vars: dict[str, str] = {}
    used_vars = {v for v, _ in nodes}
    for var, label in nodes:
        label_vars.setdefault(label, var)

    def _var_for_label(label: str) -> str:
        if label in label_vars:
            return label_vars[label]
        base = label[0].lower()
        candidate = base
        n = 1
        while candidate in used_vars:
            candidate = f"{base}{n}"
            n += 1
        used_vars.add(candidate)
        label_vars[label] = candidate
        return candidate

    segments = [f"({nodes[0][0]}:{nodes[0][1]})"]
    for from_var, from_label, rel, to_var, to_label in edge_list:
        if (from_label, rel, to_label) in schema_set:
            segments.append(f"-[:{rel}]->({to_var}:{to_label})")
            label_vars.setdefault(to_label, to_var)
            continue

        sp = _shortest_schema_path(from_label, to_label, schema_edges)
        if not sp:
            segments.append(f"-[:{rel}]->({to_var}:{to_label})")
            continue

        prev_var = from_var
        for i, (_a, step_rel, step_to) in enumerate(sp):
            if i == len(sp) - 1:
                step_var = to_var
            else:
                step_var = _var_for_label(step_to)
            segments.append(f"-[:{step_rel}]->({step_var}:{step_to})")
            label_vars[step_to] = step_var
            prev_var = step_var

    return "".join(segments)


def _repair_schema_paths(cypher: str, schema: str) -> str:
    """Repair invalid relationship hops in MATCH clauses using schema paths."""
    schema_edges = _parse_schema_relationships(schema)
    if not schema_edges:
        return cypher

    schema_set = set(schema_edges)
    match_m = re.search(r"\bMATCH\s+(.+?)(?=\s+(?:WHERE|WITH|RETURN)\b)", cypher, re.I | re.S)
    if not match_m:
        return cypher

    parts = [p.strip() for p in match_m.group(1).split(",")]
    repaired: list[str] = []
    for part in parts:
        part = _repair_hanging_reverse_paths(part, schema_set)
        part = _repair_invalid_edges(part, schema_edges)
        repaired.append(part)

    new_match = "MATCH " + ", ".join(repaired)
    return cypher[: match_m.start()] + new_match + cypher[match_m.end() :]


class StructuredRetriever:
    def __init__(
        self,
        uri: str = NEO4J_URI,
        user: str = NEO4J_USER,
        password: str = NEO4J_PASSWORD,
        user_context: Optional[UserContext] = None,
    ):
        self.driver = get_neo4j_driver(uri, user, password)
        self.user_context = user_context or DEFAULT_PUBLIC_CONTEXT
        self.rbac = GraphRBAC(uri, user, password, driver=self.driver)
        self._schema_cache: Optional[str] = None
        self._rbac_cache: dict[tuple[str, str], bool] = {}
        self._executor = StructuredCypherExecutor(max_attempts=5)

    def close(self) -> None:
        """No-op: driver is process-wide; use close_neo4j_driver() on shutdown."""

    def retrieve(
        self,
        query: str,
        limit: int = 5,
        user_context: Optional[UserContext] = None,
    ) -> dict:
        ctx = user_context or self.user_context
        clarification = self._needs_clarification(query)
        if clarification:
            return clarification
        # Multistep LLM planner only for nested analytics (regex gate) unless forced via env.
        schema = self._fetch_schema()
        if STRUCTURED_ALWAYS_MULTISTEP_PLAN or likely_needs_multistep_plan(query):
            plan = self._plan_multistep(query, schema)
            if plan and plan.needs_multistep and plan.steps:
                chunks = self._execute_multistep(plan, user_context=ctx)
                return self._format_response(query, chunks, strategy="multistep")

        chunks = self._text2cypher(query, limit, user_context=ctx)
        return self._format_response(query, chunks, strategy="text2cypher")

    def _plan_multistep(self, query: str, schema: str) -> Optional[_MultiStepPlan]:
        """
        Ask the LLM whether this query needs multi-step execution.

        This is intentionally schema-driven and avoids hardcoding specific query phrases.
        """
        try:
            prompt = load_prompt("structured_multistep_plan", schema=schema, query=query)
        except Exception:
            return None
        try:
            resp = provider.chat_completion(
                model=LLM_MODEL,
                temperature=0,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=_multistep_plan_token_budget(query, schema),
            )
            raw = (resp.choices[0].message.content or "").strip()
            js = _extract_json(raw)
            data = json.loads(js)
            plan = _MultiStepPlan.model_validate(data)
            # Guardrails: keep plans small and safe.
            if len(plan.steps) > 4:
                plan.steps = plan.steps[:4]
            # If LLM didn't provide valid cypher, bail out.
            if plan.needs_multistep and any((not s.cypher or len(s.cypher) < 10) for s in plan.steps):
                return None
            return plan
        except (json.JSONDecodeError, ValidationError):
            return None
        except Exception:
            return None

    def _execute_multistep(self, plan: _MultiStepPlan, user_context: UserContext) -> list[dict[str, Any]]:
        """
        Execute each step sequentially and return chunks for synthesis.

        No hardcoded business logic: each step is a Cypher query produced by the planner.
        """
        out: list[dict[str, Any]] = []
        # RBAC is already checked inside _text2cypher; reuse same rule here.
        if not self._can_query_structured(user_context.user_id):
            return [{
                "id": "access_denied",
                "title": "Access Denied",
                "text": f"User {user_context.user_id} does not have permission to query structured data.",
                "score": 0.0,
                "related": [],
            }]

        # Simple runtime context for later steps (optional).
        ctx: dict[str, Any] = {"plan_reason": plan.reason, "final_hint": plan.final_answer_hint}

        with self.driver.session() as session:
            for idx, step in enumerate(plan.steps, 1):
                cypher = (step.cypher or "").strip()
                issue = _sql_cypher_issue(cypher)
                if issue:
                    # regenerate once if the step contains known-invalid patterns
                    schema = self._fetch_schema()
                    cypher = _regenerate_cypher_for_issue(self, step.purpose, schema, 10, cypher, issue)
                rows: list[dict[str, Any]] = []
                norm_rows: list[dict[str, Any]] = []
                try:
                    params: dict[str, Any] = {}

                    # Always pass prior step rows as `$<step_id>_rows` for UNWIND-based chaining.
                    for k, v in ctx.items():
                        if isinstance(v, dict) and isinstance(v.get("rows"), list):
                            params[f"{k}_rows"] = v["rows"]
                    param_names = _find_param_names(cypher)
                    if param_names:
                        for pn in param_names:
                            if pn in params:
                                continue
                            # Prefer plural → look for singular in prior rows too.
                            candidates = [pn, pn.rstrip("s")]
                            vals: list[Any] = []
                            for cand in candidates:
                                vals = _collect_values_from_ctx(ctx, cand)
                                if vals:
                                    break
                            if vals:
                                # If query uses UNWIND $x, pass list; else pass scalar if single.
                                if re.search(rf"\bUNWIND\s+\${re.escape(pn)}\b", cypher, re.I):
                                    params[pn] = vals
                                else:
                                    params[pn] = vals[0] if len(vals) == 1 else vals

                    if param_names and not params:
                        raise RuntimeError(f"Step requires parameters {param_names} but no values were found from prior steps.")

                    # If we have a list param but the query likely expects scalar (e.g. {id:$id}),
                    # run once per value and merge rows. This keeps it generic and avoids hardcoding.
                    scalar_like = any(re.search(rf"\${re.escape(pn)}\b", cypher) for pn in params)
                    list_param_names = [k for k, v in params.items() if isinstance(v, list)]
                    if list_param_names and scalar_like and not re.search(r"\bUNWIND\b", cypher, re.I):
                        merged: list[dict] = []
                        # Only iterate over the first list param; others (if any) stay fixed.
                        first = list_param_names[0]
                        for v in params[first]:
                            p2 = {**params, first: v}
                            merged.extend([sanitize_row(r.data()) for r in session.run(cypher, p2)])
                        rows = merged
                    else:
                        rows = [sanitize_row(r.data()) for r in session.run(cypher, params)]
                except Exception as exc:
                    out.append({
                        "id": f"{step.id}_error",
                        "title": f"{step.id} error",
                        "text": f"Step failed: {exc}\nCypher: {cypher}",
                        "score": 0.0,
                        "related": [],
                        "cypher": cypher,
                    })
                    return out

                # Normalize keys so later steps can reference row.country, row.customerID, etc.
                norm_rows = [_normalize_row_keys(r) for r in rows if isinstance(r, dict)]

                # Store step output in ctx (for chaining and debugging).
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

    def _needs_clarification(self, query: str) -> Optional[dict]:
        """
        Ask a follow-up when the metric is ambiguous.

        Example: "avg order price" could mean freight, computed order total, or unit price.
        """
        ql = (query or "").strip().lower()
        if not ql:
            return None

        # Only trigger when user asks for an average and uses vague "order price".
        if "order" in ql and "price" in ql and ("avg" in ql or "average" in ql):
            # If they already specified one of the interpretations, don't ask.
            if any(k in ql for k in ("freight", "shipping", "ship cost", "order total", "total", "unitprice", "unit price", "line item")):
                return None
            options = [
                {
                    "id": "order_total",
                    "label": "Order total (recommended)",
                    "detail": "Sum of line items per order: unitPrice × quantity × (1 - discount)",
                    "aliases": ["total", "order total", "line item total", "sum items", "items total"],
                },
                {
                    "id": "freight",
                    "label": "Freight / shipping cost",
                    "detail": "Use Order.freight (shipping cost) per order",
                    "aliases": ["freight", "shipping", "shipping cost", "ship cost"],
                },
                {
                    "id": "unit_price",
                    "label": "Average unit price",
                    "detail": "Average of line-item unitPrice (not the order total)",
                    "aliases": ["unit price", "unitprice", "item price"],
                },
            ]
            prompt = (
                "When you say **average order price**, which metric do you mean?\n\n"
                "Reply with 1, 2, or 3 (or the option name)."
            )
            answer = format_clarification_answer(prompt, options)
            return {
                "query": query,
                "strategy": "clarification",
                "mode": "needs_clarification",
                "original_question": query,
                "clarification_kind": "structured_order_price",
                "clarification_options": options,
                "chunks": [
                    {
                        "id": "clarification",
                        "title": "Clarification",
                        "text": answer,
                        "score": 1.0,
                        "related": [],
                    }
                ],
                "total_available": 1,
            }

        return None

    def get_schema(self) -> dict:
        schema = self._fetch_schema()
        return {
            "query": "schema",
            "chunks": [{"id": "schema", "title": "Graph Schema", "text": schema, "related": []}],
            "total_available": 1,
        }

    def _can_query_structured(self, user_id: str) -> bool:
        key = (user_id, "structured")
        if key not in self._rbac_cache:
            self._rbac_cache[key] = bool(self.rbac.can_query_knowledge_area(user_id, "structured"))
        return self._rbac_cache[key]

    def _text2cypher(self, query: str, limit: int, user_context: Optional[UserContext] = None) -> list[dict]:
        ctx = user_context or self.user_context
        if not self._can_query_structured(ctx.user_id):
            return [{
                "id": "access_denied",
                "title": "Access Denied",
                "text": f"User {ctx.user_id} does not have permission to query structured data.",
                "score": 0.0,
                "related": [],
            }]

        schema = self._fetch_schema()
        max_tokens = (
            STRUCTURED_TEXT2CYPHER_LONG_MAX_TOKENS
            if len(query) > STRUCTURED_TEXT2CYPHER_LONG_QUERY_CHARS
            else STRUCTURED_TEXT2CYPHER_MAX_TOKENS
        )
        cypher = self._generate_cypher(query, schema, limit, max_tokens=max_tokens)
        if not cypher:
            return []

        for _ in range(4):
            issue = _sql_cypher_issue(cypher)
            if not issue:
                break
            cypher = _regenerate_cypher_for_issue(self, query, schema, limit, cypher, issue)
        cypher = _fix_relationship_directions(cypher, schema)
        cypher = _repair_schema_paths(cypher, schema)

        def _execute_once(c: str) -> list[dict]:
            with self.driver.session() as session:
                result = session.run(c)
                return [sanitize_row(r.data()) for r in result]

        def _regenerate(prev: str, err: str) -> Optional[str]:
            return self._generate_cypher(
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
            sql_issue=_sql_cypher_issue,
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
            corrected = _fix_relationship_directions(cypher, schema)
            if corrected.strip() != cypher.strip():
                rows2, cypher2, err2 = self._execute_cypher_rows(
                    corrected, query, schema=schema, limit=limit
                )
                if not err2 and rows2:
                    rows = rows2
                    cypher = cypher2
        if not rows:
            # Empty results often mean wrong relationship direction or bad joins.
            for attempt, retry_msg in enumerate(_EMPTY_RESULT_HINTS, start=1):
                fixed = self._generate_cypher(
                    query,
                    schema,
                    limit,
                    previous_cypher=cypher,
                    execution_error=retry_msg,
                )
                if not fixed or fixed.strip() == cypher.strip():
                    continue
                fixed = _fix_relationship_directions(fixed, schema)
                fixed = _repair_schema_paths(fixed, schema)
                rows2, cypher2, err2 = self._execute_cypher_rows(
                    fixed, query, schema=schema, limit=limit
                )
                if err2:
                    continue
                cypher = cypher2
                if rows2:
                    rows = rows2
                    break
        return self._rows_to_chunks(rows, cypher)

    def _rows_to_chunks(self, rows: list[dict], cypher: str) -> list[dict]:
        out: list[dict] = []
        for i, row in enumerate(rows):
            clean = sanitize_row(row)
            out.append({
                "id": f"row_{i}",
                "title": self._row_title(clean),
                "text": self._row_to_text(clean),
                "raw": clean,
                "score": 1.0,
                "cypher": cypher,
                "related": [],
            })
        return out

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
                with self.driver.session() as session:
                    result = session.run(cypher)
                    return [sanitize_row(r.data()) for r in result], cypher, None
            except Exception as e:
                last_err = str(e)
                if attempt == 0 and schema:
                    fixed = self._generate_cypher(
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

    def _generate_cypher(
        self,
        query: str,
        schema: str,
        limit: int,
        *,
        previous_cypher: Optional[str] = None,
        execution_error: Optional[str] = None,
        max_tokens: int = STRUCTURED_TEXT2CYPHER_MAX_TOKENS,
    ) -> Optional[str]:
        retry_block = ""
        if execution_error and previous_cypher:
            retry_block = f"""
PREVIOUS QUERY (FAILED OR EMPTY):
{previous_cypher}

ISSUE:
{execution_error}

Fix the query. Use toFloat()/toInteger() when multiplying numeric fields that may be stored as strings.
If error mentions \"Variable not defined\", a WITH clause dropped a variable — keep it in WITH or aggregate in RETURN without referencing dropped aliases.
If 0 rows: reverse any relationship whose direction does not match RELATIONSHIP TYPES, then retry.
"""

        n = analytics_result_limit(query, limit)
        prompt = load_prompt(
            "structured_text2cypher",
            schema=schema,
            retry_block=retry_block,
            query=query,
            n=n,
        )

        response = provider.chat_completion(
            model=LLM_MODEL,
            temperature=0,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
        )
        cypher = response.choices[0].message.content.strip()
        cypher = cypher.replace("```cypher", "").replace("```", "").strip()
        return cypher or None

    def _fetch_schema(self) -> str:
        if self._schema_cache:
            return self._schema_cache
        with self.driver.session() as session:
            labels_result = session.run(
                """
                CALL db.schema.nodeTypeProperties()
                YIELD nodeType, propertyName, propertyTypes
                RETURN nodeType, collect(propertyName + ': ' + propertyTypes[0]) AS properties
                """
            )
            nodes = [f"{r['nodeType']} {{{', '.join(r['properties'])}}}" for r in labels_result]

            patterns_result = session.run(
                """
                MATCH (a)-[r]->(b)
                RETURN DISTINCT labels(a)[0] AS from, type(r) AS rel, labels(b)[0] AS to
                """
            )
            patterns = [f"(:{r['from']})-[:{r['rel']}]->(:{r['to']})" for r in patterns_result]

            rel_lines: list[str] = []
            try:
                rel_props = session.run(
                    """
                    CALL db.schema.relTypeProperties()
                    YIELD relType, propertyName, propertyTypes
                    RETURN relType, collect(propertyName + ': ' + propertyTypes[0]) AS properties
                    """
                )
                rel_lines = [f"{r['relType']} {{{', '.join(r['properties'])}}}" for r in rel_props]
            except Exception:
                rel_lines = ["(relationship properties unavailable)"]

        schema = (
            "NODE TYPES:\n" + "\n".join(nodes) +
            "\n\nRELATIONSHIP TYPES:\n" + "\n".join(sorted(set(patterns))) +
            "\n\nRELATIONSHIP PROPERTIES:\n" + "\n".join(rel_lines)
        )
        self._schema_cache = schema
        return schema

    def _row_title(self, row: dict) -> str:
        for key in ("productName", "name", "title", "companyName", "categoryName", "customerID", "orderID"):
            if key in row and row[key] is not None:
                return str(row[key])
        return str(list(row.values())[0]) if row else "Result"

    def _row_to_text(self, row: dict) -> str:
        return "\n".join(f"{k}: {v}" for k, v in row.items() if v is not None)

    def _format_response(self, query: str, items: list, strategy: str) -> dict:
        return {
            "query": query,
            "strategy": strategy,
            "chunks": items,
            "total_available": len(items),
        }

