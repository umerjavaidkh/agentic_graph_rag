"""
retrieval/structured/retriever.py — Structured Neo4j retriever.

Schema introspection + LLM Text-to-Cypher + execute/repair.
"""

import re
from typing import Optional

from neo4j import GraphDatabase

from ...auth.rbac_setup import GraphRBAC
from ...auth.roles import DEFAULT_PUBLIC_CONTEXT, UserContext
from ...config.settings import (
    CHAT_MODEL,
    MODEL_PROVIDER,
    NEO4J_PASSWORD,
    NEO4J_URI,
    NEO4J_USER,
    OPENAI_API_KEY,
)
from ...model_providers.factory import get_model_provider
from .neo4j_sanitize import sanitize_row
from .query_intent import analytics_result_limit

provider = get_model_provider(MODEL_PROVIDER, OPENAI_API_KEY)
LLM_MODEL = CHAT_MODEL

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
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.user_context = user_context or DEFAULT_PUBLIC_CONTEXT
        self.rbac = GraphRBAC(uri, user, password)
        self._schema_cache: Optional[str] = None
        self._rbac_cache: dict[tuple[str, str], bool] = {}

    def close(self) -> None:
        self.driver.close()

    def retrieve(
        self,
        query: str,
        limit: int = 5,
        user_context: Optional[UserContext] = None,
    ) -> dict:
        ctx = user_context or self.user_context
        chunks = self._text2cypher(query, limit, user_context=ctx)
        return self._format_response(query, chunks, strategy="text2cypher")

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
        cypher = self._generate_cypher(query, schema, limit)
        if not cypher:
            return []
        cypher = _fix_relationship_directions(cypher, schema)
        cypher = _repair_schema_paths(cypher, schema)

        rows, cypher, err = self._execute_cypher_rows(cypher, query, schema=schema, limit=limit)
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
        prompt = f"""You are a Neo4j Cypher expert. Generate a Cypher query for the question below.

GRAPH SCHEMA:
{schema}

RULES:
- Return ONLY the Cypher query, no explanation, no markdown
- Use only labels/relationships/properties that appear in the schema
- Relationship direction is critical: RELATIONSHIP TYPES shows exact arrows. If schema lists (:A)-[:R]->(:B), only match (a:A)-[:R]->(b:B); never reverse the arrow unless the question explicitly asks for the inverse
- Multi-hop questions: chain patterns using schema directions (e.g. Order-[:ORDER_CONTAINS]->Product-[:SUPPLIED_BY]->Supplier). Each relationship must connect the node types shown in RELATIONSHIP TYPES — never attach a relationship to a node label that does not appear in that relationship's schema pattern
- Shipment/location filters: use Order-[:SHIPPED_TO]->Address and filter Address country/city properties with toLower() CONTAINS
- Aggregations / rankings ("most", "greatest", "highest", "top"): MATCH the full path, WITH groupKey, COUNT(DISTINCT countedNode) AS metric (use DISTINCT when counting orders/customers/entities that can repeat across joined rows), RETURN groupKey, metric ORDER BY metric DESC LIMIT {n}
- For ranked top-N results: ORDER BY metric DESC then LIMIT {n}
- For text search: toLower() and CONTAINS
- Numeric safety: wrap CSV-backed fields with toFloat() or toInteger() before arithmetic
- For dates: if orderDate includes time, parse via substring(toString(orderDate), 0, 10) before date()
{retry_block}
QUESTION: {query}

CYPHER:"""

        response = provider.chat_completion(
            model=LLM_MODEL,
            temperature=0,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500,
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

