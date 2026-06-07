"""Schema-driven Cypher path and relationship repair."""
from __future__ import annotations

import re
from collections import deque


def parse_schema_relationships(schema: str) -> list[tuple[str, str, str]]:
    """Extract (from_label, rel_type, to_label) triples from schema text."""
    patterns: list[tuple[str, str, str]] = []
    for line in schema.splitlines():
        m = re.search(r"\(:?(\w+)\)-\[:(\w+)\]->\(:?(\w+)\)", line)
        if m:
            patterns.append((m.group(1), m.group(2), m.group(3)))
    return patterns


def fix_relationship_directions(cypher: str, schema: str) -> str:
    """
    Flip relationship arrows that contradict schema directions.

    Schema (:A)-[:R]->(:B) allows (a:A)-[:R]->(b:B) or (b:B)<-[:R]-(a:A).
    Only flip explicit wrong traversals: (b:B)-[:R]->(a:A) or (a:A)<-[:R]-(b:B).
    """
    fixed = cypher
    for from_label, rel, to_label in parse_schema_relationships(schema):
        wrong_fwd = re.compile(
            rf"\((\w+)\s*:\s*{re.escape(to_label)}\)\s*-\[:{re.escape(rel)}\]\s*->\s*\((\w+)\s*:\s*{re.escape(from_label)}\)",
            re.IGNORECASE,
        )
        fixed = wrong_fwd.sub(rf"(\2:{from_label})-[:{rel}]->(\1:{to_label})", fixed)

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

        for i, (_a, step_rel, step_to) in enumerate(sp):
            if i == len(sp) - 1:
                step_var = to_var
            else:
                step_var = _var_for_label(step_to)
            segments.append(f"-[:{step_rel}]->({step_var}:{step_to})")
            label_vars[step_to] = step_var

    return "".join(segments)


def repair_schema_paths(cypher: str, schema: str) -> str:
    """Repair invalid relationship hops in MATCH clauses using schema paths."""
    schema_edges = parse_schema_relationships(schema)
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
