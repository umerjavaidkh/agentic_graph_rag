"""
Deterministic tenant_id filter injection for LLM-generated Cypher.

Letting an LLM "remember" to filter by tenant_id is not a safe security
boundary — a forgotten filter is a cross-tenant data leak. This module
mechanically rewrites every MATCH/OPTIONAL MATCH clause to require a
tenant_id predicate on every labeled node variable, the same way
repair.py's fix_relationship_directions/repair_schema_paths mechanically
rewrite LLM output today — never relying on the LLM to get it right.

Node variables with no label at all are treated as needing a filter too
(fail closed: we can't prove they're exempt). Only a small, fixed set of
RBAC/control-plane labels (deployment-wide by design, not per-tenant
content) are exempt.
"""
from __future__ import annotations

import re
from typing import Callable, Optional

TENANT_EXEMPT_LABELS = frozenset({"User", "Role", "KnowledgeArea", "Policy", "Entity", "Chunk"})

_CLAUSE_RE = re.compile(
    r"(?P<kind>OPTIONAL\s+MATCH|MATCH)\s+(?P<body>.+?)"
    r"(?:\s+WHERE\s+(?P<where>.+?))?"
    r"(?=\s+(?:OPTIONAL\s+MATCH|MATCH|WITH|RETURN|SET|DELETE|CREATE|MERGE|UNWIND|CALL)\b|\s*$)",
    re.I | re.S,
)

# EXISTS {...} / NOT EXISTS {...} / COUNT {...} subqueries embed their own
# MATCH — left unprotected, _CLAUSE_RE mistakes that nested MATCH for a new
# top-level clause and corrupts the query. One level of {} nesting covers
# realistic generated Cypher; deeper nesting still fails closed via
# missing_tenant_filter_issue rather than executing something unproven.
_SUBQUERY_BLOCK_RE = re.compile(
    r"((?:NOT\s+)?EXISTS|COUNT)\s*\{([^{}]*)\}",
    re.I | re.S,
)

_NODE_RE = re.compile(r"\(\s*([A-Za-z_]\w*)?((?::[A-Za-z_]\w*)*)\s*(\{[^{}]*\})?\s*\)")


def _node_labels(node_match: re.Match) -> list[str]:
    return re.findall(r":(\w+)", node_match.group(2) or "")


def _needs_filter(labels: list[str]) -> bool:
    if not labels:
        return True  # unlabeled — can't prove exempt, fail closed
    return not any(label in TENANT_EXEMPT_LABELS for label in labels)


def _clause_node_vars(body: str) -> list[tuple[str, list[str]]]:
    """(variable_name, labels) for every node slot, assigning synthetic names to anonymous nodes."""
    out: list[tuple[str, list[str]]] = []
    counter = 0
    for nm in _NODE_RE.finditer(body):
        var = nm.group(1)
        labels = _node_labels(nm)
        if not var:
            counter += 1
            var = f"_tf{counter}"
        out.append((var, labels))
    return out


def _patch_anonymous_nodes(body: str) -> str:
    """Assign a synthetic variable name to every anonymous node slot (so it's referenceable)."""
    counter = [0]

    def _sub(m: re.Match) -> str:
        var, labels_str, props = m.group(1), m.group(2) or "", m.group(3) or ""
        if not var:
            counter[0] += 1
            var = f"_tf{counter[0]}"
        return f"({var}{labels_str} {props})" if props else f"({var}{labels_str})"

    return _NODE_RE.sub(_sub, body)


def _missing_filter_vars(node_vars: list[tuple[str, list[str]]], existing_where: str) -> list[str]:
    missing = []
    for var, labels in node_vars:
        if not _needs_filter(labels):
            continue
        if re.search(rf"\b{re.escape(var)}\.tenant_id\b", existing_where or ""):
            continue
        missing.append(var)
    return missing


def _rewrite_clause(match: re.Match, tenant_id_param: str) -> str:
    kind = match.group("kind")
    body = match.group("body")
    where = match.group("where") or ""

    node_vars = _clause_node_vars(body)
    filter_vars = _missing_filter_vars(node_vars, where)
    new_body = _patch_anonymous_nodes(body)

    if not filter_vars:
        return f"{kind} {new_body} WHERE {where}" if where else f"{kind} {new_body}"

    is_optional = kind.strip().upper().startswith("OPTIONAL")
    if is_optional:
        predicate = " AND ".join(
            f"({v} IS NULL OR {v}.tenant_id = {tenant_id_param})" for v in filter_vars
        )
    else:
        predicate = " AND ".join(f"{v}.tenant_id = {tenant_id_param}" for v in filter_vars)

    if where:
        return f"{kind} {new_body} WHERE {where} AND {predicate}"
    return f"{kind} {new_body} WHERE {predicate}"


def _protect_subquery_blocks(cypher: str, on_inner: Callable[[str], str]) -> tuple[str, list[str]]:
    """
    Replace EXISTS{...}/NOT EXISTS{...}/COUNT{...} blocks with placeholder
    tokens so clause segmentation never sees their nested MATCH, running
    on_inner(inner_text) against each block's contents first. Returns the
    placeholder-substituted text plus the (already-processed) blocks to
    restore, in order.
    """
    blocks: list[str] = []

    def _extract(m: re.Match) -> str:
        prefix, inner = m.group(1), m.group(2)
        blocks.append(f"{prefix} {{{on_inner(inner)}}}")
        return f"__TENANT_BLOCK_{len(blocks) - 1}__"

    protected = _SUBQUERY_BLOCK_RE.sub(_extract, cypher)
    return protected, blocks


def _restore_subquery_blocks(text: str, blocks: list[str]) -> str:
    for i, block in enumerate(blocks):
        text = text.replace(f"__TENANT_BLOCK_{i}__", block)
    return text


def inject_tenant_filters(cypher: str, tenant_id_param: str = "$tenant_id") -> str:
    """Mechanically add a tenant_id predicate to every MATCH/OPTIONAL MATCH clause."""
    if not cypher or not cypher.strip():
        return cypher
    protected, blocks = _protect_subquery_blocks(
        cypher, lambda inner: inject_tenant_filters(inner, tenant_id_param)
    )
    injected = _CLAUSE_RE.sub(lambda m: _rewrite_clause(m, tenant_id_param), protected)
    return _restore_subquery_blocks(injected, blocks)


def _clause_issues(cypher: str) -> list[str]:
    issues: list[str] = []
    for m in _CLAUSE_RE.finditer(cypher):
        body = m.group("body")
        where = m.group("where") or ""
        node_vars = _clause_node_vars(body)
        missing = _missing_filter_vars(node_vars, where)
        if missing:
            preview = " ".join(cypher.split())[:160]
            issues.append(
                f"Could not prove tenant isolation for variable(s) {', '.join(missing)} "
                f"in clause '{m.group('kind')} {' '.join(body.split())[:60]}...': {preview}..."
            )
    return issues


def missing_tenant_filter_issue(cypher: str) -> Optional[str]:
    """
    Fail-closed check: after inject_tenant_filters has run, verify every
    MATCH/OPTIONAL MATCH clause's non-exempt node variables carry a
    tenant_id predicate. Returns a message if any clause can't be proven
    tenant-scoped — the caller must NOT execute in that case.
    """
    if not cypher or not cypher.strip():
        return None
    protected, blocks = _protect_subquery_blocks(cypher, lambda inner: inner)
    issues = _clause_issues(protected)
    for inner in blocks:
        # Strip the "PREFIX {" / trailing "}" wrapper back to bare inner text.
        inner_body = inner.split("{", 1)[1].rsplit("}", 1)[0]
        issues.extend(_clause_issues(inner_body))
    return issues[0] if issues else None
