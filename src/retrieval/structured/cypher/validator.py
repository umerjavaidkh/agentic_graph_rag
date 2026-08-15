"""Cypher static validation (SQL idioms, temporal filters)."""
from __future__ import annotations

import re
from typing import Optional

from ....config.settings import MULTI_TENANCY_ENABLED
from .tenant_injection import missing_tenant_filter_issue

# Schema-agnostic hints when a query executes but returns no rows.
# Every hint has to carry the escape clause below, and that is the whole
# point of it. These are retry instructions for a model that has just been
# told its query returned nothing; phrased only as "restructure until rows
# come back", they presuppose the answer exists and pressure it into finding
# SOME column that returns data. Observed: asked for an average salary the
# graph has no property for, the retry returned avg(created_at) aliased as
# `averageSalary`, and the user was told employees earn 1,786,771,191,163.
# An empty result is a legitimate outcome -- a question about data the graph
# does not hold has no answer, and inventing one is the worst option
# available, because a plausible figure is indistinguishable from a real one.
_NO_SUBSTITUTION = (
    "If no label or property in the schema actually represents what was asked "
    "for, the graph does not contain it: return the query unchanged rather "
    "than measuring a DIFFERENT property. Never alias one property as another "
    "concept (e.g. returning avg(created_at) AS averageSalary). Zero rows is a "
    "correct answer when the data does not hold the thing being asked about."
)

EMPTY_RESULT_HINTS = (
    (
        "Query executed successfully but returned 0 rows. "
        "Verify every relationship direction matches RELATIONSHIP TYPES exactly "
        "(if schema shows (:A)-[:R]->(:B), traverse A-[:R]->B, never B-[:R]->A). "
        "Remove unnecessary WHERE filters. " + _NO_SUBSTITUTION
    ),
    (
        "Query still returned 0 rows. "
        "Rebuild the MATCH path by chaining RELATIONSHIP TYPES from source to target. "
        "When counting unique orders/customers/entities across joins, use COUNT(DISTINCT node). "
        + _NO_SUBSTITUTION
    ),
)

# SQL idioms that are invalid or fragile in Cypher — trigger regeneration before execute.
SQL_CYPHER_ISSUES: list[tuple[str, str]] = [
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
        # `node.SOME_REL.prop` -- reaching through a relationship as if it
        # were a nested field. Keyed on the SCREAMING_SNAKE convention for
        # relationship types rather than any particular type name, so it
        # holds for any schema. The (?-i:) scope is load-bearing: these
        # patterns are matched with re.I, under which a plain [A-Z] also
        # matches lowercase and this swallows every `apoc.date.format(...)`
        # call, shadowing the apoc rule below it.
        r"\b\w+\s*\.\s*(?-i:[A-Z][A-Z0-9_]{2,})\s*\.\s*\w+",
        "Cypher has no nested field access through a relationship. Bind the "
        "relationship to a variable -- (a)-[r:REL_TYPE]->(b) -- then read "
        "r.property, or read the property from whichever node the schema "
        "lists it under.",
    ),
    (
        r"\bAS\s+\w+\)\s+AS\s+\w+",
        "Cypher syntax: you have an extra ')' before an AS alias (e.g. 'AS x) AS y'). Remove the extra parenthesis.",
    ),
    (
        r"\bapoc\.coll\.sortNodes\b",
        "Do not use apoc.coll.sortNodes: it only sorts a LIST<NODE> and takes exactly 2 args, so it fails on lists of maps. "
        "To get top-K within a group, ORDER BY the metric DESC BEFORE collect(...), then slice the collected list: "
        "WITH groupKey, item ORDER BY metric DESC WITH groupKey, collect(item)[0..K] AS topItems.",
    ),
    (
        r"\bapoc\.date\.format\s*\(\s*(?:datetime|date|localdatetime)\s*\(",
        "apoc.date.format() expects an epoch-millis Integer/Long as its first argument, not a "
        "datetime()/date()/localdatetime() temporal value — passing one directly causes a type "
        "mismatch error. For month/day bucketing over a date/datetime property, do not use "
        "apoc.date.format at all: use substring(toString(prop), 0, 7) for a YYYY-MM month key, or "
        "substring(toString(prop), 0, 10) for a YYYY-MM-DD day key.",
    ),
]

# Node label references appear right after "(" (optionally preceded by a
# variable name): "(e:Employee)", "(e:Employee {name: 'x'})", "(e:Employee:Person)".
# Relationship TYPE references use "[...]" instead ("[r:ORDERED]") and are a
# completely different namespace -- this pattern only matches "(" so it never
# confuses the two.
_NODE_LABEL_RE = re.compile(r"\(\s*\w*((?:\s*:\s*[A-Za-z_][A-Za-z0-9_]*)+)\s*[){]")

_QUESTION_YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")

_WITH_MISSING_ALIAS_MSG = (
    "Cypher syntax: every expression in WITH must be aliased using AS "
    "(e.g. `WITH p.productName AS productName`)."
)


def _split_top_level_commas(expr: str) -> list[str]:
    parts: list[str] = []
    depth = 0
    start = 0
    for i, ch in enumerate(expr):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth = max(0, depth - 1)
        elif ch == "," and depth == 0:
            parts.append(expr[start:i])
            start = i + 1
    parts.append(expr[start:])
    return parts


def _with_missing_alias_issue(cypher: str) -> Optional[str]:
    """Flag bare `n.prop` items in WITH lists (not inside COUNT(...) etc.)."""
    for match in re.finditer(
        r"\bWITH\s+(?:DISTINCT\s+)?(.+?)(?=\s+(?:WHERE|MATCH|RETURN|SET|DELETE|CREATE|MERGE|UNWIND|CALL)\b|\s*$)",
        cypher,
        flags=re.I | re.S,
    ):
        body = match.group(1)
        order_m = re.search(r"\s+ORDER\s+BY\s+", body, re.I)
        main = body[: order_m.start()] if order_m else body
        for part in _split_top_level_commas(main):
            token = part.strip()
            if re.fullmatch(r"\w+\.\w+", token):
                return _WITH_MISSING_ALIAS_MSG
    return None


def sql_cypher_issue(cypher: str) -> Optional[str]:
    for pattern, msg in SQL_CYPHER_ISSUES:
        if re.search(pattern, cypher, re.I | re.S):
            return msg
    alias_issue = _with_missing_alias_issue(cypher)
    if alias_issue:
        return alias_issue
    if MULTI_TENANCY_ENABLED:
        return missing_tenant_filter_issue(cypher)
    return None


def unknown_label_issue(cypher: str, known_labels: set[str]) -> Optional[str]:
    """
    Catch a generated Cypher query that MATCHes a node label absent from this
    graph's actual schema -- an LLM hallucinating a shape it expects (e.g. a
    first-class `Employee` node, the classic Northwind schema) rather than the
    one it was given, when this particular graph only has an `employeeID`
    property on `Order`. Neo4j doesn't error on an unknown label -- the MATCH
    just silently matches nothing -- so without this the query "succeeds"
    with 0 rows and the caller has no specific signal to regenerate against,
    only a generic "0 rows" hint that doesn't name the actual problem.

    Schema-agnostic: `known_labels` comes from live introspection
    (SchemaProvider.known_labels()), not any hardcoded label list, so this
    works for any graph/domain, not just Northwind.
    """
    if not known_labels:
        return None
    unknown: list[str] = []
    seen: set[str] = set()
    for m in _NODE_LABEL_RE.finditer(cypher or ""):
        for label in m.group(1).split(":"):
            label = label.strip()
            if label and label not in known_labels and label not in seen:
                seen.add(label)
                unknown.append(label)
    if not unknown:
        return None
    labels_str = ", ".join(unknown)
    return (
        f"Label(s) {labels_str} do not exist in this graph's schema -- the "
        f"MATCH will silently return 0 rows, not an error. Only the labels "
        f"listed under NODE TYPES in the schema exist. If you were trying to "
        f"reference an entity that isn't its own node here, check whether it "
        f"is instead just a property on a related node type (e.g. an id or "
        f"name field), and match/group on that property directly instead of "
        f"inventing a node label for it."
    )


_VAR_NODE_RE = re.compile(r"\(\s*([A-Za-z_]\w*)\s*:\s*([`\w:]+)")
_VAR_REL_RE = re.compile(r"\[\s*([A-Za-z_]\w*)\s*:\s*([`\w|]+)")
_PROP_REF_RE = re.compile(r"\b([A-Za-z_]\w*)\.([A-Za-z_]\w*)\b")


_NO_SUCH_DATA_RE = re.compile(r"^\s*(?:--\s*)?NO_SUCH_DATA\s*:\s*(.+?)\s*$", re.I | re.M)


def no_such_data_subject(cypher: str) -> Optional[str]:
    """The subject of a NO_SUCH_DATA reply, or None for an ordinary query.

    Tolerates the model prefixing it as a Cypher comment, which it does when
    the surrounding instruction is "return only Cypher".
    """
    m = _NO_SUCH_DATA_RE.search(cypher or "")
    if not m:
        return None
    subject = m.group(1).strip().strip("`<>").strip()
    return subject or "the requested information"


def unknown_property_issue(cypher: str, known_properties: dict[str, set[str]]) -> Optional[str]:
    """
    Catch a generated query reading a property that its variable's type does
    not have -- the same silent-failure class as unknown_label_issue, one
    level down, and considerably nastier.

    Neo4j returns null for a missing property instead of erroring. Null then
    propagates through aggregation as absence, so `sum(x)` over a wholly
    wrong property is 0 and `avg(x)` is null. Crucially an aggregate ALWAYS
    returns exactly one row, so the empty-result retry never fires: the
    pipeline sees a successful query with data and reports a confident wrong
    figure. Two observed cases, both from real questions:

      * `MATCH (:Order)-[li:CONTAINS]->(:OrderItem)
         RETURN sum(toFloat(li.freight))` -- `li` is the RELATIONSHIP, and
        `freight` lives on the OrderItem node, so the total came back as
        "zero" against a true value of 2,251,910.
      * asking for a `salary` property no label has produced an empty first
        result, and the empty-result retry then "fixed" it by averaging
        `created_at` under the alias `averageSalary` -- reported to the user
        as a salary of 1,786,771,191,163.

    That second case is why this rejects rather than hints: left to retry, an
    empty result pressures the model into finding SOME column that returns
    data, and a fabricated answer is worse than no answer.

    Only variables whose type is known and whose property set is non-empty
    are checked, so an unrecognised binding (a WITH alias, a map projection)
    is passed over rather than guessed at.
    """
    if not known_properties:
        return None
    # A type present in the schema with NO properties is the important case,
    # not one to skip: `[li:CONTAINS]` where CONTAINS carries nothing means
    # every `li.<anything>` is wrong. So membership decides whether a variable
    # is checkable, and the property set only decides what passes -- keyed on
    # `is not None` rather than on the set being non-empty.
    bound: dict[str, set[str]] = {}
    for rx, sep in ((_VAR_NODE_RE, ":"), (_VAR_REL_RE, "|")):
        for m in rx.finditer(cypher or ""):
            allowed: set[str] = set()
            known = False
            for type_name in m.group(2).replace("`", "").split(sep):
                props = known_properties.get(type_name.strip())
                if props is not None:
                    known = True
                    allowed |= props
            if known:
                bound.setdefault(m.group(1), set()).update(allowed)

    bad: list[str] = []
    seen: set[str] = set()
    for m in _PROP_REF_RE.finditer(cypher or ""):
        var, prop = m.group(1), m.group(2)
        ref = f"{var}.{prop}"
        if var in bound and prop not in bound[var] and ref not in seen:
            seen.add(ref)
            bad.append(ref)
    if not bad:
        return None
    detail = "; ".join(
        f"{ref} (available: {', '.join(sorted(bound[ref.split('.')[0]])) or 'none'})"
        for ref in bad
    )
    return (
        f"Property reference(s) {detail} -- these do not exist on that "
        f"variable's type. Neo4j returns null rather than an error, so the "
        f"query would report 0 or null as if it were the real answer. Check "
        f"the schema: the property may belong to a DIFFERENT variable in the "
        f"pattern (properties of a node cannot be read off the relationship "
        f"that points at it, so bind the node itself). If no type in the "
        f"schema has this property, the data does not contain it -- say so "
        f"instead of substituting a different property."
    )


def dropped_year_filter_issue(cypher: str, query: str) -> Optional[str]:
    """
    Corpus-agnostic guard against multistep steps that silently drop a temporal
    filter the question requires.
    """
    years = set(_QUESTION_YEAR_RE.findall(query or ""))
    if not years:
        return None
    c = cypher or ""
    traverses = bool(re.search(r"\bMATCH\b", c, re.I)) and bool(re.search(r"-\s*\[", c))
    if not traverses:
        return None
    if any(y in c for y in years):
        return None
    yrs = ", ".join(sorted(years))
    first = sorted(years)[0]
    return (
        f"This step re-matches the graph but is missing the date filter for {yrs} "
        f"that the question requires. Every step that traverses the graph MUST repeat "
        f"the {yrs} filter (e.g. WHERE o.orderDate STARTS WITH '{first}'), or instead "
        f"UNWIND the prior step's already-filtered rows. Add the missing {yrs} filter."
    )
