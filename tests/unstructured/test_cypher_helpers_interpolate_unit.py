"""
tests/unstructured/test_cypher_helpers_interpolate_unit.py — a helper call
that never ran.

The scope predicates are spliced into Cypher by f-string interpolation:

    AND {tenant_filter("n")} AND {language_filter("n")}

Leave the `f` off the string and Python does not complain. Neither does
any unit test with a fake session, because a fake session accepts any
text at all. The query reaches Neo4j with the literal characters
`{tenant_filter("n")}` in it and fails there:

    Invalid input '(': expected an expression

That is exactly what happened when `match_key_cypher` was spliced into
lexical.py. The query is built by CONCATENATION -- an f-string, a plain
string, and another plain string joined with `+` -- so the first two
segments interpolated and the third did not. Reading the code, it looks
like one f-string. It is three, and only some of them are.

The check is mechanical: a real f-string compiles to `ast.JoinedStr` and
its `{...}` becomes a `FormattedValue`, so the literal text can only
survive inside a plain `ast.Constant`. Anything found there is a query
that will fail the first time it is run against a real database.

Run with:
    python -m pytest tests/unstructured/test_cypher_helpers_interpolate_unit.py -v
"""
from __future__ import annotations

import ast
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

# Helpers that emit Cypher and are always spliced by interpolation.
_CYPHER_HELPERS = (
    "match_key_cypher",
    "tenant_filter",
    "language_filter",
    "content_scope_where",
    "content_scope_where_multi",
    "content_match_cypher",
    "lifecycle_active",
    "_doc_scope_cypher",
)


def _docstring_ids(tree: ast.AST) -> set[int]:
    """Identity of every docstring node.

    Excluded by POSITION, not by reading them: `tenancy.py`'s docstrings
    quote the idiom they document -- `AND {tenant_filter("n")}` -- and any
    text-based heuristic that skips them would also skip a real query that
    happened to be worded like prose.
    """
    out = set()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        body = getattr(node, "body", None) or []
        if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
            if isinstance(body[0].value.value, str):
                out.add(id(body[0].value))
    return out


def _literal_helper_calls(path: Path) -> list[str]:
    tree = ast.parse(path.read_text())
    docstrings = _docstring_ids(tree)
    found = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Constant) and isinstance(node.value, str)):
            continue
        if id(node) in docstrings:
            continue
        text = node.value
        for helper in _CYPHER_HELPERS:
            if "{" + helper in text:
                try:
                    where = path.relative_to(REPO)
                except ValueError:
                    where = path  # a temp file, from the self-check below
                found.append(f"{where}:{node.lineno} -> {{{helper}(...)}}")
    return found


def test_no_cypher_helper_is_left_uninterpolated():
    """Every splice must sit in an f-string, including the third segment
    of a concatenated query that reads as though it were one string."""
    offenders: list[str] = []
    for path in (REPO / "src").rglob("*.py"):
        offenders.extend(_literal_helper_calls(path))
    assert not offenders, (
        "Cypher helper calls left as literal text -- these queries fail at "
        "Neo4j and pass every fake-session test:\n  " + "\n  ".join(offenders)
    )


def test_the_check_can_actually_see_the_bug():
    """A guard nobody has proved catches anything is not a guard.

    This is the exact shape that shipped: an f-string concatenated with a
    plain one, where only the plain segment holds the call.
    """
    import tempfile

    source = '''
def q():
    return (
        f"""
        MATCH (n) WHERE {tenant_filter("n")}
        AND x = '""" + "ACTIVE" + """'
        WITH n, size([p IN $ps WHERE {match_key_cypher("n")} CONTAINS p]) AS hits
        RETURN n
        """
    )
'''
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as fh:
        fh.write(source)
        tmp = Path(fh.name)
    try:
        found = _literal_helper_calls(tmp)
        assert found, "the check failed to see a known-bad concatenated query"
        assert any("match_key_cypher" in f for f in found)
    finally:
        tmp.unlink()
