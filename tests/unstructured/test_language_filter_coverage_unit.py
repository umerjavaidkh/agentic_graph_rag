"""
tests/unstructured/test_language_filter_coverage_unit.py — the scope predicates
travel together, everywhere, forever.

This repository's most expensive recurring bug is a fix that lands in one
of several equivalent paths. Five structural fast-paths each resolved
their own document; the document-scoping fix landed in one of them and
there were six, so TOC questions answered from the wrong document until
PR #120 deleted the lot. `language_filter` is spliced into 38 places and
is exactly the same shape of hazard: the 39th scoped query someone adds
will be written by copying a neighbour, and if that neighbour is the one
site that was missed, the miss propagates.

So this does not test a behaviour. It tests an invariant over the source:

    wherever a document query scopes by tenant, it scopes by language too,
    on the same alias, and supplies both parameters.

A new scoped query cannot be added without language scoping, and the
failure arrives as a red test naming the file and line rather than as an
Arabic query quietly returning English documents.

Run with:
    python -m pytest tests/unstructured/test_language_filter_coverage_unit.py -v
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
_SERVICES = sorted((REPO / "src" / "unstructured" / "retrieval").rglob("*.py"))

# Only the f-string interpolation form, `{tenant_filter("n")}`, is a USE of
# the predicate. Matching the bare call name would also match the several
# docstrings that discuss the idiom, and a test that fails on prose teaches
# people to weaken it.
_CALL = re.compile(r'\{(tenant|language)_filter\(\s*(?P<alias>"[^"]*")?\s*\)\}')


def _scope_calls(text: str) -> list[tuple[int, str, str]]:
    """(line number, which filter, alias) for every scope predicate use."""
    out = []
    for i, line in enumerate(text.splitlines(), start=1):
        for m in _CALL.finditer(line):
            out.append((i, m.group(1), m.group("alias") or '"n"'))
    return out


def test_every_tenant_scoped_query_is_language_scoped_on_the_same_alias():
    """Both are deployment-level scoping on the same node; one without the
    other is an asymmetry nobody decided on."""
    missing = []
    for path in _SERVICES:
        calls = _scope_calls(path.read_text())
        for lineno, which, alias in calls:
            if which != "tenant":
                continue
            partner = [
                c for c in calls
                if c[0] == lineno and c[1] == "language" and c[2] == alias
            ]
            if not partner:
                missing.append(f"{path.relative_to(REPO)}:{lineno} tenant_filter({alias})")
    assert not missing, (
        "tenant-scoped with no language scope on the same alias:\n  "
        + "\n  ".join(missing)
    )


def test_every_language_scope_has_a_tenant_scope_beside_it():
    """The converse, so the pair cannot drift apart from either side."""
    orphans = []
    for path in _SERVICES:
        calls = _scope_calls(path.read_text())
        for lineno, which, alias in calls:
            if which != "language":
                continue
            partner = [
                c for c in calls
                if c[0] == lineno and c[1] == "tenant" and c[2] == alias
            ]
            if not partner:
                orphans.append(f"{path.relative_to(REPO)}:{lineno} language_filter({alias})")
    assert not orphans, (
        "language-scoped with no tenant scope on the same alias:\n  "
        + "\n  ".join(orphans)
    )


def test_every_query_supplying_a_tenant_also_supplies_a_language():
    """A predicate that names $language in a query given no `language`
    parameter fails with ParameterMissing -- and only once a second
    language is enabled, which is the worst possible time to find out."""
    missing = []
    for path in _SERVICES:
        lines = path.read_text().splitlines()
        for i, line in enumerate(lines):
            if not re.match(r"\s*tenant_id=tenant_id,\s*$", line):
                continue
            window = "\n".join(lines[max(0, i - 2): i + 3])
            if "language=language" not in window:
                missing.append(f"{path.relative_to(REPO)}:{i + 1}")
    assert not missing, (
        "queries passing tenant_id with no language parameter:\n  " + "\n  ".join(missing)
    )


def test_every_service_method_taking_a_tenant_takes_a_language():
    """Otherwise the parameter stops at a signature and the query below it
    silently scopes to the default language."""
    missing = []
    for path in _SERVICES:
        tree = ast.parse(path.read_text())
        for fn in [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]:
            names = {a.arg for a in fn.args.args} | {a.arg for a in fn.args.kwonlyargs}
            if "tenant_id" not in names:
                continue
            body = ast.get_source_segment(path.read_text(), fn) or ""
            if "language_filter(" not in body:
                continue  # not a scoped query builder
            if "language" not in names:
                missing.append(f"{path.relative_to(REPO)}:{fn.lineno} {fn.name}")
    assert not missing, (
        "scoped methods taking tenant_id but not language:\n  " + "\n  ".join(missing)
    )


def test_the_invariant_is_actually_covering_something():
    """A guard that matches nothing passes forever and protects nothing."""
    total = sum(
        len([c for c in _scope_calls(p.read_text()) if c[1] == "language"])
        for p in _SERVICES
    )
    assert total >= 30, f"expected the splice to cover ~38 sites, found {total}"
