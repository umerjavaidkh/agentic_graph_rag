"""
tests/interface/test_query_request_language_unit.py — the request-side seam.

Asserted by parsing schemas.py rather than by importing it. This suite
runs with `fastapi` and `pydantic` stubbed (tests/interface/
test_scalable_pipeline_unit.py, tests/unstructured/
test_ingestion_triage_unit.py), so importing the module gets a fake
BaseModel whose validators never run -- the assertion would pass without
testing anything. Same reason and same technique as
test_route_imports_resolve_unit.py, which parses rather than imports.

What the validators DO is tested where it is real: `get_profile` in
tests/shared/test_language_scoping_unit.py and `fold` in
tests/shared/test_unicode_text_unit.py. What is left to check is that
they are wired to the request at all, and that both entry paths go
through the model that carries them.

Run with:
    python -m pytest tests/interface/test_query_request_language_unit.py -v
"""
from __future__ import annotations

import ast
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
_SCHEMAS = REPO / "src" / "interface" / "schemas.py"
_QUERY_ROUTES = REPO / "src" / "interface" / "routes" / "query.py"


def _query_request() -> ast.ClassDef:
    tree = ast.parse(_SCHEMAS.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "QueryRequest":
            return node
    raise AssertionError("QueryRequest not found in schemas.py")


def _validated_fields(cls: ast.ClassDef) -> dict[str, set[str]]:
    """Field name -> the validator function names attached to it."""
    out: dict[str, set[str]] = {}
    for item in cls.body:
        if not isinstance(item, ast.FunctionDef):
            continue
        for dec in item.decorator_list:
            if not isinstance(dec, ast.Call):
                continue
            name = getattr(dec.func, "id", None) or getattr(dec.func, "attr", None)
            if name != "field_validator":
                continue
            for arg in dec.args:
                if isinstance(arg, ast.Constant):
                    out.setdefault(arg.value, set()).add(item.name)
    return out


def test_the_request_carries_a_language():
    fields = [
        item.target.id
        for item in _query_request().body
        if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name)
    ]
    assert "language" in fields


def test_the_language_is_validated_rather_than_trusted():
    """An unrecognised code must resolve to something this deployment has,
    not reach retrieval and scope to a language with no documents."""
    assert "language" in _validated_fields(_query_request())


def test_the_language_default_is_validated_too():
    """Pydantic skips validators on defaults unless asked.

    Without `validate_default`, the omitted case -- which is every existing
    caller -- is the one path that reaches retrieval unnormalised, and it
    is the path least likely to be noticed.
    """
    for item in _query_request().body:
        if isinstance(item, ast.AnnAssign) and getattr(item.target, "id", "") == "language":
            source = ast.unparse(item)
            assert "validate_default=True" in source
            return
    raise AssertionError("language field not found")


def test_the_question_is_normalised():
    assert "question" in _validated_fields(_query_request())


def test_both_query_entry_paths_take_the_model_that_carries_it():
    """One validator covers /query and /query/stream only while both take
    QueryRequest. If a route ever took a different model, the normalisation
    would silently apply to one path and not the other."""
    tree = ast.parse(_QUERY_ROUTES.read_text())
    handlers = {}
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for dec in node.decorator_list:
            if not isinstance(dec, ast.Call):
                continue
            for arg in dec.args:
                if isinstance(arg, ast.Constant) and str(arg.value).startswith("/query"):
                    annotations = {
                        getattr(a.annotation, "id", None) for a in node.args.args
                    }
                    handlers[str(arg.value)] = annotations

    assert "/query" in handlers and "/query/stream" in handlers
    for route, annotations in handlers.items():
        assert "QueryRequest" in annotations, f"{route} does not take QueryRequest"
