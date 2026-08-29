"""
tests/interface/test_language_reaches_state_unit.py — the request's language
must survive the trip to ESGState.

`language_filter()` scopes on `state["language"]`. Seven places in the
entry layer build that state dict, and each one is reached by a different
route: sync, streaming, hybrid, the document fallback from a
low-confidence structured answer. Miss one and queries on that path
silently search the default corpus -- and only once a second language is
enabled, so it would ship looking fine.

ESGState is a TypedDict and LangGraph drops any key the schema does not
name, so the failure is silent twice over: no exception at the
construction site, and no exception downstream either.

Structured state dicts are deliberately exempt. The business graph has no
language dimension -- its labels and properties are schema, not prose --
so requiring a language there would be requiring a lie.

Run with:
    python -m pytest tests/interface/test_language_reaches_state_unit.py -v
"""
from __future__ import annotations

import ast
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
_ENTRY_FILES = [
    REPO / "src" / "interface" / "routing.py",
    REPO / "src" / "interface" / "streaming" / "query_stream.py",
    REPO / "src" / "interface" / "handlers" / "documents.py",
    REPO / "src" / "interface" / "handlers" / "data.py",
    REPO / "src" / "interface" / "handlers" / "hybrid.py",
    REPO / "src" / "unstructured" / "streaming.py",
]

# Names that mark the enclosing function as driving DOCUMENT retrieval.
_UNSTRUCTURED_MARKERS = {"esg_agent", "doc_retrieve_node", "iter_document_stream"}
_STRUCTURED_MARKERS = {"structured_agent", "struct_retrieve_node"}


def _state_dicts(path: Path):
    """(function, node, keys) for each `state = {...}` in the file.

    Both assignment forms, deliberately. Three of these seven sites are
    written `state: dict[str, Any] = {...}`, which is an ast.AnnAssign and
    not an ast.Assign -- a detector that only knew the plain form silently
    skipped them and this guard passed while covering barely half of what
    it claimed to. That is the exact failure this file exists to prevent,
    so it is worth the four extra lines.
    """
    tree = ast.parse(path.read_text())
    for fn in [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]:
        for node in ast.walk(fn):
            if isinstance(node, ast.Assign):
                targets = node.targets
            elif isinstance(node, ast.AnnAssign):
                targets = [node.target]
            else:
                continue
            if not isinstance(node.value, ast.Dict):
                continue
            if not any(isinstance(t, ast.Name) and t.id == "state" for t in targets):
                continue
            keys = {k.value for k in node.value.keys if isinstance(k, ast.Constant)}
            if "question" not in keys:
                continue
            yield fn, node, keys


def _names_in(fn) -> set[str]:
    return {n.id for n in ast.walk(fn) if isinstance(n, ast.Name)} | {
        n.attr for n in ast.walk(fn) if isinstance(n, ast.Attribute)
    }


def test_every_document_state_carries_a_language():
    """The hop that makes the whole scoping chain live."""
    missing = []
    for path in _ENTRY_FILES:
        for fn, node, keys in _state_dicts(path):
            names = _names_in(fn)
            if not (names & _UNSTRUCTURED_MARKERS):
                continue
            if "language" not in keys:
                missing.append(f"{path.relative_to(REPO)}:{node.lineno} in {fn.name}()")
    assert not missing, (
        "document state built without a language -- these paths silently search "
        "the default corpus:\n  " + "\n  ".join(missing)
    )


def test_purely_structured_state_is_left_alone():
    """Requiring a language on the business graph would be requiring a lie.

    Asserted rather than merely omitted, so that a later blanket sweep that
    "helpfully" adds language everywhere fails here and has to justify
    itself.
    """
    for path in _ENTRY_FILES:
        for fn, node, keys in _state_dicts(path):
            names = _names_in(fn)
            if (names & _STRUCTURED_MARKERS) and not (names & _UNSTRUCTURED_MARKERS):
                assert "language" not in keys, (
                    f"{path.relative_to(REPO)}:{node.lineno} in {fn.name}() is a "
                    "structured-only state and must not carry a language"
                )


def test_every_document_entry_point_accepts_a_language():
    """A state key set from a name the function never received is a NameError
    that no unit test would necessarily reach."""
    broken = []
    for path in _ENTRY_FILES:
        for fn, node, keys in _state_dicts(path):
            if "language" not in keys:
                continue
            params = {a.arg for a in fn.args.args} | {a.arg for a in fn.args.kwonlyargs}
            assigned = {
                n.id for n in ast.walk(fn)
                if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)
            }
            if "language" not in params | assigned:
                broken.append(f"{path.relative_to(REPO)}:{fn.lineno} {fn.name}()")
    assert not broken, "sets state['language'] without having one:\n  " + "\n  ".join(broken)


def test_the_guard_is_covering_something():
    """A guard that matches nothing passes forever."""
    found = sum(
        1
        for path in _ENTRY_FILES
        for fn, _, keys in _state_dicts(path)
        if "language" in keys
    )
    assert found >= 5, f"expected several document state builders, found {found}"
