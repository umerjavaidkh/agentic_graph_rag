"""
tests/unstructured/test_language_scope_isolation_unit.py — the ways a document
can be reached that are NOT a scoped Cypher query.

Splicing `language_filter()` into 38 queries closed every path that goes
through Cypher. Three did not, and each was found only by running two
languages against a live corpus:

  * the vector scoping pass filtered the ANN by tenant but not language
  * the document-name index is built once per process and matched
    in-memory, so it never saw a WHERE clause at all
  * the scope cache was keyed by (tenant, query) -- so the FIRST caller's
    language decided the answer for every later caller asking the same
    words, and the second language got the first one's document back with
    zero chunks: correctly scoped everywhere except in the memo sitting in
    front of the scoping

The last one is the reason this file exists. A cache keyed by less than
the thing it is caching is invisible to every test that calls once, and
the failure it produces reads as a retrieval bug rather than a caching
one.

Run with:
    python -m pytest tests/unstructured/test_language_scope_isolation_unit.py -v
"""
from __future__ import annotations

import ast
import inspect
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
_STRATEGY = REPO / "src" / "unstructured" / "retrieval" / "strategies" / "vector_first_hybrid.py"
_CANDIDATES = REPO / "src" / "unstructured" / "retrieval" / "services" / "candidate_docs.py"


def _fn(path: Path, name: str) -> ast.FunctionDef:
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"{name} not found in {path.name}")


def _src(path: Path, name: str) -> str:
    return ast.get_source_segment(path.read_text(), _fn(path, name)) or ""


def test_the_scope_cache_is_keyed_by_language():
    """The bug this file is named for.

    `cache_key = (tenant_id, query)` let an Arabic question cache its
    document and an identical English-scoped question read it back. Both
    scopes are part of what the answer depends on, so both belong in the
    key.
    """
    body = _src(_STRATEGY, "_scope_for_query")
    assert "cache_key" in body, "the scope cache moved; re-point this test"
    line = next(l for l in body.splitlines() if "cache_key = " in l)
    assert "language" in line, f"scope cache key is not language-aware: {line.strip()}"
    assert "tenant_id" in line, f"scope cache key lost its tenant scope: {line.strip()}"


def test_the_document_name_index_carries_a_language():
    """Naming is the one path that never reaches a WHERE clause.

    The index is built once per process and matched in memory, so a
    language predicate spliced into Cypher cannot reach it, and a question
    in one language resolved a document in the other by title alone.
    """
    body = _src(_STRATEGY, "_name_index")
    assert "language" in body, "the name index does not select a language"
    matcher = _src(_STRATEGY, "_named_document")
    assert "language" in matcher, "the name matcher does not filter by language"


def test_every_document_resolution_call_passes_a_language():
    """A resolver method whose language argument has a default is silently
    scoped to the default language when the caller omits it -- which is
    wrong in the language that is not the default, and invisible in the
    one that is."""
    body = _src(_STRATEGY, "_scope_for_query")
    tree = ast.parse(body.strip())
    missing = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        target = ast.unparse(node.func)
        if "_document_resolver" not in target:
            continue
        args = {ast.unparse(a) for a in node.args} | {
            (k.arg or "") for k in node.keywords
        }
        if "language" not in args:
            missing.append(target)
    assert not missing, f"resolver calls with no language: {missing}"


def test_the_vector_scoping_pass_filters_on_language():
    """Filtered at ANN time, not after.

    probe_k is a fixed 200. A language-blind probe spends that budget on
    the wrong language as the second corpus grows, and post-filtering
    cannot recover chunks that were never fetched.
    """
    body = _src(_CANDIDATES, "candidates")
    assert 'filters["language"]' in body, "the ANN query is not language-filtered"
    assert "language" in {
        a.arg for a in _fn(_CANDIDATES, "candidates").args.args
    } | {a.arg for a in _fn(_CANDIDATES, "candidates").args.kwonlyargs}


def test_the_vector_store_can_migrate_a_payload():
    """Adding a scoping field to points already written must not require
    recomputing their embeddings, and must exist on every backend rather
    than being reached around on one."""
    from src.shared.storage.vector.base import VectorStore
    from src.shared.storage.vector.memory_store import InMemoryVectorStore

    assert hasattr(VectorStore, "set_payload_by_filter")

    store = InMemoryVectorStore()
    store.upsert("a", [0.1, 0.2], metadata={"logical_doc_id": "doc_x"})
    store.upsert("b", [0.3, 0.4], metadata={"logical_doc_id": "doc_y"})

    updated = store.set_payload_by_filter({"logical_doc_id": ["doc_x"]}, {"language": "ar"})
    assert updated == 1
    assert store._metadata["a"]["language"] == "ar"
    assert "language" not in store._metadata["b"]
