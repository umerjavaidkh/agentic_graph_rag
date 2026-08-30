"""
tests/unstructured/test_name_index_tenancy_unit.py — VectorFirstHybridStrategy's
in-process document-name index is scoped to one tenant.

`_named_document` matches a question against this index and returns whatever
document its title or logical id matches. An unscoped index therefore hands
one tenant a document belonging to another; and because the index is cached
for the process lifetime, a cache that is not keyed by tenant leaks the same
way even once the query itself is filtered.

Run with:
    python -m pytest tests/unstructured/test_name_index_tenancy_unit.py -v
"""
from __future__ import annotations

import src.shared.neo4j.tenancy as tenancy_mod
from src.unstructured.retrieval.strategies.vector_first_hybrid import (
    VectorFirstHybridStrategy,
)


class _Row(dict):
    pass


def _strategy(rows_by_tenant):
    """A bare strategy whose only Neo4j behaviour is the name-index read."""
    s = object.__new__(VectorFirstHybridStrategy)
    s._names = {}
    calls = []

    class _Session:
        def run(self, cypher, **params):
            calls.append((cypher, params))
            tid = params.get("tenant_id")
            if not tenancy_mod.MULTI_TENANCY_ENABLED:
                return [r for rows in rows_by_tenant.values() for r in rows]
            return list(rows_by_tenant.get(tid, []))

    s._neo4j_session_call = lambda fn, *a, **kw: fn(_Session(), *a, **kw)
    return s, calls


ROWS = {
    "acme": [_Row(id="doc_acme_10k_2024", title="Acme Annual Report")],
    "globex": [_Row(id="doc_globex_10k_2024", title="Globex Annual Report")],
}


def test_query_carries_the_tenant_filter_and_parameter(monkeypatch):
    monkeypatch.setattr(tenancy_mod, "MULTI_TENANCY_ENABLED", True)
    s, calls = _strategy(ROWS)
    s._name_index("acme")

    cypher, params = calls[0]
    assert "dl.tenant_id = $tenant_id" in cypher
    assert params["tenant_id"] == "acme"


def test_another_tenants_document_is_not_in_the_index(monkeypatch):
    monkeypatch.setattr(tenancy_mod, "MULTI_TENANCY_ENABLED", True)
    s, _ = _strategy(ROWS)

    ids = {doc_id for doc_id, _, _ in s._name_index("acme")}
    assert ids == {"doc_acme_10k_2024"}


def test_a_named_document_does_not_resolve_across_tenants(monkeypatch):
    monkeypatch.setattr(tenancy_mod, "MULTI_TENANCY_ENABLED", True)
    s, _ = _strategy(ROWS)

    assert s._named_document("acme", "what is in the Acme Annual Report?") == (
        "doc_acme_10k_2024"
    )
    # The title belongs to globex; acme must not be handed it.
    assert s._named_document("acme", "what is in the Globex Annual Report?") is None


def test_the_cache_does_not_serve_one_tenants_corpus_to_another(monkeypatch):
    monkeypatch.setattr(tenancy_mod, "MULTI_TENANCY_ENABLED", True)
    s, calls = _strategy(ROWS)

    s._name_index("acme")
    globex = {doc_id for doc_id, _, _ in s._name_index("globex")}

    assert globex == {"doc_globex_10k_2024"}
    assert len(calls) == 2, "each tenant needs its own read, not the first one's rows"


def test_the_index_is_still_read_once_per_tenant(monkeypatch):
    monkeypatch.setattr(tenancy_mod, "MULTI_TENANCY_ENABLED", True)
    s, calls = _strategy(ROWS)

    s._name_index("acme")
    s._name_index("acme")

    assert len(calls) == 1


def test_single_tenant_deployments_are_unaffected(monkeypatch):
    monkeypatch.setattr(tenancy_mod, "MULTI_TENANCY_ENABLED", False)
    s, calls = _strategy(ROWS)

    ids = {doc_id for doc_id, _, _ in s._name_index("")}

    assert ids == {"doc_acme_10k_2024", "doc_globex_10k_2024"}
    assert "tenant_id" not in calls[0][0]
