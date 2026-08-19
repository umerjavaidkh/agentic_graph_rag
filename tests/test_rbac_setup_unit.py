"""
tests/test_rbac_setup_unit.py — GraphRBAC tenant checks + Cypher parameterization.

Covers two fixes made during the multi-tenancy pass:
1. can_view_document now also requires a tenant_id match (a role that would
   otherwise allow viewing must never grant access to a different tenant's doc).
2. build_cypher_with_access_check/build_document_filter_cypher no longer
   f-string-interpolate user_id into Cypher — they return (cypher, params)
   tuples for parameterized execution.

Run with:
    python -m pytest tests/test_rbac_setup_unit.py -v
"""
from __future__ import annotations

import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest


def _stub_module(name: str) -> types.ModuleType:
    mod = types.ModuleType(name)
    sys.modules[name] = mod
    return mod


if "neo4j" not in sys.modules:
    _stub_module("neo4j")
sys.modules["neo4j"].GraphDatabase = MagicMock()
sys.modules["neo4j"].Driver = object

# A previously-collected test file may have stubbed src.auth/src.auth.rbac_setup
# (MagicMock GraphRBAC) or src.graph (no tenancy submodule) — this file needs
# the REAL classes/modules. Only clear entries that are actually fake stubs
# (a hand-built types.ModuleType has neither __file__ nor __path__); if another
# file already imported the real package, reuse that exact module object
# instead of reimporting — a second, divergent import would leave other test
# files' already-bound references (and any unittest.mock.patch targeting
# sys.modules by name) pointing at two different module objects.
for _mod_name in list(sys.modules):
    if _mod_name.startswith("src.auth") or _mod_name.startswith("src.graph"):
        _mod = sys.modules[_mod_name]
        if getattr(_mod, "__file__", None) is None and getattr(_mod, "__path__", None) is None:
            del sys.modules[_mod_name]

from src.auth.rbac_setup import GraphRBAC


class _FakeResult:
    def __init__(self, single_value):
        self._single_value = single_value

    def single(self):
        return self._single_value


class _FakeSession:
    def __init__(self, single_value=None):
        self._single_value = single_value
        self.calls: list[dict] = []

    def run(self, cypher, **params):
        self.calls.append({"cypher": cypher, "params": params})
        return _FakeResult(self._single_value)

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class _FakeDriver:
    def __init__(self, session):
        self._session = session

    def session(self):
        return self._session


def _rbac_with_session(session) -> GraphRBAC:
    return GraphRBAC(driver=_FakeDriver(session))


def test_can_view_document_binds_tenant_id_as_param():
    session = _FakeSession({"has_access": True})
    rbac = _rbac_with_session(session)

    result = rbac.can_view_document("user_1", "doc_1", tenant_id="tenant_a")

    assert result is True
    assert session.calls[0]["params"]["tenant_id"] == "tenant_a"
    assert session.calls[0]["params"]["doc_id"] == "doc_1"
    assert session.calls[0]["params"]["user_id"] == "user_1"
    # user_id must never be string-interpolated into the query text.
    assert "user_1" not in session.calls[0]["cypher"]


def test_can_view_document_defaults_tenant_id_to_empty_string():
    session = _FakeSession({"has_access": False})
    rbac = _rbac_with_session(session)

    rbac.can_view_document("user_1", "doc_1")

    assert session.calls[0]["params"]["tenant_id"] == ""


def test_build_cypher_with_access_check_returns_parameterized_tuple():
    session = _FakeSession({"has_access": True})
    rbac = _rbac_with_session(session)

    result = rbac.build_cypher_with_access_check(
        "user_123", "structured", "MATCH (n:Product) RETURN n LIMIT 10"
    )

    assert result is not None
    cypher, params = result
    assert params == {"user_id": "user_123", "ka_id": "structured"}
    # The raw values must never be spliced directly into the Cypher text.
    assert "user_123" not in cypher
    assert "structured" not in cypher
    assert "$user_id" in cypher
    assert "$ka_id" in cypher


def test_build_cypher_with_access_check_returns_none_when_denied():
    session = _FakeSession({"has_access": False})
    rbac = _rbac_with_session(session)

    result = rbac.build_cypher_with_access_check(
        "user_123", "structured", "MATCH (n:Product) RETURN n LIMIT 10"
    )
    assert result is None


def test_build_document_filter_cypher_returns_parameterized_tuple():
    session = _FakeSession()
    rbac = _rbac_with_session(session)

    clause, params = rbac.build_document_filter_cypher("user_123")

    assert params == {"user_id": "user_123"}
    assert "user_123" not in clause
    assert "$user_id" in clause
