"""tests/test_ingestion_cache_invalidation_unit.py — ingestion completion
busts the structured-query-path caches.

Covers the fix for two dead invalidation hooks: SchemaProvider.clear_cache()
and routing.clear_structured_entity_cache() both existed but were never
called anywhere, so a completed ingestion job's new node/relationship types
stayed invisible to the structured (Text-to-Cypher) path's cached schema and
entity summary until the next process restart.

Run with:
    python -m pytest tests/test_ingestion_cache_invalidation_unit.py -v
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


for _mod_name in list(sys.modules):
    if _mod_name.startswith("src.pipeline.ingestion") or _mod_name.startswith("src.unstructured.document"):
        del sys.modules[_mod_name]

# Always create a fresh, private stub for these and overwrite whatever's in
# sys.modules — never conditionally reuse-then-mutate. fastapi in particular
# is a real installed dependency: "stub only if absent, then unconditionally
# set an attribute" would patch a MagicMock onto the REAL fastapi module's
# UploadFile class in place if some other test file already real-imported
# it first, corrupting it for every test that runs afterward in the same
# process — an import-order-dependent bug (mirrors the one this file itself
# used to have with src.shared.auth.rbac_setup, caught via full-suite runs, not
# single-file ones).
_stub_module("neo4j").GraphDatabase = MagicMock()
_stub_module("neo4j.exceptions").ClientError = type("ClientError", (Exception,), {"message": "", "code": ""})

_stub_module("fastapi").UploadFile = MagicMock()

_stub_module("src.shared.auth")
# Stubbing the auth PACKAGE without its submodules leaves it without a __path__,
# so a later module importing src.shared.auth.roles fails with "is not a
# package" -- and the file that breaks is an innocent one that stubs nothing.
_auth_roles = _stub_module("src.shared.auth.roles")
_auth_roles.UserContext = MagicMock
_auth_roles.DEFAULT_PUBLIC_CONTEXT = MagicMock(role=MagicMock(value="public"))
_stub_module("src.shared.auth.rbac_setup").GraphRBAC = MagicMock()
_stub_module("src.shared.neo4j.driver").get_neo4j_driver = MagicMock()

_STUBBED_MODULE_NAMES = (
    "neo4j", "neo4j.exceptions", "fastapi",
    "src.shared.auth.rbac_setup", "src.shared.auth",
    "src.shared.neo4j.driver", )


def teardown_module(module) -> None:
    """See test_ingestion_manager_di_unit.py's identical hook for why: this
    file's fakes must not survive into a later-collected test file's
    sys.modules."""
    for _n in _STUBBED_MODULE_NAMES:
        sys.modules.pop(_n, None)


from src.unstructured.ingestion.service import IngestionManager


@pytest.fixture()
def fake_structured_graph(monkeypatch):
    mod = types.ModuleType("src.structured.retrieval.graph")
    fake_retriever = MagicMock()
    mod.retriever = fake_retriever
    monkeypatch.setitem(sys.modules, "src.structured.retrieval.graph", mod)
    return fake_retriever


@pytest.fixture()
def fake_routing(monkeypatch):
    mod = types.ModuleType("src.interface.routing")
    mod.clear_structured_entity_cache = MagicMock()
    monkeypatch.setitem(sys.modules, "src.interface.routing", mod)
    return mod.clear_structured_entity_cache


def test_clears_both_caches_on_success(fake_structured_graph, fake_routing):
    IngestionManager._clear_structured_query_caches()
    fake_structured_graph.clear_schema_cache.assert_called_once()
    fake_routing.assert_called_once()


def test_schema_cache_failure_does_not_block_entity_cache_clear(fake_structured_graph, fake_routing):
    fake_structured_graph.clear_schema_cache.side_effect = RuntimeError("boom")
    IngestionManager._clear_structured_query_caches()  # must not raise
    fake_routing.assert_called_once()


def test_entity_cache_failure_is_swallowed(fake_structured_graph, fake_routing):
    fake_routing.side_effect = RuntimeError("boom")
    IngestionManager._clear_structured_query_caches()  # must not raise
    fake_structured_graph.clear_schema_cache.assert_called_once()


def test_missing_structured_graph_module_does_not_block_entity_cache_clear(monkeypatch, fake_routing):
    monkeypatch.delitem(sys.modules, "src.structured.retrieval.graph", raising=False)
    real_import = __import__

    def _raise_for_structured_graph(name, *args, **kwargs):
        if name == "src.structured.retrieval.graph" or name.endswith("retrieval.structured.graph"):
            raise ImportError("not available in this deployment")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", _raise_for_structured_graph)
    IngestionManager._clear_structured_query_caches()  # must not raise
    fake_routing.assert_called_once()
