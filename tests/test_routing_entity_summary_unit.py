"""
tests/test_routing_entity_summary_unit.py — schema-derived router tool descriptions.

Covers the fix for a real "not universal" bug: the router's LLM prompt and
MCP tool descriptions used to hardcode "Northwind business graph" /
"photo credits, photographers" (this repo's demo vocabulary), which would
misroute questions for any deployment with a different schema or document
set. structured_entity_summary() replaces that with a live schema lookup,
filtered to exclude the ingested-document node labels so it only describes
the structured business graph.

Run with:
    python -m pytest tests/test_routing_entity_summary_unit.py -v
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

for _n in ["src.shared.auth", "src.shared.auth.roles"]:
    if _n not in sys.modules:
        _stub_module(_n)
sys.modules["src.shared.auth.roles"].UserContext = MagicMock

if "src.interface.routing" in sys.modules:
    del sys.modules["src.interface.routing"]

import src.interface.routing as routing_mod


@pytest.fixture(autouse=True)
def _reset_cache():
    routing_mod.clear_structured_entity_cache()
    yield
    routing_mod.clear_structured_entity_cache()


def _fake_driver(labels):
    rows = [{"label": l} for l in labels]

    session = MagicMock()
    session.run.return_value = rows
    session.__enter__ = MagicMock(return_value=session)
    session.__exit__ = MagicMock(return_value=False)

    driver = MagicMock()
    driver.session.return_value = session
    return driver


def test_excludes_document_graph_labels(monkeypatch):
    labels = ["Product", "Order", "Customer", "Document", "Section", "Page", "DocumentLogical", "DocRevision"]
    fake_driver = _fake_driver(labels)
    monkeypatch.setattr("src.shared.neo4j.driver.get_neo4j_driver", lambda: fake_driver)

    summary = routing_mod.structured_entity_summary()

    assert "Product" in summary
    assert "Order" in summary
    assert "Customer" in summary
    for doc_label in ("Document", "Section", "Page", "DocumentLogical", "DocRevision"):
        assert doc_label not in summary.split(", ")


def test_falls_back_to_generic_string_on_error(monkeypatch):
    def _boom():
        raise RuntimeError("no db")

    monkeypatch.setattr("src.shared.neo4j.driver.get_neo4j_driver", _boom)

    summary = routing_mod.structured_entity_summary()

    assert summary == "structured graph data"


def test_caches_across_calls(monkeypatch):
    call_count = {"n": 0}

    def _get_driver():
        call_count["n"] += 1
        return _fake_driver(["Product"])

    monkeypatch.setattr("src.shared.neo4j.driver.get_neo4j_driver", _get_driver)

    first = routing_mod.structured_entity_summary()
    second = routing_mod.structured_entity_summary()

    assert first == second
    assert call_count["n"] == 1


def test_no_hardcoded_domain_vocabulary_in_tool_descriptions(monkeypatch):
    monkeypatch.setattr("src.shared.neo4j.driver.get_neo4j_driver", lambda: _fake_driver(["Product", "Order"]))

    tools = routing_mod._build_mcp_route_tools()
    descriptions = " ".join(t["function"]["description"] for t in tools)

    assert "northwind" not in descriptions.lower()
    assert "photographer" not in descriptions.lower()
    assert "whistleblow" not in descriptions.lower()
    assert "Product" in descriptions
