"""
tests/test_graph_tenancy_unit.py — graph/tenancy.py's tenant_filter() helper.

Mirrors graph/versioning.py's lifecycle_active() idiom: degrades to a
harmless "true" when MULTI_TENANCY_ENABLED is off, so every call site can
splice `AND {tenant_filter(...)}` unconditionally with zero behavior change
for single-tenant deployments.

Run with:
    python -m pytest tests/test_graph_tenancy_unit.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path


import src.shared.neo4j.tenancy as tenancy_mod
from src.shared.neo4j.tenancy import tenant_filter


def test_tenant_filter_is_noop_when_disabled(monkeypatch):
    monkeypatch.setattr(tenancy_mod, "MULTI_TENANCY_ENABLED", False)
    assert tenant_filter("n") == "true"


def test_tenant_filter_enforces_when_enabled(monkeypatch):
    monkeypatch.setattr(tenancy_mod, "MULTI_TENANCY_ENABLED", True)
    assert tenant_filter("n") == "n.tenant_id = $tenant_id"


def test_tenant_filter_uses_given_alias_and_param(monkeypatch):
    monkeypatch.setattr(tenancy_mod, "MULTI_TENANCY_ENABLED", True)
    assert tenant_filter("seed", "$tid") == "seed.tenant_id = $tid"
