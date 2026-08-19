"""Cypher helpers for tenant-scoped retrieval. Mirrors versioning.py's lifecycle_active()."""

from ..config.settings import MULTI_TENANCY_ENABLED


def tenant_filter(alias: str = "n", param: str = "$tenant_id") -> str:
    """
    A WHERE-clause conjunct enforcing tenant isolation on `alias`.

    Degrades to a harmless "true" when MULTI_TENANCY_ENABLED is off, so every
    call site can splice `AND {tenant_filter(...)}` unconditionally — the same
    idiom as lifecycle_active()'s legacy-node handling — with zero behavior
    change for single-tenant deployments.
    """
    if not MULTI_TENANCY_ENABLED:
        return "true"
    return f"{alias}.tenant_id = {param}"
