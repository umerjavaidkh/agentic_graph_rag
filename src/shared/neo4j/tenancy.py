"""Cypher helpers for scoped retrieval. Mirrors versioning.py's lifecycle_active()."""

from ..config.settings import MULTI_TENANCY_ENABLED
from ..language import configured_languages


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


def language_filter(alias: str = "n", param: str = "$language") -> str:
    """
    A WHERE-clause conjunct scoping `alias` to one language.

    Same idiom as tenant_filter above, and for the same reason: it degrades
    to a harmless "true" while fewer than two languages are live in this
    deployment, so every scope call site can splice `AND {language_filter()}`
    unconditionally today and pick up real scoping the day a second language
    is enabled — with zero behaviour change until then.

    That is what makes "English is byte-identical" a property of the code
    rather than a claim about the test suite: with one language configured
    this returns a literal that Neo4j folds away, so there is no filter to
    get wrong.

    Property-based, not a traversal. The `(:Language)-[:HAS_DOCUMENT]->` edge
    is the authority for config, cascade and reporting; a scoped query must
    not have to hop to find its own scope. Same denormalisation logical_doc_id
    and lifecycle_status already use.

    For DOCUMENT retrieval only. This lives beside tenant_filter, which the
    structured path does use, so the difference is worth stating: the
    structured business graph has no language dimension. Its labels and
    properties are schema, not prose -- an Arabic question about orders is
    still answered by `MATCH (o:Order)` -- so splicing this into a
    structured query would scope a graph that has nothing to scope.
    """
    if len(configured_languages()) < 2:
        return "true"
    return f"{alias}.language = {param}"
