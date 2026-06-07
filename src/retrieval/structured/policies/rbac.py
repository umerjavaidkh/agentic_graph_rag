"""RBAC cache for structured knowledge area."""
from __future__ import annotations

from ....auth.rbac_setup import GraphRBAC


class StructuredRbac:
    def __init__(self, rbac: GraphRBAC):
        self._rbac = rbac
        self._cache: dict[tuple[str, str], bool] = {}

    def can_query(self, user_id: str) -> bool:
        key = (user_id, "structured")
        if key not in self._cache:
            self._cache[key] = bool(self._rbac.can_query_knowledge_area(user_id, "structured"))
        return self._cache[key]
