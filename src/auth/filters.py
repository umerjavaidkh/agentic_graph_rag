"""
src/auth/filters.py — DEPRECATED: Legacy Python-level filters.

⚠️  This module is deprecated. Use src/auth/rbac_setup.GraphRBAC instead.

Graph-based RBAC (role → knowledge area → document relationships) is now the
primary access control mechanism. All retrievers use GraphRBAC.can_query_knowledge_area()
and GraphRBAC.can_view_document() for access checks.

This file is kept for reference and backwards compatibility.
"""

from typing import List
from .roles import Role, UserContext


class RoleFilter:
    """
    ⚠️  DEPRECATED: Use GraphRBAC from src/auth/rbac_setup instead.
    
    Legacy role-based filter manager. Kept for reference.
    """

    # Sensitivity levels reference (legacy)
    SENSITIVITY_LEVELS = {
        "public": [Role.PUBLIC, Role.REGULAR_OFFICE, Role.COMPLIANCE_OFFICER, Role.ADMIN],
        "internal": [Role.REGULAR_OFFICE, Role.COMPLIANCE_OFFICER, Role.ADMIN],
        "sensitive": [Role.COMPLIANCE_OFFICER, Role.ADMIN],
        "confidential": [Role.ADMIN],
    }

    @staticmethod
    def _get_allowed_sensitivity_levels(user_context: UserContext) -> List[str]:
        """Get list of sensitivity levels accessible to the user (legacy)."""
        sensitivity_list = []
        for level, allowed_roles in RoleFilter.SENSITIVITY_LEVELS.items():
            if user_context.role in allowed_roles:
                sensitivity_list.append(level)
        return sensitivity_list
