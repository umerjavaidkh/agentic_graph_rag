"""
auth/ — Role-based access control and user context management.

This module provides:
- Role definitions and validation
- User context management
- Graph-based RBAC via Neo4j relationships (primary)
- Legacy role-based filters (deprecated)
"""

from .roles import Role, UserContext, validate_role
from .rbac_setup import GraphRBAC, initialize_rbac_schema
from .filters import RoleFilter

__all__ = ["Role", "UserContext", "validate_role", "GraphRBAC", "initialize_rbac_schema", "RoleFilter"]
