"""Just-in-time User + Role nodes in Neo4j RBAC graph."""
from __future__ import annotations

import logging
from typing import Optional

from ..rbac_setup import GraphRBAC
from ..roles import Role

logger = logging.getLogger(__name__)


def ensure_user_in_graph(
    rbac: GraphRBAC,
    *,
    user_id: str,
    role: Role,
    email: Optional[str] = None,
    name: Optional[str] = None,
    department: Optional[str] = None,
) -> None:
    """MERGE User and HAS_ROLE; idempotent."""
    cypher = """
        MERGE (u:User {user_id: $user_id})
        ON CREATE SET
          u.created_at = timestamp(),
          u.email = $email,
          u.name = $name,
          u.department = $department
        ON MATCH SET
          u.email = coalesce($email, u.email),
          u.name = coalesce($name, u.name),
          u.department = coalesce($department, u.department)
        WITH u
        MATCH (r:Role {name: $role_name})
        MERGE (u)-[:HAS_ROLE]->(r)
    """
    try:
        with rbac.driver.session() as session:
            session.run(
                cypher,
                user_id=user_id,
                email=email,
                name=name,
                department=department,
                role_name=role.value,
            ).consume()
    except Exception:
        logger.warning("JIT user provision failed for %s", user_id, exc_info=True)
