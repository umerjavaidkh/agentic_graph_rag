"""
src/auth/roles.py — Role definitions and user context.
"""

from enum import Enum
from dataclasses import dataclass
from typing import Optional


class Role(str, Enum):
    """Available user roles."""
    COMPLIANCE_OFFICER = "compliance_officer"      # Full access to all sensitive data
    REGULAR_OFFICE = "regular_office"              # Access to general/approved data only
    PUBLIC = "public"                              # Minimal access, public data only
    ADMIN = "admin"                                # Unrestricted access


@dataclass
class UserContext:
    """
    User context for access control.
    
    Attributes:
        user_id: Unique user identifier
        role: User's role (determines access level)
        department: Optional department (for additional filtering)
    """
    user_id: str
    role: Role
    department: Optional[str] = None

    def has_role(self, required_role: Role) -> bool:
        """Check if user has at least the required role level."""
        role_hierarchy = {
            Role.PUBLIC: 0,
            Role.REGULAR_OFFICE: 1,
            Role.COMPLIANCE_OFFICER: 2,
            Role.ADMIN: 3,
        }
        return role_hierarchy.get(self.role, 0) >= role_hierarchy.get(required_role, 0)

    def __repr__(self):
        return f"UserContext(user_id={self.user_id}, role={self.role.value}, dept={self.department})"


def validate_role(role_str: str) -> Role:
    """Validate and convert role string to Role enum."""
    try:
        return Role(role_str.lower())
    except ValueError:
        raise ValueError(f"Invalid role: {role_str}. Must be one of: {', '.join(r.value for r in Role)}")


# Default user contexts for testing
DEFAULT_PUBLIC_CONTEXT = UserContext(user_id="public_user", role=Role.PUBLIC)
DEFAULT_OFFICE_CONTEXT = UserContext(user_id="office_user", role=Role.REGULAR_OFFICE, department="Operations")
DEFAULT_COMPLIANCE_CONTEXT = UserContext(
    user_id="compliance_user",
    role=Role.COMPLIANCE_OFFICER,
    department="Compliance"
)
