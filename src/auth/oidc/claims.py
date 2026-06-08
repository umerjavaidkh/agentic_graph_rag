"""Verified JWT claims → UserContext (same shape for Google or corporate OIDC)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from ..rbac_setup import GraphRBAC
from ..roles import Role, UserContext
from .config import OidcAuthConfig
from .provision import ensure_user_in_graph


@dataclass(frozen=True)
class VerifiedClaims:
    user_id: str
    email: Optional[str]
    name: Optional[str]
    department: Optional[str]
    groups: tuple[str, ...]
    issuer: str
    subject: str


def _claim_list(claims: dict[str, Any], key: str) -> list[str]:
    raw = claims.get(key)
    if raw is None:
        return []
    if isinstance(raw, str):
        return [raw]
    if isinstance(raw, (list, tuple)):
        return [str(x) for x in raw if x is not None]
    return []


def parse_verified_claims(claims: dict[str, Any]) -> VerifiedClaims:
    sub = str(claims.get("sub") or "").strip()
    email = (claims.get("email") or claims.get("preferred_username") or "").strip() or None
    name = (claims.get("name") or "").strip() or None
    department = (claims.get("department") or claims.get("hd") or "").strip() or None
    groups = tuple(
        _claim_list(claims, "groups")
        + _claim_list(claims, "roles")
        + _claim_list(claims, "role")
    )
    issuer = str(claims.get("iss") or "")
    user_id = sub or email or "unknown"
    return VerifiedClaims(
        user_id=user_id,
        email=email.lower() if email else None,
        name=name,
        department=department,
        groups=groups,
        issuer=issuer,
        subject=sub,
    )


_ROLE_RANK = {
    Role.PUBLIC: 0,
    Role.REGULAR_OFFICE: 1,
    Role.COMPLIANCE_OFFICER: 2,
    Role.ADMIN: 3,
}


def _highest_role(candidates: list[Role]) -> Role:
    if not candidates:
        return Role.PUBLIC
    return max(candidates, key=lambda r: _ROLE_RANK.get(r, 0))


def _role_from_graph(rbac: GraphRBAC, user_id: str) -> Optional[Role]:
    names = rbac.get_user_roles(user_id)
    roles: list[Role] = []
    for name in names:
        try:
            roles.append(Role(str(name).lower()))
        except ValueError:
            continue
    if not roles:
        return None
    return _highest_role(roles)


def _role_from_claims(cfg: OidcAuthConfig, verified: VerifiedClaims) -> Role:
    if verified.email and verified.email in cfg.email_role_map:
        return cfg.email_role_map[verified.email]

    if verified.email and cfg.trusted_email_domains and cfg.domain_default_role:
        domain = verified.email.split("@")[-1]
        if domain in cfg.trusted_email_domains:
            return cfg.domain_default_role

    mapped: list[Role] = []
    for group in verified.groups:
        if group in cfg.claim_role_map:
            mapped.append(cfg.claim_role_map[group])
    if mapped:
        return _highest_role(mapped)

    return cfg.default_role


def build_user_context(
    claims: dict[str, Any],
    *,
    cfg: OidcAuthConfig,
    rbac: Optional[GraphRBAC] = None,
) -> UserContext:
    """Map verified OIDC claims to UserContext; optional Neo4j JIT + graph role override."""
    verified = parse_verified_claims(claims)
    role = cfg.default_role
    if rbac is not None:
        graph_role = _role_from_graph(rbac, verified.user_id)
        if graph_role is not None:
            role = graph_role
        else:
            role = _role_from_claims(cfg, verified)
            if cfg.jit_provision:
                ensure_user_in_graph(
                    rbac,
                    user_id=verified.user_id,
                    role=role,
                    email=verified.email,
                    name=verified.name,
                    department=verified.department,
                )
    else:
        role = _role_from_claims(cfg, verified)

    return UserContext(
        user_id=verified.user_id,
        role=role,
        department=verified.department,
    )
