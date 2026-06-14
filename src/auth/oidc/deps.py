"""FastAPI dependencies: resolve trusted UserContext from JWT or dev fallback."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from fastapi import Header, HTTPException

from ..rbac_setup import GraphRBAC
from ..thread_scope import scoped_thread_id
from ..roles import Role, UserContext, validate_role
from ...config.settings import NEO4J_PASSWORD, NEO4J_URI, NEO4J_USER
from .claims import VerifiedClaims, build_user_context, parse_verified_claims
from .config import OidcAuthConfig, load_oidc_config
from .verifier import AuthenticationError, extract_bearer_token, verify_bearer_token

logger = logging.getLogger(__name__)

_rbac: Optional[GraphRBAC] = None
_config: Optional[OidcAuthConfig] = None


def get_oidc_config() -> OidcAuthConfig:
    global _config
    if _config is None:
        _config = load_oidc_config()
    return _config


def _get_rbac() -> GraphRBAC:
    global _rbac
    if _rbac is None:
        _rbac = GraphRBAC(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
    return _rbac


@dataclass
class AuthSession:
    """Resolved principal for a request."""

    user: UserContext
    auth_mode: str  # jwt | body_fallback | anonymous
    claims: Optional[VerifiedClaims] = None


def require_bearer_session(
    *,
    authorization: Optional[str] = None,
) -> AuthSession:
    """
    Require a valid Bearer JWT. Body user_id/role is never accepted.

    Used for ingestion and other privileged operations.
    """
    cfg = get_oidc_config()
    if not cfg.enabled:
        raise HTTPException(
            status_code=401,
            detail="Authentication is required for this operation (set AUTH_ENABLED=true).",
        )

    token = extract_bearer_token(authorization)
    if not token:
        raise HTTPException(
            status_code=401,
            detail="Sign in with Google and send Authorization: Bearer <token>.",
        )

    try:
        claims_dict = verify_bearer_token(token, cfg)
        verified = parse_verified_claims(claims_dict)
        user = build_user_context(claims_dict, cfg=cfg, rbac=_get_rbac())
        return AuthSession(user=user, auth_mode="jwt", claims=verified)
    except AuthenticationError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc


def resolve_admin_session(
    *,
    authorization: Optional[str] = None,
    body_user_id: Optional[str] = None,
    body_role: Optional[str] = None,
) -> AuthSession:
    """
    Admin principal for ingestion and /admin/*.

    - AUTH_ENABLED + Bearer JWT: verified token, admin role required.
    - AUTH_ENABLED=false (default branch): body user_id + role=admin (dev sidebar).
    """
    cfg = get_oidc_config()
    token = extract_bearer_token(authorization)

    if token:
        try:
            claims_dict = verify_bearer_token(token, cfg)
            verified = parse_verified_claims(claims_dict)
            user = build_user_context(claims_dict, cfg=cfg, rbac=_get_rbac())
            if not user.has_role(Role.ADMIN):
                raise HTTPException(
                    status_code=403,
                    detail="Admin role required. Ingestion is limited to mapped admin accounts.",
                )
            return AuthSession(user=user, auth_mode="jwt", claims=verified)
        except AuthenticationError as exc:
            raise HTTPException(status_code=401, detail=str(exc)) from exc

    if cfg.enabled:
        raise HTTPException(
            status_code=401,
            detail="Authentication required. Sign in and send Authorization: Bearer <token>.",
        )

    try:
        role = validate_role(body_role or "public")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if role != Role.ADMIN:
        raise HTTPException(
            status_code=403,
            detail="Admin role required. Set role=admin on the upload page (dev mode).",
        )
    user = UserContext(
        user_id=(body_user_id or "admin_001").strip(),
        role=role,
    )
    return AuthSession(user=user, auth_mode="body_fallback")


def require_admin_session(
    *,
    authorization: Optional[str] = None,
) -> AuthSession:
    """JWT-only admin (legacy). Prefer resolve_admin_session for ingest routes."""
    return resolve_admin_session(authorization=authorization)


def resolve_user_context(
    *,
    authorization: Optional[str] = None,
    body_user_id: Optional[str] = None,
    body_role: Optional[str] = None,
    body_department: Optional[str] = None,
) -> AuthSession:
    """
    Build UserContext for /query and related endpoints.

    When AUTH_ENABLED: requires valid Bearer JWT unless AUTH_ALLOW_BODY_FALLBACK.
    When disabled: uses body fields (eval / local dev).
    """
    cfg = get_oidc_config()
    token = extract_bearer_token(authorization)

    if token:
        try:
            claims_dict = verify_bearer_token(token, cfg)
            verified = parse_verified_claims(claims_dict)
            user = build_user_context(claims_dict, cfg=cfg, rbac=_get_rbac())
            return AuthSession(user=user, auth_mode="jwt", claims=verified)
        except AuthenticationError as exc:
            raise HTTPException(status_code=401, detail=str(exc)) from exc

    if cfg.enabled and not cfg.allow_body_fallback:
        raise HTTPException(
            status_code=401,
            detail="Authentication required. Sign in and send Authorization: Bearer <token>.",
        )

    if body_user_id or body_role or not cfg.enabled:
        try:
            role = validate_role(body_role or "public")
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        user = UserContext(
            user_id=(body_user_id or "public_001").strip(),
            role=role,
            department=body_department,
        )
        mode = "body_fallback" if cfg.enabled else "anonymous"
        return AuthSession(user=user, auth_mode=mode)

    raise HTTPException(status_code=401, detail="Authentication required.")


def resolve_scoped_thread_id(
    session: AuthSession,
    client_thread_id: Optional[str] = None,
) -> str:
    """Namespace follow-up memory to the resolved user (JWT or dev body user)."""
    return scoped_thread_id(session.user.user_id, client_thread_id)


def auth_public_config() -> dict:
    """Non-secret config for the chat UI."""
    cfg = get_oidc_config()
    return {
        "enabled": cfg.enabled,
        "provider": cfg.provider,
        "google_client_id": cfg.google_client_id if cfg.provider == "google" else "",
        "allow_body_fallback": cfg.allow_body_fallback,
        "default_role": cfg.default_role.value,
    }
