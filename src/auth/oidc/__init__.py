"""OIDC/JWT authentication — Google by default, corporate IdP via OIDC_* env."""

from .claims import VerifiedClaims, build_user_context
from .config import OidcAuthConfig, load_oidc_config
from .deps import (
    AuthSession,
    auth_public_config,
    get_oidc_config,
    require_admin_session,
    require_bearer_session,
    resolve_scoped_thread_id,
    resolve_user_context,
)
from .verifier import AuthenticationError, verify_bearer_token

__all__ = [
    "AuthSession",
    "AuthenticationError",
    "OidcAuthConfig",
    "VerifiedClaims",
    "auth_public_config",
    "build_user_context",
    "get_oidc_config",
    "load_oidc_config",
    "require_admin_session",
    "require_bearer_session",
    "resolve_scoped_thread_id",
    "resolve_user_context",
    "verify_bearer_token",
]
