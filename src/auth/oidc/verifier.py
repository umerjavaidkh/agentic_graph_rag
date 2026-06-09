"""JWT verification via JWKS (Google or any OIDC issuer)."""
from __future__ import annotations

import logging
from typing import Any, Optional

import jwt
from jwt import PyJWKClient

from .config import OidcAuthConfig

logger = logging.getLogger(__name__)

_jwk_clients: dict[str, PyJWKClient] = {}


def _get_jwk_client(jwks_url: str) -> PyJWKClient:
    client = _jwk_clients.get(jwks_url)
    if client is None:
        client = PyJWKClient(jwks_url, cache_keys=True)
        _jwk_clients[jwks_url] = client
    return client


class AuthenticationError(Exception):
    """Invalid or missing bearer token."""


def verify_bearer_token(token: str, cfg: OidcAuthConfig) -> dict[str, Any]:
    """
    Verify RS256 OIDC JWT and return claims.

    Raises AuthenticationError on failure.
    """
    if not token or not token.strip():
        raise AuthenticationError("Missing bearer token.")

    audiences = cfg.audiences
    if not audiences:
        raise AuthenticationError(
            "Auth is enabled but audience is not configured "
            "(set GOOGLE_CLIENT_ID or OIDC_AUDIENCE)."
        )

    try:
        jwk_client = _get_jwk_client(cfg.jwks_url)
        signing_key = jwk_client.get_signing_key_from_jwt(token)
        claims = jwt.decode(
            token,
            signing_key.key,
            algorithms=["RS256"],
            options={
                "verify_aud": False,
                "verify_iss": False,
                "verify_exp": True,
            },
        )
    except jwt.ExpiredSignatureError as exc:
        raise AuthenticationError("Token expired.") from exc
    except jwt.InvalidTokenError as exc:
        raise AuthenticationError("Invalid token.") from exc
    except Exception as exc:
        logger.warning("JWT verification error: %s", exc, exc_info=True)
        raise AuthenticationError("Token verification failed.") from exc

    iss = str(claims.get("iss") or "")
    allowed_issuers = cfg.issuers()
    if allowed_issuers and iss not in allowed_issuers and iss.rstrip("/") not in allowed_issuers:
        raise AuthenticationError("Invalid token issuer.")

    aud = claims.get("aud")
    aud_list: list[str] = []
    if isinstance(aud, str):
        aud_list = [aud]
    elif isinstance(aud, (list, tuple)):
        aud_list = [str(a) for a in aud]

    if not any(a in audiences for a in aud_list):
        raise AuthenticationError("Invalid token audience.")

    return claims


def extract_bearer_token(authorization: Optional[str]) -> Optional[str]:
    if not authorization:
        return None
    parts = authorization.strip().split(None, 1)
    if len(parts) != 2 or parts[0].lower() != "bearer":
        return None
    return parts[1].strip() or None
