"""OIDC / JWT authentication configuration (Google default, generic OIDC supported)."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Optional

from ..roles import Role


def _bool(key: str, default: str = "false") -> bool:
    return os.environ.get(key, default).lower() in ("1", "true", "yes")


def _parse_email_role_map(raw: str) -> dict[str, Role]:
    out: dict[str, Role] = {}
    for part in (raw or "").split(","):
        piece = part.strip()
        if not piece or "=" not in piece:
            continue
        email, role_name = piece.split("=", 1)
        email = email.strip().lower()
        role_name = role_name.strip().lower()
        try:
            out[email] = Role(role_name)
        except ValueError:
            continue
    return out


def _parse_claim_role_map(raw: str) -> dict[str, Role]:
    """Map IdP group/role claim values → RAG Role. JSON or comma k=v pairs."""
    raw = (raw or "").strip()
    if not raw:
        return {}
    if raw.startswith("{"):
        try:
            data = json.loads(raw)
            return {str(k): Role(str(v)) for k, v in data.items()}
        except (json.JSONDecodeError, ValueError):
            return {}
    out: dict[str, Role] = {}
    for part in raw.split(","):
        if "=" not in part:
            continue
        claim_val, role_name = part.split("=", 1)
        try:
            out[claim_val.strip()] = Role(role_name.strip().lower())
        except ValueError:
            continue
    return out


@dataclass(frozen=True)
class OidcAuthConfig:
    enabled: bool
    provider: str  # google | oidc
    google_client_id: str
    oidc_issuer: str
    oidc_audience: str
    oidc_jwks_url: str
    role_claim: str
    default_role: Role
    email_role_map: dict[str, Role]
    claim_role_map: dict[str, Role]
    trusted_email_domains: tuple[str, ...]
    domain_default_role: Optional[Role]
    jit_provision: bool
    allow_body_fallback: bool
    google_issuers: tuple[str, ...] = (
        "https://accounts.google.com",
        "accounts.google.com",
    )

    @property
    def jwks_url(self) -> str:
        if self.provider == "google":
            return "https://www.googleapis.com/oauth2/v3/certs"
        if self.oidc_jwks_url:
            return self.oidc_jwks_url
        issuer = self.oidc_issuer.rstrip("/")
        return f"{issuer}/.well-known/jwks.json"

    @property
    def audiences(self) -> tuple[str, ...]:
        if self.provider == "google" and self.google_client_id:
            return (self.google_client_id,)
        if self.oidc_audience:
            return (self.oidc_audience,)
        return ()

    def issuers(self) -> tuple[str, ...]:
        if self.provider == "google":
            return self.google_issuers
        if self.oidc_issuer:
            return (self.oidc_issuer.rstrip("/"),)
        return ()


def load_oidc_config() -> OidcAuthConfig:
    provider = (os.environ.get("AUTH_PROVIDER") or "google").strip().lower()
    default_role_name = (os.environ.get("AUTH_DEFAULT_ROLE") or "public").strip().lower()
    try:
        default_role = Role(default_role_name)
    except ValueError:
        default_role = Role.PUBLIC

    domain_role_raw = (os.environ.get("AUTH_DOMAIN_DEFAULT_ROLE") or "").strip().lower()
    domain_default_role = None
    if domain_role_raw:
        try:
            domain_default_role = Role(domain_role_raw)
        except ValueError:
            domain_default_role = None

    enabled = _bool("AUTH_ENABLED", "false")
    google_client_id = (os.environ.get("GOOGLE_CLIENT_ID") or "").strip()

    return OidcAuthConfig(
        enabled=enabled,
        provider=provider,
        google_client_id=google_client_id,
        oidc_issuer=(os.environ.get("OIDC_ISSUER") or "").strip(),
        oidc_audience=(os.environ.get("OIDC_AUDIENCE") or "").strip(),
        oidc_jwks_url=(os.environ.get("OIDC_JWKS_URL") or "").strip(),
        role_claim=(os.environ.get("AUTH_ROLE_CLAIM") or "groups").strip(),
        default_role=default_role,
        email_role_map=_parse_email_role_map(os.environ.get("AUTH_EMAIL_ROLE_MAP", "")),
        claim_role_map=_parse_claim_role_map(os.environ.get("AUTH_CLAIM_ROLE_MAP", "")),
        trusted_email_domains=tuple(
            d.strip().lower()
            for d in (os.environ.get("AUTH_TRUSTED_EMAIL_DOMAINS") or "").split(",")
            if d.strip()
        ),
        domain_default_role=domain_default_role,
        jit_provision=_bool("AUTH_JIT_PROVISION", "true"),
        allow_body_fallback=_bool("AUTH_ALLOW_BODY_FALLBACK", "false" if enabled else "true"),
    )
