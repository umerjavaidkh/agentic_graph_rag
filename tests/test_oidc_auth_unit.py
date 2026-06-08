"""Unit tests for OIDC claims → UserContext mapping."""
from __future__ import annotations

import unittest

from src.auth.oidc.claims import build_user_context, parse_verified_claims
from src.auth.oidc.config import OidcAuthConfig
from src.auth.roles import Role


class TestOidcClaims(unittest.TestCase):
    def _cfg(self, **kwargs) -> OidcAuthConfig:
        base = dict(
            enabled=True,
            provider="google",
            google_client_id="test.apps.googleusercontent.com",
            oidc_issuer="",
            oidc_audience="",
            oidc_jwks_url="",
            role_claim="groups",
            default_role=Role.PUBLIC,
            email_role_map={},
            claim_role_map={},
            trusted_email_domains=(),
            domain_default_role=None,
            jit_provision=False,
            allow_body_fallback=False,
        )
        base.update(kwargs)
        return OidcAuthConfig(**base)

    def test_parse_google_like_claims(self):
        verified = parse_verified_claims({
            "sub": "12345",
            "email": "Alice@Corp.COM",
            "name": "Alice",
            "iss": "https://accounts.google.com",
        })
        self.assertEqual(verified.user_id, "12345")
        self.assertEqual(verified.email, "alice@corp.com")

    def test_email_role_map(self):
        cfg = self._cfg(email_role_map={"alice@corp.com": Role.REGULAR_OFFICE})
        ctx = build_user_context(
            {"sub": "x", "email": "alice@corp.com"},
            cfg=cfg,
            rbac=None,
        )
        self.assertEqual(ctx.role, Role.REGULAR_OFFICE)

    def test_claim_group_map(self):
        cfg = self._cfg(claim_role_map={"RAG-Admin": Role.ADMIN})
        ctx = build_user_context(
            {"sub": "x", "groups": ["RAG-Admin"]},
            cfg=cfg,
            rbac=None,
        )
        self.assertEqual(ctx.role, Role.ADMIN)

    def test_default_role(self):
        cfg = self._cfg(default_role=Role.PUBLIC)
        ctx = build_user_context({"sub": "new-user"}, cfg=cfg, rbac=None)
        self.assertEqual(ctx.role, Role.PUBLIC)


if __name__ == "__main__":
    unittest.main()
