"""Unit tests for OIDC claims → UserContext mapping."""
from __future__ import annotations

import sys
import unittest
from unittest.mock import MagicMock, patch

# Drop any stale fake stub a previously-collected test file may have left in
# sys.modules (several test files stub src.shared.auth/neo4j/fastapi as a bare
# types.ModuleType for their own narrow needs and never restore them) — this
# file needs the real packages, including the src.shared.auth.oidc subpackage and
# neo4j.Driver. A real module always has __file__ or __path__; a hand-built
# stub has neither.
for _mod_name in list(sys.modules):
    if (
        _mod_name == "src.shared.auth"
        or _mod_name.startswith("src.shared.auth.")
        or _mod_name == "src.unstructured.graph"
        or _mod_name.startswith("src.unstructured.graph.")
        or _mod_name in ("neo4j", "neo4j.exceptions", "fastapi")
    ):
        _mod = sys.modules[_mod_name]
        if getattr(_mod, "__file__", None) is None and getattr(_mod, "__path__", None) is None:
            del sys.modules[_mod_name]

from src.shared.auth.oidc.claims import build_user_context, parse_verified_claims
from src.shared.auth.oidc.config import OidcAuthConfig
from src.shared.auth.oidc.deps import require_admin_session, require_bearer_session
from src.shared.auth.roles import Role, UserContext


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
            default_role=Role.REGULAR_OFFICE,
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

    def test_admin_email_role_map(self):
        cfg = self._cfg(
            email_role_map={"admin@example.com": Role.ADMIN},
            default_role=Role.COMPLIANCE_OFFICER,
        )
        ctx = build_user_context(
            {"sub": "g1", "email": "admin@example.com"},
            cfg=cfg,
            rbac=None,
        )
        self.assertEqual(ctx.role, Role.ADMIN)

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
        cfg = self._cfg(default_role=Role.REGULAR_OFFICE)
        ctx = build_user_context({"sub": "new-user"}, cfg=cfg, rbac=None)
        self.assertEqual(ctx.role, Role.REGULAR_OFFICE)

    def test_jit_sync_overrides_stale_graph_role(self):
        from unittest.mock import MagicMock, patch

        rbac = MagicMock()
        rbac.get_user_roles.return_value = ["regular_office"]
        cfg = self._cfg(default_role=Role.ADMIN, jit_provision=True)
        with patch("src.shared.auth.oidc.claims.ensure_user_in_graph") as ensure:
            ctx = build_user_context(
                {"sub": "101180639787655800606", "email": "user@example.com"},
                cfg=cfg,
                rbac=rbac,
            )
        self.assertEqual(ctx.role, Role.ADMIN)
        ensure.assert_called_once()
        self.assertEqual(ensure.call_args.kwargs["role"], Role.ADMIN)


class TestIngestAuth(unittest.TestCase):
    def test_require_bearer_rejects_missing_token(self):
        with patch("src.shared.auth.oidc.deps.get_oidc_config") as cfg:
            cfg.return_value = MagicMock(enabled=True)
            with self.assertRaises(Exception) as ctx:
                require_bearer_session(authorization=None)
            self.assertEqual(ctx.exception.status_code, 401)

    def test_require_admin_rejects_compliance_officer(self):
        # require_admin_session delegates to resolve_admin_session, which verifies
        # the token itself (not via require_bearer_session) then gates on role.
        with patch("src.shared.auth.oidc.deps.get_oidc_config") as cfg, patch(
            "src.shared.auth.oidc.deps.verify_bearer_token"
        ) as verify, patch("src.shared.auth.oidc.deps.build_user_context") as build:
            cfg.return_value = MagicMock(enabled=True)
            verify.return_value = {"sub": "u1"}
            build.return_value = UserContext(
                user_id="u1", role=Role.COMPLIANCE_OFFICER, tenant_id="default"
            )
            with self.assertRaises(Exception) as ctx:
                require_admin_session(authorization="Bearer x")
            self.assertEqual(ctx.exception.status_code, 403)

    def test_require_admin_allows_admin(self):
        with patch("src.shared.auth.oidc.deps.get_oidc_config") as cfg, patch(
            "src.shared.auth.oidc.deps.verify_bearer_token"
        ) as verify, patch("src.shared.auth.oidc.deps.build_user_context") as build:
            cfg.return_value = MagicMock(enabled=True)
            verify.return_value = {"sub": "u1"}
            build.return_value = UserContext(user_id="u1", role=Role.ADMIN, tenant_id="default")
            out = require_admin_session(authorization="Bearer x")
            self.assertEqual(out.user.role, Role.ADMIN)


if __name__ == "__main__":
    unittest.main()
