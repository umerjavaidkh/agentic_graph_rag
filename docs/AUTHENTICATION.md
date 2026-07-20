# Authentication & roles

## Branches

| Branch | Auth in UI | Use case |
|--------|------------|----------|
| **`master`** (default) | Sidebar **User ID** + **Role** — no Google sign-in in `/chat` or `/upload` | Local dev, eval, demos |
| **`release/v1.0`** | **Sign in with Google** (OIDC JWT) + full production RBAC | Deployments with real identity |

Both branches share the same RAG, ingestion, feedback, and eval features. Only the login UX and default env differ.

## How identity is resolved

```mermaid
flowchart LR
  subgraph chat [Chat /query]
    G[Google JWT] --> C[claims + AUTH_EMAIL_ROLE_MAP]
    C --> JIT[Neo4j User HAS_ROLE]
    JIT --> RBAC[can_query_knowledge_area]
    B[Body user_id/role] -.->|only if no JWT and AUTH_ALLOW_BODY_FALLBACK| RBAC
  end
  subgraph ingest [Ingest /upload]
    G2[Google JWT] --> A{role = admin?}
    A -->|yes| UP[/ingest/* /admin/*]
    A -->|no| DENY[403]
  end
```

| Surface | Auth | Role required |
|---------|------|----------------|
| `/chat`, `POST /query`, `POST /query/stream` | JWT recommended; body fallback optional | Any role with data access (`compliance_officer`+ for both graphs) |
| `/upload`, `POST /ingest/*`, `GET /ingest/*` | **JWT required** (no body fallback) | **`admin`** only |
| `GET /audit/events` | JWT or dev fallback | **`admin`** only |
| `POST /admin/reset-neo4j` | **JWT required** | **`admin`** (+ `ALLOW_DB_RESET=true`) |

## RBAC by role (Neo4j knowledge areas)

| Role | Documents | Structured | Ingestion |
|------|-----------|------------|-----------|
| `public` | ✅ | ❌ | ❌ |
| `regular_office` | ❌ | ✅ | ❌ |
| `compliance_officer` | ✅ | ✅ | ❌ |
| `admin` | ✅ | ✅ | ✅ |

The document-access knowledge-area id defaults to `esg` (matching the seeded demo RBAC schema) and is configurable via `DOCUMENT_KNOWLEDGE_AREA_ID` — see [CONFIGURATION.md](CONFIGURATION.md).

## Seeded demo users (eval / `AUTH_ALLOW_BODY_FALLBACK`)

| User ID | Role | Documents | Structured |
|---------|------|-----------|------------|
| `public_001` | `public` | ✅ | ❌ |
| `regular_001` | `regular_office` | ❌ | ✅ |
| `compliance_001` | `compliance_officer` | ✅ | ✅ |
| `admin_001` | `admin` | ✅ | ✅ |

## Environment variables

**`master` (default):** `AUTH_ENABLED=false` — sidebar `user_id` + `role` for `/query`; `/upload` and `/admin/*` require `role=admin` via form/query params.

**`release/v1.0`:** `AUTH_ENABLED=true` + `GOOGLE_CLIENT_ID` — Sign in with Google in the UI; JWT overrides body identity.

| Variable | Default (`master`) | Description |
|----------|-------------------|-------------|
| `AUTH_ENABLED` | `false` | `true` → chat/upload require Google (or OIDC); use on `release/v1.0` |
| `GOOGLE_CLIENT_ID` | *(empty)* | OAuth 2.0 Web client ID — **GIS button on `release/v1.0` only** |
| `AUTH_DEFAULT_ROLE` | `compliance_officer` | Role assigned to new Google users (chat: documents + structured) |
| `AUTH_EMAIL_ROLE_MAP` | *(empty)* | Comma-separated `email=role` overrides, e.g. `you@corp.com=admin` |
| `AUTH_JIT_PROVISION` | `true` | On each login, sync User + `HAS_ROLE` in Neo4j from config/maps |
| `AUTH_ALLOW_BODY_FALLBACK` | `true` when auth off | `true` → unsigned `/query` may send `user_id`/`role` in JSON (eval/dev). Set **`false` in production**. |
| `AUTH_PROVIDER` | `google` | `google` or `oidc` (corporate IdP via `OIDC_ISSUER`, `OIDC_AUDIENCE`) |
| `AUTH_CLAIM_ROLE_MAP` | *(empty)* | Optional IdP group → role map (JSON or `Group=role` pairs) |
| `MULTI_TENANCY_ENABLED` | `false` | Property-based `tenant_id` isolation — a user needs both sufficient role AND matching tenant |

**Admin configuration (ingestion on `release/v1.0`):** map operator Google emails to `admin` in `.env`:

```env
# release/v1.0 production example
AUTH_ENABLED=true
GOOGLE_CLIENT_ID=your-client-id.apps.googleusercontent.com
AUTH_DEFAULT_ROLE=compliance_officer
AUTH_EMAIL_ROLE_MAP=you@company.com=admin
AUTH_ALLOW_BODY_FALLBACK=false   # production: Google only for /query
```

**Dev defaults on `master`:**

```env
AUTH_ENABLED=false
AUTH_ALLOW_BODY_FALLBACK=true
```

- **`admin`** — ingestion (`/upload`, `/ingest/*`), `/admin/reset-neo4j`, audit log, full RBAC.
- **`compliance_officer`** — chat only (`/query`, `/query/stream`): documents + structured data.
- **`regular_office`** — structured data only (demo user `regular_001`).
- **`public`** — document graph only (demo user `public_001`).

When a Bearer JWT is present, the server **ignores** sidebar/body `user_id` and `role` — identity comes from Google claims + maps above.

Public config for the UI: `GET /auth/config` · current principal: `GET /auth/me` (with `Authorization: Bearer …`).

## Follow-up memory

`thread_id` is scoped per user as `{user_id}:{session_uuid}` on the server. Two different Google users never share follow-up context; **New chat** only clears the current user's thread. Today this is a **single last-turn** snapshot (`conversation/thread_memory.py`). Per-user short and long memory are on the roadmap.

## Troubleshooting

**Google OAuth `origin_mismatch`:** the browser origin must exactly match an **Authorized JavaScript origin** (e.g. `http://localhost:8000`, not `127.0.0.1` unless both are registered).

**Access denied on structured queries** — with Google sign-in, use `compliance_officer` or `admin` (default: `AUTH_DEFAULT_ROLE=compliance_officer`). In dev sidebar mode: structured needs `regular_001` / `regular_office`; documents need `public_001` / `public`; both graphs need `compliance_001` or `admin_001`.

**Ingestion returns 401/403** — sign in on `/upload` with an email listed in `AUTH_EMAIL_ROLE_MAP=…=admin`. Body `role`/`user_id` fields are **not** accepted on ingest routes.
