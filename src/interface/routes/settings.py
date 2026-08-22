"""Admin settings screen: read and change tuning knobs without a redeploy.

Deliberately an allow-list, not "every environment variable". Anything that
grants access or holds a credential stays out: an API key must never be
readable through a web page, and ALLOW_CYPHER_INGEST / ALLOW_DB_RESET are
safety switches whose whole value is that flipping them takes a deploy.

Every entry here is a performance or quality dial -- how many documents
ingest at once, which model does entity extraction, how much work Axis 2
does per document. Changing one badly makes ingestion slow or coarse; it
cannot expose data or destroy any.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel, Field

from ...shared.config.settings_schema import (
    SETTINGS,
    Setting,
    _BY_NAME,
    _TRUE,
    _coerce,
)

from ...shared.auth.oidc import resolve_admin_session
from ...shared.config import overrides

log = logging.getLogger(__name__)
router = APIRouter()


class SettingsUpdate(BaseModel):
    values: Dict[str, str] = Field(default_factory=dict)
    user_id: Optional[str] = None
    role: Optional[str] = None
    tenant_id: Optional[str] = None


def _current() -> Dict[str, Any]:
    """Each setting with its effective value and where that value came from."""
    import os

    stored = overrides.load()
    rows = []
    for setting in SETTINGS:
        effective = stored.get(setting.name, os.environ.get(setting.name, setting.default))
        rows.append({
            **setting.model_dump(),
            "value": str(effective),
            "source": "override" if setting.name in stored else "environment",
        })
    return {"settings": rows, "override_count": len(stored)}


@router.get("/admin/settings")
async def get_settings(
    authorization: Optional[str] = Header(default=None),
    user_id: Optional[str] = None,
    role: Optional[str] = None,
    tenant_id: Optional[str] = None,
):
    """Current values for every tunable setting, with where each came from."""
    resolve_admin_session(
        authorization=authorization, body_user_id=user_id,
        body_role=role, body_tenant_id=tenant_id,
    )
    return _current()


@router.put("/admin/settings")
async def update_settings(
    update: SettingsUpdate,
    authorization: Optional[str] = Header(default=None),
):
    """Store overrides for the named settings.

    Values are validated here rather than at read time, because a bad value
    read at import would take down a worker on start with a traceback
    instead of being refused at the screen that set it.
    """
    resolve_admin_session(
        authorization=authorization, body_user_id=update.user_id,
        body_role=update.role, body_tenant_id=update.tenant_id,
    )

    unknown = sorted(set(update.values) - set(_BY_NAME))
    if unknown:
        raise HTTPException(
            status_code=400,
            detail=f"Not settable through this screen: {', '.join(unknown)}. "
                   "Only the listed tuning settings can be changed here; anything "
                   "granting access or holding a credential is deliberately excluded.",
        )

    cleaned: Dict[str, str] = {}
    for name, raw in update.values.items():
        try:
            cleaned[name] = _coerce(_BY_NAME[name], raw)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    merged = {**overrides.load(), **cleaned}
    try:
        overrides.save(merged)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    needs = sorted({_BY_NAME[n].applies_to for n in cleaned})
    return {
        "status": "ok",
        "saved": cleaned,
        "restart_required": True,
        "applies_to": needs,
        "message": (
            "Saved. Settings are read once when a process starts, so this takes "
            "effect after a restart"
            + (" — and WORKER_REPLICAS needs `docker compose up -d`, not just a restart."
               if "compose" in needs else ".")
        ),
    }


@router.delete("/admin/settings")
async def reset_settings(
    authorization: Optional[str] = Header(default=None),
    user_id: Optional[str] = None,
    role: Optional[str] = None,
    tenant_id: Optional[str] = None,
):
    """Drop every override and fall back to the deployed environment."""
    resolve_admin_session(
        authorization=authorization, body_user_id=user_id,
        body_role=role, body_tenant_id=tenant_id,
    )
    overrides.clear()
    return {"status": "ok", "restart_required": True,
            "message": "All overrides cleared. Takes effect after a restart."}
