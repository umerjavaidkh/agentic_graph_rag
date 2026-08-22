"""Runtime setting overrides, stored where every container can see them.

Settings come from the environment, and the environment comes from
`env_file:` in docker-compose -- which passes *values* into containers, not
the file. `/app/.env` does not exist inside the app or the workers, so a
settings screen cannot simply rewrite it. Overrides therefore live in
Redis, which the API and every worker already share, and are applied into
`os.environ` before `settings.py` reads it.

That last part is why this module is imported at the top of settings.py
rather than called by a route: the settings are module-level constants
evaluated once at import, so an override has to be in place before that
import happens. The consequence is honest and worth stating in the UI --
**a change takes effect when the process restarts**, not the moment it is
saved.

Only names in the settings screen's own allow-list are ever written here;
this module does not decide what is safe to change, it only stores it.
"""
from __future__ import annotations

import json
import logging
import os
from typing import Dict

log = logging.getLogger(__name__)

#: One key, one JSON object. Overrides are read on every process start and
#: are few enough that splitting them across keys would buy nothing.
REDIS_KEY = "graphrag:settings:overrides"


def _client():
    """A Redis client, or None when Redis is not configured or reachable.

    Never raises: a settings store that is down must not stop the
    application from starting with its ordinary environment.
    """
    url = os.environ.get("REDIS_URL")
    if not url:
        return None
    try:
        import redis

        client = redis.from_url(url, decode_responses=True)
        client.ping()
        return client
    except Exception:
        return None


def load() -> Dict[str, str]:
    """Stored overrides, or {} when there are none or the store is down."""
    client = _client()
    if client is None:
        return {}
    try:
        raw = client.get(REDIS_KEY)
        return json.loads(raw) if raw else {}
    except Exception:
        log.warning("Could not read settings overrides; using environment as-is.")
        return {}


def save(values: Dict[str, str]) -> Dict[str, str]:
    """Replace the stored overrides with `values`. Returns what was stored."""
    client = _client()
    if client is None:
        raise RuntimeError(
            "Settings cannot be saved: no Redis connection. Overrides are stored "
            "in Redis so the API and every worker read the same values."
        )
    cleaned = {str(k): str(v) for k, v in values.items()}
    client.set(REDIS_KEY, json.dumps(cleaned))
    return cleaned


def clear() -> None:
    client = _client()
    if client is not None:
        client.delete(REDIS_KEY)


def apply_to_environ() -> Dict[str, str]:
    """Put stored overrides into os.environ, then report what was applied.

    Overrides win over the container's environment: the point of the screen
    is to change a value without editing compose and redeploying, so a
    stored value the user set deliberately must beat the baked-in default.
    """
    applied = load()
    for key, value in applied.items():
        os.environ[key] = value
    return applied
