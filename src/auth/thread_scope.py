"""Per-user thread_id namespacing for in-memory follow-up context."""
from __future__ import annotations

import re
from typing import Optional

_THREAD_SUFFIX_RE = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")
_MAX_USER_LEN = 64
_MAX_THREAD_LEN = 128


def _sanitize_user_id(user_id: str) -> str:
    uid = (user_id or "anonymous").strip()
    uid = re.sub(r"[^a-zA-Z0-9_@.-]", "_", uid)
    return uid[:_MAX_USER_LEN] or "anonymous"


def _sanitize_suffix(raw_suffix: str) -> str:
    suffix = (raw_suffix or "default").strip()[:64]
    if not suffix or not _THREAD_SUFFIX_RE.match(suffix):
        return "default"
    return suffix


def scoped_thread_id(user_id: str, client_thread_id: Optional[str] = None) -> str:
    """
    Build a thread key namespaced to the authenticated/dev user.

    Client may send only a suffix (``abc``) or a full id (``user:suffix``).
    The user prefix always comes from the trusted ``user_id`` — never from the
    client — so one user cannot read or clear another user's follow-up memory.
    """
    uid = _sanitize_user_id(user_id)
    raw = (client_thread_id or "default").strip()

    if ":" in raw:
        suffix = raw.split(":", 1)[1]
    else:
        suffix = raw

    tid = f"{uid}:{_sanitize_suffix(suffix)}"
    return tid[:_MAX_THREAD_LEN]
