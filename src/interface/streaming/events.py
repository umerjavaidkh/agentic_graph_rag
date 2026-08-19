"""NDJSON stream event helpers."""
from __future__ import annotations

import json
from typing import Any


def stream_event(**payload: Any) -> str:
    return json.dumps(payload, default=str, ensure_ascii=False) + "\n"
