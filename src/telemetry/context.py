from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class TelemetryEvent:
    kind: str  # chat | embeddings | structured_execute | unstructured_retrieve | other
    model: Optional[str] = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class Telemetry:
    events: list[TelemetryEvent] = field(default_factory=list)

    def add(self, event: TelemetryEvent) -> None:
        self.events.append(event)

    def summary(self) -> dict[str, Any]:
        totals = {
            "chat_calls": 0,
            "embed_calls": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }
        models: dict[str, dict[str, int]] = {}

        for e in self.events:
            if e.kind == "chat":
                totals["chat_calls"] += 1
            if e.kind == "embeddings":
                totals["embed_calls"] += 1
            totals["prompt_tokens"] += int(e.prompt_tokens or 0)
            totals["completion_tokens"] += int(e.completion_tokens or 0)
            totals["total_tokens"] += int(e.total_tokens or 0)
            if e.model:
                m = models.setdefault(e.model, {"calls": 0, "total_tokens": 0})
                m["calls"] += 1
                m["total_tokens"] += int(e.total_tokens or 0)

        return {
            "totals": totals,
            "by_model": models,
            "events": [
                {
                    "kind": e.kind,
                    "model": e.model,
                    "prompt_tokens": e.prompt_tokens,
                    "completion_tokens": e.completion_tokens,
                    "total_tokens": e.total_tokens,
                    "meta": e.meta,
                }
                for e in self.events
            ],
        }


_telemetry_var: ContextVar[Optional[Telemetry]] = ContextVar("telemetry", default=None)


def start_telemetry() -> Telemetry:
    t = Telemetry()
    _telemetry_var.set(t)
    return t


def get_telemetry() -> Optional[Telemetry]:
    return _telemetry_var.get()


def clear_telemetry() -> None:
    _telemetry_var.set(None)

