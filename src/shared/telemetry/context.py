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
class PipelineStepRecord:
    step: str
    status: str  # ok | error
    duration_ms: int = 0
    meta: dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


@dataclass
class Telemetry:
    events: list[TelemetryEvent] = field(default_factory=list)
    pipeline: list[PipelineStepRecord] = field(default_factory=list)
    errors: list[dict[str, Any]] = field(default_factory=list)
    route: dict[str, Any] = field(default_factory=dict)

    def add(self, event: TelemetryEvent) -> None:
        self.events.append(event)

    def record_step(
        self,
        step: str,
        *,
        status: str = "ok",
        duration_ms: int = 0,
        meta: Optional[dict[str, Any]] = None,
        error: Optional[str] = None,
    ) -> None:
        self.pipeline.append(
            PipelineStepRecord(
                step=step,
                status=status,
                duration_ms=duration_ms,
                meta=meta or {},
                error=error,
            )
        )

    def record_error(self, step: str, exc: BaseException) -> None:
        self.errors.append(
            {
                "step": step,
                "type": type(exc).__name__,
                "message": str(exc),
            }
        )

    def set_route(self, tool: str, method: str, **extra: Any) -> None:
        self.route = {"tool": tool, "method": method, **extra}

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

        failed = next((s for s in self.pipeline if s.status == "error"), None)
        return {
            "totals": totals,
            "by_model": models,
            "route": dict(self.route),
            "pipeline": [
                {
                    "step": s.step,
                    "status": s.status,
                    "duration_ms": s.duration_ms,
                    "meta": s.meta,
                    "error": s.error,
                }
                for s in self.pipeline
            ],
            "errors": list(self.errors),
            "failed_step": failed.step if failed else None,
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
    existing = _telemetry_var.get()
    if existing is not None:
        return existing
    t = Telemetry()
    _telemetry_var.set(t)
    return t


def get_telemetry() -> Optional[Telemetry]:
    return _telemetry_var.get()


def clear_telemetry() -> None:
    _telemetry_var.set(None)

