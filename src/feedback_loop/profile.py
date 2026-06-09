"""Project pipeline telemetry into feedback events."""
from __future__ import annotations

from typing import Any, Optional

from .models import RetrievalFeedbackEvent
from .pattern import pattern_hash, question_hash, retrieval_pattern

_RETRIEVAL_STEPS = frozenset(
    {
        "route.select",
        "agent.invoke",
        "agent.fallback",
        "document.graph.retrieve",
        "document.hybrid.merge",
        "structured.retrieve",
        "structured.text2cypher",
        "structured.multistep.plan",
        "structured.multistep.execute",
        "structured.graph.retrieve",
        "structured.graph.generate",
        "feedback.routing",
    }
)


def compact_pipeline(pipeline: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    """Keep retrieval-relevant steps only; drop token-heavy event payloads."""
    if not pipeline:
        return []
    out: list[dict[str, Any]] = []
    for step in pipeline:
        name = step.get("step") or ""
        if name not in _RETRIEVAL_STEPS:
            continue
        row: dict[str, Any] = {
            "step": name,
            "status": step.get("status"),
            "duration_ms": step.get("duration_ms"),
        }
        meta = step.get("meta") or {}
        if meta:
            row["meta"] = dict(meta)
        if step.get("error"):
            row["error"] = step.get("error")
        out.append(row)
    return out


def _step_meta(pipeline: list[dict[str, Any]], step_name: str) -> dict[str, Any]:
    for step in pipeline:
        if step.get("step") == step_name:
            return dict(step.get("meta") or {})
    return {}


def extract_retrieval_profile(
    telemetry: dict[str, Any],
    *,
    agent: str = "",
    strategy: str = "",
) -> tuple[str, dict[str, Any]]:
    """Return (retrieval_mode, profile dict) from telemetry.pipeline."""
    pipeline = telemetry.get("pipeline") or []
    compact = compact_pipeline(pipeline)

    merge = _step_meta(compact, "document.hybrid.merge")
    if merge:
        mode = str(merge.get("mode") or "graph_rag")
        return mode, {"path": "document", **merge}

    if any(s.get("step") == "structured.multistep.execute" for s in compact):
        meta = _step_meta(compact, "structured.multistep.execute")
        reason = meta.get("reason") or "gate"
        return f"structured_multistep:{reason}", {
            "path": "structured",
            "strategy": "multistep",
            **meta,
        }

    if any(s.get("step") == "structured.text2cypher" for s in compact):
        gen = _step_meta(compact, "structured.graph.generate")
        exec_meta = next(
            (
                e.get("meta") or {}
                for e in telemetry.get("events") or []
                if e.get("kind") == "structured_execute"
            ),
            {},
        )
        mode = strategy or "text2cypher"
        return mode, {
            "path": "structured",
            "strategy": mode,
            "attempts": exec_meta.get("attempts"),
            **gen,
        }

    route = telemetry.get("route") or {}
    if agent == "hybrid":
        return "hybrid", {"path": "hybrid", "route_tool": route.get("tool")}
    if agent == "structured":
        return strategy or "structured", {"path": "structured", "strategy": strategy}
    if agent == "unstructured":
        return strategy or "document", {"path": "document", "strategy": strategy}

    return strategy or "unknown", {"path": agent or "unknown", "strategy": strategy}


def build_feedback_event(
    *,
    request_id: str,
    ts: str,
    question: str,
    telemetry: dict[str, Any],
    agent: str,
    strategy: str,
    route_tool: str,
    route_method: str,
    outcome: Optional[bool] = None,
    case_id: Optional[str] = None,
    source: str = "query",
    store_question_preview: bool = False,
) -> RetrievalFeedbackEvent:
    pattern = retrieval_pattern(question, agent=agent)
    p_hash = pattern_hash(pattern)
    mode, profile = extract_retrieval_profile(telemetry, agent=agent, strategy=strategy)
    preview = (question or "")[:120] if store_question_preview else None

    return RetrievalFeedbackEvent(
        request_id=request_id,
        ts=ts,
        question_hash=question_hash(question),
        question_preview=preview,
        agent=agent or "",
        strategy=strategy or "",
        route_tool=route_tool or "",
        route_method=route_method or "",
        pattern=pattern,
        pattern_hash=p_hash,
        retrieval_mode=mode,
        retrieval_profile=profile,
        pipeline=compact_pipeline(telemetry.get("pipeline")),
        outcome=outcome,
        case_id=case_id,
        source=source,
    )
