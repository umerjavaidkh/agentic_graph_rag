"""Build feedback-loop dashboard payloads for the ops UI."""
from __future__ import annotations

from typing import Any, Optional

from .config import FeedbackConfig
from .hints import ModeHint, best_mode_for_question
from .models import AggregateDimension
from .store import FeedbackStore, JsonlFeedbackStore, RedisFeedbackStore, get_feedback_store


def _hint_dict(hint: Optional[ModeHint]) -> Optional[dict[str, Any]]:
    if hint is None:
        return None
    return {
        "mode": hint.mode,
        "pass_rate": round(hint.pass_rate, 4),
        "samples": hint.samples,
        "confidence": round(hint.confidence, 4),
    }


def _pass_rate(passed: int, failed: int) -> Optional[float]:
    labeled = passed + failed
    if labeled <= 0:
        return None
    return round(passed / labeled, 4)


def build_dashboard_overview(
    *,
    recent_limit: int = 50,
    pattern_limit: int = 25,
    config: Optional[FeedbackConfig] = None,
    store: Optional[FeedbackStore] = None,
) -> dict[str, Any]:
    """Aggregate store data for GET /feedback/dashboard."""
    cfg = config or FeedbackConfig.load()
    fb_store = store or get_feedback_store()
    events = fb_store.recent_events(limit=max(recent_limit, pattern_limit * 4))

    totals = {
        "events": 0,
        "labeled": 0,
        "pass": 0,
        "fail": 0,
        "unlabeled": 0,
        "patterns": 0,
        "routing_applied": 0,
    }
    mode_totals: dict[str, dict[str, int]] = {}
    route_totals: dict[str, dict[str, int]] = {}
    pattern_rows: dict[str, dict[str, Any]] = {}

    for event in events:
        totals["events"] += 1
        if event.outcome is True:
            totals["labeled"] += 1
            totals["pass"] += 1
        elif event.outcome is False:
            totals["labeled"] += 1
            totals["fail"] += 1
        else:
            totals["unlabeled"] += 1

        for step in event.pipeline or []:
            if step.get("step") == "feedback.routing":
                totals["routing_applied"] += 1
                break

        p_hash = event.pattern_hash
        row = pattern_rows.setdefault(
            p_hash,
            {
                "pattern": event.pattern,
                "pattern_hash": p_hash,
                "queries": 0,
                "labeled": 0,
                "pass": 0,
                "fail": 0,
                "agents": set(),
                "modes": {},
                "routes": {},
            },
        )
        row["queries"] += 1
        row["agents"].add(event.agent or "unknown")
        if event.outcome is True:
            row["labeled"] += 1
            row["pass"] += 1
        elif event.outcome is False:
            row["labeled"] += 1
            row["fail"] += 1

        mode = event.retrieval_mode or "unknown"
        mb = row["modes"].setdefault(mode, {"pass": 0, "fail": 0, "queries": 0})
        mb["queries"] += 1
        if event.outcome is True:
            mb["pass"] += 1
        elif event.outcome is False:
            mb["fail"] += 1

        route = event.route_tool or "unknown"
        rb = row["routes"].setdefault(route, {"pass": 0, "fail": 0, "queries": 0})
        rb["queries"] += 1
        if event.outcome is True:
            rb["pass"] += 1
        elif event.outcome is False:
            rb["fail"] += 1

        mt = mode_totals.setdefault(mode, {"pass": 0, "fail": 0, "queries": 0})
        mt["queries"] += 1
        if event.outcome is True:
            mt["pass"] += 1
        elif event.outcome is False:
            mt["fail"] += 1

        rt = route_totals.setdefault(route, {"pass": 0, "fail": 0, "queries": 0})
        rt["queries"] += 1
        if event.outcome is True:
            rt["pass"] += 1
        elif event.outcome is False:
            rt["fail"] += 1

    totals["patterns"] = len(pattern_rows)

    patterns_out: list[dict[str, Any]] = []
    for row in sorted(pattern_rows.values(), key=lambda r: -r["queries"])[:pattern_limit]:
        agent = next(iter(row["agents"]), "")
        if "structured" in row["agents"]:
            agent = "structured"
        elif "unstructured" in row["agents"]:
            agent = "unstructured"

        retrieval_hint = best_mode_for_question(
            _pattern_probe(row["pattern"]),
            agent=agent,
            dimension=AggregateDimension.RETRIEVAL_MODE,
            min_samples=cfg.min_samples,
            min_margin=cfg.min_margin,
            cache_sec=0,
        )
        route_hint = best_mode_for_question(
            _pattern_probe(row["pattern"]),
            agent="",
            dimension=AggregateDimension.ROUTE_TOOL,
            min_samples=cfg.min_samples,
            min_margin=cfg.min_margin,
            cache_sec=0,
        )
        patterns_out.append(
            {
                "pattern": row["pattern"],
                "pattern_hash": row["pattern_hash"],
                "queries": row["queries"],
                "labeled": row["labeled"],
                "pass": row["pass"],
                "fail": row["fail"],
                "pass_rate": _pass_rate(row["pass"], row["fail"]),
                "by_mode": row["modes"],
                "by_route": row["routes"],
                "retrieval_hint": _hint_dict(retrieval_hint),
                "route_hint": _hint_dict(route_hint),
                "would_apply": cfg.routing_enabled
                and (retrieval_hint is not None or route_hint is not None),
            }
        )

    store_type = "unknown"
    if isinstance(fb_store, RedisFeedbackStore):
        store_type = "redis"
    elif isinstance(fb_store, JsonlFeedbackStore):
        store_type = "jsonl"

    recent_out = []
    for event in events[:recent_limit]:
        applied = any(
            (step.get("step") == "feedback.routing") for step in (event.pipeline or [])
        )
        recent_out.append(
            {
                "request_id": event.request_id,
                "ts": event.ts,
                "agent": event.agent,
                "pattern": event.pattern,
                "retrieval_mode": event.retrieval_mode,
                "route_tool": event.route_tool,
                "route_method": event.route_method,
                "outcome": event.outcome,
                "source": event.source,
                "routing_applied": applied,
                "question_preview": event.question_preview,
            }
        )

    return {
        "config": {
            "enabled": cfg.enabled,
            "routing_enabled": cfg.routing_enabled,
            "store_question": cfg.store_question,
            "store_type": store_type,
            "min_samples": cfg.min_samples,
            "min_margin": cfg.min_margin,
            "data_dir": cfg.data_dir if store_type == "jsonl" else None,
            "redis_stream": cfg.redis_stream if store_type == "redis" else None,
        },
        "totals": totals,
        "mode_totals": mode_totals,
        "route_totals": route_totals,
        "patterns": patterns_out,
        "recent_events": recent_out,
    }


def _pattern_probe(pattern: str) -> str:
    """Dashboard hints need a question string; use pattern flags as a stable probe."""
    return pattern.replace("|", " ").replace("agent:", "")
