"""LLM multistep query planner."""
from __future__ import annotations

import json
from typing import Optional

from pydantic import ValidationError

from ....config.prompts import load_prompt
from ....config.settings import (
    CHAT_MODEL,
    MODEL_PROVIDER,
    OPENAI_API_KEY,
    STRUCTURED_MODEL,
    STRUCTURED_PLAN_MAX_TOKENS,
    STRUCTURED_PLAN_QUERY_LARGE_CHARS,
    STRUCTURED_PLAN_QUERY_MEDIUM_CHARS,
    STRUCTURED_PLAN_SCHEMA_LARGE_CHARS,
    STRUCTURED_PLAN_SCHEMA_MEDIUM_CHARS,
    STRUCTURED_PLAN_TOKENS_LARGE,
    STRUCTURED_PLAN_TOKENS_MEDIUM,
    STRUCTURED_PLAN_TOKENS_SMALL,
)
from ....model_providers.factory import get_model_provider
from .context import extract_json
from .models import MultiStepPlan

LLM_MODEL = STRUCTURED_MODEL or CHAT_MODEL
_provider = get_model_provider(MODEL_PROVIDER, OPENAI_API_KEY)


def multistep_plan_token_budget(query: str, schema: str) -> int:
    """
    Token budget for the multistep planner.

    We keep this dynamic because long questions require longer JSON plans.
    Allows override via STRUCTURED_PLAN_MAX_TOKENS.
    """
    if STRUCTURED_PLAN_MAX_TOKENS.isdigit():
        return max(300, min(int(STRUCTURED_PLAN_MAX_TOKENS), 4000))

    q_len = len((query or "").strip())
    s_len = len((schema or "").strip())
    if q_len > STRUCTURED_PLAN_QUERY_LARGE_CHARS or s_len > STRUCTURED_PLAN_SCHEMA_LARGE_CHARS:
        return STRUCTURED_PLAN_TOKENS_LARGE
    if q_len > STRUCTURED_PLAN_QUERY_MEDIUM_CHARS or s_len > STRUCTURED_PLAN_SCHEMA_MEDIUM_CHARS:
        return STRUCTURED_PLAN_TOKENS_MEDIUM
    return STRUCTURED_PLAN_TOKENS_SMALL


class MultiStepPlanner:
    def __init__(self, model: str = LLM_MODEL, provider=_provider):
        self._model = model
        self._provider = provider

    def plan(self, query: str, schema: str) -> Optional[MultiStepPlan]:
        """
        Ask the LLM whether this query needs multi-step execution.

        This is intentionally schema-driven and avoids hardcoding specific query phrases.
        """
        try:
            prompt = load_prompt("structured_multistep_plan", schema=schema, query=query)
        except Exception:
            return None
        try:
            resp = self._provider.chat_completion(
                model=self._model,
                temperature=0,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=multistep_plan_token_budget(query, schema),
            )
            raw = (resp.choices[0].message.content or "").strip()
            js = extract_json(raw)
            data = json.loads(js)
            plan = MultiStepPlan.model_validate(data)
            if len(plan.steps) > 4:
                plan.steps = plan.steps[:4]
            if plan.needs_multistep and any((not s.cypher or len(s.cypher) < 10) for s in plan.steps):
                return None
            return plan
        except (json.JSONDecodeError, ValidationError):
            return None
        except Exception:
            return None
