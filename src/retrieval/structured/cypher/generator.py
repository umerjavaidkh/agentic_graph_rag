"""LLM Text-to-Cypher generation."""
from __future__ import annotations

from typing import Optional, Protocol

from ....config.prompts import load_prompt
from ....config.settings import (
    CHAT_MODEL,
    MODEL_PROVIDER,
    OPENAI_API_KEY,
    STRUCTURED_MODEL,
    STRUCTURED_TEXT2CYPHER_MAX_TOKENS,
)
from ....model_providers.factory import get_model_provider
from ..query_intent import analytics_result_limit

LLM_MODEL = STRUCTURED_MODEL or CHAT_MODEL
_provider = get_model_provider(MODEL_PROVIDER, OPENAI_API_KEY)


class CypherGenerator(Protocol):
    def generate(
        self,
        query: str,
        schema: str,
        limit: int,
        *,
        previous_cypher: Optional[str] = None,
        execution_error: Optional[str] = None,
        max_tokens: int = STRUCTURED_TEXT2CYPHER_MAX_TOKENS,
    ) -> Optional[str]: ...


def regenerate_for_issue(
    generate: CypherGenerator,
    query: str,
    schema: str,
    limit: int,
    cypher: str,
    issue: str,
) -> str:
    fixed = generate.generate(
        query,
        schema,
        limit,
        previous_cypher=cypher,
        execution_error=issue,
    )
    return fixed.strip() if fixed else cypher


class OpenAICypherGenerator:
    """Default Cypher generator using the configured model provider."""

    def __init__(self, model: str = LLM_MODEL, provider=_provider):
        self._model = model
        self._provider = provider

    def generate(
        self,
        query: str,
        schema: str,
        limit: int,
        *,
        previous_cypher: Optional[str] = None,
        execution_error: Optional[str] = None,
        max_tokens: int = STRUCTURED_TEXT2CYPHER_MAX_TOKENS,
    ) -> Optional[str]:
        retry_block = ""
        if execution_error and previous_cypher:
            retry_block = f"""
PREVIOUS QUERY (FAILED OR EMPTY):
{previous_cypher}

ISSUE:
{execution_error}

Fix the query. Use toFloat()/toInteger() when multiplying numeric fields that may be stored as strings.
If error mentions \"Variable not defined\", a WITH clause dropped a variable — keep it in WITH or aggregate in RETURN without referencing dropped aliases.
If 0 rows: reverse any relationship whose direction does not match RELATIONSHIP TYPES, then retry.
"""

        n = analytics_result_limit(query, limit)
        prompt = load_prompt(
            "structured_text2cypher",
            schema=schema,
            retry_block=retry_block,
            query=query,
            n=n,
        )

        response = self._provider.chat_completion(
            model=self._model,
            temperature=0,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
        )
        cypher = response.choices[0].message.content.strip()
        cypher = cypher.replace("```cypher", "").replace("```", "").strip()
        return cypher or None
