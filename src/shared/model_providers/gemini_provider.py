"""gemini_provider.py — Gemini backend for chat/completion.

Uses the current Google-recommended `google-genai` SDK (`from google import
genai`), not the older/deprecated `google-generativeai` package.

No embeddings() here even though Gemini's API does support embeddings —
the confirmed design keeps embeddings OpenAI-only regardless of
MODEL_PROVIDER (see model_providers/factory.py's get_embedding_provider()),
so wiring up Gemini embeddings would be dead code today.
"""
import os
from typing import Iterator

from .base import ModelProvider
from ._shim import ShimResponse, ShimUsage
from ..telemetry.context import TelemetryEvent, get_telemetry

_ROLE_MAP = {"assistant": "model", "user": "user"}


def _to_gemini_contents(messages: list[dict]) -> tuple[str, list[dict]]:
    """Split out system-role messages (Gemini takes system_instruction as a
    separate config field, same shape mismatch as Anthropic) and translate
    the OpenAI-style 'assistant' role to Gemini's 'model' role."""
    system_parts: list[str] = []
    contents: list[dict] = []
    for m in messages:
        role = m.get("role")
        content = str(m.get("content") or "")
        if role == "system":
            system_parts.append(content)
        else:
            contents.append({"role": _ROLE_MAP.get(role, "user"), "parts": [{"text": content}]})
    return "\n\n".join(system_parts), contents


class GeminiProvider(ModelProvider):
    def __init__(self, api_key: str | None = None):
        from google import genai  # optional dependency, imported lazily

        key = api_key or os.environ.get("GOOGLE_API_KEY", "")
        self.client = genai.Client(api_key=key) if key else genai.Client()
        self._genai = genai

    def _config(self, system: str, **kwargs):
        types = self._genai.types
        cfg_kwargs: dict = {}
        if system:
            cfg_kwargs["system_instruction"] = system
        if "temperature" in kwargs:
            cfg_kwargs["temperature"] = kwargs.pop("temperature")
        if "max_tokens" in kwargs:
            cfg_kwargs["max_output_tokens"] = kwargs.pop("max_tokens")
        cfg_kwargs.update(kwargs)
        return types.GenerateContentConfig(**cfg_kwargs) if cfg_kwargs else None

    def chat_completion(self, model: str, messages: list[dict], **kwargs):
        system, contents = _to_gemini_contents(messages)
        call_kwargs = dict(kwargs)
        config = self._config(system, **call_kwargs)
        resp = self.client.models.generate_content(model=model, contents=contents, config=config)
        text = resp.text or ""
        usage_meta = getattr(resp, "usage_metadata", None)
        usage = ShimUsage(
            prompt_tokens=int(getattr(usage_meta, "prompt_token_count", 0) or 0) if usage_meta else 0,
            completion_tokens=int(getattr(usage_meta, "candidates_token_count", 0) or 0) if usage_meta else 0,
            total_tokens=int(getattr(usage_meta, "total_token_count", 0) or 0) if usage_meta else 0,
        )
        t = get_telemetry()
        if t is not None:
            t.add(
                TelemetryEvent(
                    kind="chat",
                    model=model,
                    prompt_tokens=usage.prompt_tokens,
                    completion_tokens=usage.completion_tokens,
                    total_tokens=usage.total_tokens,
                )
            )
        return ShimResponse(text, usage=usage)

    def chat_completion_stream(
        self,
        model: str,
        messages: list[dict],
        **kwargs,
    ) -> Iterator[str]:
        system, contents = _to_gemini_contents(messages)
        call_kwargs = dict(kwargs)
        config = self._config(system, **call_kwargs)
        prompt_tokens = 0
        completion_tokens = 0
        total_tokens = 0
        for chunk in self.client.models.generate_content_stream(model=model, contents=contents, config=config):
            usage_meta = getattr(chunk, "usage_metadata", None)
            if usage_meta is not None:
                prompt_tokens = int(getattr(usage_meta, "prompt_token_count", 0) or 0)
                completion_tokens = int(getattr(usage_meta, "candidates_token_count", 0) or 0)
                total_tokens = int(getattr(usage_meta, "total_token_count", 0) or 0)
            delta = chunk.text
            if delta:
                yield delta
        tel = get_telemetry()
        if tel is not None and (prompt_tokens or completion_tokens or total_tokens):
            tel.add(
                TelemetryEvent(
                    kind="chat",
                    model=model,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=total_tokens or (prompt_tokens + completion_tokens),
                )
            )

    def embeddings(self, model: str, input: list[str] | str, **kwargs):
        raise NotImplementedError(
            "Gemini embeddings are not wired up — embeddings always use "
            "model_providers.factory.get_embedding_provider() (OpenAI-backed), regardless of MODEL_PROVIDER."
        )

    def close(self) -> None:
        if hasattr(self.client, "close"):
            self.client.close()
