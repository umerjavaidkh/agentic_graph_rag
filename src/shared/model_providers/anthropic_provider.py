"""anthropic_provider.py — Claude backend for chat/completion.

No embeddings() — Anthropic has no embeddings API. Callers that need
embeddings must go through get_embedding_provider() (always OpenAI), never
whatever get_chat_provider() resolved to; see model_providers/factory.py.
"""
import os
from typing import Iterator

from .base import ModelProvider
from ..config.settings import LLM_MAX_RETRIES, LLM_REQUEST_TIMEOUT_SEC
from ._shim import ShimResponse, ShimUsage
from ..telemetry.context import TelemetryEvent, get_telemetry

_DEFAULT_MAX_TOKENS = 4096


def _split_system(messages: list[dict]) -> tuple[str, list[dict]]:
    """Anthropic takes the system prompt as a separate top-level param, not
    a 'system'-role message in the list — every caller in this codebase
    builds messages OpenAI-style with a leading {"role": "system", ...}, so
    pull it out rather than asking 14 call sites to branch on provider."""
    system_parts: list[str] = []
    rest: list[dict] = []
    for m in messages:
        if m.get("role") == "system":
            system_parts.append(str(m.get("content") or ""))
        else:
            rest.append(m)
    return "\n\n".join(system_parts), rest


class AnthropicProvider(ModelProvider):
    def __init__(self, api_key: str | None = None):
        import anthropic  # optional dependency, imported lazily

        key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        # Same reasoning as OpenAIProvider: cap the wall clock so one
        # stalled request cannot block a caller indefinitely.
        opts = {"timeout": LLM_REQUEST_TIMEOUT_SEC, "max_retries": LLM_MAX_RETRIES}
        self.client = anthropic.Anthropic(api_key=key, **opts) if key else anthropic.Anthropic(**opts)

    def chat_completion(self, model: str, messages: list[dict], **kwargs):
        system, rest = _split_system(messages)
        call_kwargs = dict(kwargs)
        max_tokens = call_kwargs.pop("max_tokens", None) or _DEFAULT_MAX_TOKENS
        resp = self.client.messages.create(
            model=model,
            max_tokens=max_tokens,
            messages=rest,
            system=system,
            **call_kwargs,
        )
        text = "".join(block.text for block in resp.content if getattr(block, "type", "") == "text")
        usage = ShimUsage(
            prompt_tokens=int(getattr(resp.usage, "input_tokens", 0) or 0),
            completion_tokens=int(getattr(resp.usage, "output_tokens", 0) or 0),
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
        system, rest = _split_system(messages)
        call_kwargs = dict(kwargs)
        max_tokens = call_kwargs.pop("max_tokens", None) or _DEFAULT_MAX_TOKENS
        with self.client.messages.stream(
            model=model,
            max_tokens=max_tokens,
            messages=rest,
            system=system,
            **call_kwargs,
        ) as stream:
            for delta in stream.text_stream:
                if delta:
                    yield delta
            final = stream.get_final_message()
            tel = get_telemetry()
            if tel is not None and final is not None:
                usage = getattr(final, "usage", None)
                pt = int(getattr(usage, "input_tokens", 0) or 0) if usage else 0
                ct = int(getattr(usage, "output_tokens", 0) or 0) if usage else 0
                if pt or ct:
                    tel.add(
                        TelemetryEvent(
                            kind="chat", model=model, prompt_tokens=pt, completion_tokens=ct, total_tokens=pt + ct
                        )
                    )

    def embeddings(self, model: str, input: list[str] | str, **kwargs):
        raise NotImplementedError(
            "Anthropic has no embeddings API. Use model_providers.factory.get_embedding_provider() "
            "for embeddings — it always returns an OpenAI-backed provider regardless of MODEL_PROVIDER."
        )

    def close(self) -> None:
        if hasattr(self.client, "close"):
            self.client.close()
