import os
from typing import Iterator

from openai import OpenAI

from .base import ModelProvider
from ..telemetry.context import TelemetryEvent, get_telemetry


class OpenAIProvider(ModelProvider):
    def __init__(self, api_key: str | None = None):
        key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.client = OpenAI(api_key=key) if key else OpenAI()

    def chat_completion(self, model: str, messages: list[dict], **kwargs):
        resp = self.client.chat.completions.create(model=model, messages=messages, **kwargs)
        t = get_telemetry()
        if t is not None:
            usage = getattr(resp, "usage", None)
            pt = int(getattr(usage, "prompt_tokens", 0) or 0) if usage else 0
            ct = int(getattr(usage, "completion_tokens", 0) or 0) if usage else 0
            tt = int(getattr(usage, "total_tokens", 0) or (pt + ct)) if usage else 0
            t.add(TelemetryEvent(kind="chat", model=model, prompt_tokens=pt, completion_tokens=ct, total_tokens=tt))
        return resp

    def chat_completion_stream(
        self,
        model: str,
        messages: list[dict],
        **kwargs,
    ) -> Iterator[str]:
        stream_kwargs = dict(kwargs)
        stream_kwargs["stream"] = True
        stream_kwargs.setdefault("stream_options", {"include_usage": True})
        stream = self.client.chat.completions.create(
            model=model,
            messages=messages,
            **stream_kwargs,
        )
        prompt_tokens = 0
        completion_tokens = 0
        total_tokens = 0
        for chunk in stream:
            usage = getattr(chunk, "usage", None)
            if usage is not None:
                prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
                completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
                total_tokens = int(getattr(usage, "total_tokens", 0) or 0)
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta.content
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
        resp = self.client.embeddings.create(model=model, input=input, **kwargs)
        t = get_telemetry()
        if t is not None:
            usage = getattr(resp, "usage", None)
            tt = int(getattr(usage, "total_tokens", 0) or 0) if usage else 0
            t.add(TelemetryEvent(kind="embeddings", model=model, total_tokens=tt))
        return resp

    def close(self) -> None:
        if hasattr(self.client, "close"):
            self.client.close()
