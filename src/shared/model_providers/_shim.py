"""_shim.py — OpenAI-response-shape adapter shared by non-OpenAI providers.

11 call sites across the codebase read `response.choices[0].message.content`
directly (and 3 more read `resp.data[0].embedding` for embeddings) instead of
going through any abstraction — a pre-existing pattern, not something this
module changes. Rather than touching every one of those call sites for each
new provider, AnthropicProvider/GeminiProvider build one of these shim
objects from their native SDK response, so every existing caller keeps
working unchanged regardless of which provider actually served the request.
"""
from __future__ import annotations

from typing import Optional


class ShimUsage:
    def __init__(self, prompt_tokens: int = 0, completion_tokens: int = 0, total_tokens: int = 0):
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.total_tokens = total_tokens or (prompt_tokens + completion_tokens)


class ShimMessage:
    def __init__(self, content: str):
        self.content = content


class ShimChoice:
    def __init__(self, content: str):
        self.message = ShimMessage(content)


class ShimResponse:
    """Mimics an OpenAI ChatCompletion response: .choices[0].message.content, .usage.*"""

    def __init__(self, content: str, *, usage: Optional[ShimUsage] = None):
        self.choices = [ShimChoice(content)]
        self.usage = usage or ShimUsage()
