from typing import Optional
from .base import ModelProvider
from .openai_provider import OpenAIProvider

# Friendly aliases -> canonical provider name.
_PROVIDER_ALIASES = {
    "openai": "openai",
    "anthropic": "anthropic",
    "claude": "anthropic",
    "gemini": "gemini",
    "google": "gemini",
}


def get_model_provider(name: str = "openai", api_key: Optional[str] = None) -> ModelProvider:
    """Explicit provider constructor. Prefer get_chat_provider()/
    get_embedding_provider() below for normal call sites — those also
    resolve the correct API key for whichever provider is configured; this
    stays available for tests and any call site that genuinely needs to
    construct a specific provider by name."""
    canonical = _PROVIDER_ALIASES.get(name.lower())
    if canonical is None:
        raise ValueError(f"Unsupported model provider: {name}")

    if canonical == "openai":
        return OpenAIProvider(api_key=api_key)
    if canonical == "anthropic":
        from .anthropic_provider import AnthropicProvider

        return AnthropicProvider(api_key=api_key)
    if canonical == "gemini":
        from .gemini_provider import GeminiProvider

        return GeminiProvider(api_key=api_key)

    raise ValueError(f"Unsupported model provider: {name}")  # pragma: no cover — unreachable


_chat_provider_singleton: ModelProvider | None = None
_embedding_provider_singleton: ModelProvider | None = None


def get_chat_provider() -> ModelProvider:
    """Process-wide chat/completion provider, resolved from settings.MODEL_PROVIDER
    and the matching API key — the single source of truth every chat call
    site should use instead of constructing get_model_provider(...) directly.

    Fixes a real bug: most call sites previously did
    get_model_provider(MODEL_PROVIDER, OPENAI_API_KEY) — always passing the
    OpenAI key regardless of which provider MODEL_PROVIDER actually named —
    and a few called get_model_provider() with no arguments at all, silently
    defaulting to "openai" and ignoring MODEL_PROVIDER entirely.
    """
    global _chat_provider_singleton
    if _chat_provider_singleton is not None:
        return _chat_provider_singleton

    from ..config.settings import ANTHROPIC_API_KEY, GOOGLE_API_KEY, MODEL_PROVIDER, OPENAI_API_KEY

    canonical = _PROVIDER_ALIASES.get(MODEL_PROVIDER.lower(), "openai")
    key = {
        "openai": OPENAI_API_KEY,
        "anthropic": ANTHROPIC_API_KEY,
        "gemini": GOOGLE_API_KEY,
    }[canonical]
    _chat_provider_singleton = get_model_provider(canonical, key)
    return _chat_provider_singleton


def get_embedding_provider() -> ModelProvider:
    """Process-wide embedding provider — always OpenAI, independent of
    MODEL_PROVIDER. Anthropic has no embeddings API at all, and Neo4j's
    vector index hardcodes dimension 1536 (not parameterized), so embedding
    provider/model swapping is deliberately out of scope; this is the one
    place that decision is enforced."""
    global _embedding_provider_singleton
    if _embedding_provider_singleton is not None:
        return _embedding_provider_singleton

    from ..config.settings import OPENAI_API_KEY

    _embedding_provider_singleton = OpenAIProvider(api_key=OPENAI_API_KEY)
    return _embedding_provider_singleton
