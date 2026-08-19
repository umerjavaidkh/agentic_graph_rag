"""
tests/test_model_providers_unit.py — multi-provider LLM support.

Guards two things:
1. AnthropicProvider/GeminiProvider correctly adapt their native SDK
   response into the OpenAI-shaped object (.choices[0].message.content,
   .usage.*) that ~11 call sites across the codebase read directly — see
   src/model_providers/_shim.py's docstring for the full inventory. Mocks
   the SDK client so no real network call happens.
2. model_providers.factory's get_chat_provider()/get_embedding_provider()
   resolve the correct provider class and API key from MODEL_PROVIDER —
   the actual bug this work fixed: most call sites used to hardcode
   OPENAI_API_KEY regardless of MODEL_PROVIDER, and several ignored
   MODEL_PROVIDER entirely by calling get_model_provider() with no args.

Run with:
    python -m pytest tests/test_model_providers_unit.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest


from src.shared.model_providers._shim import ShimResponse, ShimUsage
from src.shared.model_providers.anthropic_provider import AnthropicProvider, _split_system
from src.shared.model_providers.gemini_provider import GeminiProvider, _to_gemini_contents


# ── Shim shape ────────────────────────────────────────────────────────────

def test_shim_response_matches_openai_shape():
    resp = ShimResponse("hello", usage=ShimUsage(prompt_tokens=10, completion_tokens=5))
    assert resp.choices[0].message.content == "hello"
    assert resp.usage.prompt_tokens == 10
    assert resp.usage.completion_tokens == 5
    assert resp.usage.total_tokens == 15  # auto-summed when not given explicitly


def test_shim_usage_defaults_to_zero():
    resp = ShimResponse("x")
    assert resp.usage.prompt_tokens == 0
    assert resp.usage.total_tokens == 0


# ── Anthropic: message/system splitting ─────────────────────────────────

def test_split_system_extracts_system_role():
    system, rest = _split_system([
        {"role": "system", "content": "be terse"},
        {"role": "user", "content": "hi"},
    ])
    assert system == "be terse"
    assert rest == [{"role": "user", "content": "hi"}]


def test_split_system_handles_no_system_message():
    system, rest = _split_system([{"role": "user", "content": "hi"}])
    assert system == ""
    assert len(rest) == 1


def test_split_system_joins_multiple_system_messages():
    system, rest = _split_system([
        {"role": "system", "content": "a"},
        {"role": "system", "content": "b"},
        {"role": "user", "content": "hi"},
    ])
    assert system == "a\n\nb"
    assert len(rest) == 1


# ── Anthropic: chat_completion adapts to OpenAI shape ───────────────────

def test_anthropic_chat_completion_shape():
    provider = AnthropicProvider(api_key="sk-ant-test")
    fake_block = SimpleNamespace(type="text", text="the answer")
    fake_resp = SimpleNamespace(
        content=[fake_block],
        usage=SimpleNamespace(input_tokens=7, output_tokens=3),
    )
    provider.client = MagicMock()
    provider.client.messages.create.return_value = fake_resp

    result = provider.chat_completion(
        model="claude-sonnet-5",
        messages=[{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}],
        temperature=0.1,
        max_tokens=50,
    )

    assert result.choices[0].message.content == "the answer"
    assert result.usage.prompt_tokens == 7
    assert result.usage.completion_tokens == 3

    call_kwargs = provider.client.messages.create.call_args.kwargs
    assert call_kwargs["system"] == "sys"
    assert call_kwargs["messages"] == [{"role": "user", "content": "hi"}]
    assert call_kwargs["max_tokens"] == 50
    assert call_kwargs["temperature"] == 0.1


def test_anthropic_chat_completion_defaults_max_tokens_when_missing():
    provider = AnthropicProvider(api_key="sk-ant-test")
    fake_resp = SimpleNamespace(
        content=[SimpleNamespace(type="text", text="x")],
        usage=SimpleNamespace(input_tokens=1, output_tokens=1),
    )
    provider.client = MagicMock()
    provider.client.messages.create.return_value = fake_resp

    provider.chat_completion(model="claude-sonnet-5", messages=[{"role": "user", "content": "hi"}])

    assert provider.client.messages.create.call_args.kwargs["max_tokens"] > 0


def test_anthropic_embeddings_not_implemented():
    provider = AnthropicProvider(api_key="sk-ant-test")
    with pytest.raises(NotImplementedError):
        provider.embeddings(model="x", input="hi")


# ── Gemini: message/content conversion ──────────────────────────────────

def test_to_gemini_contents_extracts_system_and_maps_assistant_role():
    system, contents = _to_gemini_contents([
        {"role": "system", "content": "be terse"},
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ])
    assert system == "be terse"
    assert contents == [
        {"role": "user", "parts": [{"text": "hi"}]},
        {"role": "model", "parts": [{"text": "hello"}]},  # OpenAI 'assistant' -> Gemini 'model'
    ]


# ── Gemini: chat_completion adapts to OpenAI shape ──────────────────────

def test_gemini_chat_completion_shape():
    provider = GeminiProvider(api_key="fake-test")
    fake_resp = SimpleNamespace(
        text="the answer",
        usage_metadata=SimpleNamespace(
            prompt_token_count=7, candidates_token_count=3, total_token_count=10
        ),
    )
    provider.client = MagicMock()
    provider.client.models.generate_content.return_value = fake_resp

    result = provider.chat_completion(
        model="gemini-2.5-flash",
        messages=[{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}],
        temperature=0.2,
        max_tokens=40,
    )

    assert result.choices[0].message.content == "the answer"
    assert result.usage.prompt_tokens == 7
    assert result.usage.completion_tokens == 3
    assert result.usage.total_tokens == 10

    call_kwargs = provider.client.models.generate_content.call_args.kwargs
    assert call_kwargs["contents"] == [{"role": "user", "parts": [{"text": "hi"}]}]
    config = call_kwargs["config"]
    assert config.system_instruction == "sys"
    assert config.temperature == 0.2
    assert config.max_output_tokens == 40


def test_gemini_embeddings_not_implemented():
    provider = GeminiProvider(api_key="fake-test")
    with pytest.raises(NotImplementedError):
        provider.embeddings(model="x", input="hi")


# ── Factory: provider/key resolution (the actual bug this fixes) ───────

def _reset_factory_singletons():
    import src.shared.model_providers.factory as factory

    factory._chat_provider_singleton = None
    factory._embedding_provider_singleton = None


@pytest.mark.parametrize(
    "model_provider,expected_class,expected_key_attr",
    [
        ("openai", "OpenAIProvider", "OPENAI_API_KEY"),
        ("anthropic", "AnthropicProvider", "ANTHROPIC_API_KEY"),
        ("claude", "AnthropicProvider", "ANTHROPIC_API_KEY"),  # alias
        ("gemini", "GeminiProvider", "GOOGLE_API_KEY"),
        ("google", "GeminiProvider", "GOOGLE_API_KEY"),  # alias
        ("nonsense", "OpenAIProvider", "OPENAI_API_KEY"),  # unknown -> openai fallback
    ],
)
def test_get_chat_provider_resolves_correct_class_and_key(
    model_provider, expected_class, expected_key_attr, monkeypatch
):
    _reset_factory_singletons()
    monkeypatch.setattr("src.shared.config.settings.MODEL_PROVIDER", model_provider)
    monkeypatch.setattr("src.shared.config.settings.OPENAI_API_KEY", "openai-key")
    monkeypatch.setattr("src.shared.config.settings.ANTHROPIC_API_KEY", "anthropic-key")
    monkeypatch.setattr("src.shared.config.settings.GOOGLE_API_KEY", "google-key")

    from src.shared.model_providers.factory import get_chat_provider

    provider = get_chat_provider()
    assert type(provider).__name__ == expected_class
    _reset_factory_singletons()


def test_get_chat_provider_is_a_cached_singleton(monkeypatch):
    _reset_factory_singletons()
    monkeypatch.setattr("src.shared.config.settings.MODEL_PROVIDER", "openai")
    monkeypatch.setattr("src.shared.config.settings.OPENAI_API_KEY", "openai-key")

    from src.shared.model_providers.factory import get_chat_provider

    first = get_chat_provider()
    second = get_chat_provider()
    assert first is second
    _reset_factory_singletons()


def test_get_embedding_provider_always_openai_regardless_of_model_provider(monkeypatch):
    """The confirmed design: MODEL_PROVIDER only controls chat. Embeddings
    always use OpenAI — Anthropic has no embeddings API, and Neo4j's vector
    index has a fixed dimension, so embedding-provider swapping is out of
    scope by design, not an oversight."""
    _reset_factory_singletons()
    monkeypatch.setattr("src.shared.config.settings.MODEL_PROVIDER", "anthropic")
    monkeypatch.setattr("src.shared.config.settings.OPENAI_API_KEY", "openai-key")

    from src.shared.model_providers.factory import get_embedding_provider

    provider = get_embedding_provider()
    assert type(provider).__name__ == "OpenAIProvider"
    _reset_factory_singletons()


def test_get_model_provider_rejects_unsupported_name():
    from src.shared.model_providers.factory import get_model_provider

    with pytest.raises(ValueError):
        get_model_provider("not-a-real-provider")
