from typing import Optional
from .base import ModelProvider
from .openai_provider import OpenAIProvider


def get_model_provider(name: str = "openai", api_key: Optional[str] = None) -> ModelProvider:
    name = name.lower()
    if name == "openai":
        return OpenAIProvider(api_key=api_key)

    raise ValueError(f"Unsupported model provider: {name}")
