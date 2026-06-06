"""Shared LLM/embedding provider for unstructured retrieval."""
from __future__ import annotations

from ...config.settings import MODEL_PROVIDER, OPENAI_API_KEY
from ...model_providers.factory import get_model_provider

provider = get_model_provider(MODEL_PROVIDER, OPENAI_API_KEY)
