"""Shared LLM provider for unstructured retrieval synthesis (chat only).

Embeddings do NOT come from this module — they always go through
model_providers.factory.get_embedding_provider() (OpenAI-backed) regardless
of MODEL_PROVIDER; see graph_seeds.py's get_embedding().
"""
from __future__ import annotations

from ...shared.model_providers.factory import get_chat_provider

provider = get_chat_provider()
