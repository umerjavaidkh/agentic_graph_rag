import os
from openai import OpenAI
from .base import ModelProvider


class OpenAIProvider(ModelProvider):
    def __init__(self, api_key: str | None = None):
        key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.client = OpenAI(api_key=key) if key else OpenAI()

    def chat_completion(self, model: str, messages: list[dict], **kwargs):
        return self.client.chat.completions.create(model=model, messages=messages, **kwargs)

    def embeddings(self, model: str, input: list[str] | str, **kwargs):
        return self.client.embeddings.create(model=model, input=input, **kwargs)

    def close(self) -> None:
        if hasattr(self.client, "close"):
            self.client.close()
