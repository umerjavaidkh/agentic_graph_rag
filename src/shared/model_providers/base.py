from abc import ABC, abstractmethod
from typing import Any, Iterator, Mapping


class ModelProvider(ABC):
    @abstractmethod
    def chat_completion(self, model: str, messages: list[Mapping[str, Any]], **kwargs: Any) -> Any:
        pass

    @abstractmethod
    def embeddings(self, model: str, input: list[str] | str, **kwargs: Any) -> Any:
        pass

    def chat_completion_stream(
        self,
        model: str,
        messages: list[Mapping[str, Any]],
        **kwargs: Any,
    ) -> Iterator[str]:
        """Yield text deltas; default falls back to one-shot completion."""
        resp = self.chat_completion(model=model, messages=messages, **kwargs)
        content = resp.choices[0].message.content or ""
        if content:
            yield content

    def close(self) -> None:
        return None
