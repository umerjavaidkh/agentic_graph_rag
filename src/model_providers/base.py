from abc import ABC, abstractmethod
from typing import Any, Iterable, Mapping


class ModelProvider(ABC):
    @abstractmethod
    def chat_completion(self, model: str, messages: list[Mapping[str, Any]], **kwargs: Any) -> Any:
        pass

    @abstractmethod
    def embeddings(self, model: str, input: list[str] | str, **kwargs: Any) -> Any:
        pass

    def close(self) -> None:
        return None
