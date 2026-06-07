from functools import lru_cache
from pathlib import Path
from typing import Any

PROMPTS_DIR = Path(__file__).resolve().parents[1] / "prompts"


@lru_cache(maxsize=None)
def _read_prompt_template(name: str) -> str:
    """Load raw prompt text once per process (templates are static at runtime)."""
    path = PROMPTS_DIR / f"{name}.txt"
    return path.read_text(encoding="utf-8")


def load_prompt(name: str, **kwargs: Any) -> str:
    """Load a named prompt template and format it with values."""
    return _read_prompt_template(name).format(**kwargs)


def clear_prompt_cache() -> None:
    """Clear in-memory prompt templates (tests or hot-reload only)."""
    _read_prompt_template.cache_clear()
