from functools import lru_cache
from pathlib import Path
from typing import Any

# src/shared/config/prompts.py -> parents[2] is src/, where the
# templates live. Deliberately not derived from PROJECT_ROOT: prompts
# ship inside the package, not beside it.
PROMPTS_DIR = Path(__file__).resolve().parents[2] / "prompts"


@lru_cache(maxsize=None)
def _read_prompt_template(name: str) -> str:
    """Load raw prompt text once per process (templates are static at runtime)."""
    path = PROMPTS_DIR / f"{name}.txt"
    return path.read_text(encoding="utf-8")


# Prompts that produce an ANSWER a person reads. The language directive is
# appended to these and to nothing else: route_query and the text2cypher
# prompts emit a tool name or Cypher, where "answer in the language of the
# question" is at best noise and at worst an instruction to emit Arabic
# identifiers into a query.
#
# Structured prompts are absent deliberately. The business graph has no
# language dimension, and its synthesis prompt describes rows, not prose
# quoted from a document.
_ANSWER_PROMPTS = frozenset({
    "document_default",
    "document_low_confidence",
    "document_overview",
    "document_page",
    "document_section_content",
    "document_structural",
    "document_synthesis",
    "document_table",
    "document_toc",
    "document_visual",
    "chapter_summary",
})

# Appended verbatim, not templated into each file: a `{placeholder}` in a
# template raises KeyError in every caller that does not pass it, and there
# are twelve callers that have no reason to know this exists.
_LANGUAGE_DIRECTIVE = (
    "\n\nAnswer in the same language as the question. "
    "Quote the document VERBATIM in its own language and never translate a "
    "quotation, a heading, a table cell or a number -- a translated citation "
    "cannot be checked against the page it cites."
)


def language_directive() -> str:
    """The answer-language instruction, or nothing at all.

    Empty while one language is configured, for the same reason
    `language_filter()` compiles to "true" there: a single-language
    deployment gets byte-identical prompts, so the English answers cannot
    move under it. A prompt change is not free even when it is logically a
    no-op -- the model is not a pure function of intent -- so it is not
    made until there is a second language to make it for.
    """
    # Imported here rather than at module scope: config.settings is
    # imported by almost everything, and language.py imports settings.
    from ..language import configured_languages

    return _LANGUAGE_DIRECTIVE if len(configured_languages()) > 1 else ""


def load_prompt(name: str, **kwargs: Any) -> str:
    """Load a named prompt template and format it with values."""
    text = _read_prompt_template(name).format(**kwargs)
    if name in _ANSWER_PROMPTS:
        text += language_directive()
    return text


def clear_prompt_cache() -> None:
    """Clear in-memory prompt templates (tests or hot-reload only)."""
    _read_prompt_template.cache_clear()
