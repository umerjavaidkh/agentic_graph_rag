"""Errors from a model provider that a user needs to see, not just a log line."""
from __future__ import annotations

import re

#: "Please try again in 8.64s" / "try again in 1m30s" — providers phrase the
#: wait differently, so the reset is reported only when it is actually given.
_RETRY_AFTER = re.compile(r"try again in ([\d.]+\s*[a-z]+(?:\s*[\d.]+\s*[a-z]+)?)", re.I)
_LIMIT_KIND = re.compile(r"on (requests|tokens) per (day|min|minute|hour)", re.I)
_MODEL = re.compile(r"for ([\w.\-]+) in (?:organization|project)", re.I)


class ModelRateLimitError(RuntimeError):
    """The provider refused the call because a rate limit is exhausted.

    Distinct from every other provider error because the correct response is
    the opposite one: a malformed or oversized response is retried by
    splitting the work into smaller pieces, which for a rate limit means
    issuing *more* requests against the limit that just refused you. One
    429 became two, then four -- a rate limit measured in requests per day
    was hit ~5,800 times in a single ingest, each retry making it worse.
    """

    def __init__(self, message: str, *, model: str = "", limit: str = "", retry_after: str = ""):
        self.model, self.limit, self.retry_after = model, limit, retry_after
        super().__init__(message)

    @classmethod
    def from_exception(cls, exc: Exception) -> "ModelRateLimitError":
        """Build one from a provider exception, keeping whatever it told us."""
        text = str(exc)
        model = (_MODEL.search(text) or [None, ""])[1] if _MODEL.search(text) else ""
        kind = _LIMIT_KIND.search(text)
        limit = f"{kind.group(1)} per {kind.group(2)}" if kind else ""
        retry = _RETRY_AFTER.search(text)
        retry_after = retry.group(1) if retry else ""
        return cls(cls._human(model, limit, retry_after, text),
                   model=model, limit=limit, retry_after=retry_after)

    @staticmethod
    def _human(model: str, limit: str, retry_after: str, raw: str) -> str:
        """A sentence that says what to do, not just what happened."""
        who = f"for {model}" if model else "for the configured model"
        what = f" ({limit})" if limit else ""
        # OpenAI reports "try again in 8.64s" even when the exhausted bucket
        # is a per-day one, which reads as "nearly over" when the real wait
        # is hours. Repeating it would be worse than saying nothing.
        when = f" Resets in {retry_after}." if retry_after and "day" not in limit else ""
        advice = (
            " A per-day limit will not clear by waiting a few seconds -- either use a model "
            "without a daily cap, raise the account tier, or reduce requests per document "
            "(AXIS2_NER_BATCH_SIZE, AXIS2_MAX_LLM_PAIRS)."
            if "day" in limit
            else " Reduce concurrency (worker count, AXIS2_NER_CONCURRENCY) or batch more per request."
        )
        return f"Model API rate limit reached {who}{what}.{when}{advice}"


def is_rate_limit(exc: Exception) -> bool:
    """Whether a provider exception is a rate limit, without importing SDKs.

    Matched by shape rather than by class so this holds for OpenAI,
    Anthropic and Gemini alike, and does not break when an SDK reorganises
    its exception hierarchy.
    """
    if type(exc).__name__ in ("RateLimitError", "ResourceExhausted", "TooManyRequests"):
        return True
    if getattr(exc, "status_code", None) == 429 or getattr(
        getattr(exc, "response", None), "status_code", None
    ) == 429:
        return True
    text = str(exc).lower()
    return "rate limit" in text or "429" in text and "rate" in text
