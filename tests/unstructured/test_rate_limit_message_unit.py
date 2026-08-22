"""A rate limit must stop the job and say so, not multiply itself.

Two behaviours, both learned the hard way on a real ingest: the NER retry
path treated a 429 as a "response too large" failure and split the batch,
reissuing two requests against the limit that had just refused one (~5,800
429s in a single run), and the resulting failure surfaced to the user as a
generic crash with no mention of a quota.
"""
import pytest

from src.shared.model_providers.errors import ModelRateLimitError, is_rate_limit

OPENAI_RPD = (
    "Error code: 429 - {'error': {'message': 'Rate limit reached for gpt-4o-mini in "
    "organization org-BEy on requests per day (RPD): Limit 10000, Used 10000, "
    "Requested 1. Please try again in 8.64s.', 'type': 'requests'}}"
)
OPENAI_RPM = (
    "Error code: 429 - {'error': {'message': 'Rate limit reached for gpt-4.1-mini in "
    "organization org-BEy on requests per min (RPM): Limit 500. Please try again in 2s.'}}"
)


class _Err(Exception):
    pass


class _WithStatus(Exception):
    status_code = 429


@pytest.mark.parametrize("exc", [
    _Err(OPENAI_RPD),
    _Err("rate limit exceeded"),
    _WithStatus("something"),
    type("RateLimitError", (Exception,), {})("anything"),        # OpenAI SDK
    type("ResourceExhausted", (Exception,), {})("anything"),     # Gemini
])
def test_rate_limits_are_recognised_across_providers(exc):
    """Matched by shape, so this survives an SDK reorganising its classes."""
    assert is_rate_limit(exc)


@pytest.mark.parametrize("exc", [
    _Err("Expecting ',' delimiter: line 1 column 1220"),
    _Err("connection reset by peer"),
    _Err("model not found"),
])
def test_ordinary_failures_are_not_mistaken_for_rate_limits(exc):
    """A malformed response must still take the split-and-retry path."""
    assert not is_rate_limit(exc)


def test_the_message_names_the_model_and_the_limit():
    err = ModelRateLimitError.from_exception(_Err(OPENAI_RPD))
    text = str(err)
    assert "gpt-4o-mini" in text
    assert "requests per day" in text
    assert err.model == "gpt-4o-mini"


def test_a_daily_limit_does_not_repeat_the_providers_misleading_countdown():
    """OpenAI says "try again in 8.64s" for a limit that resets tomorrow.
    Echoing that reads as "nearly over" when the real wait is ~24 hours."""
    text = str(ModelRateLimitError.from_exception(_Err(OPENAI_RPD)))
    assert "8.64" not in text
    assert "per-day limit will not clear" in text


def test_a_per_minute_limit_does_keep_the_countdown():
    text = str(ModelRateLimitError.from_exception(_Err(OPENAI_RPM)))
    assert "2s" in text
    assert "concurrency" in text


def test_the_message_says_what_to_change():
    """An error the user can act on names the knob, not just the symptom."""
    text = str(ModelRateLimitError.from_exception(_Err(OPENAI_RPD)))
    assert "AXIS2_NER_BATCH_SIZE" in text or "account tier" in text


def test_an_unparseable_provider_message_still_produces_something_usable():
    text = str(ModelRateLimitError.from_exception(_Err("429 rate limit")))
    assert "rate limit" in text.lower()
    assert len(text) > 40
