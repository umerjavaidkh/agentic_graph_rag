"""The settings screen changes tuning dials — and nothing else.

The risk in exposing configuration over HTTP is not that someone sets a
batch size badly; it is that the same door opens onto credentials and the
safety switches. So the allow-list is the security boundary, and these
tests exist mainly to hold that line.
"""
import pytest

from src.shared.config.settings_schema import (
    SETTINGS,
    Setting,
    _BY_NAME,
    _coerce,
)

#: Anything matching these must never be settable through a web page.
FORBIDDEN = [
    "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GEMINI_API_KEY",
    "NEO4J_PASSWORD", "NEO4J_URI", "NEO4J_USER",
    "ALLOW_CYPHER_INGEST", "ALLOW_DB_RESET",
    "OIDC_CLIENT_SECRET", "MINIO_SECRET_KEY", "REDIS_URL",
]


@pytest.mark.parametrize("name", FORBIDDEN)
def test_credentials_and_safety_switches_are_not_exposed(name):
    assert name not in _BY_NAME, f"{name} must not be settable from the UI"


def test_no_setting_looks_like_a_secret():
    """Catches a future addition nobody thought about."""
    for setting in SETTINGS:
        upper = setting.name.upper()
        assert not any(w in upper for w in ("KEY", "SECRET", "PASSWORD", "TOKEN", "CREDENTIAL")), \
            f"{setting.name} looks like a credential"


def test_every_setting_explains_itself():
    """A dial with no explanation is a dial nobody can safely turn."""
    for setting in SETTINGS:
        assert len(setting.help) > 30, f"{setting.name} has no useful help text"
        assert setting.default != "", f"{setting.name} has no default"


def test_numeric_settings_are_bounded():
    """An unbounded worker count or batch size is a footgun, not a setting."""
    for setting in SETTINGS:
        if setting.kind in ("int", "float"):
            assert setting.minimum is not None, f"{setting.name} has no minimum"
            assert setting.maximum is not None, f"{setting.name} has no maximum"


# ── validation ────────────────────────────────────────────────────────────

def _s(**kw):
    base = dict(name="X", group="g", kind="int", help="h" * 40, default="1",
                minimum=1, maximum=10)
    base.update(kw)
    return Setting(**base)


@pytest.mark.parametrize("raw, expected", [("5", "5"), ("5.0", "5"), (" 7 ", "7")])
def test_integers_are_normalised(raw, expected):
    assert _coerce(_s(), raw) == expected


@pytest.mark.parametrize("raw", ["0", "11", "-3"])
def test_out_of_range_is_refused(raw):
    with pytest.raises(ValueError, match="at least|at most"):
        _coerce(_s(), raw)


def test_non_numeric_is_refused():
    with pytest.raises(ValueError, match="must be a number"):
        _coerce(_s(), "abc")


@pytest.mark.parametrize("raw, expected", [
    ("true", "true"), ("TRUE", "true"), ("1", "true"), ("yes", "true"),
    ("false", "false"), ("no", "false"), ("", "false"),
])
def test_booleans_accept_what_people_actually_type(raw, expected):
    assert _coerce(_s(kind="bool", minimum=None, maximum=None), raw) == expected


def test_a_choice_outside_the_list_is_refused():
    setting = _s(kind="choice", choices=["openai", "gemini"], minimum=None, maximum=None)
    assert _coerce(setting, "gemini") == "gemini"
    with pytest.raises(ValueError, match="must be one of"):
        _coerce(setting, "hackerllm")


def test_empty_text_is_refused():
    """An empty model name would fail later, at a much less obvious place."""
    with pytest.raises(ValueError, match="cannot be empty"):
        _coerce(_s(kind="text", minimum=None, maximum=None), "   ")


def test_worker_replicas_says_it_needs_compose():
    """Restarting is not enough for this one — it changes container count."""
    assert _BY_NAME["WORKER_REPLICAS"].applies_to == "compose"


def test_worker_side_settings_are_marked_as_such():
    """Restarting only the API would appear to do nothing for these."""
    assert _BY_NAME["AXIS2_NER_BATCH_SIZE"].applies_to == "workers"
