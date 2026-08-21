"""The tabular ingest endpoint: admin-gated, dry-run by default, no secrets out.

Structured data had no HTTP surface at all -- CSVs and databases could only be
loaded from the command line, while documents had two endpoints and a UI tab.
"""
from pathlib import Path

REPO = next(p for p in Path(__file__).resolve().parents if (p / "src").is_dir())
ROUTES = (REPO / "src" / "interface" / "routes" / "ingest.py").read_text()
SCHEMAS = (REPO / "src" / "interface" / "schemas.py").read_text()


def _endpoint_body() -> str:
    decorator = '@router.post("/ingest/tabular"'
    assert decorator in ROUTES, "POST /ingest/tabular is not registered"
    return ROUTES.split(decorator, 1)[1].split("\n@router.", 1)[0]


def test_endpoint_is_registered_and_admin_gated():
    """It writes to the graph and a database source carries credentials, so it
    cannot be reachable by an ordinary user."""
    assert "resolve_admin_session" in _endpoint_body()


def test_writing_is_opt_in():
    """Labels are inferred from table names, so a fixture and a production
    dataset can infer the same label and the second load rewrites the first.
    Reviewing the plan first is the safety property, so `load` must default to
    False -- a caller who omits it must never write."""
    assert "load: bool = Field(\n        default=False," in SCHEMAS
    body = _endpoint_body()
    assert "if request.load:" in body      # audit only on a real write
    assert "dry_run=not request.load" in body


def test_the_response_carries_the_sanitised_source():
    """source_tag strips the password; the raw URL must never be echoed, since
    this response is logged and rendered in a browser."""
    body = _endpoint_body()
    assert "source_tag" in ROUTES
    assert "request.source" not in body.split("return TabularIngestResponse")[1]


def test_failures_are_reported_as_client_or_upstream_errors():
    """An unreadable source is not a bug in this service. A 500 would say it
    is, and would put the exception text -- which can contain the connection
    string -- into the response."""
    body = _endpoint_body()
    assert "status_code=400" in body      # unsupported source shape
    assert "status_code=502" in body      # could not connect / read
    assert "type(exc).__name__" in body   # class only, never the message


def test_the_work_runs_off_the_event_loop():
    """Reflection opens a connection and a load streams every row; either on
    the loop would block every other request."""
    assert "run_in_executor" in _endpoint_body()
