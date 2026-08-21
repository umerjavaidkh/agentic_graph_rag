"""The container entrypoint must run the command it is given.

docker-compose hands the worker `rq worker ingest`. An entrypoint that ends
at `exec uvicorn` regardless accepts that command and silently discards it:
every scaled worker starts a second API, nothing consumes the ingest queue,
and the only clue is that ingestion still works -- because the API falls
back to running jobs in-process when it cannot reach the queue.

Nothing else in the suite covers this: it is shell, it runs only in a
container, and the failure mode is a container that starts cleanly.
"""
from pathlib import Path

import pytest

ENTRYPOINT = Path(__file__).resolve().parents[2] / "scripts" / "docker-entrypoint.sh"


@pytest.fixture(scope="module")
def script() -> str:
    return ENTRYPOINT.read_text()


def test_the_entrypoint_exists_and_is_executable():
    assert ENTRYPOINT.exists()
    assert ENTRYPOINT.stat().st_mode & 0o111, "not executable"


def test_a_given_command_is_run_instead_of_the_api(script):
    assert 'exec "$@"' in script, "the entrypoint never runs the command it is given"


def test_the_command_wins_over_the_default(script):
    """Order is the whole bug: `exec uvicorn` first makes `exec "$@"` dead code."""
    assert script.index('exec "$@"') < script.index("exec uvicorn"), (
        "the API is started before the given command is dispatched, "
        "so a worker container would run the API instead"
    )


def test_the_api_is_still_the_default_with_no_command(script):
    assert "exec uvicorn" in script


def test_workers_do_not_race_to_seed_demo_data(script):
    """With `--scale worker=N`, seeding before dispatch means N containers
    load the same fixtures at once."""
    assert script.index('exec "$@"') < script.index("init_demo_data.py"), (
        "demo-data seeding runs before the command dispatch, so every "
        "scaled worker would seed too"
    )


def test_neo4j_is_awaited_for_workers_too(script):
    """A worker that starts before Neo4j is up fails its first job."""
    assert script.index("wait_for_neo4j.py") < script.index('exec "$@"')
