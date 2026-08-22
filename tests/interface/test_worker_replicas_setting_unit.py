"""Ingestion concurrency must be configuration, not a remembered flag.

rq runs one job per worker process, so the worker replica count *is* the
number of documents ingested at once. It used to be settable only by
passing `--scale worker=N` on the command line, which meant a laptop and a
cloud deployment differed by what someone typed rather than by their
config -- and that a plain `docker compose up -d` silently reset it.
"""
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
COMPOSE = ROOT / "docker-compose.yml"


@pytest.fixture(scope="module")
def compose() -> str:
    return COMPOSE.read_text()


@pytest.fixture(scope="module")
def worker_block(compose: str) -> str:
    """Just the worker service, so assertions cannot match another service."""
    match = re.search(r"\n  worker:\n(.*?)(?=\n  [a-z_-]+:\n)", compose, re.S)
    assert match, "worker service not found in docker-compose.yml"
    return match.group(1)


def test_the_worker_count_comes_from_an_env_var(worker_block: str):
    assert "WORKER_REPLICAS" in worker_block


def test_it_has_a_default_so_a_fresh_clone_still_runs(worker_block: str):
    """Someone with no .env entry must still get workers, not zero."""
    match = re.search(r"replicas:\s*\$\{WORKER_REPLICAS:-(\d+)\}", worker_block)
    assert match, "replicas is not defaulted"
    assert int(match.group(1)) >= 1


def test_the_default_is_modest_enough_for_a_laptop(worker_block: str):
    """Each replica multiplies with AXIS2_NER_CONCURRENCY, so the default
    must not saturate a provider's per-minute limit out of the box."""
    default = int(re.search(r"replicas:\s*\$\{WORKER_REPLICAS:-(\d+)\}", worker_block).group(1))
    assert default <= 4


def test_the_env_example_documents_it():
    """A knob nobody can find is not configurable."""
    example = (ROOT / ".env.example").read_text()
    assert "WORKER_REPLICAS" in example
    # It multiplies with the per-worker model concurrency; saying so is the
    # difference between a usable knob and one that quietly hits a 429.
    assert "AXIS2_NER_CONCURRENCY" in example


def test_the_worker_still_runs_rq_not_the_api(worker_block: str):
    """Guards the regression where the entrypoint discarded this command and
    every 'worker' started a second copy of the API."""
    assert "rq" in worker_block and "worker" in worker_block
