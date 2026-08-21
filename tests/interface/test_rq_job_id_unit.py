"""The RQ job id must satisfy RQ's own rules.

rq 2.x rejects any job id outside [A-Za-z0-9_-]. `enqueue_ingest` converts
every failure into None, and the dispatcher reads None as "no queue
configured" and runs the job in-process. So an invalid id does not surface
as an error: ingestion still completes, serially, in the web process, while
the workers sit idle waiting for jobs that were never enqueued.
"""
import re
import sys
import types

import pytest

from src.pipeline.ingestion import queue as queue_module

#: rq's own constraint (rq.job.Job.__init__ / _id setter).
RQ_ALLOWED_JOB_ID = re.compile(r"^[A-Za-z0-9_-]+$")


class _CapturingQueue:
    def __init__(self):
        self.kwargs = None

    def enqueue(self, *args, **kwargs):
        self.kwargs = kwargs
        return type("Job", (), {"id": kwargs.get("job_id")})()


@pytest.fixture
def rq_available(monkeypatch):
    """rq is a deployment dependency, not a test one.

    `monkeypatch.setitem` rather than assigning into sys.modules directly, so
    the stub is removed afterwards instead of leaking into every module
    imported by a later test.
    """
    rq_stub = types.ModuleType("rq")
    rq_stub.Retry = lambda **kwargs: ("retry", kwargs)
    monkeypatch.setitem(sys.modules, "rq", rq_stub)

    tasks_stub = types.ModuleType("src.pipeline.ingestion.tasks")
    tasks_stub.run_ingest_job = lambda job_id: None
    monkeypatch.setitem(sys.modules, "src.pipeline.ingestion.tasks", tasks_stub)


@pytest.fixture
def captured(monkeypatch, rq_available):
    q = _CapturingQueue()
    monkeypatch.setattr(queue_module, "get_ingest_queue", lambda: q)
    return q


@pytest.mark.parametrize(
    "job_id",
    [
        "69de71eee139410c9917ec42a5b8ad5f",   # a real uuid4().hex
        "abc123",
        "a" * 32,
    ],
)
def test_the_enqueued_job_id_is_one_rq_accepts(captured, job_id):
    queue_module.enqueue_ingest(job_id)

    assert captured.kwargs is not None, "enqueue was never called"
    assert RQ_ALLOWED_JOB_ID.match(captured.kwargs["job_id"]), (
        f"rq 2.x rejects {captured.kwargs['job_id']!r}; every enqueue would "
        "fail and silently fall back to in-process ingestion"
    )


def test_the_id_still_identifies_the_job_it_came_from(captured):
    queue_module.enqueue_ingest("69de71eee139410c9917ec42a5b8ad5f")

    assert "69de71eee139410c9917ec42a5b8ad5f" in captured.kwargs["job_id"]


def test_a_failed_enqueue_returns_none_rather_than_raising(monkeypatch):
    """Documents the fallback that hid this: the caller cannot tell an
    invalid id from an absent queue."""
    class _Failing:
        def enqueue(self, *a, **k):
            raise ValueError("Job ID must only contain letters, numbers, underscores and dashes")

    monkeypatch.setattr(queue_module, "get_ingest_queue", lambda: _Failing())
    assert queue_module.enqueue_ingest("x") is None
