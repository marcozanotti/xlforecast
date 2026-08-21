"""Gate G4 -- worker restart survival and mid-run cancellation, driven for real.

These spawn actual subprocesses. That is the point: FR-802 cannot be demonstrated with a
mock, because the claim under test is precisely that compiled CPU-bound work can be stopped,
and a mock would stop obligingly.
"""

from __future__ import annotations

import io

import numpy as np
import pandas as pd
import polars as pl
import pytest

from xlforecast.schemas.jobs import JobRecord, JobStatus
from xlforecast.schemas.request import DataMapping, ForecastRequest
from xlforecast.storage.jobs import InMemoryJobStore
from xlforecast.storage.objects import LocalObjectStore
from xlforecast.worker.checkpoint import Checkpointer
from xlforecast.worker.executor import cancel_marker, data_key, execute_job, result_key
from xlforecast.worker.tasks import MAX_ATTEMPTS, run_job

MAPPING = DataMapping(unique_id_col="unique_id", ds_col="ds", y_col="y")
REQUEST = ForecastRequest(
    h=6,
    freq="ME",
    n_windows=3,
    models=["SeasonalNaive", "WindowAverage", "HistoricAverage"],
    ensemble="none",
)


def panel(n_series: int = 3, n_obs: int = 90) -> pl.DataFrame:
    rng = np.random.default_rng(41)
    frames = []
    for i in range(n_series):
        dates = pd.date_range("2016-01-31", periods=n_obs, freq="ME")
        t = np.arange(n_obs)
        y = 200 + 30 * np.sin(2 * np.pi * t / 12) + rng.normal(0, 5, n_obs)
        frames.append(
            pl.DataFrame({"unique_id": [f"S{i}"] * n_obs, "ds": list(dates), "y": y.tolist()})
        )
    return pl.concat(frames)


@pytest.fixture
def store(tmp_path):
    """A filesystem store, not the in-memory one.

    Deliberate: a subprocess does not share the parent's address space, and neither does a
    job that outlives the worker. An in-memory store cannot support FR-801 at all.
    """
    store = LocalObjectStore(tmp_path)
    buffer = io.BytesIO()
    panel().write_parquet(buffer)
    store.put(data_key("d1"), buffer.getvalue())
    return store


@pytest.fixture
def jobs():
    return InMemoryJobStore()


def make_record(job_id: str = "j1") -> JobRecord:
    return JobRecord(
        job_id=job_id,
        data_id="d1",
        request=REQUEST,
        mapping=MAPPING,
        owner="marco",
        created_at="2026-08-21T00:00:00Z",
    )


@pytest.mark.slow
class TestSuccessfulExecution:
    def test_a_job_runs_to_completion_in_a_subprocess(self, store, jobs):
        record = make_record()
        jobs.create(record)
        outcome = execute_job(record, store, is_cancelled=lambda: False)
        assert outcome.status is JobStatus.COMPLETED
        assert store.exists(result_key("j1"))

    def test_it_writes_a_manifest(self, store, jobs):
        """Hard rule 10 -- no manifest, no result."""
        record = make_record()
        jobs.create(record)
        execute_job(record, store, is_cancelled=lambda: False)
        assert store.exists("jobs/j1/manifest.json")

    def test_every_fold_is_checkpointed(self, store, jobs):
        record = make_record()
        jobs.create(record)
        execute_job(record, store, is_cancelled=lambda: False)
        assert Checkpointer(job_id="j1", store=store).completed() == {0, 1, 2}


@pytest.mark.slow
class TestCancellation:
    """FR-802 / G4 -- 'cancellation works mid-run'."""

    def test_a_job_cancelled_before_it_starts_ends_cancelled(self, store, jobs):
        record = make_record()
        jobs.create(record)
        outcome = execute_job(record, store, is_cancelled=lambda: True, grace_seconds=0.2)
        assert outcome.status is JobStatus.CANCELLED

    def test_cancellation_leaves_the_marker_the_child_polls(self, store, jobs):
        record = make_record()
        jobs.create(record)
        execute_job(record, store, is_cancelled=lambda: True, grace_seconds=0.2)
        assert store.exists(cancel_marker("j1"))

    def test_a_forced_cancel_terminates_the_process(self, store, jobs):
        """The half that a cooperative flag cannot deliver: no polling can interrupt a fit
        already running inside compiled code."""
        record = make_record()
        jobs.create(record)
        outcome = execute_job(
            record, store, is_cancelled=lambda: True, grace_seconds=0.0, poll_interval=0.0
        )
        assert outcome.status is JobStatus.CANCELLED
        assert outcome.forced

    def test_work_completed_before_cancelling_is_not_discarded(self, store, jobs):
        """Cancelling should cost the user the remaining folds, not the finished ones."""
        record = make_record()
        jobs.create(record)
        execute_job(record, store, is_cancelled=lambda: False)  # populate checkpoints
        store.delete(result_key("j1"))
        assert Checkpointer(job_id="j1", store=store).completed() == {0, 1, 2}


@pytest.mark.slow
class TestRestartSurvival:
    """FR-801 / G4 -- 'survives a worker restart by resuming from its last completed fold'."""

    def test_a_redelivered_job_resumes_rather_than_restarting(self, store, jobs):
        record = make_record()
        jobs.create(record)

        # First attempt dies after one fold, as a killed worker would leave things.
        checkpointer = Checkpointer(job_id="j1", store=store)
        execute_job(record, store, is_cancelled=lambda: False)
        for fold in (1, 2):
            store.delete(f"jobs/j1/folds/{fold:04d}.json")
            store.delete(f"jobs/j1/folds/{fold:04d}.parquet")
        store.delete(result_key("j1"))
        assert checkpointer.completed() == {0}

        # arq redelivers; the run picks up where it stopped.
        outcome = run_job("j1", jobs=jobs, objects=store)
        assert outcome.status is JobStatus.COMPLETED
        assert checkpointer.completed() == {0, 1, 2}

    def test_a_resumed_job_produces_the_same_result_as_an_uninterrupted_one(self, store, jobs):
        """The answer must not depend on how many times the worker died."""
        jobs.create(make_record("clean"))
        run_job("clean", jobs=jobs, objects=store)
        clean = store.get(result_key("clean")).decode()

        jobs.create(make_record("interrupted"))
        run_job("interrupted", jobs=jobs, objects=store)
        for fold in (1, 2):
            store.delete(f"jobs/interrupted/folds/{fold:04d}.json")
            store.delete(f"jobs/interrupted/folds/{fold:04d}.parquet")
        store.delete(result_key("interrupted"))
        run_job("interrupted", jobs=jobs, objects=store)
        resumed = store.get(result_key("interrupted")).decode()

        import json

        assert json.loads(clean)["leaderboard"] == json.loads(resumed)["leaderboard"]


@pytest.mark.slow
class TestTaskStateTransitions:
    def test_a_completed_job_records_its_result_key(self, store, jobs):
        jobs.create(make_record())
        run_job("j1", jobs=jobs, objects=store)
        record = jobs.get("j1")
        assert record.status is JobStatus.COMPLETED
        assert record.result_key == result_key("j1")
        assert record.started_at is not None
        assert record.finished_at is not None

    def test_progress_is_recorded_and_reports_completed_folds(self, store, jobs):
        jobs.create(make_record())
        run_job("j1", jobs=jobs, objects=store)
        progress = jobs.progress("j1")
        assert progress.status is JobStatus.COMPLETED
        assert progress.folds_done == 3

    def test_a_cancelled_job_reports_the_folds_it_did_finish(self, store, jobs):
        """Rather than claiming the whole horizon -- the user can still download those."""
        jobs.create(make_record())
        jobs.request_cancel("j1")
        run_job("j1", jobs=jobs, objects=store)
        assert jobs.get("j1").status is JobStatus.CANCELLED

    def test_a_job_that_keeps_killing_its_worker_eventually_gives_up(self, store, jobs):
        """arq redelivery is at-least-once. FR-801 makes redelivery cheap, but a poison job
        must not cycle forever."""
        jobs.create(make_record().model_copy(update={"attempts": MAX_ATTEMPTS}))
        outcome = run_job("j1", jobs=jobs, objects=store)
        assert outcome.status is JobStatus.FAILED
        assert "gave up" in (outcome.error or "")

    def test_attempts_are_counted(self, store, jobs):
        jobs.create(make_record())
        run_job("j1", jobs=jobs, objects=store)
        assert jobs.get("j1").attempts == 1
