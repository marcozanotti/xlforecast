"""NFR-08 -- deleting panel data after the retention window."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from xlforecast.schemas.jobs import JobRecord, JobStatus
from xlforecast.schemas.request import DataMapping, ForecastRequest
from xlforecast.storage.jobs import InMemoryJobStore
from xlforecast.storage.objects import MemoryObjectStore
from xlforecast.storage.retention import RetentionPolicy

MAPPING = DataMapping(unique_id_col="a", ds_col="b", y_col="c")
NOW = datetime(2026, 8, 21, tzinfo=UTC)


def record(job_id: str, *, age_days: int, status: JobStatus = JobStatus.COMPLETED) -> JobRecord:
    finished = (NOW - timedelta(days=age_days)).isoformat()
    return JobRecord(
        job_id=job_id,
        data_id=f"data-{job_id}",
        request=ForecastRequest(h=4, freq="ME"),
        mapping=MAPPING,
        owner="marco",
        created_at=finished,
        finished_at=None if status is JobStatus.RUNNING else finished,
        status=status,
    )


@pytest.fixture
def setup():
    objects, jobs = MemoryObjectStore(), InMemoryJobStore()
    policy = RetentionPolicy(objects=objects, jobs=jobs, retention_days=30)
    return objects, jobs, policy


def populate(objects, job_id: str) -> None:
    objects.put(f"data/data-{job_id}.parquet", b"panel")
    objects.put(f"jobs/{job_id}/folds/0000.json", b"[]")
    objects.put(f"jobs/{job_id}/folds/0000.parquet", b"preds")
    objects.put(f"jobs/{job_id}/result.json", b"{}")
    objects.put(f"jobs/{job_id}/manifest.json", b"{}")


class TestSweep:
    def test_an_expired_job_loses_its_panel_and_checkpoints(self, setup):
        objects, jobs, policy = setup
        jobs.create(record("old", age_days=40))
        populate(objects, "old")

        report = policy.sweep("marco", now=NOW)
        assert report.panels_deleted == 1
        assert report.checkpoints_deleted == 2
        assert not objects.exists("data/data-old.parquet")

    def test_results_and_manifests_survive(self, setup):
        """Kept deliberately. The v2 forecast-stability feature compares a forecast against a
        previous cycle's, and deleting manifests would make that impossible without a
        migration. A leaderboard is error metrics, not customer observations."""
        objects, jobs, policy = setup
        jobs.create(record("old", age_days=40))
        populate(objects, "old")
        policy.sweep("marco", now=NOW)
        assert objects.exists("jobs/old/result.json")
        assert objects.exists("jobs/old/manifest.json")

    def test_a_job_inside_the_window_is_untouched(self, setup):
        objects, jobs, policy = setup
        jobs.create(record("recent", age_days=10))
        populate(objects, "recent")
        report = policy.sweep("marco", now=NOW)
        assert report.panels_deleted == 0
        assert objects.exists("data/data-recent.parquet")

    def test_a_running_job_is_skipped_however_old(self, setup):
        """Deleting the panel out from under a live job would fail it in a way the user
        could not act on."""
        objects, jobs, policy = setup
        jobs.create(record("live", age_days=99, status=JobStatus.RUNNING))
        populate(objects, "live")
        report = policy.sweep("marco", now=NOW)
        assert report.jobs_skipped_active == 1
        assert objects.exists("data/data-live.parquet")

    def test_the_window_is_configurable(self, setup):
        objects, jobs, _ = setup
        jobs.create(record("job", age_days=10))
        populate(objects, "job")
        strict = RetentionPolicy(objects=objects, jobs=jobs, retention_days=7)
        assert strict.sweep("marco", now=NOW).panels_deleted == 1

    def test_the_report_states_what_happened(self, setup):
        """Returned rather than logged, so the FR-805 audit line carries real numbers."""
        objects, jobs, policy = setup
        jobs.create(record("a", age_days=40))
        jobs.create(record("b", age_days=1))
        populate(objects, "a")
        populate(objects, "b")
        report = policy.sweep("marco", now=NOW)
        assert report.jobs_considered == 2
        assert report.panels_deleted == 1

    def test_another_owners_data_is_untouched(self, setup):
        objects, jobs, policy = setup
        jobs.create(record("mine", age_days=40))
        other = record("theirs", age_days=40).model_copy(update={"owner": "someone-else"})
        jobs.create(other)
        populate(objects, "mine")
        populate(objects, "theirs")
        policy.sweep("marco", now=NOW)
        assert objects.exists("data/data-theirs.parquet")

    def test_sweeping_twice_is_safe(self, setup):
        objects, jobs, policy = setup
        jobs.create(record("old", age_days=40))
        populate(objects, "old")
        policy.sweep("marco", now=NOW)
        assert policy.sweep("marco", now=NOW).panels_deleted == 0


class TestRetainsPredicate:
    @pytest.mark.parametrize(
        ("key", "kept"),
        [
            ("data/data-1.parquet", False),
            ("jobs/j/folds/0000.json", False),
            ("jobs/j/folds/0000.parquet", False),
            ("jobs/j/result.json", True),
            ("jobs/j/manifest.json", True),
        ],
    )
    def test_classifies_keys(self, setup, key, kept):
        _, _, policy = setup
        assert policy.retains(key) is kept
