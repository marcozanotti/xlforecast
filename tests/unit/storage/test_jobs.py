"""Job state (FR-801, FR-802, FR-803)."""

from __future__ import annotations

import pytest

from xlforecast.schemas.jobs import JobProgress, JobRecord, JobStatus
from xlforecast.schemas.request import DataMapping, ForecastRequest
from xlforecast.storage.jobs import InMemoryJobStore, UnknownJobError

MAPPING = DataMapping(unique_id_col="a", ds_col="b", y_col="c")


def record(job_id: str = "j", owner: str = "marco", **kw) -> JobRecord:
    return JobRecord(
        job_id=job_id,
        data_id="d",
        request=ForecastRequest(h=4, freq="ME"),
        mapping=MAPPING,
        owner=owner,
        created_at="2026-08-21T00:00:00Z",
        **kw,
    )


@pytest.fixture
def store():
    return InMemoryJobStore()


class TestLifecycle:
    def test_create_then_get(self, store):
        store.create(record())
        assert store.get("j").owner == "marco"

    def test_unknown_job_raises_with_a_remedy(self, store):
        with pytest.raises(UnknownJobError) as exc:
            store.get("ghost")
        assert exc.value.fix

    def test_update_replaces_the_record(self, store):
        store.create(record())
        store.update(store.get("j").model_copy(update={"status": JobStatus.RUNNING}))
        assert store.get("j").status is JobStatus.RUNNING


class TestCancellation:
    """FR-802 -- a flag the engine polls, not an asyncio cancel it would never notice."""

    def test_cancel_sets_the_flag(self, store):
        store.create(record())
        assert not store.cancel_requested("j")
        store.request_cancel("j")
        assert store.cancel_requested("j")

    def test_cancelling_an_unknown_job_raises_rather_than_silently_doing_nothing(self, store):
        with pytest.raises(UnknownJobError):
            store.request_cancel("ghost")


class TestQuotaAccounting:
    """FR-803 -- concurrency is counted over non-terminal jobs."""

    @pytest.mark.parametrize(
        ("status", "counts"),
        [
            (JobStatus.QUEUED, True),
            (JobStatus.RUNNING, True),
            (JobStatus.COMPLETED, False),
            (JobStatus.FAILED, False),
            (JobStatus.CANCELLED, False),
            (JobStatus.QUOTA_EXHAUSTED, False),
        ],
    )
    def test_only_live_jobs_count_towards_concurrency(self, store, status, counts):
        store.create(record(status=status))
        assert store.active_count("marco") == (1 if counts else 0)

    def test_other_owners_do_not_count(self, store):
        store.create(record("a", owner="marco"))
        store.create(record("b", owner="someone-else"))
        assert store.active_count("marco") == 1


class TestProgress:
    def test_progress_round_trips(self, store):
        store.create(record())
        store.set_progress(
            JobProgress(
                job_id="j",
                status=JobStatus.RUNNING,
                folds_total=3,
                folds_done=1,
                models_total=5,
                models_done_in_fold=2,
                current_model="AutoETS",
                updated_at="2026-08-21T00:00:01Z",
            )
        )
        assert store.progress("j").current_model == "AutoETS"

    def test_absent_progress_is_none_rather_than_a_zeroed_record(self, store):
        store.create(record())
        assert store.progress("j") is None

    def test_fraction_is_computed_from_folds_and_models(self):
        progress = JobProgress(
            job_id="j",
            status=JobStatus.RUNNING,
            folds_total=4,
            folds_done=2,
            models_total=5,
            models_done_in_fold=1,
            updated_at="t",
        )
        assert progress.fraction == pytest.approx(11 / 20)

    def test_fraction_of_an_empty_job_is_zero_not_a_division_error(self):
        progress = JobProgress(
            job_id="j",
            status=JobStatus.QUEUED,
            folds_total=0,
            folds_done=0,
            models_total=0,
            models_done_in_fold=0,
            updated_at="t",
        )
        assert progress.fraction == 0.0


def test_terminal_states_are_exactly_the_finished_ones():
    assert {s for s in JobStatus if s.terminal} == {
        JobStatus.COMPLETED,
        JobStatus.FAILED,
        JobStatus.CANCELLED,
        JobStatus.QUOTA_EXHAUSTED,
    }
