"""The Redis backends, exercised against fakeredis.

These are the production implementations of Protocols the rest of the system already uses, so
what matters is that they behave identically to the in-memory ones -- otherwise the difference
shows up in production as an environment-dependent bug rather than here as a test failure.
"""

from __future__ import annotations

import pytest

from xlforecast.api.security import ConfirmationError, TokenService
from xlforecast.schemas.jobs import JobProgress, JobRecord, JobStatus
from xlforecast.schemas.request import DataMapping, ForecastRequest
from xlforecast.storage.jobs import UnknownJobError
from xlforecast.storage.redis_backend import RedisJobStore, RedisReplayStore

fakeredis = pytest.importorskip("fakeredis")

MAPPING = DataMapping(unique_id_col="a", ds_col="b", y_col="c")
REQUEST = ForecastRequest(h=4, freq="ME")


@pytest.fixture
def client():
    return fakeredis.FakeStrictRedis()


@pytest.fixture
def store(client):
    return RedisJobStore(client=client)


def record(job_id: str = "j", owner: str = "marco", **kw) -> JobRecord:
    return JobRecord(
        job_id=job_id,
        data_id="d",
        request=REQUEST,
        mapping=MAPPING,
        owner=owner,
        created_at="2026-08-21T00:00:00Z",
        **kw,
    )


class TestJobStoreParity:
    """Same behaviour as InMemoryJobStore, which the API and worker are written against."""

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
                updated_at="t",
            )
        )
        assert store.progress("j").current_model == "AutoETS"

    def test_absent_progress_is_none(self, store):
        store.create(record())
        assert store.progress("j") is None

    def test_cancellation_flag(self, store):
        store.create(record())
        assert not store.cancel_requested("j")
        store.request_cancel("j")
        assert store.cancel_requested("j")

    def test_cancelling_an_unknown_job_raises(self, store):
        with pytest.raises(UnknownJobError):
            store.request_cancel("ghost")

    def test_owner_index_and_active_count(self, store):
        store.create(record("a"))
        store.create(record("b", status=JobStatus.COMPLETED))
        store.create(record("c", owner="someone-else"))
        assert {r.job_id for r in store.list_for_owner("marco")} == {"a", "b"}
        assert store.active_count("marco") == 1

    def test_a_record_expiring_out_of_the_owner_index_is_tolerated(self, store, client):
        """The index is a hint. A stale member must not fail a quota check."""
        store.create(record("a"))
        client.delete("xlf:job:a")
        assert store.list_for_owner("marco") == []
        assert store.active_count("marco") == 0


class TestSharedReplayStore:
    """The bug this fixes: two API instances would each accept the same token once."""

    def test_a_signature_can_be_claimed_only_once(self, client):
        replay = RedisReplayStore(client=client)
        assert replay.claim("sig")
        assert not replay.claim("sig")

    def test_two_instances_share_the_claim(self, client):
        """Both point at the same Redis, so whichever gets there first wins -- which is the
        property an in-process set cannot provide."""
        first = TokenService(secret=b"s", replay=RedisReplayStore(client=client))
        second = TokenService(secret=b"s", replay=RedisReplayStore(client=client))
        token = first.mint("d1", REQUEST)
        first.redeem(token, "d1", REQUEST)
        with pytest.raises(ConfirmationError, match="already been used"):
            second.redeem(token, "d1", REQUEST)

    def test_without_a_shared_store_two_instances_both_accept(self, client):
        """Documents precisely what is wrong with the in-process fallback, so nobody deploys
        several instances on it by accident."""
        first = TokenService(secret=b"s")
        second = TokenService(secret=b"s")
        token = first.mint("d1", REQUEST)
        first.redeem(token, "d1", REQUEST)
        second.redeem(token, "d1", REQUEST)  # accepted -- the flaw, made explicit

    def test_distinct_signatures_are_independent(self, client):
        replay = RedisReplayStore(client=client)
        assert replay.claim("one")
        assert replay.claim("two")
