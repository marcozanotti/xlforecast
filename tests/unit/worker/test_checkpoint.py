"""FR-801 / FR-802 -- resume and cancellation, at the engine level.

Gate G4 asserts these over HTTP; this file asserts the mechanism they rest on, because a
resume that silently recomputes is indistinguishable from a working one at the API boundary.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl
import pytest

from xlforecast.engine.run import run_from_frame
from xlforecast.schemas.request import DataMapping, ForecastRequest
from xlforecast.storage.objects import LocalObjectStore, MemoryObjectStore
from xlforecast.worker.checkpoint import Checkpointer, RunControl

MAPPING = DataMapping(unique_id_col="unique_id", ds_col="ds", y_col="y")
REQUEST = ForecastRequest(
    h=6,
    freq="ME",
    n_windows=3,
    models=["SeasonalNaive", "WindowAverage", "HistoricAverage"],
    ensemble="none",
)


def panel(n_series: int = 3, n_obs: int = 90) -> pl.DataFrame:
    rng = np.random.default_rng(31)
    frames = []
    for i in range(n_series):
        dates = pd.date_range("2016-01-31", periods=n_obs, freq="ME")
        t = np.arange(n_obs)
        y = 200 + 30 * np.sin(2 * np.pi * t / 12) + rng.normal(0, 5, n_obs)
        frames.append(
            pl.DataFrame({"unique_id": [f"S{i}"] * n_obs, "ds": list(dates), "y": y.tolist()})
        )
    return pl.concat(frames)


class TestCheckpointStore:
    def test_a_saved_fold_round_trips(self):
        store = MemoryObjectStore()
        cp = Checkpointer(job_id="j", store=store)
        run_from_frame(
            panel(),
            request=REQUEST,
            mapping=MAPPING,
            job_id="j",
            control=RunControl(checkpointer=cp),
        )
        assert cp.completed() == {0, 1, 2}
        loaded = cp.load(0)
        assert loaded is not None
        scores, predictions = loaded
        assert all(s.fold_index == 0 for s in scores)
        # The predictions half matters as much as the scores: without it the conformal
        # layer has no residuals and a resumed run silently loses its intervals.
        assert not predictions.is_empty()
        assert set(predictions.columns) >= {"unique_id", "ds", "model", "y_hat"}

    def test_a_fold_missing_its_predictions_is_not_resumable(self):
        """Scores alone would resume without error and produce a run with no intervals --
        a silent degradation, so a half-written fold must simply be recomputed."""
        store = MemoryObjectStore()
        cp = Checkpointer(job_id="j", store=store)
        run_from_frame(
            panel(),
            request=REQUEST,
            mapping=MAPPING,
            job_id="j",
            control=RunControl(checkpointer=cp),
        )
        store.delete("jobs/j/folds/0001.parquet")
        assert cp.completed() == {0, 2}
        assert cp.load(1) is None

    def test_a_resumed_run_still_produces_intervals(self):
        """The regression this whole mechanism nearly shipped: resume succeeded, and the
        forecast came back bare."""
        store = MemoryObjectStore()
        run_from_frame(
            panel(),
            request=REQUEST,
            mapping=MAPPING,
            job_id="j",
            control=RunControl(checkpointer=Checkpointer(job_id="j", store=store)),
        )
        resumed = run_from_frame(
            panel(),
            request=REQUEST,
            mapping=MAPPING,
            job_id="j",
            control=RunControl(checkpointer=Checkpointer(job_id="j", store=store)),
        )
        quantities = {r.quantity for r in resumed.forecast.rows}
        assert {"point", "lo", "hi"} <= quantities
        assert resumed.calibration

    def test_completed_is_empty_before_anything_runs(self):
        assert Checkpointer(job_id="fresh", store=MemoryObjectStore()).completed() == set()

    def test_clear_removes_every_fold(self):
        store = MemoryObjectStore()
        cp = Checkpointer(job_id="j", store=store)
        run_from_frame(
            panel(),
            request=REQUEST,
            mapping=MAPPING,
            job_id="j",
            control=RunControl(checkpointer=cp),
        )
        cp.clear()
        assert cp.completed() == set()

    def test_checkpoints_are_written_atomically(self, tmp_path):
        """A torn checkpoint is worse than a missing one: a missing fold simply re-runs,
        a half-written one deserialises into nonsense."""
        store = LocalObjectStore(tmp_path)
        store.put("a/b.json", b"[]")
        assert not list(tmp_path.rglob("*.partial"))
        assert store.get("a/b.json") == b"[]"

    def test_keys_cannot_escape_the_store_root(self, tmp_path):
        """Job ids arrive from requests; '../' in one must not write outside the store."""
        store = LocalObjectStore(tmp_path / "root")
        store.put("../../escape.json", b"x")
        assert (tmp_path / "root" / "escape.json").exists()
        assert not (tmp_path / "escape.json").exists()


class TestResume:
    """FR-801 -- surviving a restart means resuming, not starting again."""

    def test_a_second_run_reuses_checkpointed_folds(self):
        store = MemoryObjectStore()
        first = RunControl(checkpointer=Checkpointer(job_id="j", store=store))
        run_from_frame(panel(), request=REQUEST, mapping=MAPPING, job_id="j", control=first)
        assert first.resumed_folds == set()

        second = RunControl(checkpointer=Checkpointer(job_id="j", store=store))
        run_from_frame(panel(), request=REQUEST, mapping=MAPPING, job_id="j", control=second)
        assert second.resumed_folds == {0, 1, 2}, "every fold should have been recovered"

    def test_a_resumed_run_produces_the_same_leaderboard(self):
        """The point of resuming: the answer must not depend on how many times the worker
        died along the way."""
        store = MemoryObjectStore()
        fresh = run_from_frame(panel(), request=REQUEST, mapping=MAPPING, job_id="a")
        run_from_frame(
            panel(),
            request=REQUEST,
            mapping=MAPPING,
            job_id="j",
            control=RunControl(checkpointer=Checkpointer(job_id="j", store=store)),
        )
        resumed = run_from_frame(
            panel(),
            request=REQUEST,
            mapping=MAPPING,
            job_id="j",
            control=RunControl(checkpointer=Checkpointer(job_id="j", store=store)),
        )
        assert fresh.leaderboard.model_dump_json() == resumed.leaderboard.model_dump_json()

    def test_resume_can_be_disabled(self):
        store = MemoryObjectStore()
        run_from_frame(
            panel(),
            request=REQUEST,
            mapping=MAPPING,
            job_id="j",
            control=RunControl(checkpointer=Checkpointer(job_id="j", store=store)),
        )
        control = RunControl(checkpointer=Checkpointer(job_id="j", store=store), resume=False)
        run_from_frame(panel(), request=REQUEST, mapping=MAPPING, job_id="j", control=control)
        assert control.resumed_folds == set()

    def test_a_partial_checkpoint_resumes_only_what_exists(self):
        """The realistic case: a worker died after fold 0."""
        store = MemoryObjectStore()
        cp = Checkpointer(job_id="j", store=store)
        run_from_frame(
            panel(),
            request=REQUEST,
            mapping=MAPPING,
            job_id="j",
            control=RunControl(checkpointer=cp),
        )
        store.delete("jobs/j/folds/0002.json")
        control = RunControl(checkpointer=Checkpointer(job_id="j", store=store))
        run_from_frame(panel(), request=REQUEST, mapping=MAPPING, job_id="j", control=control)
        assert control.resumed_folds == {0, 1}


class TestCancellation:
    """FR-802 -- the between-folds half. Killing a fold in flight is the worker's job."""

    def test_a_stop_request_ends_the_run_early(self):
        control = RunControl(should_stop=lambda: True)
        result = run_from_frame(
            panel(), request=REQUEST, mapping=MAPPING, job_id="j", control=control
        )
        assert result.fold_scores == []

    def test_completed_folds_are_retained_when_stopping(self):
        """Cancelling should not throw away work already done -- FR-803 relies on the same
        behaviour when a job runs out of quota mid-run."""
        seen: list[int] = []

        def stop_after_first() -> bool:
            return len(seen) >= 1

        def record(*, fold_index: int, models_done: int, current_model: str | None) -> None:
            seen.append(fold_index)

        control = RunControl(should_stop=stop_after_first, progress=record)
        result = run_from_frame(
            panel(), request=REQUEST, mapping=MAPPING, job_id="j", control=control
        )
        assert seen == [0]
        assert {s.fold_index for s in result.fold_scores} == {0}

    def test_no_control_means_no_behaviour_change(self):
        """ADR-001 -- the engine runs standalone. The CLI passes no control and must be
        unaffected by any of this."""
        with_control = run_from_frame(
            panel(), request=REQUEST, mapping=MAPPING, job_id="a", control=RunControl()
        )
        without = run_from_frame(panel(), request=REQUEST, mapping=MAPPING, job_id="b")
        assert with_control.leaderboard.model_dump_json() == without.leaderboard.model_dump_json()


class TestProgressReporting:
    def test_progress_is_reported_per_fold(self):
        seen = []

        def record(*, fold_index: int, models_done: int, current_model: str | None) -> None:
            seen.append((fold_index, models_done))

        run_from_frame(
            panel(),
            request=REQUEST,
            mapping=MAPPING,
            job_id="j",
            control=RunControl(progress=record),
        )
        assert [f for f, _ in seen] == [0, 1, 2]
        assert all(done == 3 for _, done in seen)


def test_object_store_reports_a_missing_key_rather_than_returning_empty():
    from xlforecast.storage.objects import ObjectNotFoundError

    with pytest.raises(ObjectNotFoundError):
        MemoryObjectStore().get("absent")
