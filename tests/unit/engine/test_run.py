"""Gate G1 -- end-to-end engine behaviour.

Every assertion here corresponds to a G1 clause in `docs/03-BUILD-PLAN.md`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl
import pytest

from xlforecast.engine.run import run_from_frame
from xlforecast.schemas.request import DataMapping, ForecastRequest

MAPPING = DataMapping(unique_id_col="unique_id", ds_col="ds", y_col="y")
FAST = ["SeasonalNaive", "WindowAverage", "HistoricAverage"]


def build(
    series: dict[str, np.ndarray], start: str = "2021-01-03", freq: str = "W"
) -> pl.DataFrame:
    frames = []
    for uid, values in series.items():
        dates = pd.date_range(start, periods=len(values), freq=freq)
        frames.append(
            pl.DataFrame(
                {
                    "unique_id": [uid] * len(values),
                    "ds": list(dates),
                    "y": [float(v) for v in values],
                }
            )
        )
    return pl.concat(frames)


def seasonal_random_walk(n: int, m: int, seed: int) -> np.ndarray:
    """y_t = y_{t-m} + noise. SeasonalNaive is the optimal forecast here."""
    rng = np.random.default_rng(seed)
    y = np.zeros(n)
    y[:m] = 100 + rng.normal(0, 1, m)
    for t in range(m, n):
        y[t] = y[t - m] + rng.normal(0, 1)
    return y


def random_walk(n: int, seed: int) -> np.ndarray:
    """y_t = y_{t-1} + noise. Naive is optimal; SeasonalNaive is m/h times worse."""
    rng = np.random.default_rng(seed)
    return 100 + np.cumsum(rng.normal(0, 1, n))


@pytest.fixture(scope="module")
def seasonal_result():
    panel = build({f"S{i}": seasonal_random_walk(200, 52, i) for i in range(4)})
    request = ForecastRequest(h=13, freq="W", n_windows=3, models=FAST)
    return run_from_frame(panel, request=request, mapping=MAPPING, job_id="seasonal")


class TestEndToEnd:
    """G1: runs end to end on a panel, emits four tables plus a manifest."""

    def test_produces_a_leaderboard_a_forecast_and_a_manifest(self, seasonal_result):
        assert seasonal_result.leaderboard.rows
        assert seasonal_result.forecast.rows
        assert seasonal_result.manifest.job_id == "seasonal"

    def test_manifest_records_what_reproduction_needs(self, seasonal_result):
        m = seasonal_result.manifest
        assert len(m.data_fingerprint) == 64
        assert len(m.cutoffs) == 3
        assert m.request.season_length == 52
        assert m.autoarima_mode == "fourier"  # FR-201a at m=52
        assert m.ets_mode == "mstl"  # FR-201c at m=52
        assert m.thread_config  # NFR-02 -- byte-identity is relative to this
        assert m.package_versions["statsforecast"]

    def test_forecast_covers_every_model_and_the_full_horizon(self, seasonal_result):
        from collections import Counter

        points = Counter(r.model for r in seasonal_result.forecast.rows if r.quantity == "point")
        assert set(points.values()) == {4 * 13}, "4 series x 13 horizon steps per model"

    def test_every_point_forecast_carries_its_intervals(self, seasonal_result):
        """FR-301 -- the delivered forecast is banded, not bare. Every point gets a lo and a
        hi at every requested level."""
        from collections import Counter

        by_quantity = Counter(r.quantity for r in seasonal_result.forecast.rows)
        levels = seasonal_result.forecast.levels
        assert by_quantity["lo"] == by_quantity["point"] * len(levels)
        assert by_quantity["hi"] == by_quantity["point"] * len(levels)

    def test_intervals_nest_by_level(self, seasonal_result):
        """A 95% interval must contain the 80%, which must contain the point."""
        keyed: dict[tuple[str, str, str], dict[tuple[str, int | None], float]] = {}
        for row in seasonal_result.forecast.rows:
            keyed.setdefault((row.unique_id, row.ds, row.model), {})[(row.quantity, row.level)] = (
                row.value
            )
        checked = 0
        for values in keyed.values():
            point = values.get(("point", None))
            if point is None or ("lo", 95) not in values:
                continue
            assert values[("lo", 95)] <= values[("lo", 80)] <= point
            assert point <= values[("hi", 80)] <= values[("hi", 95)]
            checked += 1
        assert checked > 0, "no banded points found to check"

    def test_every_model_is_scored_on_every_fold(self, seasonal_result):
        for row in seasonal_result.leaderboard.rows:
            if row.scope == "panel":
                assert row.n_folds == 3


class TestTiming:
    """G1: RunTiming per model, whose parts account for the total (FR-217/217a)."""

    def test_every_model_reports_train_and_predict_separately(self, seasonal_result):
        assert seasonal_result.timing.per_model
        for t in seasonal_result.timing.per_model:
            assert t.train_cpu_seconds >= 0
            assert t.predict_cpu_seconds >= 0

    def test_final_refit_is_recorded_with_no_fold(self, seasonal_result):
        finals = [t for t in seasonal_result.timing.per_model if t.fold_index is None]
        assert len(finals) == len(FAST), "one final refit per model"

    def test_parts_account_for_the_whole(self, seasonal_result):
        timing = seasonal_result.timing
        assert timing.overhead_cpu_seconds, "FR-217a -- overhead is reported, not folded in"
        assert timing.total_cpu_seconds >= timing.model_cpu_seconds

    def test_cost_is_reported_per_model(self, seasonal_result):
        cost = seasonal_result.timing.cost_by_model()
        assert set(cost) == set(FAST)
        assert list(cost.values()) == sorted(cost.values(), reverse=True)


class TestReproducibility:
    """G1 / NFR-02: same input + spec + thread config -> byte-identical leaderboard."""

    def test_two_identical_runs_produce_identical_leaderboards(self):
        panel = build({f"S{i}": seasonal_random_walk(200, 52, i) for i in range(3)})
        request = ForecastRequest(h=13, freq="W", n_windows=2, models=FAST)
        first = run_from_frame(panel, request=request, mapping=MAPPING, job_id="a")
        second = run_from_frame(panel, request=request, mapping=MAPPING, job_id="b")
        assert first.leaderboard.model_dump_json() == second.leaderboard.model_dump_json()

    def test_the_fingerprint_is_stable_across_runs(self):
        panel = build({f"S{i}": seasonal_random_walk(200, 52, i) for i in range(3)})
        request = ForecastRequest(h=13, freq="W", n_windows=2, models=FAST)
        a = run_from_frame(panel, request=request, mapping=MAPPING, job_id="a")
        b = run_from_frame(panel, request=request, mapping=MAPPING, job_id="b")
        assert a.manifest.data_fingerprint == b.manifest.data_fingerprint

    def test_row_order_of_the_input_does_not_change_the_result(self):
        """Canonical sort (TS §4.7) exists so that how a file was written cannot change
        what the leaderboard says."""
        panel = build({f"S{i}": seasonal_random_walk(200, 52, i) for i in range(3)})
        shuffled = panel.sample(fraction=1.0, shuffle=True, seed=7)
        request = ForecastRequest(h=13, freq="W", n_windows=2, models=FAST)
        a = run_from_frame(panel, request=request, mapping=MAPPING, job_id="a")
        b = run_from_frame(shuffled, request=request, mapping=MAPPING, job_id="b")
        assert a.manifest.data_fingerprint == b.manifest.data_fingerprint
        assert a.leaderboard.model_dump_json() == b.leaderboard.model_dump_json()


class TestBaselineHonesty:
    """AC-406 / AC-406a -- the pair that the original spec got backwards."""

    def test_seasonal_random_walk_reports_that_nothing_beat_the_baseline(self):
        """AC-406. On y_t = y_{t-m} + e, SeasonalNaive IS the optimal forecast, so the
        honest result is that nothing beat it."""
        panel = build({f"S{i}": seasonal_random_walk(200, 52, i) for i in range(5)})
        request = ForecastRequest(h=13, freq="W", n_windows=2, models=FAST)
        result = run_from_frame(panel, request=request, mapping=MAPPING, job_id="srw")
        assert not result.leaderboard.any_beat_baseline
        winner = next(r for r in result.leaderboard.rows if r.scope == "panel" and r.rank == 1)
        assert winner.model == "SeasonalNaive"

    @pytest.mark.slow
    def test_pure_random_walk_lets_autoarima_beat_seasonal_naive(self):
        """AC-406a. The original spec used THIS panel for the assertion above, which would
        have failed against a correct engine: on a driftless random walk the optimal
        forecast is the last value, seasonal-naive error variance is ~m*sigma^2 against
        h*sigma^2 for naive, and AutoARIMA selects ARIMA(0,1,0) and reproduces it."""
        panel = build({f"S{i}": random_walk(200, i) for i in range(5)})
        request = ForecastRequest(
            h=13, freq="W", n_windows=2, models=["SeasonalNaive", "AutoARIMA"]
        )
        result = run_from_frame(panel, request=request, mapping=MAPPING, job_id="rw")
        assert result.leaderboard.any_beat_baseline
        winner = next(r for r in result.leaderboard.rows if r.scope == "panel" and r.rank == 1)
        assert winner.model == "AutoARIMA"
        assert winner.vs_baseline_pct is not None
        assert winner.vs_baseline_pct < 0, "negative is better"


class TestDegenerateMetrics:
    """G1 / FR-214 -- one degenerate series must not NaN an entire leaderboard row."""

    def test_a_fold_constant_series_yields_none_not_nan(self):
        """FR-105 excludes all-constant *series*; a series can still be constant within an
        early training *fold*, which zeroes the MASE denominator. Routine on intermittent
        SKU panels, and it must not poison the panel aggregate."""
        flat_then_moving = np.concatenate([np.full(150, 42.0), 42.0 + np.arange(50) * 3.0])
        panel = build(
            {
                "normal_a": seasonal_random_walk(200, 52, 1),
                "normal_b": seasonal_random_walk(200, 52, 2),
                "fold_constant": flat_then_moving,
            }
        )
        request = ForecastRequest(h=13, freq="W", n_windows=3, models=FAST)
        result = run_from_frame(panel, request=request, mapping=MAPPING, job_id="degenerate")

        panel_rows = [r for r in result.leaderboard.rows if r.scope == "panel"]
        assert panel_rows, "a degenerate series must not empty the leaderboard"
        for row in panel_rows:
            for value in (row.mase, row.rmsse, row.mae, row.rmse, row.smape):
                assert value is None or np.isfinite(value), "never NaN or inf (FR-214)"

    def test_series_metrics_may_be_none_without_removing_the_series(self):
        flat = np.concatenate([np.full(150, 5.0), 5.0 + np.arange(50) * 2.0])
        panel = build({"a": seasonal_random_walk(200, 52, 1), "flat": flat})
        request = ForecastRequest(h=13, freq="W", n_windows=3, models=FAST)
        result = run_from_frame(panel, request=request, mapping=MAPPING, job_id="none-metrics")
        series_rows = [r for r in result.leaderboard.rows if r.scope == "series"]
        assert {r.unique_id for r in series_rows} == {"a", "flat"}


class TestExclusionsAreNamed:
    """FS §6 -- silently dropping a series is a listed failure mode."""

    def test_excluded_series_carry_reasons_into_the_manifest(self):
        panel = build(
            {
                "ok_a": seasonal_random_walk(200, 52, 1),
                "ok_b": seasonal_random_walk(200, 52, 2),
                "all_zero": np.zeros(200),
                "too_short": seasonal_random_walk(60, 52, 3),
            }
        )
        request = ForecastRequest(h=13, freq="W", n_windows=3, models=FAST)
        result = run_from_frame(panel, request=request, mapping=MAPPING, job_id="excl")
        excluded = result.manifest.excluded_series
        assert excluded["all_zero"] == "all_zero"
        assert excluded["too_short"] == "insufficient_observations"
        assert all(result.profile.validation.excluded_detail[uid] for uid in excluded)
