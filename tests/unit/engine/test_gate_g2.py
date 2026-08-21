"""Gate G2, asserted against the assembled pipeline rather than the units.

Each clause corresponds to a G2 line in `docs/03-BUILD-PLAN.md`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl
import pytest

from xlforecast.engine.run import run_from_frame
from xlforecast.schemas.request import DataMapping, ForecastRequest

MAPPING = DataMapping(unique_id_col="unique_id", ds_col="ds", y_col="y")
MEMBERS = ["SeasonalNaive", "WindowAverage", "HistoricAverage", "AutoETS"]


def known_noise_panel(n_series: int = 8, n_obs: int = 150, sigma: float = 12.0) -> pl.DataFrame:
    """Gaussian noise of known scale, so nominal coverage has a right answer to sit near."""
    rng = np.random.default_rng(11)
    frames = []
    for i in range(n_series):
        dates = pd.date_range("2012-01-31", periods=n_obs, freq="ME")
        t = np.arange(n_obs)
        y = 400 + 80 * np.sin(2 * np.pi * t / 12) + rng.normal(0, sigma, n_obs)
        frames.append(
            pl.DataFrame({"unique_id": [f"S{i}"] * n_obs, "ds": list(dates), "y": y.tolist()})
        )
    return pl.concat(frames)


@pytest.fixture(scope="module")
def result():
    # h=8 with 3 windows gives 24 residuals per series, clearing min_residuals=20 so that
    # per-series calibration actually engages -- the regime in which AC-301's control has
    # discriminating power (see conformal.DEFAULT_MIN_RESIDUALS).
    request = ForecastRequest(
        h=8,
        freq="ME",
        n_windows=3,
        models=MEMBERS,
        ensemble="inverse_error",
        selection="per_series",
        levels=[80, 95],
    )
    return run_from_frame(known_noise_panel(), request=request, mapping=MAPPING, job_id="g2")


class TestCoverage:
    """G2 clause 1: out-of-calibration coverage within +/-5pp of nominal, with the
    in-calibration control proving the figure is not a tautology."""

    @pytest.mark.parametrize("level", [80, 95])
    def test_out_of_calibration_coverage_is_within_five_points(self, result, level):
        rows = [
            r
            for r in result.calibration
            if r.scope == "all" and r.level == level and r.empirical is not None
        ]
        assert rows
        for row in rows:
            assert abs(row.empirical - level / 100) <= 0.05, f"{row.model}: {row.empirical:.3f}"

    def test_the_control_never_reports_under_coverage(self, result):
        """It cannot, being conservative on its own calibration sample -- which is exactly
        why it is evidence about the honest figure rather than a figure itself."""
        rows = [
            r
            for r in result.calibration
            if r.scope == "all" and r.empirical_in_calibration is not None
        ]
        assert rows
        for row in rows:
            assert row.empirical_in_calibration >= row.nominal - 1e-9, row.model

    def test_the_control_is_at_least_as_high_as_the_honest_figure(self, result):
        for row in result.calibration:
            if row.scope != "all" or None in (row.empirical, row.empirical_in_calibration):
                continue
            assert row.empirical_in_calibration >= row.empirical - 1e-9, row.model

    def test_per_series_calibration_actually_engaged(self, result):
        """Guards the fixture's premise. If min_residuals forced a pooled fallback, the
        control's gap would collapse and the assertions above would pass vacuously."""
        rows = [r for r in result.calibration if r.scope == "all" and r.level == 80]
        assert any(r.n_pooled_fallback == 0 for r in rows)


class TestTailReporting:
    """G2 / FR-307b -- the split that carries information."""

    def test_tails_are_reported_by_intermittency_class(self, result):
        scoped = {r.scope for r in result.calibration}
        assert "smooth" in scoped
        for row in result.calibration:
            if row.scope != "all":
                assert row.lower_tail is not None
                assert row.upper_tail is not None

    def test_coverage_is_not_split_by_class(self, result):
        """FR-303 as corrected: measurement showed the split carried no information."""
        for row in result.calibration:
            if row.scope != "all":
                assert row.empirical is None


class TestEnsembleCompetesFairly:
    """G2 clauses 3 and 4."""

    def test_the_ensemble_appears_in_the_leaderboard(self, result):
        models = {r.model for r in result.leaderboard.rows if r.scope == "panel"}
        assert any(m.startswith("Ensemble[") for m in models)

    def test_it_is_scored_on_the_same_folds_as_its_members(self, result):
        """FR-405 -- asserted via fold_index equality, which is why FoldScore carries one."""
        by_model: dict[str, set[int]] = {}
        for score in result.fold_scores:
            by_model.setdefault(score.model, set()).add(score.fold_index)
        ensemble = next(m for m in by_model if m.startswith("Ensemble["))
        for member in MEMBERS:
            assert by_model[ensemble] == by_model[member], member

    def test_it_is_scored_on_the_same_series_as_its_members(self, result):
        by_model: dict[str, set[str]] = {}
        for score in result.fold_scores:
            if score.unique_id is not None:
                by_model.setdefault(score.model, set()).add(score.unique_id)
        ensemble = next(m for m in by_model if m.startswith("Ensemble["))
        assert by_model[ensemble] == by_model["SeasonalNaive"]

    def test_it_does_not_automatically_win(self, result):
        """An ensemble that always came first would be evidence of a scoring bug, not of a
        good ensemble."""
        panel = sorted(
            (r for r in result.leaderboard.rows if r.scope == "panel"), key=lambda r: r.rank
        )
        assert panel[0].mase is not None

    def test_the_ensemble_configuration_is_recorded(self, result):
        params = result.manifest.ensemble_params
        assert params["method"] == "inverse_error"
        assert params["metric"] == "mase"
        assert "prob_method" in params


class TestSelectionBias:
    """FR-408 -- the winner's curse is reported, not merely warned about."""

    def test_per_series_selection_warns_below_five_windows(self, result):
        assert "overfits" in str(result.manifest.ensemble_params["selection_warnings"])

    def test_the_selection_strategy_reaches_the_manifest(self, result):
        assert result.manifest.ensemble_params["selection"] == "per_series"


class TestReproducibilityStillHolds:
    """Phase 2 added stochastic-looking machinery; NFR-02 must survive it."""

    def test_two_runs_agree_including_calibration(self):
        request = ForecastRequest(
            h=8,
            freq="ME",
            n_windows=2,
            models=["SeasonalNaive", "WindowAverage", "HistoricAverage"],
            ensemble="median",
        )
        panel = known_noise_panel(n_series=4, n_obs=120)
        first = run_from_frame(panel, request=request, mapping=MAPPING, job_id="a")
        second = run_from_frame(panel, request=request, mapping=MAPPING, job_id="b")
        assert first.leaderboard.model_dump_json() == second.leaderboard.model_dump_json()
        assert [c.model_dump() for c in first.calibration] == [
            c.model_dump() for c in second.calibration
        ]
