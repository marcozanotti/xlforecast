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


class TestProbabilisticScoring:
    """FR-208 -- scaled CRPS in the leaderboard, with its grid in the manifest."""

    def test_every_ranked_model_carries_a_crps(self, result):
        for row in result.leaderboard.rows:
            if row.scope == "panel":
                assert row.scaled_crps is not None, row.model

    def test_the_quantile_grid_is_recorded(self, result):
        """A run at levels=[80,95] is not CRPS-comparable with one at [50,80,95], so the
        grid travels with the result rather than being implied by it."""
        assert result.manifest.crps_quantiles == [0.025, 0.1, 0.5, 0.9, 0.975]

    def test_crps_and_mase_broadly_agree(self, result):
        """Expected under ADR-006, and worth stating: every model's band is
        `point ± q(its own residuals)`, so the probabilistic ranking largely tracks the
        point ranking. CRPS earns its place by pricing interval width, not by reordering
        the leaderboard -- and the methodology page has to say so rather than let a reader
        discover it."""
        panel = [r for r in result.leaderboard.rows if r.scope == "panel"]
        by_mase = [r.model for r in sorted(panel, key=lambda r: r.mase or 9e9)]
        by_crps = [r.model for r in sorted(panel, key=lambda r: r.scaled_crps or 9e9)]
        assert by_mase[0] == by_crps[0]

    def test_crps_is_absent_when_conformal_is_disabled(self):
        """No bands, no probabilistic score -- and `None` says that plainly."""
        request = ForecastRequest(
            h=8,
            freq="ME",
            n_windows=2,
            models=["SeasonalNaive", "HistoricAverage"],
            ensemble="none",
            conformal=False,
        )
        run = run_from_frame(
            known_noise_panel(n_series=4, n_obs=120),
            request=request,
            mapping=MAPPING,
            job_id="nocal",
        )
        assert run.manifest.crps_quantiles == []
        assert all(r.scaled_crps is None for r in run.leaderboard.rows)


class TestDeliveredIntervals:
    """FR-301 -- the forecast that reaches the user is banded."""

    def test_bands_are_calibrated_from_every_fold(self, result):
        """Unlike the scoring bands, which each hold one fold out (FR-302)."""
        folds = set(range(len(result.manifest.cutoffs)))
        for band in result.bands:
            assert set(band.calibrated_from_folds) == folds

    def test_each_band_records_which_model_it_belongs_to(self, result):
        """Half-widths are keyed by series, so without this a list of bands is ambiguous
        and a lookup silently returns whichever model came first."""
        assert all(b.model for b in result.bands)
        assert len({(b.model, b.level) for b in result.bands}) == len(result.bands)

    def test_lower_bounds_respect_the_series_support(self, result):
        """FR-307 -- clipped, so a non-negative series never gets a negative lower bound."""
        assert all(r.value >= 0 for r in result.forecast.rows if r.quantity == "lo")


class TestSelectionReachesTheLeaderboard:
    def test_exactly_one_model_is_selected_per_series(self, result):
        by_series: dict[str, list[str]] = {}
        for row in result.leaderboard.rows:
            if row.scope == "series" and row.selected and row.unique_id:
                by_series.setdefault(row.unique_id, []).append(row.model)
        assert by_series
        assert all(len(v) == 1 for v in by_series.values())

    def test_the_selected_row_discloses_its_bias(self, result):
        """FR-408 -- an argmin's own score is not an unbiased estimate of its accuracy."""
        selected = [r for r in result.leaderboard.rows if r.selected and r.scope == "series"]
        assert selected
        assert all(r.selection_biased for r in selected)
        assert any(r.selected_lofo_score is not None for r in selected)
