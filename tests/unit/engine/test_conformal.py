"""FR-301/302/303/307, AC-301, AC-307 -- and gate G2.

NFR-10 puts `engine/conformal.py` at 100% line coverage alongside `folds.py` and
`guardrail.py`. It is on that list because a calibration bug does not crash: it produces a
plausible-looking interval that is quietly wrong, and the number that would normally catch it
was, in the original specification, incapable of failing.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import polars as pl
import pytest

from xlforecast.engine import conformal
from xlforecast.engine.conformal import (
    apply_bands,
    calibrate,
    collect_residuals,
    conformal_quantile,
    coverage,
    series_support,
)
from xlforecast.engine.folds import make_folds

H, SEASON, FREQ, N_WINDOWS = 8, 12, "ME", 4
NOISE = 10.0


def gaussian_panel(n_series: int = 8, n_obs: int = 120, seed: int = 0) -> pl.DataFrame:
    """Known noise distribution, so nominal coverage has a right answer to be near."""
    rng = np.random.default_rng(seed)
    frames = []
    for i in range(n_series):
        dates = pd.date_range("2014-01-31", periods=n_obs, freq=FREQ)
        t = np.arange(n_obs)
        y = 500 + 60 * np.sin(2 * np.pi * t / SEASON) + rng.normal(0, NOISE, n_obs)
        frames.append(
            pl.DataFrame({"unique_id": [f"S{i}"] * n_obs, "ds": list(dates), "y": y.tolist()})
        )
    return pl.concat(frames)


def intermittent_panel(n_series: int = 6, n_obs: int = 120, seed: int = 3) -> pl.DataFrame:
    """Non-negative counts with many zeros -- exactly the series persona P1 owns."""
    rng = np.random.default_rng(seed)
    frames = []
    for i in range(n_series):
        dates = pd.date_range("2014-01-31", periods=n_obs, freq=FREQ)
        y = np.where(rng.random(n_obs) < 0.35, rng.poisson(6, n_obs).astype(float), 0.0)
        frames.append(
            pl.DataFrame({"unique_id": [f"I{i}"] * n_obs, "ds": list(dates), "y": y.tolist()})
        )
    return pl.concat(frames)


def residuals_for(panel: pl.DataFrame, models: list[str]) -> tuple[pl.DataFrame, list]:
    from xlforecast.engine import local as local_family

    folds = make_folds(panel, h=H, n_windows=N_WINDOWS, step_size=H, freq=FREQ)
    origin = panel.get_column("ds").min()
    predictions = {}
    for fold in folds:
        preds, _ = local_family.forecast_fold(
            models, fold, h=H, freq=FREQ, season_length=SEASON, origin=origin
        )
        predictions[fold.index] = preds
    return collect_residuals(folds, predictions), folds


@pytest.fixture(scope="module")
def gaussian_residuals():
    panel = gaussian_panel()
    residuals, _ = residuals_for(panel, ["SeasonalNaive", "HistoricAverage"])
    return residuals, series_support(panel)


class TestConformalQuantile:
    def test_uses_the_finite_sample_correction(self):
        """`ceil((n+1)(1-alpha))/n`, not the plain empirical quantile. At the residual counts
        a 3-window CV produces, that is the difference between roughly nominal coverage and
        systematic under-coverage."""
        values = np.arange(1.0, 11.0)  # n=10
        # ceil(11 * 0.8) = 9 -> the 9th smallest
        assert conformal_quantile(values, 80) == 9.0

    def test_is_monotone_in_level(self):
        """FR-301 -- a 95% band can never be narrower than an 80% band."""
        values = np.random.default_rng(0).exponential(5.0, 200)
        widths = [conformal_quantile(values, lv) for lv in (50, 80, 90, 95, 99)]
        assert widths == sorted(widths)

    def test_too_few_residuals_returns_the_widest_observed_not_an_extrapolation(self):
        values = np.array([1.0, 2.0, 3.0])
        assert conformal_quantile(values, 99) == 3.0

    def test_empty_input_is_nan_not_zero(self):
        assert math.isnan(conformal_quantile(np.array([]), 80))

    def test_non_finite_residuals_are_discarded(self):
        values = np.array([1.0, 2.0, np.nan, np.inf, 3.0])
        assert conformal_quantile(values, 50) == 2.0


class TestCrossConformalProvenance:
    """FR-302 -- the band that scores fold k must not have seen fold k."""

    def test_excluded_folds_are_absent_from_the_provenance(self, gaussian_residuals):
        residuals, _ = gaussian_residuals
        folds = frozenset(residuals.get_column("fold_index").unique().to_list())
        bands = calibrate(
            residuals,
            model="SeasonalNaive",
            level=80,
            exclude_folds=frozenset({1}),
            all_folds=folds,
        )
        assert 1 not in bands.calibrated_from_folds
        assert set(bands.calibrated_from_folds) == set(folds) - {1}

    def test_the_delivered_band_uses_every_fold(self, gaussian_residuals):
        residuals, _ = gaussian_residuals
        folds = frozenset(residuals.get_column("fold_index").unique().to_list())
        bands = calibrate(residuals, model="SeasonalNaive", level=80, all_folds=folds)
        assert set(bands.calibrated_from_folds) == set(folds)

    def test_excluding_a_fold_actually_changes_the_band(self, gaussian_residuals):
        """If it did not, 'cross-conformal' would be a label rather than a mechanism."""
        residuals, _ = gaussian_residuals
        folds = frozenset(residuals.get_column("fold_index").unique().to_list())
        full = calibrate(residuals, model="SeasonalNaive", level=80, all_folds=folds)
        without = calibrate(
            residuals,
            model="SeasonalNaive",
            level=80,
            exclude_folds=frozenset({0}),
            all_folds=folds,
        )
        assert full.half_width != without.half_width


class TestCoverageIsNotATautology:
    """AC-301 and gate G2. The control is the point of this class."""

    def test_out_of_calibration_coverage_is_within_five_points_of_nominal(self, gaussian_residuals):
        residuals, support = gaussian_residuals
        for level in (80, 95):
            observed = coverage(
                residuals,
                model="SeasonalNaive",
                level=level,
                support=support,
                min_residuals=10,
            )
            assert abs(observed - level / 100) <= 0.05, f"level {level}: {observed:.3f}"

    @pytest.mark.parametrize("level", [80, 95])
    def test_the_in_calibration_control_cannot_report_under_coverage(
        self, gaussian_residuals, level
    ):
        """The assertion that proves the headline number can fail.

        Measured: at level 80 the in-calibration figure is 0.844 against the honest 0.809;
        at level 95 they are 1.000 and 0.953. The control does not sit *closer* to nominal --
        it sits systematically *above* it, because a conformal quantile computed with the
        finite-sample correction is conservative on its own calibration sample.

        That is exactly why it is worthless as evidence: it cannot come out low, so it would
        report a comfortable number for an interval that was far too narrow. The honest
        figure can land on either side of nominal, and does.
        """
        residuals, support = gaussian_residuals
        honest = coverage(
            residuals, model="SeasonalNaive", level=level, support=support, min_residuals=10
        )
        control = coverage(
            residuals,
            model="SeasonalNaive",
            level=level,
            support=support,
            min_residuals=10,
            in_calibration=True,
        )
        assert control >= level / 100, "in-calibration coverage cannot fall below nominal"
        assert control > honest, (
            f"level {level}: control {control:.3f} must exceed honest {honest:.3f}; "
            "equality would mean fold k was not held out"
        )

    def test_a_mutation_back_to_same_fold_calibration_is_detectable(self, gaussian_residuals):
        """Guards the regression directly: if someone 'simplifies' FR-302 away, the two
        numbers become identical and this fails."""
        residuals, support = gaussian_residuals
        honest = coverage(
            residuals, model="HistoricAverage", level=80, support=support, min_residuals=10
        )
        control = coverage(
            residuals,
            model="HistoricAverage",
            level=80,
            support=support,
            min_residuals=10,
            in_calibration=True,
        )
        assert honest != control, "identical figures mean fold k was not held out"


class TestSupportClipping:
    """FR-307 / AC-307."""

    def test_lower_bound_is_clipped_at_zero_for_non_negative_series(self):
        panel = intermittent_panel()
        support = series_support(panel)
        assert all(lo == 0.0 for lo, _ in support.values())

    def test_a_series_that_goes_negative_keeps_an_unbounded_lower_support(self):
        dates = pd.date_range("2014-01-31", periods=30, freq=FREQ)
        panel = pl.DataFrame(
            {"unique_id": ["neg"] * 30, "ds": list(dates), "y": [float(i) - 15 for i in range(30)]}
        )
        assert series_support(panel)["neg"][0] == -math.inf

    def test_clipping_sharpens_the_interval_without_changing_coverage(self):
        """AC-307, corrected after measurement.

        Clipping the lower bound at zero *cannot* change coverage on non-negative data: no
        observation lies below zero, so truncating there never excludes a point that was
        previously inside. Measured on an intermittent panel: 80.7% coverage clipped and
        80.7% unclipped, with mean width falling 21.6%.

        What clipping buys is sharpness and interpretability. A negative lower bound on unit
        demand is not a wider forecast, it is a nonsensical one.
        """
        panel = intermittent_panel()
        residuals, _ = residuals_for(panel, ["HistoricAverage"])
        clipped = series_support(panel)
        unclipped = dict.fromkeys(clipped, (-math.inf, math.inf))

        assert coverage(
            residuals, model="HistoricAverage", level=80, support=clipped, min_residuals=10
        ) == pytest.approx(
            coverage(
                residuals,
                model="HistoricAverage",
                level=80,
                support=unclipped,
                min_residuals=10,
            )
        )
        narrow = conformal.interval_width(
            residuals, model="HistoricAverage", level=80, support=clipped, min_residuals=10
        )
        wide = conformal.interval_width(
            residuals, model="HistoricAverage", level=80, support=unclipped, min_residuals=10
        )
        assert narrow < wide
        assert (wide - narrow) / wide > 0.10, "clipping should remove a material share of width"

    def test_symmetric_bands_leave_the_lower_tail_idle_on_count_data(self):
        """FR-307a -- the diagnostic that actually detects the problem.

        A well-behaved interval splits miscoverage roughly evenly between its tails.
        Measured: on intermittent data 0.00% of violations are lower-tail and 15.62% are
        upper-tail, against a balanced 10.2/5.5 on Gaussian data. The lower half of the
        interval does no work -- and that is invisible in the coverage figure, which reads
        ~nominal either way.
        """
        panel = intermittent_panel()
        residuals, _ = residuals_for(panel, ["HistoricAverage"])
        below, above = conformal.tail_miscoverage(
            residuals,
            model="HistoricAverage",
            level=80,
            support=series_support(panel),
            min_residuals=10,
        )
        assert below == pytest.approx(0.0, abs=1e-9)
        assert above > 0.05

    def test_gaussian_data_splits_its_miscoverage_between_both_tails(self):
        """The contrast case: where the support bound is not binding, both tails are live."""
        panel = gaussian_panel()
        residuals, _ = residuals_for(panel, ["HistoricAverage"])
        below, above = conformal.tail_miscoverage(
            residuals,
            model="HistoricAverage",
            level=80,
            support=series_support(panel),
            min_residuals=10,
        )
        assert below > 0.01
        assert above > 0.01

    def test_clip_rate_is_recorded_per_series(self):
        panel = intermittent_panel()
        residuals, _ = residuals_for(panel, ["HistoricAverage"])
        folds = frozenset(residuals.get_column("fold_index").unique().to_list())
        bands = calibrate(
            residuals, model="HistoricAverage", level=95, all_folds=folds, min_residuals=10
        )
        _, rates = apply_bands(residuals, bands, series_support(panel))
        assert rates
        assert any(rate > 0 for rate in rates.values()), "clipping must actually be happening"


class TestFallbackChain:
    """TS §5.4 -- the original spec defined two steps of a three-step chain."""

    def test_a_thin_series_falls_back_to_pooled_panel_residuals(self, gaussian_residuals):
        residuals, _ = gaussian_residuals
        bands = calibrate(
            residuals,
            model="SeasonalNaive",
            level=80,
            min_residuals=100,
            all_folds=frozenset(residuals.get_column("fold_index").unique().to_list()),
        )
        assert bands.pooled_fallback, "every series should have fallen back"
        assert all(math.isfinite(v) for v in bands.half_width.values())
        assert len(set(bands.half_width.values())) == 1, "pooled fallback shares one width"

    def test_when_the_panel_is_also_too_thin_the_level_is_unavailable_not_wrong(self):
        """The terminal fallback. A NaN half-width says 'we cannot certify this level',
        which is the honest answer; a number would be a fabricated one."""
        dates = pd.date_range("2014-01-31", periods=40, freq=FREQ)
        panel = pl.DataFrame(
            {"unique_id": ["a"] * 40, "ds": list(dates), "y": np.arange(40.0).tolist()}
        )
        residuals, _ = residuals_for(panel, ["HistoricAverage"])
        bands = calibrate(
            residuals,
            model="HistoricAverage",
            level=80,
            min_residuals=10_000_000,
            all_folds=frozenset({0}),
        )
        assert all(math.isnan(v) for v in bands.half_width.values())

    def test_pooled_fallback_is_recorded_for_diagnostics(self, gaussian_residuals):
        residuals, _ = gaussian_residuals
        bands = calibrate(
            residuals,
            model="SeasonalNaive",
            level=80,
            min_residuals=100,
            all_folds=frozenset({0, 1, 2, 3}),
        )
        assert set(bands.pooled_fallback) == set(bands.half_width)


class TestResidualCollection:
    def test_residuals_carry_the_fold_and_horizon_step(self, gaussian_residuals):
        residuals, _ = gaussian_residuals
        assert set(residuals.columns) >= {"fold_index", "horizon_step", "abs_residual"}
        assert residuals.get_column("horizon_step").max() == H

    def test_absent_predictions_produce_an_empty_frame_not_a_crash(self):
        panel = gaussian_panel(n_series=2, n_obs=60)
        folds = make_folds(panel, h=H, n_windows=2, step_size=H, freq=FREQ)
        empty = collect_residuals(folds, {})
        assert empty.is_empty()
        assert "abs_residual" in empty.columns

    def test_predictions_that_do_not_align_are_dropped_not_mismatched(self):
        panel = gaussian_panel(n_series=2, n_obs=60)
        folds = make_folds(panel, h=H, n_windows=2, step_size=H, freq=FREQ)
        bogus = pl.DataFrame(
            {
                "unique_id": ["ghost"],
                "ds": [pd.Timestamp("1999-01-31")],
                "model": ["X"],
                "y_hat": [1.0],
            }
        ).with_columns(pl.col("ds").cast(pl.Datetime("us")))
        assert collect_residuals(folds, {f.index: bogus for f in folds}).is_empty()


def test_coverage_of_an_unknown_model_is_nan_not_an_exception(gaussian_residuals):
    residuals, support = gaussian_residuals
    assert math.isnan(coverage(residuals, model="NotFitted", level=80, support=support))


def test_module_exposes_a_default_minimum(gaussian_residuals):
    assert conformal.DEFAULT_MIN_RESIDUALS == 20


class TestEmptyAndDegenerateInputs:
    """The guards exist because a calibration failure must degrade to `NaN`, meaning
    'we cannot certify this', rather than to a number nobody can trace."""

    def test_tail_miscoverage_of_an_unknown_model_is_nan(self, gaussian_residuals):
        residuals, support = gaussian_residuals
        below, above = conformal.tail_miscoverage(
            residuals, model="NotFitted", level=80, support=support
        )
        assert math.isnan(below)
        assert math.isnan(above)

    def test_interval_width_of_an_unknown_model_is_nan(self, gaussian_residuals):
        residuals, support = gaussian_residuals
        assert math.isnan(
            conformal.interval_width(residuals, model="NotFitted", level=80, support=support)
        )

    def test_coverage_is_nan_when_every_band_hit_the_terminal_fallback(self, gaussian_residuals):
        """With no certifiable band anywhere, there is nothing to report -- and reporting
        `0.0` would read as catastrophic miscalibration rather than as absence."""
        residuals, support = gaussian_residuals
        observed = coverage(
            residuals,
            model="SeasonalNaive",
            level=80,
            support=support,
            min_residuals=10_000_000,
        )
        assert math.isnan(observed)

    def test_width_and_tails_are_nan_under_the_terminal_fallback(self, gaussian_residuals):
        residuals, support = gaussian_residuals
        assert math.isnan(
            conformal.interval_width(
                residuals,
                model="SeasonalNaive",
                level=80,
                support=support,
                min_residuals=10_000_000,
            )
        )


class TestQuantileFrame:
    """FR-208 -- the quantile grid that probabilistic scoring consumes."""

    def test_grid_is_derived_from_the_requested_levels(self):
        assert list(conformal.quantile_levels([80, 95])) == [0.025, 0.1, 0.5, 0.9, 0.975]
        assert list(conformal.quantile_levels([80])) == [0.1, 0.5, 0.9]

    def test_the_median_is_always_present_because_it_is_the_point_forecast(self):
        for levels in ([50], [80, 95], [10, 50, 90, 99]):
            assert 0.5 in conformal.quantile_levels(levels)

    def test_quantiles_are_monotone_across_the_grid(self, gaussian_residuals):
        residuals, support = gaussian_residuals
        frame, columns = conformal.quantile_frame(
            residuals,
            model="SeasonalNaive",
            levels=[80, 95],
            support=support,
            min_residuals=10,
        )
        values = frame.select(columns).to_numpy()
        assert (np.diff(values, axis=1) >= -1e-9).all(), "quantile crossing"

    def test_the_median_column_is_the_point_forecast(self, gaussian_residuals):
        residuals, support = gaussian_residuals
        frame, _ = conformal.quantile_frame(
            residuals, model="SeasonalNaive", levels=[80], support=support, min_residuals=10
        )
        assert frame.get_column("q0.5000").to_list() == frame.get_column("y_hat").to_list()

    def test_an_unknown_model_yields_an_empty_frame_with_the_column_names(self, gaussian_residuals):
        """The columns are still returned so a caller can build an empty result without
        special-casing the absence."""
        residuals, support = gaussian_residuals
        frame, columns = conformal.quantile_frame(
            residuals, model="NotFitted", levels=[80, 95], support=support
        )
        assert frame.is_empty()
        assert columns == ["q0.0250", "q0.1000", "q0.5000", "q0.9000", "q0.9750"]

    def test_lower_quantiles_respect_the_series_support(self):
        """FR-307 applies to the quantile grid too, not only to the reported band."""
        panel = intermittent_panel()
        residuals, _ = residuals_for(panel, ["HistoricAverage"])
        frame, columns = conformal.quantile_frame(
            residuals,
            model="HistoricAverage",
            levels=[80, 95],
            support=series_support(panel),
            min_residuals=10,
        )
        assert (frame.get_column(columns[0]).to_numpy() >= 0).all()
