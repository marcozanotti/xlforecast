"""FR-203a/b/c -- the ML adapter and the matched-pair invariant.

The pair `LocalX` / `GlobalX` is the leaderboard's most defensible single claim: same
learner, same features, same folds, differing only in the information set, so the difference
between them *is* the effect of pooling. That only holds if nothing else drifts, which is
what this file guards.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl
import pytest

from xlforecast.engine import ml
from xlforecast.engine.folds import make_folds
from xlforecast.engine.registry import FeatureRecipe, build_ml_plan
from xlforecast.errors import InvalidModelNameError

H, SEASON, FREQ = 6, 12, "ME"
PAIRS = [("LocalLinear", "GlobalLinear"), ("LocalLGBM", "GlobalLGBM"), ("LocalXGB", "GlobalXGB")]


def panel(n_series: int = 3, n_obs: int = 90) -> pl.DataFrame:
    rng = np.random.default_rng(4)
    frames = []
    for i in range(n_series):
        dates = pd.date_range("2016-01-31", periods=n_obs, freq=FREQ)
        t = np.arange(n_obs)
        y = 200 + 40 * np.sin(2 * np.pi * t / SEASON) + rng.normal(0, 5, n_obs)
        frames.append(
            pl.DataFrame({"unique_id": [f"S{i}"] * n_obs, "ds": list(dates), "y": y.tolist()})
        )
    return pl.concat(frames)


@pytest.fixture(scope="module")
def fold():
    return make_folds(panel(), h=H, n_windows=2, step_size=H, freq=FREQ)[0]


class TestMatchedPairs:
    """FR-203b -- the invariant that makes the comparison controlled."""

    @pytest.mark.parametrize(("local_name", "global_name"), PAIRS)
    def test_pair_shares_one_feature_recipe(self, local_name, global_name):
        a = build_ml_plan(local_name, freq=FREQ, season_length=SEASON, seed=42)
        b = build_ml_plan(global_name, freq=FREQ, season_length=SEASON, seed=42)
        assert a.recipe == b.recipe, "features must be held constant across a pair"

    @pytest.mark.parametrize(("local_name", "global_name"), PAIRS)
    def test_pair_differs_only_in_information_set(self, local_name, global_name):
        a = build_ml_plan(local_name, freq=FREQ, season_length=SEASON, seed=42)
        b = build_ml_plan(global_name, freq=FREQ, season_length=SEASON, seed=42)
        assert a.information_set == "own_series"
        assert b.information_set == "panel"
        assert type(a.estimator) is type(b.estimator), "same learner class"

    @pytest.mark.parametrize(("local_name", "global_name"), PAIRS)
    def test_both_halves_predict_the_same_shape(self, fold, local_name, global_name):
        preds, _ = ml.forecast_fold(
            [local_name, global_name], fold, h=H, freq=FREQ, season_length=SEASON, seed=42
        )
        counts = preds.group_by("model").len().sort("model")
        assert counts.get_column("len").n_unique() == 1


class TestSmallDataHyperparameters:
    """FR-203c -- global defaults on ~91 rows produce near-constant predictions, which reads
    as a bug rather than a finding."""

    def test_local_lgbm_gets_conservative_settings(self):
        local = build_ml_plan("LocalLGBM", freq=FREQ, season_length=SEASON, seed=1).estimator
        glob = build_ml_plan("GlobalLGBM", freq=FREQ, season_length=SEASON, seed=1).estimator
        assert local.min_child_samples < glob.min_child_samples
        assert local.num_leaves < glob.num_leaves

    def test_local_xgb_gets_a_shallower_tree(self):
        local = build_ml_plan("LocalXGB", freq=FREQ, season_length=SEASON, seed=1).estimator
        glob = build_ml_plan("GlobalXGB", freq=FREQ, season_length=SEASON, seed=1).estimator
        assert local.max_depth < glob.max_depth

    def test_lgbm_is_configured_for_reproducibility(self):
        """NFR-02 -- LightGBM's histogram construction is thread-order dependent otherwise."""
        est = build_ml_plan("GlobalLGBM", freq=FREQ, season_length=SEASON, seed=7).estimator
        assert est.deterministic is True
        assert est.force_row_wise is True
        assert est.random_state == 7


class TestInformationSetBehaviour:
    def test_global_fits_once_over_the_panel(self, fold):
        _, timings = ml.forecast_fold(
            ["GlobalLinear"], fold, h=H, freq=FREQ, season_length=SEASON, seed=42
        )
        assert timings[0].n_series_fitted == len(fold.series)

    def test_local_fits_once_per_series(self, fold):
        _, timings = ml.forecast_fold(
            ["LocalLinear"], fold, h=H, freq=FREQ, season_length=SEASON, seed=42
        )
        assert timings[0].n_series_fitted == len(fold.series)

    def test_series_shorter_than_max_lag_is_skipped_not_crashed(self):
        """A series with fewer rows than the longest lag yields no training data at all.
        FR-215's common-support rule then keeps the panel aggregate honest about which
        series each model was actually scored on."""
        short = panel(n_series=1, n_obs=90).with_columns(pl.lit("tiny").alias("unique_id")).head(8)
        combined = pl.concat([panel(2, 90), short])
        folds = make_folds(combined, h=H, n_windows=1, step_size=H, freq=FREQ)
        preds, timings = ml.forecast_fold(
            ["LocalLinear"], folds[0], h=H, freq=FREQ, season_length=SEASON, seed=42
        )
        assert "tiny" not in set(preds.get_column("unique_id").to_list())
        assert timings[0].n_series_fitted >= 1


class TestEffectiveTrainRows:
    def test_counts_rows_surviving_dropna(self):
        frame = pd.DataFrame({"unique_id": ["a"] * 30 + ["b"] * 20, "ds": range(50), "y": 1.0})
        assert ml.effective_train_rows(frame, max_lag=12) == (30 - 12) + (20 - 12)

    def test_never_goes_negative(self):
        frame = pd.DataFrame({"unique_id": ["a"] * 5, "ds": range(5), "y": 1.0})
        assert ml.effective_train_rows(frame, max_lag=12) == 0


class TestFeatureRecipe:
    @pytest.mark.parametrize(
        ("freq", "expected"),
        [
            ("W-SUN", ("week", "month")),
            ("D", ("dayofweek", "month")),
            ("ME", ("month", "quarter")),
            ("QE-DEC", ("quarter",)),
        ],
    )
    def test_calendar_features_follow_the_frequency(self, freq, expected):
        assert FeatureRecipe.for_freq(freq, 12).date_features == expected

    def test_seasonal_lag_is_included(self):
        assert 52 in FeatureRecipe.for_freq("W-SUN", 52).lags

    def test_non_seasonal_frequency_gets_short_lags_only(self):
        assert FeatureRecipe.for_freq("YE-DEC", 1).lags == (1, 2, 3, 4)


class TestFullRefit:
    def test_forecast_full_reports_no_fold_index(self):
        preds, timings = ml.forecast_full(
            ["GlobalLinear"], panel(), h=H, freq=FREQ, season_length=SEASON, seed=42
        )
        assert all(t.fold_index is None for t in timings)
        assert preds.height == 3 * H


def test_statistical_models_are_rejected_by_the_ml_builder():
    with pytest.raises(InvalidModelNameError):
        build_ml_plan("AutoETS", freq=FREQ, season_length=SEASON, seed=1)
