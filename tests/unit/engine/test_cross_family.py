"""AC-205 / AC-206 -- the identical-test-index property, across model families.

CLAUDE.md hard rule 2: these tests must never be skipped or marked xfail.

`test_folds.py` proves the folds themselves are panel-wide. This file proves the property
that actually matters: that a **local** model and a **global** model, run through the real
adapters, are scored on element-wise identical `(unique_id, ds)` sets on a deliberately
ragged panel.

`TestWhyNotLibraryCV` is the control. It demonstrates that the Nixtla libraries' own
cross-validation does *not* have this property, so the assertions above are a statement about
our engine rather than something we would get for free. Without the control, a future
refactor that "simplifies" back to `MLForecast.cross_validation` would keep every other test
green while silently reintroducing look-ahead leakage.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl
import pytest

from xlforecast.engine import local as local_family
from xlforecast.engine import ml as ml_family
from xlforecast.engine.evaluate import score_fold
from xlforecast.engine.folds import make_folds

H = 13
N_WINDOWS = 3
STEP = 13
SEASON = 52
FREQ = "W"

LOCAL_MODELS = ["SeasonalNaive", "HistoricAverage"]
GLOBAL_MODELS = ["GlobalLinear"]
LOCAL_ML_MODELS = ["LocalLinear"]


def ragged_panel() -> pl.DataFrame:
    """Series ending on three different dates.

    This is the normal case for SKU data with new and discontinued products, and it is the
    case in which per-series cutoffs leak: a global model trained at the short series' cutoff
    would have seen the long series' later observations.
    """
    rng = np.random.default_rng(5)
    frames = []
    for uid, n in (("long", 200), ("mid", 175), ("short", 160)):
        dates = pd.date_range("2021-01-03", periods=n, freq=FREQ)
        t = np.arange(n)
        y = 100 + 20 * np.sin(2 * np.pi * t / SEASON) + rng.normal(0, 3, n)
        frames.append(pl.DataFrame({"unique_id": [uid] * n, "ds": list(dates), "y": y.tolist()}))
    return pl.concat(frames)


@pytest.fixture(scope="module")
def folds():
    return make_folds(ragged_panel(), h=H, n_windows=N_WINDOWS, step_size=STEP, freq=FREQ)


@pytest.fixture(scope="module")
def family_predictions(folds):
    """Run both families through the real adapters on the same Fold objects."""
    panel = ragged_panel()
    origin = panel.get_column("ds").min()
    out = {}
    for fold in folds:
        loc, _ = local_family.forecast_fold(
            LOCAL_MODELS, fold, h=H, freq=FREQ, season_length=SEASON, origin=origin
        )
        glb, _ = ml_family.forecast_fold(
            GLOBAL_MODELS + LOCAL_ML_MODELS,
            fold,
            h=H,
            freq=FREQ,
            season_length=SEASON,
            seed=42,
        )
        out[fold.index] = (loc, glb)
    return out


def scored_index(predictions: pl.DataFrame, fold, model: str) -> list[tuple[str, pd.Timestamp]]:
    """The `(unique_id, ds)` pairs this model is actually graded on for this fold."""
    joined = (
        predictions.filter(pl.col("model") == model)
        .join(fold.test.select(["unique_id", "ds"]), on=["unique_id", "ds"], how="inner")
        .select(["unique_id", "ds"])
    )
    return sorted((r["unique_id"], r["ds"]) for r in joined.iter_rows(named=True))


class TestIdenticalTestIndexAcrossFamilies:
    """AC-205. Never skip. Never xfail."""

    def test_local_and_global_are_scored_on_identical_pairs(self, folds, family_predictions):
        for fold in folds:
            loc, glb = family_predictions[fold.index]
            local_index = scored_index(loc, fold, "SeasonalNaive")
            global_index = scored_index(glb, fold, "GlobalLinear")
            assert local_index == global_index, f"fold {fold.index} diverged across families"

    def test_local_ml_matches_its_global_twin(self, folds, family_predictions):
        """FR-203b -- the matched pair differs in the information set and nothing else, so
        the two halves must be graded on exactly the same points."""
        for fold in folds:
            _, glb = family_predictions[fold.index]
            assert scored_index(glb, fold, "LocalLinear") == scored_index(glb, fold, "GlobalLinear")

    def test_the_scored_index_equals_the_fold_support(self, folds, family_predictions):
        for fold in folds:
            loc, _ = family_predictions[fold.index]
            assert scored_index(loc, fold, "SeasonalNaive") == fold.test_index()

    def test_each_cutoff_is_one_calendar_date_for_the_whole_panel(self, folds):
        """FR-206. The property the libraries do not give us."""
        for fold in folds:
            assert isinstance(fold.cutoff, pd.Timestamp)
            train_max = fold.train.get_column("ds").max()
            assert train_max <= fold.cutoff

    def test_no_family_sees_an_observation_after_the_cutoff(self, folds):
        """The leakage assertion stated directly: nothing in any training slice postdates
        the fold's single panel-wide cutoff, for any series."""
        for fold in folds:
            latest = fold.train.group_by("unique_id").agg(pl.col("ds").max())
            assert all(row[1] <= fold.cutoff for row in latest.iter_rows())

    def test_evaluation_scores_both_families_on_the_same_series(self, folds, family_predictions):
        for fold in folds:
            loc, glb = family_predictions[fold.index]
            scores = score_fold(
                fold,
                pl.concat([loc, glb]),
                models=LOCAL_MODELS + GLOBAL_MODELS + LOCAL_ML_MODELS,
                season_length=SEASON,
            )
            by_model: dict[str, set[str]] = {}
            for s in scores:
                if s.unique_id is not None:
                    by_model.setdefault(s.model, set()).add(s.unique_id)
            assert len(set(map(frozenset, by_model.values()))) == 1, (
                "every model must be scored on the same series within a fold"
            )


class TestEffectiveTrainingRows:
    """AC-206 -- identical cutoffs still mean different training samples."""

    def test_the_dropna_asymmetry_is_recorded_not_hidden(self, folds):
        """mlforecast drops `max_lag` rows per series for lag construction; statsforecast
        trains on the full pre-cutoff history. A local-vs-global comparison that does not
        surface this is silently confounded by it."""
        panel = ragged_panel()
        origin = panel.get_column("ds").min()
        fold = folds[0]

        _, local_timings = local_family.forecast_fold(
            ["SeasonalNaive"], fold, h=H, freq=FREQ, season_length=SEASON, origin=origin
        )
        _, ml_timings = ml_family.forecast_fold(
            ["GlobalLinear"], fold, h=H, freq=FREQ, season_length=SEASON, seed=42
        )
        assert local_timings[0].n_rows_trained > ml_timings[0].n_rows_trained
        # The gap is exactly max_lag rows per series in the fold's support -- derived from
        # the support rather than hardcoded, because a ragged panel's support shrinks in
        # later folds as short series run out.
        max_lag = SEASON
        expected_gap = len(fold.series) * max_lag
        assert local_timings[0].n_rows_trained - ml_timings[0].n_rows_trained == expected_gap

    def test_the_asymmetry_grows_with_the_folds_support(self, folds):
        """Sanity on the above: a fold containing more series loses more rows to lag
        construction, so the confound is not a fixed offset that could be waved away."""
        panel = ragged_panel()
        origin = panel.get_column("ds").min()
        gaps = []
        for fold in folds:
            _, loc = local_family.forecast_fold(
                ["SeasonalNaive"], fold, h=H, freq=FREQ, season_length=SEASON, origin=origin
            )
            _, mlt = ml_family.forecast_fold(
                ["GlobalLinear"], fold, h=H, freq=FREQ, season_length=SEASON, seed=42
            )
            gaps.append((len(fold.series), loc[0].n_rows_trained - mlt[0].n_rows_trained))
        assert all(gap == n * SEASON for n, gap in gaps)


class TestWhyNotLibraryCV:
    """The control: the libraries' own CV does NOT have the property asserted above.

    If this class ever fails, the libraries changed their semantics and FR-206a should be
    re-read before anything is 'simplified'.
    """

    def test_library_backtest_splits_produce_per_series_cutoffs(self):
        """`utilsforecast.processing.backtest_splits` derives cutoffs from each series' own
        max date, so a ragged panel gets a different fold-1 date per series. This is the
        leakage FR-206a describes, demonstrated rather than asserted."""
        from utilsforecast.processing import backtest_splits

        panel = ragged_panel().to_pandas().sort_values(["unique_id", "ds"])
        cutoffs_seen = []
        for cutoffs, _train, _valid in backtest_splits(
            panel,
            n_windows=N_WINDOWS,
            h=H,
            id_col="unique_id",
            time_col="ds",
            freq=pd.tseries.frequencies.to_offset(FREQ),
            step_size=STEP,
        ):
            cutoffs_seen.append(cutoffs["cutoff"].nunique())

        assert max(cutoffs_seen) > 1, (
            "the libraries would give this ragged panel more than one cutoff per fold; "
            "if this now fails, re-read FR-206a before changing engine/folds.py"
        )

    def test_our_folds_collapse_those_to_one_date_each(self, folds):
        assert len({f.cutoff for f in folds}) == N_WINDOWS
        for fold in folds:
            assert isinstance(fold.cutoff, pd.Timestamp)
