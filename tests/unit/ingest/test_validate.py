"""FR-105 / FR-105a -- per-series validation with named reasons.

FS §6: silently dropping a series is a listed failure mode, so every rejection here must
carry both a reason enum and a sentence naming the series.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl

from xlforecast.ingest.profile import profile_panel
from xlforecast.ingest.validate import validate_panel
from xlforecast.schemas.enums import ExclusionReason
from xlforecast.schemas.request import DataMapping, ForecastRequest

MAPPING = DataMapping(unique_id_col="a", ds_col="b", y_col="c")
REQUEST = ForecastRequest(h=13, freq="W", n_windows=3)
SEASON = 52
LONG = 200


def series(uid: str, values, dates=None) -> pl.DataFrame:
    dates = (
        dates if dates is not None else pd.date_range("2021-01-03", periods=len(values), freq="W")
    )
    return pl.DataFrame(
        {
            "unique_id": [uid] * len(values),
            "ds": list(dates),
            "y": [None if v is None else float(v) for v in values],
        }
    )


def healthy(uid: str = "ok", n: int = LONG) -> pl.DataFrame:
    rng = np.random.default_rng(abs(hash(uid)) % 1000)
    return series(uid, 100 + rng.normal(0, 5, n))


def report_for(panel: pl.DataFrame):
    profile = profile_panel(panel, data_id="d", mapping=MAPPING, freq="W")
    return validate_panel(panel, request=REQUEST, profile=profile, season_length=SEASON)


class TestExclusionReasons:
    def test_healthy_series_survive(self):
        assert report_for(healthy()).n_excluded == 0

    def test_duplicate_timestamps(self):
        dates = list(pd.date_range("2021-01-03", periods=LONG, freq="W"))
        dates[-1] = dates[0]
        panel = pl.concat([healthy(), series("dupes", np.ones(LONG), dates)])
        assert report_for(panel).excluded["dupes"] == ExclusionReason.DUPLICATE_TIMESTAMPS

    def test_non_monotonic_timestamps(self):
        dates = list(reversed(pd.date_range("2021-01-03", periods=LONG, freq="W")))
        panel = pl.concat([healthy(), series("backwards", np.arange(LONG), dates)])
        assert report_for(panel).excluded["backwards"] == ExclusionReason.NON_MONOTONIC

    def test_all_zero(self):
        panel = pl.concat([healthy(), series("zeros", np.zeros(LONG))])
        assert report_for(panel).excluded["zeros"] == ExclusionReason.ALL_ZERO

    def test_all_constant(self):
        panel = pl.concat([healthy(), series("flat", np.full(LONG, 7.0))])
        assert report_for(panel).excluded["flat"] == ExclusionReason.ALL_CONSTANT

    def test_excess_missing(self):
        """FR-105 says '>50% missing'. Exactly 50% is not excess, so the fixture must clear
        the threshold -- and the observed values must vary, or all_constant fires first."""
        rng = np.random.default_rng(1)
        values = [None if i % 3 else float(rng.normal(100, 5)) for i in range(LONG)]
        panel = pl.concat([healthy(), series("holey", values)])
        assert report_for(panel).excluded["holey"] == ExclusionReason.EXCESS_MISSING

    def test_exactly_half_missing_is_not_excess(self):
        rng = np.random.default_rng(2)
        values = [None if i % 2 == 0 else float(rng.normal(100, 5)) for i in range(LONG)]
        panel = pl.concat([healthy(), series("borderline_missing", values)])
        assert "borderline_missing" not in report_for(panel).excluded

    def test_too_short(self):
        panel = pl.concat([healthy(), healthy("stub", n=60)])
        assert report_for(panel).excluded["stub"] == ExclusionReason.TOO_SHORT

    def test_off_grid_timestamps(self):
        dates = list(pd.date_range("2021-01-03", periods=LONG, freq="W"))
        dates[10] = dates[10] + pd.Timedelta(days=2)
        panel = pl.concat([healthy(), series("offgrid", np.arange(LONG), dates)])
        assert report_for(panel).excluded["offgrid"] == ExclusionReason.FREQ_MISMATCH


class TestLengthThreshold:
    """FR-105 -- the naive 2m + h admits series that then vanish inside cross-validation."""

    def test_threshold_is_2m_plus_h_plus_the_earlier_windows(self):
        required = REQUEST.min_observations(SEASON)
        assert required == 2 * SEASON + 13 + 2 * 13

    def test_a_series_between_the_two_thresholds_is_excluded_not_silently_dropped(self):
        """117 observations clears the naive 2m + h but not the CV configuration. Under the
        original rule it would have passed ingestion and then disappeared mid-run."""
        naive = 2 * SEASON + 13
        assert naive == 117
        panel = pl.concat([healthy(), healthy("borderline", n=naive)])
        report = report_for(panel)
        assert report.excluded["borderline"] == ExclusionReason.TOO_SHORT
        assert "143 are required" in report.excluded_detail["borderline"]


class TestReportQuality:
    def test_every_exclusion_carries_a_sentence_naming_the_series(self):
        panel = pl.concat([healthy(), series("zeros", np.zeros(LONG)), healthy("stub", n=30)])
        report = report_for(panel)
        for uid in report.excluded:
            assert uid in report.excluded_detail[uid]
            assert report.excluded_detail[uid].endswith(".")

    def test_counts_reconcile(self):
        panel = pl.concat([healthy(), series("zeros", np.zeros(LONG)), healthy("stub", n=30)])
        report = report_for(panel)
        assert report.n_series_in - report.n_excluded == report.n_series_out

    def test_the_first_fault_wins_so_the_user_fixes_one_thing_at_a_time(self):
        """A series that is both too short and all-zero reports one reason, not a pile."""
        panel = pl.concat([healthy(), series("both", np.zeros(30))])
        report = report_for(panel)
        assert report.excluded["both"] == ExclusionReason.ALL_ZERO


def test_validation_needs_the_season_length_profiling_infers():
    """FR-105a -- the threshold depends on season_length, which profiling produces. The
    original `ingest -> validate -> profile` order was circular."""
    # 140 observations clears the threshold at m=12 (63) but not at m=52 (143), so which
    # season_length profiling infers decides whether this series survives at all.
    panel = healthy("only", n=140)
    profile = profile_panel(panel, data_id="d", mapping=MAPPING, freq="W")
    inferred = profile.season_length_candidates[0]
    assert inferred == 52
    assert REQUEST.min_observations(52) == 143
    assert REQUEST.min_observations(12) == 63

    tight = validate_panel(panel, request=REQUEST, profile=profile, season_length=inferred)
    loose = validate_panel(panel, request=REQUEST, profile=profile, season_length=12)
    assert tight.n_excluded == 1
    assert loose.n_excluded == 0
