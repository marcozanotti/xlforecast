"""FR-104/106/108/502 -- profiling, and the orderings that are easy to get backwards."""

from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl
import pytest

from xlforecast.ingest.profile import (
    classify_intermittency,
    infer_freq,
    profile_panel,
    seasonality_strength,
)
from xlforecast.schemas.request import DataMapping

MAPPING = DataMapping(unique_id_col="a", ds_col="b", y_col="c")


def build(series: dict[str, list[float]], freq: str = "W", start: str = "2021-01-03"):
    frames = []
    for uid, values in series.items():
        dates = pd.date_range(start, periods=len(values), freq=freq)
        frames.append(
            pl.DataFrame({"unique_id": [uid] * len(values), "ds": list(dates), "y": values})
        )
    return pl.concat(frames)


class TestIntermittencyQuadrants:
    """FR-108 -- Syntetos-Boylan, cut points ADI 1.32 and CV-squared 0.49.

    Deterministic inputs, because a random draw does not reliably land in the quadrant you
    label it with -- which is how an earlier version of this check misled me.
    """

    def test_smooth_is_frequent_and_steady(self):
        klass, adi, cv2 = classify_intermittency(np.full(100, 50.0))
        assert klass == "smooth"
        assert adi == pytest.approx(1.0)
        assert cv2 == pytest.approx(0.0)

    def test_erratic_is_frequent_but_variable(self):
        values = np.tile([10.0, 200.0], 50)  # every period, hugely variable
        klass, adi, cv2 = classify_intermittency(values)
        assert klass == "erratic"
        assert adi < 1.32
        assert cv2 >= 0.49

    def test_intermittent_is_sparse_but_steady(self):
        values = np.zeros(100)
        values[::4] = 50.0  # one demand in four, identical size
        klass, adi, cv2 = classify_intermittency(values)
        assert klass == "intermittent"
        assert adi >= 1.32
        assert cv2 < 0.49

    def test_lumpy_is_sparse_and_variable(self):
        values = np.zeros(100)
        values[::4] = np.tile([5.0, 300.0], 13)[:25]
        klass, adi, cv2 = classify_intermittency(values)
        assert klass == "lumpy"
        assert adi >= 1.32
        assert cv2 >= 0.49

    def test_an_all_zero_series_does_not_divide_by_zero(self):
        klass, _, cv2 = classify_intermittency(np.zeros(50))
        assert klass == "lumpy"
        assert cv2 == 0.0


class TestFrequencyInference:
    """FR-104."""

    @pytest.mark.parametrize(
        ("freq", "expected"), [("W", "W-SUN"), ("D", "D"), ("ME", "ME"), ("h", "h")]
    )
    def test_common_frequencies_are_inferred(self, freq, expected):
        panel = build({"A": [1.0] * 40}, freq=freq)
        inferred, confidence = infer_freq(panel)
        assert inferred == expected
        assert confidence == pytest.approx(1.0)

    def test_gaps_do_not_reduce_confidence(self):
        """Confidence measures alignment, not completeness. A weekly panel with a missing
        week is still unambiguously weekly -- the gap is FR-106's problem, and gap-filling
        handles it -- so refusing to trust the inference would be the wrong response."""
        dates = list(pd.date_range("2021-01-03", periods=20, freq="W"))
        del dates[5]
        panel = pl.DataFrame({"unique_id": ["A"] * 19, "ds": dates, "y": [1.0] * 19})
        inferred, confidence = infer_freq(panel)
        assert inferred == "W-SUN"
        assert confidence == pytest.approx(1.0)

    def test_off_grid_timestamps_do_reduce_confidence(self):
        """Timestamps falling between grid points are a genuine frequency problem, and
        validation rejects those series with FREQ_MISMATCH."""
        dates = list(pd.date_range("2021-01-03", periods=20, freq="W"))
        dates[7] = dates[7] + pd.Timedelta(days=3)
        dates[12] = dates[12] + pd.Timedelta(days=2)
        panel = pl.DataFrame({"unique_id": ["A"] * 20, "ds": dates, "y": [1.0] * 20})
        _, confidence = infer_freq(panel)
        assert confidence < 1.0

    def test_monthly_panels_are_not_penalised_for_uneven_month_lengths(self):
        """The bug this replaced: month-ends are 28-31 days apart, so a modal-gap measure
        scored a perfectly regular monthly panel at ~0.56."""
        panel = pl.DataFrame(
            {
                "unique_id": ["A"] * 40,
                "ds": list(pd.date_range("2021-01-31", periods=40, freq="ME")),
                "y": [1.0] * 40,
            }
        )
        inferred, confidence = infer_freq(panel)
        assert inferred == "ME"
        assert confidence == pytest.approx(1.0)

    def test_a_single_observation_has_no_inferable_frequency(self):
        panel = build({"A": [1.0]})
        _, confidence = infer_freq(panel)
        assert confidence == 0.0


class TestSeasonalityStrength:
    def test_a_strongly_seasonal_series_scores_high(self):
        t = np.arange(120)
        y = 100 + 30 * np.sin(2 * np.pi * t / 12)
        seasonal, _ = seasonality_strength(y, 12)
        assert seasonal is not None
        assert seasonal > 0.9

    def test_a_series_too_short_to_decompose_returns_none(self):
        """An honest absence rather than a fabricated zero -- diagnostics show these to the
        user, and 0.0 would read as 'measured, no seasonality'."""
        assert seasonality_strength(np.arange(10.0), 12) == (None, None)


class TestProfilePanel:
    def test_ragged_panels_are_flagged(self):
        """The single most consequential fact about a panel for leakage purposes."""
        panel = pl.concat([build({"A": [1.0] * 30}), build({"B": [1.0] * 20})])
        assert profile_panel(panel, data_id="d", mapping=MAPPING, freq="W").ragged

    def test_aligned_panels_are_not_flagged(self):
        panel = build({"A": [1.0] * 30, "B": [2.0] * 30})
        assert not profile_panel(panel, data_id="d", mapping=MAPPING, freq="W").ragged

    def test_intermittency_is_classified_before_gap_filling(self):
        """FR-106 -- `gap_fill='zero'` manufactures intermittency, so classifying afterwards
        would route a smooth-but-gappy series to Croston."""
        smooth = [50.0] * 40
        prefill = build({"A": smooth})
        # The same series after zero-filling a large hole: now full of zeros.
        filled_values = smooth[:20] + [0.0] * 20
        filled = build({"A": filled_values})

        with_prefill = profile_panel(
            filled, data_id="d", mapping=MAPPING, freq="W", prefill=prefill
        )
        without = profile_panel(filled, data_id="d", mapping=MAPPING, freq="W")
        assert with_prefill.series[0].intermittency == "smooth"
        assert without.series[0].intermittency != "smooth"

    def test_series_are_profiled_in_sorted_order(self):
        panel = build({"B": [1.0] * 30, "A": [2.0] * 30})
        profile = profile_panel(panel, data_id="d", mapping=MAPPING, freq="W")
        assert [s.unique_id for s in profile.series] == ["A", "B"]

    def test_the_profile_carries_no_observations(self):
        """NFR-07 -- this is the object that crosses the LLM trust boundary."""
        panel = build({"A": [123.456] * 30})
        payload = profile_panel(panel, data_id="d", mapping=MAPPING, freq="W").model_dump_json()
        assert "123.456" not in payload

    def test_season_length_falls_back_to_the_frequency_default(self):
        panel = build({"A": [1.0] * 60}, freq="ME")
        profile = profile_panel(panel, data_id="d", mapping=MAPPING, freq="ME")
        assert profile.season_length_candidates == [12]

    def test_intermittent_share_counts_intermittent_and_lumpy(self):
        sparse = [0.0, 0.0, 0.0, 50.0] * 15
        panel = build({"smooth": [50.0] * 60, "sparse": sparse})
        profile = profile_panel(panel, data_id="d", mapping=MAPPING, freq="W")
        assert profile.intermittent_share == pytest.approx(0.5)
