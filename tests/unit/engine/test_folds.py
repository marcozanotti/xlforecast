"""FR-205/206/206a/206b -- the single most important correctness property in the system.

`test_cutoffs_are_single_calendar_dates_on_a_ragged_panel` and the test-index assertions in
this file must never be skipped or marked xfail (CLAUDE.md hard rule 2).
"""

from __future__ import annotations

from itertools import pairwise

import pandas as pd
import polars as pl
import pytest

from xlforecast.engine.folds import InsufficientHistoryError, make_cutoffs, make_folds, split


def panel(spec: dict[str, int], *, start: str = "2021-01-03", freq: str = "W") -> pl.DataFrame:
    """Build a panel from {series: n_obs}. Series share a start and end at different dates,
    which is what makes a panel ragged."""
    frames = []
    for uid, n in spec.items():
        dates = pd.date_range(start, periods=n, freq=freq)
        frames.append(
            pl.DataFrame(
                {"unique_id": [uid] * n, "ds": list(dates), "y": [float(i) for i in range(n)]}
            )
        )
    return pl.concat(frames)


def ragged() -> pl.DataFrame:
    """Three different end dates -- the normal case for SKU data with new and discontinued
    products, and the case that makes per-series cutoffs leak.

    Lengths chosen against the actual window boundaries: with 156 periods, h=13, n_windows=3
    and step 13, the cutoffs land at grid indices 116/129/142 and the test windows are
    117-129, 130-142 and 143-155. So `short` (135) is scored in folds 0 and 1 but has ended
    before fold 2, which is exactly the ragged behaviour worth testing.
    """
    return panel({"long": 156, "mid": 145, "short": 135})


class TestCutoffs:
    def test_cutoffs_are_ascending_and_correctly_spaced(self):
        cuts = make_cutoffs(panel({"A": 156}), h=13, n_windows=3, step_size=13, freq="W")
        assert cuts == sorted(cuts)
        assert len(cuts) == 3
        assert all((b - a).days == 13 * 7 for a, b in pairwise(cuts))

    def test_last_cutoff_leaves_exactly_h_periods_to_score(self):
        p = panel({"A": 156})
        cuts = make_cutoffs(p, h=13, n_windows=3, step_size=13, freq="W")
        end = p.get_column("ds").max()
        assert len(pd.date_range(cuts[-1], end, freq="W")) - 1 == 13

    def test_cutoffs_are_identical_for_every_series_on_a_ragged_panel(self):
        """FR-206a -- the property the Nixtla libraries do NOT give us. Their
        `backtest_splits` derives cutoffs from each series' own max date, so `long` and
        `short` would get different fold-1 dates and a global model trained at `short`'s
        cutoff would have seen `long`'s later observations."""
        cuts = make_cutoffs(ragged(), h=13, n_windows=3, step_size=13, freq="W")
        for fold in make_folds(ragged(), h=13, n_windows=3, step_size=13, freq="W"):
            assert fold.cutoff in cuts
            # one date for the whole panel, not one per series
            assert isinstance(fold.cutoff, pd.Timestamp)

    def test_ragged_panel_cutoffs_are_anchored_to_the_panel_not_the_longest_series(self):
        wide = make_cutoffs(ragged(), h=13, n_windows=3, step_size=13, freq="W")
        just_long = make_cutoffs(panel({"long": 156}), h=13, n_windows=3, step_size=13, freq="W")
        assert wide == just_long, "panel max date defines the calendar"

    @pytest.mark.parametrize(
        ("freq", "n", "h"), [("W", 156, 13), ("ME", 60, 6), ("D", 400, 30), ("QE", 40, 4)]
    )
    def test_positional_indexing_handles_anchored_frequencies(self, freq, n, h):
        """Cutoffs are chosen by position on the calendar grid rather than by offset
        arithmetic, so month-end and quarter-end need no special casing."""
        cuts = make_cutoffs(panel({"A": n}, freq=freq), h=h, n_windows=2, step_size=h, freq=freq)
        assert len(cuts) == 2

    def test_too_short_for_the_horizon_is_a_named_error(self):
        with pytest.raises(InsufficientHistoryError) as exc:
            make_cutoffs(panel({"A": 10}), h=13, n_windows=1, step_size=13, freq="W")
        assert exc.value.fix

    def test_too_short_for_the_windows_states_what_is_needed(self):
        with pytest.raises(InsufficientHistoryError) as exc:
            make_cutoffs(panel({"A": 20}), h=13, n_windows=3, step_size=13, freq="W")
        assert "at least" in str(exc.value)


class TestSplit:
    def test_train_is_inclusive_of_the_cutoff_and_test_is_exclusive(self):
        p = panel({"A": 156})
        cutoff = make_cutoffs(p, h=13, n_windows=1, step_size=13, freq="W")[0]
        train, test = split(p, cutoff, h=13, freq="W")
        assert train.get_column("ds").max() == cutoff
        assert test.get_column("ds").min() > cutoff

    def test_test_window_is_bounded_above_by_the_horizon(self):
        """Bounded on both sides, so a ragged series with a late gap cannot contribute
        observations from beyond the horizon."""
        p = panel({"A": 200})
        cutoff = make_cutoffs(p, h=13, n_windows=1, step_size=13, freq="W")[0]
        _, test = split(p, cutoff, h=13, freq="W")
        assert test.height == 13

    def test_no_observation_appears_in_both_train_and_test(self):
        p = ragged()
        for cutoff in make_cutoffs(p, h=13, n_windows=3, step_size=13, freq="W"):
            train, test = split(p, cutoff, h=13, freq="W")
            overlap = set(map(tuple, train.select(["unique_id", "ds"]).rows())) & set(
                map(tuple, test.select(["unique_id", "ds"]).rows())
            )
            assert not overlap


class TestFoldSupport:
    def test_every_family_would_receive_the_identical_test_index(self):
        """AC-205. The folds are built once and handed to both adapters, so the evaluation
        set cannot diverge between families -- which is the property that actually makes the
        leaderboard comparable. Identical cutoffs alone do not imply this."""
        folds = make_folds(ragged(), h=13, n_windows=3, step_size=13, freq="W")
        for fold in folds:
            index = fold.test_index()
            assert index == sorted(index)
            assert len(index) == len(set(index)), "no duplicate (unique_id, ds) pairs"
            assert {uid for uid, _ in index} == fold.series

    def test_series_absent_from_a_fold_are_excluded_for_every_model(self):
        """FR-206b -- a model may not be scored on a fold another model was excluded from."""
        folds = make_folds(ragged(), h=13, n_windows=3, step_size=13, freq="W")
        early = folds[0]
        assert "short" in early.series  # 135 obs reaches into the first test window
        late = folds[-1]
        assert "short" not in late.series, "short series ended before the last window"
        assert "short" in late.excluded

    def test_excluded_series_always_carry_a_reason(self):
        """FS §6 -- silently dropping a series is a listed failure mode."""
        for fold in make_folds(ragged(), h=13, n_windows=3, step_size=13, freq="W"):
            assert all(reason for reason in fold.excluded.values())

    def test_support_is_the_intersection_of_trained_and_tested(self):
        for fold in make_folds(ragged(), h=13, n_windows=3, step_size=13, freq="W"):
            trained = set(fold.train.get_column("unique_id").unique().to_list())
            tested = set(fold.test.get_column("unique_id").unique().to_list())
            assert trained == tested == set(fold.series)

    def test_folds_are_deterministic(self):
        """NFR-02 -- the same panel and spec must produce the same folds, every time."""
        a = make_folds(ragged(), h=13, n_windows=3, step_size=13, freq="W")
        b = make_folds(ragged(), h=13, n_windows=3, step_size=13, freq="W")
        assert [f.cutoff for f in a] == [f.cutoff for f in b]
        assert [f.test_index() for f in a] == [f.test_index() for f in b]

    def test_training_windows_grow_monotonically(self):
        folds = make_folds(panel({"A": 156}), h=13, n_windows=3, step_size=13, freq="W")
        heights = [f.train.height for f in folds]
        assert heights == sorted(heights)
        assert len(set(heights)) == 3

    def test_overlapping_windows_are_supported(self):
        """step_size < h gives overlapping test windows, which is legitimate."""
        folds = make_folds(panel({"A": 156}), h=13, n_windows=3, step_size=4, freq="W")
        assert len(folds) == 3
        assert folds[1].cutoff > folds[0].cutoff
