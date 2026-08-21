"""Cross-validation folds -- the single source of truth for cutoffs (TS §5.1, FR-205/206).

**Why this module drives the loop instead of calling the libraries' `cross_validation`.**

Neither `StatsForecast.cross_validation` nor `MLForecast.cross_validation` accepts a cutoff
array (verified against statsforecast 2.1.1 / mlforecast 1.1.0). Both derive cutoffs
internally from *each series' own last timestamp* --
`utilsforecast.processing.backtest_splits` computes `max_dates = groupby(id).ds.max()`, and
statsforecast walks `range(-test_size, -h+1, step_size)` per series group.

On a ragged panel -- the normal case for SKU data with new and discontinued products -- that
gives every series a different fold-1 date. A global model trained at series A's cutoff has
then seen series B's observations from after it. That is look-ahead leakage in exactly the
family FR-207 wants compared fairly against local models, and no assertion about cutoff
*equality* can detect it, because per-series cutoffs are trivially equal across families.

So cutoffs here are **panel-wide calendar dates**, and every family is handed the same
pre-sliced train/test frames (FR-206a).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd
import polars as pl

from xlforecast.errors import XLForecastError
from xlforecast.panel import DS, ID, span
from xlforecast.schemas.enums import ExclusionReason

__all__ = ["Fold", "InsufficientHistoryError", "make_cutoffs", "make_folds", "split"]


class InsufficientHistoryError(XLForecastError):
    """The panel is too short for the requested CV configuration."""


@dataclass(frozen=True, slots=True)
class Fold:
    """One CV window, identical for every model family.

    `series` is the fold's common support: the exact set every model is scored on. Handing
    each family the same pre-sliced frames is what makes AC-205's test-index assertion a
    statement about the engine rather than about two libraries agreeing by coincidence.
    """

    index: int
    cutoff: pd.Timestamp
    train: pl.DataFrame
    test: pl.DataFrame
    series: frozenset[str]
    excluded: dict[str, ExclusionReason] = field(default_factory=dict)

    def test_index(self) -> list[tuple[str, pd.Timestamp]]:
        """The `(unique_id, ds)` pairs being scored, sorted.

        This is the property AC-205 asserts on. Identical cutoffs do not imply identical
        evaluation sets, and the evaluation set is what makes a leaderboard comparable.
        """
        return sorted(
            (row[ID], row[DS]) for row in self.test.select([ID, DS]).iter_rows(named=True)
        )


def _calendar(panel: pl.DataFrame, freq: str) -> pd.DatetimeIndex:
    """The panel's canonical calendar at `freq`.

    Cutoffs are chosen by *position* on this grid rather than by offset arithmetic on a
    timestamp, so month-end, quarter-end and anchored weekly frequencies behave without
    special cases.
    """
    lo, hi = span(panel)
    return pd.date_range(start=lo, end=hi, freq=freq)


def make_cutoffs(
    panel: pl.DataFrame, *, h: int, n_windows: int, step_size: int, freq: str
) -> list[pd.Timestamp]:
    """Panel-wide calendar cutoffs, computed once per job (FR-206).

    Returned ascending. Fold `i` trains on `ds <= cutoffs[i]` and is scored on the `h`
    calendar steps after it.
    """
    grid = _calendar(panel, freq)
    last = len(grid) - 1 - h  # the latest cutoff that still leaves h steps to score
    if last < 0:
        raise InsufficientHistoryError(
            f"the panel spans {len(grid)} periods at freq '{freq}', which cannot support a "
            f"horizon of {h}.",
            fix=f"Shorten the horizon below {len(grid)}, or supply more history.",
        )
    first = last - (n_windows - 1) * step_size
    if first < 0:
        needed = h + (n_windows - 1) * step_size + 1
        raise InsufficientHistoryError(
            f"{n_windows} windows of horizon {h} at step {step_size} need at least {needed} "
            f"periods; the panel spans {len(grid)}.",
            fix=f"Reduce n_windows or the horizon, or supply at least {needed} periods.",
        )
    return [grid[first + i * step_size] for i in range(n_windows)]


def split(
    panel: pl.DataFrame, cutoff: pd.Timestamp, *, h: int, freq: str
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Split at a panel-wide `cutoff` into (train, test).

    Train is everything up to and including the cutoff; test is the next `h` calendar steps.
    The test window is bounded on both sides so a ragged series with a late gap cannot
    contribute observations from beyond the horizon.
    """
    horizon_end = pd.date_range(start=cutoff, periods=h + 1, freq=freq)[-1]
    train = panel.filter(pl.col(DS) <= cutoff)
    test = panel.filter((pl.col(DS) > cutoff) & (pl.col(DS) <= horizon_end))
    return train, test


def make_folds(
    panel: pl.DataFrame, *, h: int, n_windows: int, step_size: int, freq: str
) -> list[Fold]:
    """Build every fold once. Both families consume these same objects.

    A series is in a fold's support only if it has training history at the cutoff *and* at
    least one observation to be scored on. Series failing either are excluded from that fold
    **for every model** (FR-206b) -- a model may not be scored on a fold from which another
    model was excluded, or the panel aggregate silently compares different things.
    """
    cutoffs = make_cutoffs(panel, h=h, n_windows=n_windows, step_size=step_size, freq=freq)
    all_series = set(panel.get_column(ID).unique().to_list())

    folds: list[Fold] = []
    for index, cutoff in enumerate(cutoffs):
        train, test = split(panel, cutoff, h=h, freq=freq)
        trained = set(train.get_column(ID).unique().to_list())
        tested = set(test.get_column(ID).unique().to_list())
        support = trained & tested

        excluded: dict[str, ExclusionReason] = {}
        for uid in sorted(all_series - support):
            excluded[uid] = (
                ExclusionReason.TOO_SHORT if uid not in trained else ExclusionReason.EXCESS_MISSING
            )

        folds.append(
            Fold(
                index=index,
                cutoff=cutoff,
                train=train.filter(pl.col(ID).is_in(list(support))),
                test=test.filter(pl.col(ID).is_in(list(support))),
                series=frozenset(support),
                excluded=excluded,
            )
        )
    return folds
