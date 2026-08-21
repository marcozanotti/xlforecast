"""Per-series validation with named exclusion reasons (FR-105, FR-105a).

Every rejection carries a reason and a sentence naming the series. Silently dropping a
series is a listed failure mode (FS §6), so there is no path through this module that
removes a series without recording why.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl

from xlforecast.panel import DS, ID, Y, span
from xlforecast.schemas.enums import ExclusionReason
from xlforecast.schemas.profile import DataProfile, ValidationReport
from xlforecast.schemas.request import ForecastRequest

__all__ = ["MISSING_THRESHOLD", "validate_panel"]

MISSING_THRESHOLD = 0.5  # FR-105: ">50% missing"

_SENTENCES: dict[ExclusionReason, str] = {
    ExclusionReason.DUPLICATE_TIMESTAMPS: "has {n} duplicated timestamp(s)",
    ExclusionReason.NON_MONOTONIC: "has timestamps that are not in ascending order",
    ExclusionReason.TOO_SHORT: "has {n} observations but {need} are required",
    ExclusionReason.ALL_ZERO: "is entirely zero",
    ExclusionReason.ALL_CONSTANT: "never changes value",
    ExclusionReason.EXCESS_MISSING: "is {pct:.0%} missing",
    ExclusionReason.FREQ_MISMATCH: "has {n} timestamp(s) off the {freq} calendar",
}

_FIXES: dict[ExclusionReason, str] = {
    ExclusionReason.DUPLICATE_TIMESTAMPS: "Aggregate or de-duplicate the rows for this series.",
    ExclusionReason.NON_MONOTONIC: "Sort the rows by date before uploading.",
    ExclusionReason.TOO_SHORT: "Shorten the horizon, reduce CV windows, or supply more history.",
    ExclusionReason.ALL_ZERO: "Remove the series, or forecast it as a constant zero.",
    ExclusionReason.ALL_CONSTANT: "Remove the series; a competition cannot rank a constant.",
    ExclusionReason.EXCESS_MISSING: "Supply more history, or enable gap filling.",
    ExclusionReason.FREQ_MISMATCH: "Resample the series onto a regular calendar.",
}


def _grid(panel: pl.DataFrame, freq: str) -> set[pd.Timestamp]:
    lo, hi = span(panel)
    return set(pd.date_range(start=lo, end=hi, freq=freq))


def validate_panel(
    panel: pl.DataFrame, *, request: ForecastRequest, profile: DataProfile, season_length: int
) -> ValidationReport:
    """Apply FR-105 per series.

    The length threshold is `2*m + h + (n_windows-1)*step_size`, not the naive `2*m + h`:
    the *earliest* CV training window must itself satisfy `2*m`. Series between the two
    thresholds would otherwise pass ingestion and then vanish inside cross-validation --
    which is the silent drop FS §6 forbids. See `ForecastRequest.min_observations`.
    """
    required = request.min_observations(season_length)
    grid = _grid(panel, profile.freq_inferred)

    excluded: dict[str, ExclusionReason] = {}
    detail: dict[str, str] = {}

    def reject(uid: str, reason: ExclusionReason, **fmt: object) -> None:
        if uid in excluded:  # first fault wins; the user fixes one thing at a time
            return
        excluded[uid] = reason
        excluded_sentence = _SENTENCES[reason].format(**fmt)
        detail[uid] = f"Series '{uid}' {excluded_sentence}. {_FIXES[reason]}"

    for (uid,), frame in sorted(panel.group_by([ID]), key=lambda kv: kv[0]):
        dates = frame.get_column(DS)
        values = frame.get_column(Y).to_numpy().astype(float)

        n_dupes = int(dates.len() - dates.n_unique())
        if n_dupes:
            reject(uid, ExclusionReason.DUPLICATE_TIMESTAMPS, n=n_dupes)
            continue

        if not dates.to_pandas().is_monotonic_increasing:
            reject(uid, ExclusionReason.NON_MONOTONIC)
            continue

        off_grid = int(sum(1 for d in dates.to_pandas() if d not in grid))
        if off_grid:
            reject(uid, ExclusionReason.FREQ_MISMATCH, n=off_grid, freq=profile.freq_inferred)
            continue

        n_missing = int(np.isnan(values).sum())
        if values.size and n_missing / values.size > MISSING_THRESHOLD:
            reject(uid, ExclusionReason.EXCESS_MISSING, pct=n_missing / values.size)
            continue

        observed = values[~np.isnan(values)]
        if observed.size and np.all(observed == 0):
            reject(uid, ExclusionReason.ALL_ZERO)
            continue
        if observed.size > 1 and np.all(observed == observed[0]):
            reject(uid, ExclusionReason.ALL_CONSTANT)
            continue

        if values.size < required:
            reject(uid, ExclusionReason.TOO_SHORT, n=values.size, need=required)
            continue

    n_in = profile.n_series
    return ValidationReport(
        n_series_in=n_in,
        n_series_out=n_in - len(excluded),
        excluded=excluded,
        excluded_detail=detail,
    )
