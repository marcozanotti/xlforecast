"""DataProfile computation (FR-104, FR-108, FR-502).

Two orderings in this module are load-bearing and easy to get backwards:

1. **Profiling precedes validation.** FR-105's length threshold depends on `season_length`,
   which is inferred here when the user does not supply it. The original pipeline order
   (`ingest -> validate -> profile`) was circular; FR-105a fixed it.
2. **Intermittency is classified on the PRE-gap-fill series.** `gap_fill="zero"` manufactures
   intermittency, so classifying afterwards would route smooth-but-gappy series to Croston
   (FR-106). Callers pass the pre-fill frame as `prefill`.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import polars as pl

from xlforecast.panel import DS, ID, Y
from xlforecast.schemas.enums import IntermittencyClass
from xlforecast.schemas.freq import default_season_length, normalise_freq
from xlforecast.schemas.profile import DataProfile, SeriesProfile, ValidationReport
from xlforecast.schemas.request import DataMapping

__all__ = ["classify_intermittency", "infer_freq", "profile_panel", "seasonality_strength"]

# Syntetos-Boylan cut points.
_ADI_CUT = 1.32
_CV2_CUT = 0.49


def infer_freq(panel: pl.DataFrame) -> tuple[str, float]:
    """Infer the panel frequency, with a confidence (FR-104).

    Confidence is the share of observed timestamps that land on the inferred frequency's
    calendar grid -- that is, it measures **alignment, not completeness**. A weekly panel
    with missing weeks still scores 1.0, because every timestamp it does have is a valid
    weekly one; the gaps are FR-106's problem and are gap-filled. A panel whose timestamps
    fall between grid points scores below 1.0, and validation rejects those series with
    `FREQ_MISMATCH`.

    Measured against the grid rather than against gap equality because calendar offsets do
    not have constant gaps: month-ends are 28, 29, 30 or 31 days apart, so a modal-gap
    measure scores a *perfectly regular* monthly panel at about 0.56, and the ingest layer
    would have spent its life distrusting the most common business frequency there is.
    """
    gaps = (
        panel.sort([ID, DS])
        .with_columns(pl.col(DS).diff().over(ID).alias("gap"))
        .drop_nulls("gap")
        .get_column("gap")
    )
    if gaps.is_empty():
        return "D", 0.0

    delta = pd.Timedelta(gaps.mode().sort()[0])
    alias = _alias_for(delta)
    return alias, _grid_confidence(panel, alias)


def _alias_for(delta: pd.Timedelta) -> str:
    for candidate, span in (("h", "1h"), ("D", "1D"), ("W", "7D")):
        if delta == pd.Timedelta(span):
            return normalise_freq(candidate)
    days = delta.days
    if 28 <= days <= 31:
        return normalise_freq("ME")
    if 89 <= days <= 92:
        return normalise_freq("QE")
    if 360 <= days <= 366:
        return normalise_freq("YE")
    if days >= 1:
        return f"{days}D"
    return f"{int(delta.total_seconds())}s"


def _grid_confidence(panel: pl.DataFrame, alias: str) -> float:
    """Share of observed timestamps that fall on the inferred calendar grid."""
    stamps = panel.get_column(DS).unique().to_pandas()
    try:
        grid = set(pd.date_range(start=stamps.min(), end=stamps.max(), freq=alias))
    except (ValueError, TypeError):  # pragma: no cover - defensive
        return 0.0
    if not grid:
        return 0.0
    return float(sum(1 for s in stamps if s in grid) / len(stamps))


def classify_intermittency(values: np.ndarray) -> tuple[IntermittencyClass, float, float]:
    """Syntetos-Boylan classification (FR-108). Returns (class, ADI, CV-squared).

    ADI is the average interval between non-zero demands; CV² is the squared coefficient of
    variation of the non-zero demands themselves. Cut points 1.32 and 0.49.
    """
    nonzero = values[values != 0]
    if nonzero.size == 0:
        return "lumpy", float(values.size), 0.0

    adi = float(values.size / nonzero.size)
    mean = float(nonzero.mean())
    cv2 = float((nonzero.std() / mean) ** 2) if mean != 0 else 0.0

    if adi < _ADI_CUT:
        return ("erratic" if cv2 >= _CV2_CUT else "smooth"), adi, cv2
    return ("lumpy" if cv2 >= _CV2_CUT else "intermittent"), adi, cv2


def seasonality_strength(
    values: np.ndarray, season_length: int
) -> tuple[float | None, float | None]:
    """STL-based seasonality and trend strength (Wang/Smith/Hyndman).

    `1 - Var(remainder) / Var(remainder + component)`, clipped to [0, 1]. Returns
    `(None, None)` where the series is too short to decompose -- an honest absence rather
    than a fabricated zero, since `Diagnostics` shows these to the user.
    """
    if season_length < 2 or values.size < 2 * season_length:
        return None, None
    try:
        from statsmodels.tsa.seasonal import STL

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = STL(values, period=season_length, robust=True).fit()
    except (ValueError, ImportError):  # pragma: no cover - short/degenerate series
        return None, None

    remainder = np.asarray(result.resid, dtype=float)

    def strength(component: np.ndarray) -> float:
        denom = float(np.var(remainder + component))
        if denom <= 0:
            return 0.0
        return float(np.clip(1.0 - np.var(remainder) / denom, 0.0, 1.0))

    return strength(np.asarray(result.seasonal)), strength(np.asarray(result.trend))


def _series_profile(
    uid: str, filled: pl.DataFrame, prefill: pl.DataFrame, season_length: int
) -> SeriesProfile:
    values = filled.get_column(Y).to_numpy().astype(float)
    observed = values[~np.isnan(values)]
    # FR-106: classify on the pre-fill series, before zero-filling invents intermittency.
    raw = prefill.get_column(Y).to_numpy().astype(float)
    raw = raw[~np.isnan(raw)]
    klass, adi, cv2 = classify_intermittency(raw if raw.size else observed)
    seasonal, trend = seasonality_strength(np.nan_to_num(values), season_length)

    n_missing = int(np.isnan(values).sum())
    return SeriesProfile(
        unique_id=uid,
        n_obs=int(values.size),
        start=str(filled.get_column(DS).min()),
        end=str(filled.get_column(DS).max()),
        n_missing=n_missing,
        pct_missing=float(n_missing / values.size) if values.size else 0.0,
        zero_share=float((observed == 0).sum() / observed.size) if observed.size else 1.0,
        intermittency=klass,
        adi=adi,
        cv2=cv2,
        seasonality_strength=seasonal,
        trend_strength=trend,
        short_history=values.size < 2 * season_length,
    )


def profile_panel(
    panel: pl.DataFrame,
    *,
    data_id: str,
    mapping: DataMapping,
    freq: str | None = None,
    season_length: int | None = None,
    prefill: pl.DataFrame | None = None,
) -> DataProfile:
    """Derived statistics only -- this is what crosses the LLM trust boundary (NFR-07).

    `prefill` is the panel before gap filling, used for intermittency classification only.
    Defaults to `panel` when no filling was done.
    """
    prefill = panel if prefill is None else prefill
    inferred, confidence = infer_freq(panel)
    resolved_freq = normalise_freq(freq) if freq else inferred
    season = season_length or default_season_length(resolved_freq)

    prefill_by_id = {uid: frame for (uid,), frame in prefill.group_by([ID])}
    series = [
        _series_profile(uid, frame, prefill_by_id.get(uid, frame), season)
        for (uid,), frame in sorted(panel.group_by([ID]), key=lambda kv: kv[0])
    ]

    ends = panel.group_by(ID).agg(pl.col(DS).max()).get_column(DS)
    total = sum(s.n_obs for s in series)
    intermittent = sum(1 for s in series if s.intermittency in ("intermittent", "lumpy"))

    return DataProfile(
        data_id=data_id,
        n_series=len(series),
        n_rows=panel.height,
        freq_inferred=resolved_freq,
        freq_confidence=confidence,
        ds_min=str(panel.get_column(DS).min()),
        ds_max=str(panel.get_column(DS).max()),
        # Drives the FR-206a fold decision, and is the single most consequential fact about
        # a panel for leakage purposes.
        ragged=bool(ends.n_unique() > 1),
        season_length_candidates=[season],
        intermittent_share=float(intermittent / len(series)) if series else 0.0,
        pct_missing_overall=float(sum(s.n_missing for s in series) / total) if total else 0.0,
        exog_available=list(mapping.exog),
        series=series,
        validation=ValidationReport(n_series_in=len(series), n_series_out=len(series)),
    )
