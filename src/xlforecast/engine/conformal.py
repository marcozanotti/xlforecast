"""Conformal calibration (TS §5.4, ADR-006 as amended, FR-301/302/303/307).

**Cross-conformal, not split-conformal.** The band used when scoring fold *k* is calibrated
only from the residuals of folds ≠ *k*. The band delivered with the final forecast uses all
folds.

This is the correction the spec review forced. Calibrating on the CV residuals and then
measuring coverage against those same residuals is circular: an empirical quantile covers its
own calibration sample at the nominal rate *by construction*, so the reported figure would
have been ≈80% for an 80% band no matter how badly the intervals were built. AC-301, success
criterion #4 and gate G2 would all have passed against a broken implementation. Cross-conformal
costs no extra model fits -- only repeated quantile computation over residuals that already
exist -- which is why `n_windows` stays at its default of 3.

**Bands are clipped to the series' observed support** (FR-307). Measured on an intermittent
panel: the unclipped lower bound sits below zero at 95.8% of points, and clipping removes
21.6% of the interval width.

Note what clipping does *not* do, corrected here after measurement: it does not change
coverage, and cannot. No observation of a non-negative series lies below zero, so truncating
the lower bound at zero never excludes a point that was previously inside. The original FR-307
rationale claimed clipping was needed because coverage "approaches 100%"; measured coverage is
80.7% clipped and 80.7% unclipped. Clipping buys sharpness and interpretability -- a negative
lower bound on unit demand is not a wider forecast, it is a nonsensical one.

The pathology that *is* real on those series is **one-sided miscoverage**: with symmetric
additive bands, 0.00% of violations fall in the lower tail and 15.62% in the upper, against a
roughly balanced 10.2/5.5 on Gaussian data. The interval's lower half does no work. That is
what `tail_miscoverage` reports, and it is the honest evidence that a symmetric additive band
is the wrong object for count data -- not the coverage number, which looks fine either way.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import polars as pl

from xlforecast.engine.folds import Fold
from xlforecast.panel import DS, ID, Y
from xlforecast.schemas.results import ConformalBands

__all__ = [
    "DEFAULT_MIN_RESIDUALS",
    "Support",
    "apply_bands",
    "calibrate",
    "collect_residuals",
    "conformal_quantile",
    "coverage",
    "interval_width",
    "series_support",
    "tail_miscoverage",
]

DEFAULT_MIN_RESIDUALS = 20
#: A series has `n_windows * h` residuals, so per-series calibration engages only when that
#: product clears `DEFAULT_MIN_RESIDUALS`. At the NFR-01 defaults (3 windows, h=13) it does,
#: at 39; at h=6 it does not, and every series falls back to the pooled panel residuals.
#:
#: That matters beyond band width. Cross-conformal needs *more* residuals than in-calibration,
#: because each fold's band is built from a strictly smaller pool -- so `min_residuals` bites
#: harder here than the number suggests. It also blunts AC-301's control: dropping one fold
#: from a large pooled set barely moves the quantile, so the honest and in-calibration figures
#: converge (measured gap +0.090 under per-series calibration, +0.007 under pooled). The
#: control is still directionally correct, but a reader should know which regime produced it.
#: `CalibrationRow.n_pooled_fallback` reports it.

#: unique_id -> (lower, upper) bound implied by the observed history.
Support = dict[str, tuple[float, float]]


def series_support(panel: pl.DataFrame) -> Support:
    """Per-series bounds for FR-307 clipping.

    A series that has never been negative is treated as non-negative, which is the honest
    reading of demand data: a lower bound below zero is not a wider interval, it is a
    meaningless one.
    """
    stats = panel.group_by(ID).agg(pl.col(Y).min().alias("lo"))
    return {
        row[ID]: (0.0 if row["lo"] is not None and row["lo"] >= 0 else -math.inf, math.inf)
        for row in stats.iter_rows(named=True)
    }


def collect_residuals(folds: list[Fold], predictions: dict[int, pl.DataFrame]) -> pl.DataFrame:
    """Join per-fold predictions to actuals and compute absolute residuals.

    `horizon_step` is retained so that coverage can be reported by step -- forecast
    uncertainty grows with the horizon, and a band that is well calibrated on average may
    still be too wide at step 1 and too narrow at step h.
    """
    frames = []
    for fold in folds:
        preds = predictions.get(fold.index)
        if preds is None or preds.is_empty():
            continue
        actual = fold.test.select([ID, DS, Y])
        joined = preds.join(actual, on=[ID, DS], how="inner")
        if joined.is_empty():
            continue
        ranked = joined.with_columns(
            pl.col(DS).rank("dense").over([ID, "model"]).cast(pl.Int32).alias("horizon_step"),
            pl.lit(fold.index).alias("fold_index"),
            (pl.col(Y) - pl.col("y_hat")).abs().alias("abs_residual"),
        )
        frames.append(
            ranked.select(
                [ID, DS, "model", "fold_index", "horizon_step", Y, "y_hat", "abs_residual"]
            )
        )
    if not frames:
        return pl.DataFrame(
            schema={
                ID: pl.Utf8,
                DS: pl.Datetime("us"),
                "model": pl.Utf8,
                "fold_index": pl.Int32,
                "horizon_step": pl.Int32,
                Y: pl.Float64,
                "y_hat": pl.Float64,
                "abs_residual": pl.Float64,
            }
        )
    return pl.concat(frames)


def conformal_quantile(values: np.ndarray, level: int) -> float:
    """The finite-sample-corrected conformal quantile of absolute residuals.

    Uses `ceil((n+1)(1-alpha)) / n` rather than the plain empirical quantile. With the small
    residual counts a 3-window CV produces, the correction is the difference between roughly
    nominal coverage and systematic under-coverage -- it is what makes the guarantee hold at
    finite `n` rather than only asymptotically.
    """
    clean = values[np.isfinite(values)]
    n = clean.size
    if n == 0:
        return math.nan
    rank = math.ceil((n + 1) * level / 100.0)
    if rank > n:
        # Too few residuals to certify this level at all: the honest answer is the widest
        # observed residual, not a number interpolated beyond the data.
        return float(np.max(clean))
    return float(np.sort(clean)[rank - 1])


@dataclass(frozen=True, slots=True)
class _Pool:
    per_series: dict[str, np.ndarray]
    pooled: np.ndarray


def _residual_pool(residuals: pl.DataFrame, *, model: str, exclude_folds: frozenset[int]) -> _Pool:
    subset = residuals.filter(pl.col("model") == model)
    if exclude_folds:
        subset = subset.filter(~pl.col("fold_index").is_in(list(exclude_folds)))
    per_series = {
        uid: frame.get_column("abs_residual").to_numpy().astype(float)
        for (uid,), frame in subset.group_by([ID])
    }
    pooled = subset.get_column("abs_residual").to_numpy().astype(float)
    return _Pool(per_series=per_series, pooled=pooled)


def calibrate(
    residuals: pl.DataFrame,
    *,
    model: str,
    level: int,
    exclude_folds: frozenset[int] = frozenset(),
    min_residuals: int = DEFAULT_MIN_RESIDUALS,
    all_folds: frozenset[int] | None = None,
) -> ConformalBands:
    """Half-widths for one model at one level.

    `exclude_folds` is what makes this cross-conformal: pass `{k}` to build the band used
    when scoring fold `k`, or nothing to build the delivered band from every fold.

    Fallback chain, fully defined (the original spec stopped one step short):
      1. the series' own residuals, when it has at least `min_residuals`;
      2. otherwise the pooled panel residuals, recorded per series in diagnostics;
      3. otherwise `NaN`, meaning the level is unavailable for that series -- never a
         silently wrong number.
    """
    pool = _residual_pool(residuals, model=model, exclude_folds=exclude_folds)
    calibrated_from = sorted((all_folds or frozenset()) - exclude_folds)

    half_width: dict[str, float] = {}
    fallback: set[str] = set()
    for uid, values in pool.per_series.items():
        usable = values[np.isfinite(values)]
        if usable.size >= min_residuals:
            half_width[uid] = conformal_quantile(usable, level)
        elif pool.pooled[np.isfinite(pool.pooled)].size >= min_residuals:
            half_width[uid] = conformal_quantile(pool.pooled, level)
            fallback.add(uid)
        else:
            half_width[uid] = math.nan
            fallback.add(uid)

    return ConformalBands(
        model=model,
        level=level,
        half_width=half_width,
        pooled_fallback=fallback,
        clipped={},
        calibrated_from_folds=calibrated_from,
    )


def apply_bands(
    frame: pl.DataFrame, bands: ConformalBands, support: Support
) -> tuple[pl.DataFrame, dict[str, float]]:
    """Attach `lo`/`hi` columns to a frame carrying `unique_id` and `y_hat`.

    Returns the frame and the per-series clip rate, which FR-307 requires in diagnostics:
    a series whose lower bound is clipped often has an interval that is not symmetric in
    practice, and its coverage is not comparable with an unclipped series'.
    """
    widths = pl.DataFrame(
        {ID: list(bands.half_width), "half_width": list(bands.half_width.values())},
        schema={ID: pl.Utf8, "half_width": pl.Float64},
    )
    lower = pl.DataFrame(
        {ID: list(support), "support_lo": [v[0] for v in support.values()]},
        schema={ID: pl.Utf8, "support_lo": pl.Float64},
    )
    joined = frame.join(widths, on=ID, how="left").join(lower, on=ID, how="left")
    out = joined.with_columns(
        (pl.col("y_hat") - pl.col("half_width")).alias("_raw_lo"),
        (pl.col("y_hat") + pl.col("half_width")).alias("hi"),
    ).with_columns(
        pl.max_horizontal("_raw_lo", pl.col("support_lo").fill_null(-math.inf)).alias("lo")
    )

    clip_rates = {
        row[ID]: float(row["rate"])
        for row in out.group_by(ID)
        .agg((pl.col("lo") > pl.col("_raw_lo")).mean().alias("rate"))
        .iter_rows(named=True)
    }
    return out.drop(["_raw_lo", "support_lo"]), clip_rates


def coverage(
    residuals: pl.DataFrame,
    *,
    model: str,
    level: int,
    support: Support,
    min_residuals: int = DEFAULT_MIN_RESIDUALS,
    in_calibration: bool = False,
) -> float:
    """Empirical coverage (FR-303).

    With `in_calibration=False` (the default and the only figure fit to show a user), fold
    `k` is scored against a band built from folds ≠ `k`.

    With `in_calibration=True` every fold is scored against a band built from all folds,
    including itself. That number is ≈nominal by construction and exists solely as the
    control in AC-301: it must come out measurably tighter to nominal than the honest one,
    which is what proves the honest one is not a tautology. It must never reach the UI.
    """
    subset = residuals.filter(pl.col("model") == model)
    if subset.is_empty():
        return math.nan

    folds = frozenset(subset.get_column("fold_index").unique().to_list())
    covered: list[float] = []
    for fold_index in sorted(folds):
        exclude = frozenset() if in_calibration else frozenset({fold_index})
        bands = calibrate(
            residuals,
            model=model,
            level=level,
            exclude_folds=exclude,
            min_residuals=min_residuals,
            all_folds=folds,
        )
        scored, _ = apply_bands(subset.filter(pl.col("fold_index") == fold_index), bands, support)
        hits = scored.filter(pl.col("half_width").is_not_nan()).select(
            ((pl.col(Y) >= pl.col("lo")) & (pl.col(Y) <= pl.col("hi"))).alias("hit")
        )
        if not hits.is_empty():
            covered.extend(hits.get_column("hit").cast(pl.Float64).to_list())
    return float(np.mean(covered)) if covered else math.nan


def _scored_with_bands(
    residuals: pl.DataFrame, *, model: str, level: int, support: Support, min_residuals: int
) -> pl.DataFrame:
    """Every fold scored against a band built from the other folds (FR-302)."""
    subset = residuals.filter(pl.col("model") == model)
    if subset.is_empty():
        return subset
    folds = frozenset(subset.get_column("fold_index").unique().to_list())
    parts = []
    for fold_index in sorted(folds):
        bands = calibrate(
            residuals,
            model=model,
            level=level,
            exclude_folds=frozenset({fold_index}),
            min_residuals=min_residuals,
            all_folds=folds,
        )
        scored, _ = apply_bands(subset.filter(pl.col("fold_index") == fold_index), bands, support)
        parts.append(scored)
    return pl.concat(parts).filter(pl.col("half_width").is_not_nan())


def tail_miscoverage(
    residuals: pl.DataFrame,
    *,
    model: str,
    level: int,
    support: Support,
    min_residuals: int = DEFAULT_MIN_RESIDUALS,
) -> tuple[float, float]:
    """Share of observations falling below `lo` and above `hi` (FR-307a).

    A well-behaved interval splits its miscoverage roughly evenly between the tails. A
    symmetric additive band on non-negative count data does not: essentially every violation
    is upper-tail, because the lower bound sits below the support and can never be crossed.

    This is the diagnostic that detects the problem FR-307 exists to address. Coverage does
    not: it reads ~nominal whether or not the lower half of the interval is doing any work.
    """
    scored = _scored_with_bands(
        residuals, model=model, level=level, support=support, min_residuals=min_residuals
    )
    if scored.is_empty():
        return math.nan, math.nan
    below = float(scored.select((pl.col(Y) < pl.col("lo")).mean()).item())
    above = float(scored.select((pl.col(Y) > pl.col("hi")).mean()).item())
    return below, above


def interval_width(
    residuals: pl.DataFrame,
    *,
    model: str,
    level: int,
    support: Support,
    min_residuals: int = DEFAULT_MIN_RESIDUALS,
) -> float:
    """Mean interval width -- the sharpness that clipping actually improves."""
    scored = _scored_with_bands(
        residuals, model=model, level=level, support=support, min_residuals=min_residuals
    )
    if scored.is_empty():
        return math.nan
    return float(scored.select((pl.col("hi") - pl.col("lo")).mean()).item())


def quantile_levels(levels: list[int]) -> np.ndarray:
    """The quantile grid implied by a set of symmetric interval levels (FR-208).

    Level 80 contributes the 0.10 and 0.90 quantiles, level 95 the 0.025 and 0.975; the
    median is always present because it is the point forecast. So `[80, 95]` gives a
    five-point grid.

    The grid goes in the manifest, because `scaled_crps` is a quantile-grid approximation:
    a run at `levels=[80, 95]` and one at `[50, 80, 95]` produce CRPS values that are **not
    comparable**, and a leaderboard that hides that is quietly dishonest.
    """
    qs = {0.5}
    for level in levels:
        alpha = 1.0 - level / 100.0
        qs.add(round(alpha / 2.0, 6))
        qs.add(round(1.0 - alpha / 2.0, 6))
    return np.array(sorted(qs), dtype=float)


def quantile_frame(
    residuals: pl.DataFrame,
    *,
    model: str,
    levels: list[int],
    support: Support,
    min_residuals: int = DEFAULT_MIN_RESIDUALS,
    cross_conformal: bool = True,
) -> tuple[pl.DataFrame, list[str]]:
    """Per-point quantile values for one model, for probabilistic scoring.

    With `cross_conformal=True` each fold is scored against a band built from the other
    folds (FR-302), so the CRPS a model earns is subject to the same held-out discipline as
    its coverage. Returns the frame and the ordered quantile column names.
    """
    grid = quantile_levels(levels)
    columns = [f"q{q:.4f}" for q in grid]
    subset = residuals.filter(pl.col("model") == model)
    if subset.is_empty():
        return subset, columns

    folds = frozenset(subset.get_column("fold_index").unique().to_list())
    parts = []
    for fold_index in sorted(folds):
        exclude = frozenset({fold_index}) if cross_conformal else frozenset()
        slice_ = subset.filter(pl.col("fold_index") == fold_index)
        widths: dict[int, ConformalBands] = {
            level: calibrate(
                residuals,
                model=model,
                level=level,
                exclude_folds=exclude,
                min_residuals=min_residuals,
                all_folds=folds,
            )
            for level in levels
        }
        frame = slice_.select([ID, DS, Y, "y_hat", "fold_index", "horizon_step"])
        lower = pl.DataFrame(
            {ID: list(support), "support_lo": [v[0] for v in support.values()]},
            schema={ID: pl.Utf8, "support_lo": pl.Float64},
        )
        frame = frame.join(lower, on=ID, how="left")
        exprs = []
        for q, name in zip(grid, columns, strict=True):
            if q == 0.5:
                exprs.append(pl.col("y_hat").alias(name))
                continue
            level = round((1 - 2 * min(q, 1 - q)) * 100)
            band = widths[level]
            hw = pl.col(ID).replace_strict(band.half_width, default=None).cast(pl.Float64)
            offset = pl.col("y_hat") + (hw if q > 0.5 else -hw)
            exprs.append(
                (
                    offset
                    if q > 0.5
                    else pl.max_horizontal(offset, pl.col("support_lo").fill_null(-math.inf))
                ).alias(name)
            )
        parts.append(frame.with_columns(exprs).drop("support_lo"))
    return pl.concat(parts), columns
