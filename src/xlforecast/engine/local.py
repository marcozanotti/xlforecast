"""statsforecast adapter, driven by our own fold loop (TS §5.1, FR-206).

Deliberately does not call `StatsForecast.cross_validation`: it accepts no cutoff array and
derives per-series cutoffs internally, which leaks panel futures into global models on ragged
panels. It receives pre-sliced `Fold` objects instead.

One `StatsForecast` instance per model, rather than one instance holding all of them. That
costs an extra pass over the grouped data per model, but FR-217 requires `train` and
`predict` reported *per model*, and a single instance fits them together and can only be
timed in aggregate. It also keeps Fourier regressors confined to AutoARIMA, since models
that do not support exogenous inputs would otherwise be handed them.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import polars as pl

from xlforecast.engine.folds import Fold
from xlforecast.engine.registry import build_local
from xlforecast.engine.timing import measure
from xlforecast.panel import DS, ID, Y
from xlforecast.schemas.results import ModelTiming

__all__ = ["FOURIER_HARMONICS", "forecast_fold", "forecast_full", "fourier_terms"]

FOURIER_HARMONICS = 3


def fourier_terms(
    dates: pd.Series, *, period: int, origin: pd.Timestamp, freq: str
) -> pd.DataFrame:
    """Fourier regressors for FR-201a's non-seasonal AutoARIMA mode.

    Positions are measured from a fixed `origin` on the panel calendar rather than from each
    series' own start, so a given calendar date gets the same regressor values in every fold
    and every series. Anything else would make the design matrix fold-dependent and break
    NFR-02.
    """
    step = pd.tseries.frequencies.to_offset(freq)
    try:
        # Fast path for fixed-span offsets (hourly, daily, minutely).
        span = pd.Timedelta(step.nanos, "ns")
        positions = np.array([(d - origin) / span for d in dates], dtype=float)
    except ValueError:
        # Anchored and calendar offsets -- W-SUN, ME, QE, YE -- have no fixed nanosecond
        # span, so position is resolved against the calendar grid instead.
        grid = pd.date_range(start=origin, end=max(dates), freq=freq)
        lookup = {d: i for i, d in enumerate(grid)}
        positions = np.array([lookup.get(d, np.nan) for d in dates], dtype=float)

    out: dict[str, np.ndarray] = {}
    for k in range(1, FOURIER_HARMONICS + 1):
        angle = 2 * np.pi * k * positions / period
        out[f"fourier_sin{k}"] = np.sin(angle)
        out[f"fourier_cos{k}"] = np.cos(angle)
    return pd.DataFrame(out, index=dates.index)


def _needs_fourier(name: str, season_length: int) -> bool:
    return name == "AutoARIMA" and season_length > 24


def _future_frame(train: pd.DataFrame, *, h: int, freq: str) -> pd.DataFrame:
    """The `h` future timestamps per series, which exogenous prediction requires."""
    rows = []
    for uid, group in train.groupby(ID, sort=True):
        future = pd.date_range(start=group[DS].max(), periods=h + 1, freq=freq)[1:]
        rows.append(pd.DataFrame({ID: uid, DS: future}))
    return pd.concat(rows, ignore_index=True)


def _run_one(
    name: str,
    train_pd: pd.DataFrame,
    *,
    h: int,
    freq: str,
    season_length: int,
    origin: pd.Timestamp,
    fold_index: int | None,
    n_jobs: int = 1,
) -> tuple[pl.DataFrame, ModelTiming]:
    from statsforecast import StatsForecast

    model = build_local(name, season_length=season_length)
    frame = train_pd
    x_future: pd.DataFrame | None = None

    if _needs_fourier(name, season_length):
        frame = train_pd.copy()
        frame = pd.concat(
            [frame, fourier_terms(frame[DS], period=season_length, origin=origin, freq=freq)],
            axis=1,
        )
        x_future = _future_frame(train_pd, h=h, freq=freq)
        x_future = pd.concat(
            [x_future, fourier_terms(x_future[DS], period=season_length, origin=origin, freq=freq)],
            axis=1,
        )

    # FR-211. statsforecast parallelises across *series*, and each series is fitted
    # independently, so forecasts are bit-identical across worker counts -- verified in
    # tests/unit/engine/test_parallel.py. Speed here costs no accuracy, unlike the
    # approximation lever measured in Phase 3 (2.95x faster, 3.3% worse MASE).
    sf = StatsForecast(models=[model], freq=freq, n_jobs=n_jobs)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with measure() as train_t:
            sf.fit(frame)
        with measure() as predict_t:
            raw = sf.predict(h=h, X_df=x_future) if x_future is not None else sf.predict(h=h)

    assert isinstance(raw, pd.DataFrame)  # narrows the library's union return type
    column = next(c for c in raw.columns if c not in (ID, DS))
    preds = (
        pl.from_pandas(raw[[ID, DS, column]])
        .rename({column: "y_hat"})
        .with_columns(
            pl.lit(name).alias("model"),
            # Normalised at the adapter boundary. XGBoost predicts float32 where the rest
            # predict float64, and mixing them fails concat outright; dtype also changes
            # serialised bytes, so leaving it to chance would put NFR-02 byte-identity at
            # the mercy of which models happened to be selected.
            pl.col("y_hat").cast(pl.Float64),
            pl.col(DS).cast(pl.Datetime("us")),
        )
    )
    timing = ModelTiming(
        model=name,
        fold_index=fold_index,
        train_cpu_seconds=train_t.cpu,
        predict_cpu_seconds=predict_t.cpu,
        train_wall_seconds=train_t.wall,
        predict_wall_seconds=predict_t.wall,
        n_series_fitted=int(train_pd[ID].nunique()),
        n_rows_trained=len(train_pd),
    )
    return preds.select([ID, DS, "model", "y_hat"]), timing


def forecast_fold(
    names: list[str],
    fold: Fold,
    *,
    h: int,
    freq: str,
    season_length: int,
    origin: pd.Timestamp,
    n_jobs: int = 1,
) -> tuple[pl.DataFrame, list[ModelTiming]]:
    """Fit on the fold's training slice and predict its horizon."""
    train_pd = fold.train.select([ID, DS, Y]).to_pandas()  # convert once, at the boundary
    frames, timings = [], []
    for name in names:
        preds, timing = _run_one(
            name,
            train_pd,
            h=h,
            freq=freq,
            season_length=season_length,
            origin=origin,
            fold_index=fold.index,
            n_jobs=n_jobs,
        )
        frames.append(preds)
        timings.append(timing)
    return pl.concat(frames) if frames else _empty(), timings


def forecast_full(
    names: list[str],
    panel: pl.DataFrame,
    *,
    h: int,
    freq: str,
    season_length: int,
    origin: pd.Timestamp,
    n_jobs: int = 1,
) -> tuple[pl.DataFrame, list[ModelTiming]]:
    """Refit on full history for the delivered forecast (`fold_index=None`)."""
    train_pd = panel.select([ID, DS, Y]).to_pandas()
    frames, timings = [], []
    for name in names:
        preds, timing = _run_one(
            name,
            train_pd,
            h=h,
            freq=freq,
            season_length=season_length,
            origin=origin,
            fold_index=None,
            n_jobs=n_jobs,
        )
        frames.append(preds)
        timings.append(timing)
    return pl.concat(frames) if frames else _empty(), timings


def _empty() -> pl.DataFrame:
    return pl.DataFrame(
        schema={ID: pl.Utf8, DS: pl.Datetime("us"), "model": pl.Utf8, "y_hat": pl.Float64}
    )
