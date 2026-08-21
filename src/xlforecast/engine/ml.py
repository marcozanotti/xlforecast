"""mlforecast adapter, parameterised by information set (FR-203a/b).

One code path produces both halves of a matched pair. `GlobalLGBM` fits once across the
panel; `LocalLGBM` fits once per series. Everything else -- learner class, feature recipe,
folds, horizon -- is held constant, which is what makes the pair a controlled comparison of
the information set rather than a comparison of two differently-specified models.

Not `statsforecast.SklearnModel`: that wrapper calls `model.fit(X, y)` with `X` = exogenous
columns only, performs no lag or date-feature engineering, and demands future exogenous
values at predict time. On a panel without exogenous columns its design matrix is empty.
"""

from __future__ import annotations

import warnings

import pandas as pd
import polars as pl

from xlforecast.engine.folds import Fold
from xlforecast.engine.registry import MLPlan, build_ml_plan
from xlforecast.engine.timing import measure
from xlforecast.panel import DS, ID, Y
from xlforecast.schemas.results import ModelTiming

__all__ = ["effective_train_rows", "forecast_fold", "forecast_full"]


def effective_train_rows(train: pd.DataFrame, max_lag: int) -> int:
    """Rows surviving `dropna=True` after lag construction (AC-206).

    Recorded because statsforecast trains on the full pre-cutoff history while mlforecast
    discards the first `max_lag` rows of every series. Identical cutoffs therefore still mean
    different training samples, and a local-vs-global comparison that does not surface this
    is silently confounded by it.
    """
    counts = train.groupby(ID, sort=False).size()
    return int((counts - max_lag).clip(lower=0).sum())


def _fit_predict(
    plan: MLPlan, train_pd: pd.DataFrame, *, h: int, freq: str
) -> tuple[pd.DataFrame, float, float, float, float]:
    from mlforecast import MLForecast

    mlf = MLForecast(
        models={plan.name: plan.estimator},
        freq=freq,
        lags=list(plan.recipe.lags),
        date_features=list(plan.recipe.date_features),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with measure() as train_t:
            mlf.fit(train_pd, static_features=[], dropna=True)
        with measure() as predict_t:
            preds = mlf.predict(h)
    return preds, train_t.cpu, train_t.wall, predict_t.cpu, predict_t.wall


def _run_one(
    plan: MLPlan, train_pd: pd.DataFrame, *, h: int, freq: str, fold_index: int | None
) -> tuple[pl.DataFrame, ModelTiming]:
    max_lag = max(plan.recipe.lags)
    frames: list[pd.DataFrame] = []
    train_cpu = train_wall = predict_cpu = predict_wall = 0.0
    fitted_series = 0

    if plan.information_set == "panel":
        preds, tc, tw, pc, pw = _fit_predict(plan, train_pd, h=h, freq=freq)
        frames.append(preds)
        train_cpu, train_wall, predict_cpu, predict_wall = tc, tw, pc, pw
        fitted_series = int(train_pd[ID].nunique())
    else:
        for uid, group in train_pd.groupby(ID, sort=True):
            # A series shorter than the longest lag yields no training rows at all. Skip it
            # rather than crash; FR-215's common-support rule then keeps the panel aggregate
            # honest about which series each model was actually scored on.
            if len(group) <= max_lag:
                continue
            preds, tc, tw, pc, pw = _fit_predict(plan, group.reset_index(drop=True), h=h, freq=freq)
            frames.append(preds)
            train_cpu += tc
            train_wall += tw
            predict_cpu += pc
            predict_wall += pw
            fitted_series += 1
            del uid

    if frames:
        combined = pd.concat(frames, ignore_index=True)
        out = (
            pl.from_pandas(combined[[ID, DS, plan.name]])
            .rename({plan.name: "y_hat"})
            .with_columns(
                pl.lit(plan.name).alias("model"),
                pl.col("y_hat").cast(pl.Float64),
                pl.col(DS).cast(pl.Datetime("us")),
            )
            .select([ID, DS, "model", "y_hat"])
        )
    else:
        out = pl.DataFrame(
            schema={ID: pl.Utf8, DS: pl.Datetime("us"), "model": pl.Utf8, "y_hat": pl.Float64}
        )

    timing = ModelTiming(
        model=plan.name,
        fold_index=fold_index,
        train_cpu_seconds=train_cpu,
        predict_cpu_seconds=predict_cpu,
        train_wall_seconds=train_wall,
        predict_wall_seconds=predict_wall,
        n_series_fitted=fitted_series,
        n_rows_trained=effective_train_rows(train_pd, max_lag),
    )
    return out, timing


def _run(
    names: list[str],
    train: pl.DataFrame,
    *,
    h: int,
    freq: str,
    season_length: int,
    seed: int,
    fold_index: int | None,
) -> tuple[pl.DataFrame, list[ModelTiming]]:
    train_pd = train.select([ID, DS, Y]).to_pandas()
    frames, timings = [], []
    for name in names:
        plan = build_ml_plan(name, freq=freq, season_length=season_length, seed=seed)
        preds, timing = _run_one(plan, train_pd, h=h, freq=freq, fold_index=fold_index)
        frames.append(preds)
        timings.append(timing)
    combined = (
        pl.concat(frames)
        if frames
        else pl.DataFrame(
            schema={ID: pl.Utf8, DS: pl.Datetime("us"), "model": pl.Utf8, "y_hat": pl.Float64}
        )
    )
    return combined, timings


def forecast_fold(
    names: list[str], fold: Fold, *, h: int, freq: str, season_length: int, seed: int
) -> tuple[pl.DataFrame, list[ModelTiming]]:
    return _run(
        names,
        fold.train,
        h=h,
        freq=freq,
        season_length=season_length,
        seed=seed,
        fold_index=fold.index,
    )


def forecast_full(
    names: list[str], panel: pl.DataFrame, *, h: int, freq: str, season_length: int, seed: int
) -> tuple[pl.DataFrame, list[ModelTiming]]:
    return _run(
        names, panel, h=h, freq=freq, season_length=season_length, seed=seed, fold_index=None
    )
