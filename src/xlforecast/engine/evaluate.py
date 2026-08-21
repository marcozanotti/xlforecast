"""Metric assembly over utilsforecast (TS §5.6, FR-208/209/214/215).

Three things the original specification assumed and the library does not provide:

1. There is no `crps` in `utilsforecast.losses`. Probabilistic scoring arrives in Phase 2 and
   will use `scaled_crps`, whose quantile grid is recorded in the manifest because two runs
   with different `levels` produce values that are not comparable.
2. `mase`/`rmsse` divide by a training-window quantity that is **zero** for a series constant
   within an early fold; `smape` divides by `|y| + |y_hat|`, zero on an all-zero window. These
   are routine on intermittent SKU panels, not exotic. FR-214 is the policy.
3. Aggregation must be nan-safe, state itself, and report the support it was computed over,
   or a metric averaged over 240 series is silently compared with one averaged over 288.
"""

from __future__ import annotations

import math
import warnings
from functools import partial

import numpy as np
import polars as pl

from xlforecast.engine.folds import Fold
from xlforecast.panel import DS, ID, Y
from xlforecast.schemas.enums import Family, InformationSet
from xlforecast.schemas.registry import MODEL_REGISTRY
from xlforecast.schemas.results import FoldScore, Leaderboard, LeaderboardRow

__all__ = [
    "ALL_METRICS",
    "METRICS",
    "PROB_METRIC",
    "build_leaderboard",
    "score_fold",
    "score_probabilistic",
]

METRICS = ("mase", "rmsse", "mae", "rmse", "smape")
#: Probabilistic metric, added once conformal bands exist (Phase 2). Aggregated alongside
#: the point metrics but kept separate in name, because it is absent when `conformal=False`
#: and because its value depends on the quantile grid recorded in the manifest.
PROB_METRIC = "scaled_crps"
ALL_METRICS = (*METRICS, PROB_METRIC)

BASELINE = "SeasonalNaive"
INCUMBENT = "WindowAverage"


def _clean(value: float | None) -> float | None:
    """FR-214: a degenerate metric is `None`, never `NaN` or `inf`.

    `NaN` is not merely untidy here -- it propagates through a naive `mean()` and takes an
    entire leaderboard row with it, and it cannot survive a JSON round trip into a bare
    float, so it would corrupt a manifest replay silently.
    """
    if value is None:
        return None
    v = float(value)
    return None if (math.isnan(v) or math.isinf(v)) else v


def score_fold(
    fold: Fold, predictions: pl.DataFrame, *, models: list[str], season_length: int
) -> list[FoldScore]:
    """Score one fold, per model per series.

    The MASE/RMSSE scale comes from *this fold's* training window, which is the correct
    definition under cross-validation: using the full history would leak later observations
    into the denominator of an earlier fold's score.
    """
    from utilsforecast.evaluation import evaluate
    from utilsforecast.losses import mae, mase, rmse, rmsse, smape

    wide = (
        predictions.filter(pl.col("model").is_in(models))
        .pivot(on="model", index=[ID, DS], values="y_hat")
        .join(fold.test.select([ID, DS, Y]), on=[ID, DS], how="inner")
    )
    present = [m for m in models if m in wide.columns]
    if wide.is_empty() or not present:
        return []

    frame = wide.select([ID, DS, Y, *present]).to_pandas()
    train_pd = fold.train.select([ID, DS, Y]).to_pandas()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        table = evaluate(
            frame,
            metrics=[
                partial(mase, seasonality=season_length),
                partial(rmsse, seasonality=season_length),
                mae,
                rmse,
                smape,
            ],
            models=present,
            train_df=train_pd,
        )

    rows: list[FoldScore] = []
    for uid, group in table.groupby(ID, sort=True):
        by_metric = group.set_index("metric")
        for model in present:
            rows.append(
                FoldScore(
                    fold_index=fold.index,
                    cutoff=str(fold.cutoff),
                    model=model,
                    unique_id=str(uid),
                    n_train_rows=len(train_pd[train_pd[ID] == uid]),
                    metrics={m: _clean(by_metric.loc[m, model]) for m in METRICS},
                )
            )
    return rows


def _mean(values: list[float | None]) -> float | None:
    """Nan-safe mean over a series' folds. All-`None` stays `None` rather than becoming 0."""
    present = [v for v in values if v is not None]
    return sum(present) / len(present) if present else None


def _family(model: str) -> tuple[Family, InformationSet]:
    spec = MODEL_REGISTRY.get(model)
    if spec is None:  # ensembles are registered later (Phase 2)
        return "ensemble", "panel"
    return spec.family, spec.information_set


def build_leaderboard(
    fold_scores: list[FoldScore], *, models: list[str], rank_metric: str = "mase"
) -> Leaderboard:
    """Aggregate fold scores into per-series and panel rows (FR-209, FR-215).

    The panel aggregate is computed over the **common support** -- the series every ranked
    model actually scored -- so a model that declined 48 short series cannot appear to beat
    one that scored all 288 by having been graded on an easier subset.
    """
    # (model, series) -> metric -> values across folds
    collected: dict[tuple[str, str], dict[str, list[float | None]]] = {}
    n_folds: dict[str, set[int]] = {}
    for score in fold_scores:
        if score.unique_id is None:
            continue
        bucket = collected.setdefault((score.model, score.unique_id), {m: [] for m in ALL_METRICS})
        for metric in ALL_METRICS:
            bucket[metric].append(score.metrics.get(metric))
        n_folds.setdefault(score.model, set()).add(score.fold_index)

    per_series: dict[tuple[str, str], dict[str, float | None]] = {
        key: {m: _mean(vals[m]) for m in ALL_METRICS} for key, vals in collected.items()
    }

    scored_by_model: dict[str, set[str]] = {}
    for model, uid in per_series:
        scored_by_model.setdefault(model, set()).add(uid)

    ranked = [m for m in models if m in scored_by_model]
    common = set.intersection(*(scored_by_model[m] for m in ranked)) if ranked else set()

    def panel_metric(model: str, metric: str) -> float | None:
        return _mean(
            [
                per_series[(model, uid)][metric]
                for uid in sorted(common)
                if (model, uid) in per_series
            ]
        )

    panel_values = {m: {k: panel_metric(m, k) for k in ALL_METRICS} for m in ranked}

    def sort_key(model: str) -> float:
        value = panel_values[model].get(rank_metric)
        return math.inf if value is None else value

    order = sorted(ranked, key=sort_key)
    baseline = panel_values.get(BASELINE, {}).get(rank_metric)
    incumbent = panel_values.get(INCUMBENT, {}).get(rank_metric)

    def relative(value: float | None, reference: float | None) -> float | None:
        if value is None or reference in (None, 0):
            return None
        return (value - reference) / reference * 100.0

    rows: list[LeaderboardRow] = []
    for rank, model in enumerate(order, start=1):
        family, info = _family(model)
        values = panel_values[model]
        rows.append(
            LeaderboardRow(
                scope="panel",
                model=model,
                family=family,
                information_set=info,
                n_folds=len(n_folds.get(model, ())),
                n_series_scored=len(scored_by_model[model]),
                n_series_common=len(common),
                rank=rank,
                vs_baseline_pct=relative(values.get(rank_metric), baseline),
                vs_incumbent_pct=relative(values.get(rank_metric), incumbent),
                mase=values["mase"],
                rmsse=values["rmsse"],
                mae=values["mae"],
                rmse=values["rmse"],
                smape=values["smape"],
                scaled_crps=values[PROB_METRIC],
            )
        )
        for uid in sorted(scored_by_model[model]):
            series_values = per_series[(model, uid)]
            rows.append(
                LeaderboardRow(
                    scope="series",
                    unique_id=uid,
                    model=model,
                    family=family,
                    information_set=info,
                    n_folds=len(n_folds.get(model, ())),
                    n_series_scored=1,
                    n_series_common=len(common),
                    rank=rank,
                    mase=series_values["mase"],
                    rmsse=series_values["rmsse"],
                    mae=series_values["mae"],
                    rmse=series_values["rmse"],
                    smape=series_values["smape"],
                    scaled_crps=series_values[PROB_METRIC],
                )
            )

    # FR-406: does anything actually beat the mandated baseline? A model with no score does
    # not count as beating it, and neither does anything when the baseline itself is
    # unscored -- in both cases the honest answer is "we cannot say", which is `False`.
    def beats_baseline(model: str) -> bool:
        if model == BASELINE or baseline is None:
            return False
        score = panel_values[model].get(rank_metric)
        return score is not None and score < baseline

    beat = any(beats_baseline(m) for m in ranked)
    return Leaderboard(rows=rows, aggregation="mean", any_beat_baseline=beat)


def score_probabilistic(
    quantiles: pl.DataFrame, *, columns: list[str], grid: np.ndarray
) -> dict[tuple[int, str], float | None]:
    """Scaled CRPS per (fold, series) from a model's quantile frame (FR-208).

    `utilsforecast` has no plain `crps`; what exists is `scaled_crps`, a quantile-grid
    approximation normalised by the sum of actuals. Two consequences the leaderboard has to
    carry honestly:

    * the value depends on the grid, so `Manifest.crps_quantiles` records it — a run at
      `levels=[80, 95]` is not CRPS-comparable with one at `[50, 80, 95]`;
    * the normaliser is **zero** on an all-zero evaluation window, routine on intermittent
      panels, so FR-214's `None` policy applies here exactly as it does to MASE.
    """
    from utilsforecast.losses import scaled_crps

    if quantiles.is_empty():
        return {}

    out: dict[tuple[int, str], float | None] = {}
    for (fold_index,), fold_frame in quantiles.group_by(["fold_index"]):
        frame = fold_frame.select([ID, DS, Y, *columns]).to_pandas()
        frame["cutoff"] = 0  # single window per call; grouping is done by us, not the loss
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                table = scaled_crps(frame, models={"m": columns}, quantiles=grid)
            except (ValueError, ZeroDivisionError, KeyError):  # pragma: no cover - defensive
                continue
        for row in table.to_dict("records"):
            out[(int(fold_index), str(row[ID]))] = _clean(row["m"])
    return out
