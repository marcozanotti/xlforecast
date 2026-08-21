"""Model selection (FR-401/402/408).

`pooled` is the default because `per_series` argmin over ~13 models on 3 folds overfits the
CV badly. FR-402 requires a warning below 5 windows; FR-408 goes further and requires the
bias to be *reported*, not merely warned about.

**The winner's curse is reported, not just flagged.** When selection is per-series, the
selected model's own CV score is an argmin over models on the very folds it is scored on, so
it is optimistically biased. Alongside it we compute a leave-one-fold-out estimate: select on
folds ≠ *k*, score on fold *k*, average. That is nearly free — selection is an argmin over
fold scores that already exist — and without it success criterion #1 ("beats seasonal naive
on the panel aggregate") could be satisfied by selection bias alone.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from xlforecast.schemas.enums import SelectionStrategy
from xlforecast.schemas.results import FoldScore

__all__ = [
    "LOW_WINDOW_THRESHOLD",
    "Selection",
    "select",
    "selected_lofo_score",
]

#: FR-402 -- below this, `per_series` selection is warned about explicitly.
LOW_WINDOW_THRESHOLD = 5


@dataclass(slots=True)
class Selection:
    """Which model wins where, and how much to trust the number attached to it."""

    strategy: SelectionStrategy
    #: unique_id -> model. Under `pooled` every series maps to the same winner.
    per_series: dict[str, str] = field(default_factory=dict)
    panel_winner: str | None = None
    #: True when the reported score for the selection is an argmin over the folds it is
    #: scored on -- i.e. optimistically biased (FR-408).
    biased: bool = False
    #: The leave-one-fold-out companion: select on folds != k, score on fold k.
    lofo_score: float | None = None
    warnings: list[str] = field(default_factory=list)


def _pooled_scores(
    fold_scores: list[FoldScore], *, metric: str, exclude_fold: int | None = None
) -> dict[str, float]:
    buckets: dict[str, list[float]] = {}
    for s in fold_scores:
        if exclude_fold is not None and s.fold_index == exclude_fold:
            continue
        value = s.metrics.get(metric)
        if value is not None:
            buckets.setdefault(s.model, []).append(value)
    return {m: float(np.mean(v)) for m, v in buckets.items() if v}


def _series_scores(
    fold_scores: list[FoldScore], *, metric: str, exclude_fold: int | None = None
) -> dict[str, dict[str, float]]:
    buckets: dict[str, dict[str, list[float]]] = {}
    for s in fold_scores:
        if s.unique_id is None:
            continue
        if exclude_fold is not None and s.fold_index == exclude_fold:
            continue
        value = s.metrics.get(metric)
        if value is not None:
            buckets.setdefault(s.unique_id, {}).setdefault(s.model, []).append(value)
    return {
        uid: {m: float(np.mean(v)) for m, v in per_model.items()}
        for uid, per_model in buckets.items()
    }


def _argmin(scores: dict[str, float], *, candidates: list[str] | None = None) -> str | None:
    pool = {m: v for m, v in scores.items() if candidates is None or m in candidates}
    if not pool:
        return None
    return min(pool, key=lambda m: pool[m])


def selected_lofo_score(
    fold_scores: list[FoldScore], *, strategy: SelectionStrategy, metric: str
) -> float | None:
    """Unbiased companion to the selected model's score (FR-408).

    For each fold `k`: choose the winner using folds ≠ `k`, then score that winner on fold
    `k`. Averaging over folds gives an estimate that selection has not been allowed to
    inflate. Requires at least two folds; with one there is nothing to hold out and the
    honest answer is that no unbiased estimate is available.
    """
    folds = sorted({s.fold_index for s in fold_scores})
    if len(folds) < 2:
        return None

    realised: list[float] = []
    for k in folds:
        if strategy == "pooled":
            winner = _argmin(_pooled_scores(fold_scores, metric=metric, exclude_fold=k))
            if winner is None:
                continue
            values = [
                s.metrics[metric]
                for s in fold_scores
                if s.fold_index == k and s.model == winner and s.metrics.get(metric) is not None
            ]
            realised.extend(v for v in values if v is not None)
        else:
            chosen = {
                uid: _argmin(per_model)
                for uid, per_model in _series_scores(
                    fold_scores, metric=metric, exclude_fold=k
                ).items()
            }
            for s in fold_scores:
                if s.fold_index != k or s.unique_id is None:
                    continue
                if chosen.get(s.unique_id) == s.model and s.metrics.get(metric) is not None:
                    value = s.metrics[metric]
                    if value is not None:
                        realised.append(value)
    return float(np.mean(realised)) if realised else None


def select(
    fold_scores: list[FoldScore],
    *,
    strategy: SelectionStrategy = "pooled",
    metric: str = "mase",
    n_windows: int = 3,
    baseline: str = "SeasonalNaive",
    any_beat_baseline: bool = True,
) -> Selection:
    """Choose the model(s) to deliver.

    When nothing beat the baseline the recommendation *is* the baseline (FR-406). That is
    not a failure state to be worked around: reporting it plainly is the product.
    """
    warnings: list[str] = []
    pooled = _pooled_scores(fold_scores, metric=metric)

    if not any_beat_baseline and baseline in pooled:
        winner = baseline
        warnings.append(
            f"No model beat {baseline}; recommending it. This is a result, not an error."
        )
        series_map = dict.fromkeys(_series_scores(fold_scores, metric=metric), winner)
        return Selection(
            strategy=strategy,
            per_series=series_map,
            panel_winner=winner,
            biased=False,
            lofo_score=None,
            warnings=warnings,
        )

    if strategy == "per_series":
        if n_windows < LOW_WINDOW_THRESHOLD:
            warnings.append(
                f"per_series selection with {n_windows} CV windows overfits: the winner for "
                f"each series is an argmin over models on {n_windows} folds. Its reported "
                f"score is optimistically biased -- see the leave-one-fold-out column."
            )
        per_series = {
            uid: chosen
            for uid, per_model in _series_scores(fold_scores, metric=metric).items()
            if (chosen := _argmin(per_model)) is not None
        }
        return Selection(
            strategy=strategy,
            per_series=per_series,
            panel_winner=_argmin(pooled),
            biased=True,
            lofo_score=selected_lofo_score(fold_scores, strategy=strategy, metric=metric),
            warnings=warnings,
        )

    if strategy == "clustered":
        warnings.append("clustered selection is not implemented in v1; using pooled.")

    pooled_winner = _argmin(pooled)
    series_map = (
        dict.fromkeys(_series_scores(fold_scores, metric=metric), pooled_winner)
        if pooled_winner is not None
        else {}
    )
    return Selection(
        strategy="pooled" if strategy == "clustered" else strategy,
        per_series=series_map,
        panel_winner=pooled_winner,
        # Pooled selection is an argmin too, over the same folds -- narrower than per-series
        # (one choice over ~13 models rather than one per series) but not unbiased.
        biased=True,
        lofo_score=selected_lofo_score(fold_scores, strategy="pooled", metric=metric),
        warnings=warnings,
    )
