"""Ensembling (TS §5.5, FR-403/403a/404/405/405a).

**Weights are estimated leave-one-fold-out.** For `inverse_error` and `best_k`, the weights
used when scoring fold *k* come only from folds ≠ *k*. `median` and `trimmed_mean` fit no
parameters and are exempt.

Without that, an ensemble whose weights are derived from the fold errors it is then scored on
is *precisely* the free pass FR-405 claims to prevent — and it is a subtler failure than the
one FS §6 originally listed, because the folds genuinely are identical. Nothing about the fold
machinery would look wrong; only the ensemble's score would be quietly optimistic.

**Vincentization and linear pooling are different objects** (FR-404). Vincentization averages
quantiles level by level, producing an interval of average *shape*. Linear pooling averages
probabilities, producing the mixture — which is wider whenever members disagree, and can be
multi-modal. Both are implemented; the choice is recorded in the manifest.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import polars as pl

from xlforecast.errors import EnsembleConfigError
from xlforecast.panel import DS, ID
from xlforecast.schemas.enums import EnsembleMethod, ProbEnsembleMethod
from xlforecast.schemas.results import FoldScore

__all__ = [
    "MIN_MEMBERS_FOR_TRIM",
    "EnsembleOutcome",
    "EnsemblePlan",
    "combine_point",
    "combine_quantiles",
    "ensemble_name",
    "lofo_weights",
    "member_errors",
]

#: Below this, a 0.2 trim removes less than one model per side and `trimmed_mean` is just a
#: mean wearing a different name -- so it degrades to `median` and says so (FR-403a).
MIN_MEMBERS_FOR_TRIM = 5


def ensemble_name(method: EnsembleMethod) -> str:
    return f"Ensemble[{method}]"


@dataclass(frozen=True, slots=True)
class EnsemblePlan:
    method: EnsembleMethod
    members: tuple[str, ...]
    trim: float = 0.2
    best_k: int = 3
    metric: str = "mase"
    prob_method: ProbEnsembleMethod = "vincentization"

    def __post_init__(self) -> None:
        if self.method != "none" and len(self.members) < 2:
            raise EnsembleConfigError(
                f"an ensemble needs at least 2 members, got {len(self.members)}.",
                fix="Enable more models, or set ensemble='none'.",
            )


@dataclass(slots=True)
class EnsembleOutcome:
    """What the ensemble actually did, for the manifest and diagnostics.

    `fallbacks` exists because FR-403a's edge cases are defined behaviour, not implementation
    detail: a user whose `best_k=5` silently became `best_k=3` should be able to find that out.
    """

    applied_method: EnsembleMethod
    weights_by_fold: dict[int | None, dict[str, float]] = field(default_factory=dict)
    fallbacks: list[str] = field(default_factory=list)


def member_errors(
    fold_scores: list[FoldScore],
    *,
    members: tuple[str, ...],
    metric: str,
    exclude_fold: int | None = None,
) -> dict[str, float]:
    """Pooled error per member, optionally holding one fold out (FR-405a).

    Degenerate per-series metrics are `None` under FR-214 and are skipped rather than
    treated as zero, which would make a model look perfect on the series it failed to score.
    """
    buckets: dict[str, list[float]] = {m: [] for m in members}
    for score in fold_scores:
        if score.model not in buckets:
            continue
        if exclude_fold is not None and score.fold_index == exclude_fold:
            continue
        value = score.metrics.get(metric)
        if value is not None:
            buckets[score.model].append(value)
    return {m: float(np.mean(v)) for m, v in buckets.items() if v}


def lofo_weights(
    fold_scores: list[FoldScore], *, plan: EnsemblePlan, exclude_fold: int | None
) -> tuple[dict[str, float], list[str]]:
    """Member weights for one fold, and any FR-403a fallbacks that fired.

    `exclude_fold=k` gives the weights used to *score* fold k; `None` gives the weights for
    the delivered forecast, which may use every fold.
    """
    fallbacks: list[str] = []
    errors = member_errors(
        fold_scores, members=plan.members, metric=plan.metric, exclude_fold=exclude_fold
    )
    available = tuple(m for m in plan.members if m in errors)

    if plan.method in ("median", "trimmed_mean", "none") or not available:
        # Parameterless: every member counts equally and no leakage is possible.
        return dict.fromkeys(plan.members, 1.0 / len(plan.members)), fallbacks

    if plan.method == "best_k":
        k = plan.best_k
        if k > len(available):
            fallbacks.append(
                f"best_k={plan.best_k} exceeds the {len(available)} scored members; used all"
            )
            k = len(available)
        chosen = sorted(available, key=lambda m: errors[m])[:k]
        return dict.fromkeys(chosen, 1.0 / k), fallbacks

    # inverse_error. A zero error would be an infinite weight, so it is treated as a tie
    # among the zero-error members rather than allowed to dominate arithmetically.
    zeros = [m for m in available if errors[m] <= 0]
    if zeros:
        fallbacks.append("zero-error member(s) present; weighted equally among them")
        return dict.fromkeys(zeros, 1.0 / len(zeros)), fallbacks
    raw = {m: 1.0 / errors[m] for m in available}
    total = sum(raw.values())
    return {m: w / total for m, w in raw.items()}, fallbacks


def _effective_method(plan: EnsemblePlan, n_members: int) -> tuple[EnsembleMethod, list[str]]:
    if plan.method == "trimmed_mean" and n_members < MIN_MEMBERS_FOR_TRIM:
        return "median", [
            f"trimmed_mean with {n_members} members trims less than one per side; "
            "degraded to median"
        ]
    return plan.method, []


def _trimmed_mean(values: np.ndarray, trim: float) -> float:
    ordered = np.sort(values)
    cut = int(np.floor(len(ordered) * trim))
    kept = ordered[cut : len(ordered) - cut] if cut else ordered
    return float(np.mean(kept if kept.size else ordered))


def combine_point(
    predictions: pl.DataFrame,
    *,
    plan: EnsemblePlan,
    weights: dict[str, float],
) -> tuple[pl.DataFrame, list[str]]:
    """Combine member point forecasts into one ensemble forecast.

    Operates on the members' predictions for a single fold, which is what makes the ensemble
    compete on the same folds as its constituents (FR-405) rather than being assembled from
    final forecasts.
    """
    members = [m for m in plan.members if m in set(predictions.get_column("model").unique())]
    if len(members) < 2:
        raise EnsembleConfigError(
            f"only {len(members)} of the ensemble's members produced forecasts.",
            fix="Check that the member models ran, or set ensemble='none'.",
        )

    method, fallbacks = _effective_method(plan, len(members))
    wide = predictions.filter(pl.col("model").is_in(members)).pivot(
        on="model", index=[ID, DS], values="y_hat"
    )
    present = [m for m in members if m in wide.columns]
    matrix = wide.select(present).to_numpy()

    if method == "median":
        combined = np.nanmedian(matrix, axis=1)
    elif method == "trimmed_mean":
        combined = np.apply_along_axis(_trimmed_mean, 1, matrix, plan.trim)
    else:
        w = np.array([weights.get(m, 0.0) for m in present], dtype=float)
        if w.sum() <= 0:  # every weight fell outside the surviving members
            w = np.ones(len(present))
        w = w / w.sum()
        combined = matrix @ w

    out = wide.select([ID, DS]).with_columns(
        pl.Series("y_hat", combined, dtype=pl.Float64),
        pl.lit(ensemble_name(plan.method)).alias("model"),
    )
    return out.select([ID, DS, "model", "y_hat"]), fallbacks


def combine_quantiles(
    quantiles: dict[str, np.ndarray],
    levels: np.ndarray,
    *,
    weights: dict[str, float],
    method: ProbEnsembleMethod,
) -> np.ndarray:
    """Combine member quantile functions at a single point (FR-404).

    `quantiles` maps member -> values at `levels`. Returns the ensemble's values at the same
    levels.

    Vincentization averages the quantiles directly. Linear pooling averages the *cumulative
    probabilities* and inverts, which is a different object: it is at least as wide as
    vincentization whenever members disagree, because it represents genuine disagreement as
    genuine uncertainty rather than averaging it away.
    """
    members = [m for m in quantiles if m in weights and weights[m] > 0]
    if not members:
        members = list(quantiles)
    w = np.array([weights.get(m, 1.0) for m in members], dtype=float)
    w = w / w.sum()
    stack = np.vstack([quantiles[m] for m in members])

    if method == "vincentization":
        return np.asarray(w @ stack, dtype=float)

    # Linear pooling: build the mixture CDF on a shared grid, then invert it.
    grid = np.linspace(stack.min(), stack.max(), 512)
    mixture = np.zeros_like(grid)
    for weight, row in zip(w, stack, strict=True):
        mixture += weight * np.interp(grid, row, levels, left=0.0, right=1.0)
    return np.asarray(np.interp(levels, mixture, grid), dtype=float)
