"""Closed vocabularies shared across every contract (TS §4.1)."""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

__all__ = [
    "EnsembleMethod",
    "ExclusionReason",
    "Family",
    "GapFill",
    "InformationSet",
    "IntermittencyClass",
    "ProbEnsembleMethod",
    "Quantity",
    "Scope",
    "SelectionStrategy",
]

# `family` and `information_set` are deliberately two axes, not one. FR-207 exists to make
# own-series-vs-panel comparisons legible, and a single local|global|ensemble literal cannot
# express that a LocalLGBM is an ML learner with own-series information, or that an ensemble
# of local models is still own-series.
Family = Literal["local", "global", "ensemble"]
InformationSet = Literal["own_series", "panel"]

Scope = Literal["panel", "series"]
Quantity = Literal["point", "lo", "hi"]
IntermittencyClass = Literal["smooth", "intermittent", "erratic", "lumpy"]

SelectionStrategy = Literal["pooled", "per_series", "clustered"]
EnsembleMethod = Literal["none", "median", "trimmed_mean", "inverse_error", "best_k"]
ProbEnsembleMethod = Literal["vincentization", "linear_pool"]
GapFill = Literal["none", "zero", "interpolate"]


class ExclusionReason(StrEnum):
    """One member per FR-105 rejection rule.

    A series is never dropped without one of these attached (FS §6).
    """

    DUPLICATE_TIMESTAMPS = "duplicate_timestamps"
    NON_MONOTONIC = "non_monotonic_timestamps"
    TOO_SHORT = "insufficient_observations"
    ALL_ZERO = "all_zero"
    ALL_CONSTANT = "all_constant"
    EXCESS_MISSING = "excess_missing"
    UNPARSEABLE_DS = "unparseable_timestamps"
    FREQ_MISMATCH = "frequency_mismatch"
    MISSING_FUTURE_EXOG = "missing_future_exog"
