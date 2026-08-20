"""Declarative model metadata — the source of truth for `ModelName` (TS §5.2).

Deliberately split from `engine/registry.py`: this module holds *metadata only* and imports
nothing from statsforecast, mlforecast, LightGBM or XGBoost, so `xlforecast.schemas` stays
importable without the heavy compiled stack. Phase 1's `engine/registry.py` binds constructors
onto these entries.

`ModelName` is validated against this table rather than being a closed `Literal`. With a
literal containing only Apache/MIT models, the `commercial_ok` gate was unreachable — a
prohibited model could never be named in the first place, so the check was dead code and its
test would have been vacuous. Registry validation also makes a v2 model addition a table row
rather than a schema change.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from xlforecast.schemas.enums import Family, InformationSet

__all__ = ["DEFAULT_MODELS", "MODEL_REGISTRY", "ModelSpec", "is_known", "spec_for"]


@dataclass(frozen=True, slots=True)
class ModelSpec:
    """Metadata for one model. `commercial_ok=False` must never reach a job (TS §5.2)."""

    name: str
    family: Family
    information_set: InformationSet
    handles_intermittent: bool
    supports_exog: bool
    licence: str
    commercial_ok: bool
    #: Minimum observations, as seasons plus an absolute floor. Local ML needs far more than
    #: the statistical models because lag construction consumes `max_lag` rows per series
    #: (FR-203c): at weekly frequency a series at the FR-105 floor leaves ~91 training rows,
    #: which the Phase 0 spike confirmed exactly.
    min_obs_seasons: float = 2.0
    min_obs_abs: int = 10
    #: Small-data hyperparameters are load-bearing for local ML, not decorative — global
    #: defaults (min_child_samples=20, num_leaves=31) produce near-constant predictions on
    #: ~91 rows, which reads as a bug rather than a finding (FR-203c).
    small_data_params: bool = False

    def min_obs(self, season_length: int) -> int:
        """Minimum observations required of a series at this seasonal period."""
        return max(self.min_obs_abs, math.ceil(self.min_obs_seasons * season_length))


def _sf(
    name: str,
    *,
    intermittent: bool = False,
    exog: bool = False,
    seasons: float = 2.0,
    abs_min: int = 10,
) -> ModelSpec:
    return ModelSpec(
        name=name,
        family="local",
        information_set="own_series",
        handles_intermittent=intermittent,
        supports_exog=exog,
        licence="Apache-2.0",
        commercial_ok=True,
        min_obs_seasons=seasons,
        min_obs_abs=abs_min,
    )


def _ml(name: str, information_set: InformationSet, licence: str) -> ModelSpec:
    local = information_set == "own_series"
    return ModelSpec(
        name=name,
        family="local" if local else "global",
        information_set=information_set,
        handles_intermittent=False,
        supports_exog=True,
        licence=licence,
        commercial_ok=True,
        # Local ML must clear max_lag plus a workable training remainder; a global model
        # pools the panel and has no such per-series constraint.
        min_obs_seasons=1.0 if local else 0.0,
        min_obs_abs=120 if local else 10,
        small_data_params=local,
    )


_ENTRIES: tuple[ModelSpec, ...] = (
    # --- baselines (FR-201, FR-201b) -------------------------------------------------
    _sf("SeasonalNaive", seasons=1.0),
    _sf("HistoricAverage", seasons=0.0),
    _sf("WindowAverage", seasons=1.0),
    # --- local statistical (FR-201) --------------------------------------------------
    _sf("AutoARIMA", exog=True),
    _sf("AutoETS"),
    _sf("AutoCES"),
    _sf("DynamicOptimizedTheta"),
    # --- intermittent (FR-202) -------------------------------------------------------
    _sf("CrostonClassic", intermittent=True, seasons=0.0),
    _sf("ADIDA", intermittent=True, seasons=0.0),
    _sf("IMAPA", intermittent=True, seasons=0.0),
    _sf("ZeroModel", intermittent=True, seasons=0.0, abs_min=1),
    # --- ML: three learners x two information sets = three matched pairs (FR-203b) ----
    _ml("LocalLinear", "own_series", "BSD-3-Clause"),
    _ml("LocalLGBM", "own_series", "MIT"),
    _ml("LocalXGB", "own_series", "Apache-2.0"),
    _ml("GlobalLinear", "panel", "BSD-3-Clause"),
    _ml("GlobalLGBM", "panel", "MIT"),
    _ml("GlobalXGB", "panel", "Apache-2.0"),
)

MODEL_REGISTRY: dict[str, ModelSpec] = {spec.name: spec for spec in _ENTRIES}

#: FR-216, frozen against the Phase 0 spike (docs/05): 2.9 min projected against a 10-minute
#: budget. Cost concentrates in LocalLGBM (54%), Fourier AutoARIMA (26%) and AutoETS (12%).
DEFAULT_MODELS: tuple[str, ...] = (
    "SeasonalNaive",
    "HistoricAverage",
    "WindowAverage",
    "AutoARIMA",
    "AutoETS",
    "DynamicOptimizedTheta",
    "CrostonClassic",
    "LocalLinear",
    "LocalLGBM",
    "LocalXGB",
    "GlobalLinear",
    "GlobalLGBM",
    "GlobalXGB",
)


def is_known(name: str) -> bool:
    return name in MODEL_REGISTRY


def spec_for(name: str) -> ModelSpec:
    return MODEL_REGISTRY[name]
