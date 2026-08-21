"""Model construction, bound onto the declarative metadata in `schemas/registry` (TS §5.2).

The split is deliberate: `schemas/registry` holds metadata and imports nothing heavy, so the
API and CLI can validate a request without paying for a 300 MB import. This module binds
constructors and is only touched once a job actually runs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from xlforecast.errors import InvalidModelNameError
from xlforecast.schemas.enums import InformationSet
from xlforecast.schemas.registry import MODEL_REGISTRY, ModelSpec

__all__ = ["FeatureRecipe", "MLPlan", "build_local", "build_ml_plan", "is_ml", "spec"]


class _Estimator(Protocol):  # pragma: no cover - structural only
    """Structural type for a scikit-learn-compatible regressor.

    Capital `X` matches scikit-learn's own signature; renaming it here would make the
    protocol fail to describe the objects it exists to describe.
    """

    def fit(self, X: Any, y: Any) -> Any: ...  # noqa: N803
    def predict(self, X: Any) -> Any: ...  # noqa: N803


def spec(name: str) -> ModelSpec:
    try:
        return MODEL_REGISTRY[name]
    except KeyError as exc:
        raise InvalidModelNameError(
            f"'{name}' is not a known model.",
            fix=f"Choose one of: {', '.join(sorted(MODEL_REGISTRY))}.",
        ) from exc


_ML_NAMES = frozenset(
    {"LocalLinear", "LocalLGBM", "LocalXGB", "GlobalLinear", "GlobalLGBM", "GlobalXGB"}
)


def is_ml(name: str) -> bool:
    return name in _ML_NAMES


def build_local(name: str, *, season_length: int) -> Any:
    """Construct a statsforecast model.

    `AutoARIMA` is constructed non-seasonally above `season_length` 24 (FR-201a); the Fourier
    regressors that replace the seasonal terms are built in `engine/local.py`, because they
    are a property of the data frame rather than of the model object.
    """
    from statsforecast import models as sf

    match name:
        case "SeasonalNaive":
            return sf.SeasonalNaive(season_length=season_length)
        case "HistoricAverage":
            return sf.HistoricAverage()
        case "WindowAverage":
            # FR-201b: at the seasonal frequency, this is the incumbent method -- a trailing
            # mean over one full cycle is what persona P1 forecasts with today.
            return sf.WindowAverage(window_size=season_length)
        case "AutoARIMA":
            return sf.AutoARIMA(season_length=1 if season_length > 24 else season_length)
        case "AutoETS":
            # FR-201c. statsforecast's ETS carries a fixed 24-slot seasonal buffer and
            # returns early above it, so AutoETS(season_length=52) silently fits a
            # non-seasonal model -- verified to give forecasts identical to
            # AutoETS(season_length=1). MSTL decomposes the seasonality out first and
            # forecasts the trend with ETS, which is the statsforecast-recommended route
            # for long periods. Measured on a weekly seasonal panel: MASE 1.123 -> 0.765,
            # from worse than SeasonalNaive to better than it.
            if season_length > 24:
                return sf.MSTL(
                    season_length=season_length,
                    trend_forecaster=sf.AutoETS(model="ZZN"),
                    alias="AutoETS",
                )
            return sf.AutoETS(season_length=season_length)
        case "AutoCES":
            return sf.AutoCES(season_length=season_length)
        case "DynamicOptimizedTheta":
            return sf.DynamicOptimizedTheta(season_length=season_length)
        case "CrostonClassic":
            return sf.CrostonClassic()
        case "ADIDA":
            return sf.ADIDA()
        case "IMAPA":
            return sf.IMAPA()
        case "ZeroModel":
            return sf.ZeroModel()
        case _:
            raise InvalidModelNameError(
                f"'{name}' is not a statsforecast model.",
                fix="Use build_ml_plan for machine-learning models.",
            )


@dataclass(frozen=True, slots=True)
class FeatureRecipe:
    """The feature set, identical for a matched pair (FR-203b).

    `LocalLGBM` and `GlobalLGBM` must differ in exactly one variable -- the information set --
    with learner, features and folds held constant. Sharing one recipe object between them is
    how that is enforced rather than merely intended.
    """

    lags: tuple[int, ...]
    date_features: tuple[str, ...]

    @classmethod
    def for_freq(cls, freq: str, season_length: int) -> FeatureRecipe:
        base = freq.split("-")[0].lstrip("0123456789")
        calendar = {
            "W": ("week", "month"),
            "D": ("dayofweek", "month"),
            "B": ("dayofweek", "month"),
            "ME": ("month", "quarter"),
            "QE": ("quarter",),
            "YE": (),
        }.get(base, ("month",))
        lags = sorted({1, 2, 3, 4, season_length}) if season_length > 1 else [1, 2, 3, 4]
        return cls(lags=tuple(lags), date_features=calendar)


@dataclass(frozen=True, slots=True)
class MLPlan:
    """How to run one ML model: the learner, its information set, and shared features."""

    name: str
    information_set: InformationSet
    estimator: _Estimator
    recipe: FeatureRecipe


def build_ml_plan(name: str, *, freq: str, season_length: int, seed: int) -> MLPlan:
    """Construct an ML model with the small-data hyperparameters its registry entry demands.

    FR-203c: local ML sees roughly 91 training rows per series at weekly frequency after lag
    construction (measured in the Phase 0 spike). LightGBM's defaults -- `min_child_samples=20`,
    `num_leaves=31` -- produce near-constant predictions at that size, which reads as a bug
    rather than a finding. Global models pool the panel and keep the standard defaults.
    """
    entry = spec(name)
    small = entry.small_data_params

    if name.endswith("Linear"):
        from sklearn.linear_model import Ridge

        estimator: Any = Ridge(alpha=1.0, random_state=seed)
    elif name.endswith("LGBM"):
        from lightgbm import LGBMRegressor

        estimator = LGBMRegressor(
            n_estimators=100,
            verbose=-1,
            random_state=seed,
            n_jobs=1,
            deterministic=True,
            # NFR-02: LightGBM histogram construction is thread-order dependent otherwise
            force_row_wise=True,
            min_child_samples=5 if small else 20,
            num_leaves=7 if small else 31,
            learning_rate=0.1,
        )
    elif name.endswith("XGB"):
        from xgboost import XGBRegressor

        estimator = XGBRegressor(
            n_estimators=100,
            verbosity=0,
            random_state=seed,
            n_jobs=1,
            max_depth=3 if small else 6,
            min_child_weight=1,
            reg_lambda=2.0 if small else 1.0,
        )
    else:
        raise InvalidModelNameError(
            f"'{name}' is not a machine-learning model.",
            fix="Use build_local for statistical models.",
        )

    return MLPlan(
        name=name,
        information_set=entry.information_set,
        estimator=estimator,
        recipe=FeatureRecipe.for_freq(freq, season_length),
    )
