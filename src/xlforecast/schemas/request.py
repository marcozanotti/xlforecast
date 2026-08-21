"""Request contracts (TS §4.2): what a job is, before anything has been computed."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, field_validator, model_validator

from xlforecast.errors import EnsembleConfigError, InvalidModelNameError, ProhibitedModelError
from xlforecast.schemas.enums import (
    EnsembleMethod,
    GapFill,
    ProbEnsembleMethod,
    SelectionStrategy,
)
from xlforecast.schemas.freq import normalise_freq
from xlforecast.schemas.registry import DEFAULT_MODELS, MODEL_REGISTRY

__all__ = ["DataMapping", "ExogSpec", "ForecastRequest", "ModelName", "ResolvedRequest"]


def _validate_model_name(name: str) -> str:
    """Validate against the registry, not a closed Literal (TS §5.2)."""
    spec = MODEL_REGISTRY.get(name)
    if spec is None:
        known = ", ".join(sorted(MODEL_REGISTRY))
        raise InvalidModelNameError(
            f"'{name}' is not a known model.",
            fix=f"Choose one of: {known}.",
        )
    if not spec.commercial_ok:
        raise ProhibitedModelError(
            f"'{name}' is licensed {spec.licence}, which forbids commercial use.",
            fix="Remove it from the model list; it cannot ship in this product.",
        )
    return name


ModelName = Annotated[str, AfterValidator(_validate_model_name)]
Freq = Annotated[str, AfterValidator(normalise_freq)]


class ExogSpec(BaseModel):
    """One exogenous column (FR-102).

    A bare `list[str]` could not carry dtype or fill policy, and could not express that a
    future-known column needs `h` future rows per series (FR-111).
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    kind: Literal["historic", "future_known"]
    dtype: Literal["float", "int", "category", "bool"] = "float"
    fill: Literal["none", "zero", "ffill", "interpolate"] = "none"


class DataMapping(BaseModel):
    """User's column mapping (FR-101, FR-102)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    unique_id_col: str
    ds_col: str
    y_col: str
    exog: list[ExogSpec] = Field(default_factory=list)

    @field_validator("exog")
    @classmethod
    def _unique_exog_names(cls, v: list[ExogSpec]) -> list[ExogSpec]:
        names = [e.name for e in v]
        dupes = {n for n in names if names.count(n) > 1}
        if dupes:
            raise ValueError(f"duplicate exogenous column(s): {sorted(dupes)}")
        return v

    @property
    def future_known(self) -> list[str]:
        return [e.name for e in self.exog if e.kind == "future_known"]


class ForecastRequest(BaseModel):
    """A job specification (TS §4.2).

    Frozen, because `Manifest` embeds one and a manifest that can be mutated after the fact
    is not a reproducibility record.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    h: Annotated[int, Field(ge=1, le=520)]
    freq: Freq
    season_length: int | None = Field(default=None, ge=1)
    # validate_default is load-bearing, not defensive: Pydantic does NOT run field
    # validators over a default_factory result, so without it the default request carried
    # DEFAULT_MODELS in declaration order while a round-tripped one came back sorted. Two
    # objects representing the same job would then differ, which is precisely the NFR-02
    # byte-identity hazard the sorting exists to prevent. Caught by the G0 property test.
    models: list[ModelName] = Field(
        default_factory=lambda: list(DEFAULT_MODELS), validate_default=True
    )
    levels: list[Annotated[int, Field(ge=1, le=99)]] = Field(
        default_factory=lambda: [80, 95], validate_default=True
    )
    n_windows: Annotated[int, Field(ge=1, le=20)] = 3
    step_size: int | None = Field(default=None, ge=1)
    selection: SelectionStrategy = "pooled"
    ensemble: EnsembleMethod = "median"
    ensemble_prob_method: ProbEnsembleMethod = "vincentization"
    ensemble_metric: Literal["mase", "rmsse", "scaled_crps"] = "mase"
    ensemble_trim: Annotated[float, Field(ge=0.0, lt=0.5)] = 0.2
    best_k: Annotated[int, Field(ge=1)] = 3
    gap_fill: GapFill = "none"
    conformal: bool = True
    seed: int = 42
    assumptions: list[str] = Field(default_factory=list)

    @field_validator("models")
    @classmethod
    def _seasonal_naive_always(cls, v: list[str]) -> list[str]:
        """FR-204 — not negotiable.

        A `field_validator`, not a mutating `mode="after"` validator. The original spec
        assigned to `self.models` inside `mode="after"`, which raises `Instance is frozen`
        under `frozen=True` (verified in the Phase 0 spike) and recurses forever under
        `validate_assignment`.

        Sorted and deduped so that list order cannot influence `best_k` tie-breaking, which
        would otherwise put NFR-02 byte-identity at the mercy of how the caller happened to
        order the argument.
        """
        return sorted({*v, "SeasonalNaive"})

    @field_validator("levels")
    @classmethod
    def _levels_sorted_unique(cls, v: list[int]) -> list[int]:
        """Deterministic column order for XLF_Forecast / XLF_Leaderboard (FR-701)."""
        if not v:
            raise ValueError("at least one interval level is required")
        return sorted(set(v))

    @model_validator(mode="after")
    def _ensemble_needs_folds(self) -> ForecastRequest:
        """FR-405a — leave-one-fold-out weighting is impossible with a single window."""
        if self.ensemble in ("inverse_error", "best_k") and self.n_windows < 2:
            raise EnsembleConfigError(
                f"ensemble '{self.ensemble}' estimates weights from fold errors, which "
                f"requires at least 2 CV windows (got {self.n_windows}).",
                fix="Raise n_windows to 2 or more, or use 'median' / 'trimmed_mean', "
                "which fit no parameters.",
            )
        return self

    @property
    def effective_step_size(self) -> int:
        """`step_size` defaults to `h` (TS §4.2)."""
        return self.step_size if self.step_size is not None else self.h

    def min_observations(self, season_length: int) -> int:
        """FR-105 length threshold.

        `2 * season_length + h + (n_windows - 1) * step_size`, not the naive
        `2 * season_length + h`: the *earliest* CV training window must itself satisfy
        `2 * season_length`, otherwise series pass ingestion and then vanish inside
        cross-validation, which FS §6 forbids.
        """
        return 2 * season_length + self.h + (self.n_windows - 1) * self.effective_step_size

    def autoarima_mode(self, season_length: int) -> Literal["seasonal", "fourier"]:
        """FR-201a. Fourier above m=24: 3.1x cheaper and better practice at long periods."""
        return "fourier" if season_length > 24 else "seasonal"

    def ets_mode(self, season_length: int) -> Literal["seasonal", "mstl"]:
        """FR-201c. MSTL above m=24, because seasonal ETS cannot represent the period.

        `statsforecast/ets.py` allocates a fixed 24-slot seasonal state buffer and returns
        early when `m > 24`, so `AutoETS(season_length=52)` is silently non-seasonal --
        measured to produce forecasts identical to `AutoETS(season_length=1)`. A model that
        quietly ignores the seasonality it was asked to fit, and then scores badly for it,
        is exactly the kind of misleading leaderboard row this project exists to prevent.
        """
        return "mstl" if season_length > 24 else "seasonal"


class ResolvedRequest(ForecastRequest):
    """A `ForecastRequest` with every inferable field filled (TS §4.2).

    The manifest stores *this*, not the user's original: replay must not re-run inference,
    or a change in `season_length` inference silently changes what "reproducing the run"
    means.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    season_length: int = Field(ge=1)
    step_size: int = Field(ge=1)
    autoarima: Literal["seasonal", "fourier"]
    ets: Literal["seasonal", "mstl"]

    @model_validator(mode="after")
    def _fourier_threshold_respected(self) -> ResolvedRequest:
        expected = "fourier" if self.season_length > 24 else "seasonal"
        if self.autoarima != expected:
            raise ValueError(
                f"autoarima={self.autoarima!r} contradicts FR-201a at "
                f"season_length={self.season_length} (expected {expected!r})"
            )
        expected_ets = "mstl" if self.season_length > 24 else "seasonal"
        if self.ets != expected_ets:
            raise ValueError(
                f"ets={self.ets!r} contradicts FR-201c at "
                f"season_length={self.season_length} (expected {expected_ets!r})"
            )
        return self

    @classmethod
    def from_request(cls, request: ForecastRequest, *, season_length: int) -> ResolvedRequest:
        """Resolve inference-dependent fields against a profiled `season_length`."""
        return cls(
            **request.model_dump(exclude={"season_length", "step_size"}),
            season_length=season_length,
            step_size=request.effective_step_size,
            autoarima=request.autoarima_mode(season_length),
            ets=request.ets_mode(season_length),
        )

    def expected_output_rows(self, n_series: int) -> int:
        """FR-708 precheck: rows in `XLF_Forecast_Long`, which holds every model's forecast.

        `series x h x models x (1 + 2*levels)`. At the 500k-row input cap this overflows
        Excel's 1,048,576 limit long before the input cap binds, which is why FR-107 alone
        was never sufficient.
        """
        per_point = 1 + 2 * len(self.levels)
        return n_series * self.h * len(self.models) * per_point

    def fits_in_excel(self, n_series: int) -> bool:
        return self.expected_output_rows(n_series) <= 1_048_576

    def cv_cutoff_count(self) -> int:
        return self.n_windows

    def total_fits_per_local_model(self, n_series: int) -> int:
        """Explanatory only. FR-217 makes measured `train + predict` time the cost proxy;
        fit counts are not a unit of work and must never be used as one."""
        return n_series * (self.n_windows + 1)
