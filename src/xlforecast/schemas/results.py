"""Result and manifest contracts (TS §4.4, §4.5)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from xlforecast.schemas.artifacts import ArtifactPack
from xlforecast.schemas.enums import ExclusionReason, Family, InformationSet, Quantity, Scope
from xlforecast.schemas.profile import DataProfile
from xlforecast.schemas.request import DataMapping, ResolvedRequest

__all__ = [
    "CalibrationRow",
    "ConformalBands",
    "FoldScore",
    "ForecastFrame",
    "ForecastRow",
    "Leaderboard",
    "LeaderboardRow",
    "Manifest",
    "ModelTiming",
    "RunResult",
    "RunTiming",
]


class FoldScore(BaseModel):
    """Per-model per-fold detail (XLF_Diagnostics block 2).

    `fold_index` exists because gate G2 asserts fold identity between an ensemble and its
    members, and nothing in the original schema set carried a fold identifier to assert on.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    fold_index: int = Field(ge=0)
    cutoff: str  #: panel-wide calendar date (FR-206), not a per-series offset
    model: str
    unique_id: str | None = None
    #: Per-family effective training rows, so the mlforecast `dropna`/`keep_last_n` asymmetry
    #: can be audited rather than silently confounding a local-vs-global comparison (AC-206).
    n_train_rows: int = Field(ge=0)
    metrics: dict[str, float | None] = Field(default_factory=dict)


class LeaderboardRow(BaseModel):
    """One ranked result (FS §5 `XLF_Leaderboard`).

    Carries no duration field. Measured time is not reproducible, so a duration here would
    make NFR-02 byte-identity unachievable by construction; timings live in `RunTiming`
    (FR-217b).
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    scope: Scope
    unique_id: str | None = None
    model: str
    family: Family
    information_set: InformationSet

    # All metrics are `float | None`, never NaN. FR-214 makes zero denominators routine
    # (a series constant within a training fold; an all-zero evaluation window), and the
    # Phase 0 spike confirmed a NaN float serialises to `null` and then fails re-validation
    # into a bare float -- a silent corruption that only surfaces on replay.
    mase: float | None = None
    rmsse: float | None = None
    mae: float | None = None
    rmse: float | None = None
    smape: float | None = None
    scaled_crps: float | None = None  #: named for what utilsforecast actually computes

    #: Out-of-calibration coverage per level (FR-303). A single panel figure: measurement
    #: showed splitting it by intermittency class carries no information (0.807 vs 0.809).
    #: The tails, which differ sharply between classes, live in `CalibrationRow`.
    coverage: dict[int, float] = Field(default_factory=dict)

    n_folds: int = Field(ge=0)
    n_series_scored: int = Field(ge=0)  #: FR-209
    n_series_common: int = Field(ge=0)  #: FR-215
    rank: int = Field(ge=1)
    #: Against SeasonalNaive, the mandated baseline (FR-204). Negative is better.
    vs_baseline_pct: float | None = None
    #: Against WindowAverage at the seasonal frequency -- the *incumbent* method persona P1
    #: forecasts with today (FR-201b). Beating SeasonalNaive is table stakes; beating the
    #: incumbent is the commercial claim, and one field cannot carry both comparisons.
    vs_incumbent_pct: float | None = None
    selected: bool = False
    #: FR-408. An argmin over models on the folds it is then scored on is optimistically
    #: biased; the naive figure must never be read as an unbiased estimate.
    selection_biased: bool = False
    selected_lofo_score: float | None = None


class Leaderboard(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    rows: list[LeaderboardRow] = Field(default_factory=list)
    aggregation: Literal["mean", "median"] = "mean"  #: FR-209 -- must be stated
    baseline_models: list[str] = Field(default_factory=lambda: ["SeasonalNaive", "WindowAverage"])
    any_beat_baseline: bool = False  #: FR-406


class ForecastRow(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    unique_id: str
    ds: str
    model: str
    quantity: Quantity
    level: int | None = None
    value: float


class ForecastFrame(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    rows: list[ForecastRow] = Field(default_factory=list)
    levels: list[int] = Field(default_factory=list)


class ConformalBands(BaseModel):
    """Calibrated half-widths at one level (TS §5.4)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    #: Which model these bands belong to. Without it a list of bands is ambiguous -- the
    #: half-widths are keyed by series, so two models' bands at the same level are
    #: indistinguishable, and a lookup silently returns whichever came first.
    model: str
    level: int = Field(ge=1, le=99)
    half_width: dict[str, float] = Field(default_factory=dict)
    pooled_fallback: set[str] = Field(default_factory=set)
    clipped: dict[str, float] = Field(default_factory=dict)  #: FR-307 clip rate per series
    #: FR-302 cross-conformal: which folds this band was calibrated from. Empty is a bug --
    #: a band calibrated from every fold and then scored on one of them reports nominal
    #: coverage by construction, which is the defect ADR-006 was amended to prevent.
    calibrated_from_folds: list[int] = Field(default_factory=list)


class CalibrationRow(BaseModel):
    """One model at one level, for XLF_Diagnostics block 3 (FR-303, FR-307a/b).

    Separate from `LeaderboardRow` because these are diagnostics rather than ranking inputs,
    and because `empirical_in_calibration` must never reach a user: it is ~nominal by
    construction and exists only as AC-301's control.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    model: str
    level: int = Field(ge=1, le=99)
    #: Which series this row summarises. FR-303's split is by intermittency class, applied
    #: to the tails rather than to coverage.
    scope: Literal["all", "smooth", "intermittent"] = "all"
    nominal: float = Field(ge=0.0, le=1.0)
    empirical: float | None = None
    #: AC-301's control. Conservative by construction -- it cannot report under-coverage,
    #: which is exactly why it is evidence about the honest figure and not a figure itself.
    empirical_in_calibration: float | None = None
    #: FR-307b. Share of observations below `lo` and above `hi`. A balanced interval splits
    #: its miscoverage between them; a symmetric band on zero-inflated counts does not, and
    #: that is inherent to putting a continuous interval around a point mass at zero rather
    #: than a defect to be corrected.
    lower_tail: float | None = None
    upper_tail: float | None = None
    mean_width: float | None = None
    n_pooled_fallback: int = 0
    mean_clip_rate: float | None = None


class ModelTiming(BaseModel):
    """FR-217. The cost proxy for NFR-01: measured `train + predict`, never fit counts.

    The split is load-bearing, not bookkeeping. The Phase 0 spike found `LocalLGBM` spends
    2.2x more in `predict` than in `train`, because recursive forecasting over `h` is `h`
    sequential predict-and-refeature passes -- invisible to any count-based proxy.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    model: str
    fold_index: int | None = None  #: None = the final refit on full history
    train_cpu_seconds: float = Field(ge=0.0)
    predict_cpu_seconds: float = Field(ge=0.0)
    train_wall_seconds: float = Field(ge=0.0)
    predict_wall_seconds: float = Field(ge=0.0)
    n_series_fitted: int = Field(ge=0)
    n_rows_trained: int = Field(ge=0)

    @property
    def total_cpu_seconds(self) -> float:
        return self.train_cpu_seconds + self.predict_cpu_seconds


class RunTiming(BaseModel):
    """Per-model timings plus the overhead that is nobody's train or predict (FR-217a)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    per_model: list[ModelTiming] = Field(default_factory=list)
    #: ingest, profile, validate, folds, conformal, ensemble, evaluate, artifacts, persist.
    #: Reported separately so the parts account for the whole; attributing only train and
    #: predict understates every run and makes the S3 estimate optimistic on exactly the
    #: large panels where a bad estimate costs most.
    overhead_cpu_seconds: dict[str, float] = Field(default_factory=dict)
    total_wall_seconds: float = Field(ge=0.0)
    n_workers: int = Field(ge=1)

    @property
    def model_cpu_seconds(self) -> float:
        return sum(t.total_cpu_seconds for t in self.per_model)

    @property
    def overhead_total(self) -> float:
        return sum(self.overhead_cpu_seconds.values())

    @property
    def total_cpu_seconds(self) -> float:
        return self.model_cpu_seconds + self.overhead_total

    def cost_by_model(self) -> dict[str, float]:
        """Total CPU seconds per model, summed over folds. The NFR-01 ranking."""
        out: dict[str, float] = {}
        for t in self.per_model:
            out[t.model] = out.get(t.model, 0.0) + t.total_cpu_seconds
        return dict(sorted(out.items(), key=lambda kv: -kv[1]))


class Manifest(BaseModel):
    """Everything needed to reproduce a run bit-for-bit (TS §4.5, hard rule 10)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    job_id: str
    engine_version: str
    package_versions: dict[str, str] = Field(default_factory=dict)
    python_version: str
    #: The RESOLVED request. Replay must not re-run inference, or a change in season_length
    #: inference quietly changes what "reproducing the run" means.
    request: ResolvedRequest
    mapping: DataMapping
    data_id: str
    data_fingerprint: str  #: sha256 of the canonical panel (TS §4.7)
    cutoffs: list[str] = Field(default_factory=list)  #: panel-wide calendar dates
    excluded_series: dict[str, ExclusionReason] = Field(default_factory=dict)
    autoarima_mode: Literal["seasonal", "fourier"]  #: FR-201a changes the fitted model
    ets_mode: Literal["seasonal", "mstl"]  #: FR-201c likewise -- see ForecastRequest.ets_mode
    crps_quantiles: list[float] = Field(default_factory=list)  #: FR-208 -- grid-dependent
    ensemble_params: dict[str, float | int | str] = Field(default_factory=dict)
    prompt_versions: dict[str, str] = Field(default_factory=dict)  #: TS §9.5 required this
    #: NFR-02: float reductions reorder under thread count, so byte-identity is defined
    #: relative to a recorded thread configuration, not absolutely.
    thread_config: dict[str, str] = Field(default_factory=dict)
    #: Lineage for v2 forecast-stability monitoring. Added in Phase 0 because there was no
    #: key on which to relate two runs of the same panel, and retrofitting one is exactly
    #: the migration the Build Plan note exists to avoid.
    previous_job_id: str | None = None
    started_at: str
    finished_at: str
    seed: int


class RunResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    job_id: str
    #: Carried through because XLF_Diagnostics block 4 (series flags) is built from it, and
    #: so is ArtifactPack.series_flags in Phase 6. Also the record of which series were
    #: excluded and why -- dropping a series without telling the user is a listed failure
    #: mode (FS §6), so the reasons must survive to the output layer.
    profile: DataProfile
    leaderboard: Leaderboard
    forecast: ForecastFrame
    fold_scores: list[FoldScore] = Field(default_factory=list)
    bands: list[ConformalBands] = Field(default_factory=list)
    calibration: list[CalibrationRow] = Field(default_factory=list)
    timing: RunTiming
    artifacts: ArtifactPack
    manifest: Manifest
