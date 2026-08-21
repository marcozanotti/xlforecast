# 02 — Technical Specification

Implements the requirements in `01-FUNCTIONAL-SPEC.md`. Decisions marked **[FIXED]** come from
ADRs in `00-PROJECT-BRIEF.md` and must not be changed without updating that document.

---

## 1. System architecture

```
┌──────────────────────────────────────────────────────────┐
│ Excel  ·  task pane (Office.js webview via xlwings)      │
│  range read/write · chunked · batched context.sync()     │
└───────────────┬──────────────────────────────────────────┘
                │ HTTPS / JSON + Arrow IPC
┌───────────────▼──────────────────────────────────────────┐
│ API  ·  FastAPI                                          │
│  auth · licence · validation · job submit · SSE stream   │
└───┬───────────────────────┬──────────────────────────────┘
    │                       │
    │ enqueue               │ read
┌───▼───────────┐   ┌───────▼──────────────────────────────┐
│ Redis         │   │ Object store (S3/Azure Blob)         │
│ queue + state │   │ inputs · results · artifacts (Parquet)│
└───┬───────────┘   └───────▲──────────────────────────────┘
    │ consume               │ write
┌───▼───────────────────────┴──────────────────────────────┐
│ Workers  ·  arq                                          │
│  xlforecast engine: ingest → CV → fit → conformal →      │
│  ensemble → select → evaluate → artifact pack            │
└───────────────────────────┬──────────────────────────────┘
                            │ tools (read-only)
                    ┌───────▼─────────┐
                    │ LLM provider    │  ← profiles & artifacts only
                    │ (pluggable)     │     never raw observations
                    └─────────────────┘
```

**Trust boundary:** raw panel data never leaves the API/worker/object-store triangle. The LLM sees
only derived profiles and artifact packs. This is what makes the enterprise story viable.

---

## 2. Technology stack

| Layer | Choice | Notes |
|---|---|---|
| Language (all server) | Python 3.11 | Every pinned dep publishes cp311 manylinux wheels. 3.12+ is not blocked by numba any more (see below) but stays out of v1 for lockfile stability |
| Package/env manager | `uv` | Lockfile committed. **The llvmlite friction is real, but it is not where this row used to say it was.** The engine is numba-free: statsforecast 2.0 dropped numba for compiled `coreforecast`, and a core install contains neither `numba` nor `llvmlite` (verified in Phase 0). It enters through the **`explain` extra**: `shap` declares `numba` and `llvmlite` with no lower bound, so the resolver backtracks to `llvmlite 0.36`, which predates cp311 wheels and then attempts to build LLVM from source. Two floors are therefore load-bearing: `statsforecast>=2.0` and `llvmlite>=0.43, numba>=0.60` |
| API | FastAPI + Uvicorn | |
| Validation | Pydantic v2 | Single schema shared by API, CLI, LLM structured output |
| Queue / job state | Redis + `arq` | `arq` over Celery for operational simplicity — **not** for being async-native, which buys nothing here: the engine is CPU-bound compiled code. Engine work runs in a `ProcessPoolExecutor`; the arq task awaits the future. This is required, not optional: `arq.abort_job` cancels an asyncio task, which cannot interrupt a synchronous `coreforecast`/LightGBM call, and moving it to a thread does not help either — `CancelledError` will not stop it. FR-802 cancellation terminates the worker **process**; FR-801 resume works from per-fold checkpoints |
| Dataframes | Polars for ingest/reshape, pandas at the Nixtla boundary | Nixtla APIs expect pandas; convert once at the edge |
| Local models | `statsforecast` | |
| Global **and local** ML models | `mlforecast` + LightGBM, XGBoost, scikit-learn `LinearRegression` | One adapter, parameterised by information set: fitted once across the panel (`GlobalX`) or once per `unique_id` (`LocalX`), with an identical feature recipe either way (FR-203a/b). **Not** `statsforecast.SklearnModel` — it does no lag or date feature engineering and needs future exogenous values at predict time |
| Evaluation | `utilsforecast` | `utilsforecast.evaluation.evaluate` + `utilsforecast.losses`. Note there is **no `crps`** in that module — the probabilistic losses are `scaled_crps`, `mqloss`, `scaled_mqloss`, `quantile_loss`, `coverage`, `calibration`. See §5.6 |
| Features / attribution | `shap` (TreeExplainer) | LightGBM path only |
| Calendars | `holidays` | |
| Storage format | Parquet (pyarrow) | |
| Excel add-in | xlwings PRO (Server) | **[FIXED]** ADR-002 |
| Task pane UI | HTML + Alpine.js + Bootstrap | Minimal JS; xlwings renders the bridge |
| LLM client | `pydantic-ai` or raw provider SDK behind an interface | Must support tool use + structured output |
| CLI | Typer | |
| Testing | pytest, pytest-asyncio, hypothesis | |
| Lint/format | ruff, mypy (strict on `engine/`) | |
| Container | Docker, multi-stage | |
| Deploy | API: Azure Container Apps or Google Cloud Run. **Worker: Container Apps (KEDA Redis scaler) or Cloud Run Jobs** | Spiky load; do not run always-on VMs. The two platforms are *not* interchangeable for the worker: Cloud Run scales on inbound requests, and a Redis-polling arq worker has none, so it needs `min-instances >= 1` and never reaches zero. Container Apps scales to zero on queue length via KEDA. On GCP the equivalent requires Cloud Run Jobs or a Pub/Sub push dispatcher — a different job-dispatch design, not a config flag |

---

## 3. Repository layout

```
xlforecast/
├── pyproject.toml
├── uv.lock
├── CLAUDE.md
├── docs/                       # these documents
├── src/xlforecast/
│   ├── schemas/                # Pydantic contracts — the spine of the system
│   │   ├── enums.py            # ModelName, Family, InformationSet, Scope, Quantity, ExclusionReason
│   │   ├── freq.py             # to_offset-backed normalisation + validation (FR-112)
│   │   ├── request.py          # ForecastRequest, DataMapping, ExogSpec
│   │   ├── results.py          # LeaderboardRow, Leaderboard, ForecastRow, ForecastFrame,
│   │   │                       #   FoldScore, ConformalBands, RunResult, Manifest
│   │   ├── artifacts.py        # ArtifactPack and members
│   │   └── profile.py          # SeriesProfile, DataProfile, ValidationReport
│   ├── ingest/
│   │   ├── readers.py          # csv/parquet/arrow → canonical panel
│   │   ├── validate.py         # FR-105 rules, per-series exclusion reasons
│   │   └── profile.py          # DataProfile computation
│   ├── engine/
│   │   ├── folds.py            # single source of truth for CV cutoffs
│   │   ├── registry.py         # model registry + licence metadata
│   │   ├── local.py            # statsforecast adapter (local statistical models)
│   │   ├── ml.py               # mlforecast adapter, parameterised by information set:
│   │   │                       #   panel-wide (GlobalX) or per-series (LocalX), same features
│   │   ├── conformal.py        # calibration layer
│   │   ├── ensemble.py         # median / trimmed / inv-error / best-k
│   │   ├── select.py           # pooled / per_series / clustered
│   │   ├── evaluate.py         # utilsforecast wrapper, metric assembly
│   │   └── run.py              # orchestrator: spec → RunResult
│   ├── explain/
│   │   ├── detect.py           # peak/drop detection
│   │   ├── decompose.py        # ETS/ARIMA components
│   │   ├── attribute.py        # SHAP over mlforecast features
│   │   ├── calendar.py         # holiday/seasonal-index matching
│   │   ├── analogues.py        # same-period-last-year, seasonal naive
│   │   └── pack.py             # assembles ArtifactPack
│   ├── llm/
│   │   ├── redact.py           # payload builder — the only path to an LLM request (hard rule 4)
│   │   ├── provider.py         # interface + OpenAI/Azure/Bedrock impls
│   │   ├── parse.py            # NL → ForecastRequest (single call)
│   │   ├── narrate.py          # artifact-grounded explanation agent
│   │   ├── tools.py            # read-only tool definitions
│   │   ├── guardrail.py        # numeric verification
│   │   └── prompts/            # versioned .md prompt templates
│   ├── api/
│   │   ├── main.py
│   │   ├── routes/             # jobs, explain, data, licence
│   │   ├── deps.py             # auth, quota, licence
│   │   └── sse.py
│   ├── worker/
│   │   ├── tasks.py            # arq task definitions
│   │   └── progress.py         # granular progress reporting
│   ├── storage/                # object store + Redis abstractions
│   └── cli.py
├── addin/                      # xlwings Server app
│   ├── app/
│   │   ├── main.py             # xlwings script entry points
│   │   ├── sheets.py           # sheet writers (FR-701..704)
│   │   ├── ranges.py           # chunked read/write
│   │   └── taskpane/           # HTML/CSS/Alpine templates
│   └── manifest.xml
├── tests/
│   ├── unit/
│   ├── integration/
│   ├── golden/                 # reproducibility fixtures
│   └── llm/                    # guardrail regression suite
└── benchmarks/                 # M3 validation harness (gate G3)
    ├── tsf.py                  # Monash .tsf reader
    ├── m3.py                   # single-origin runner + baseline comparison
    └── baselines/              # committed published results + tolerances
```

---

## 4. Core data contracts

Write these first. Everything else depends on them. **This section is normative and complete** —
Phase 0's gate is "all Pydantic schemas from §4, complete, with unit tests", so anything referenced
elsewhere in this document must be defined here. The original §4 defined `DataMapping`,
`ForecastRequest`, `LeaderboardRow` and `Manifest` only, while §3, §5 and §9 referenced
`DataProfile`, `ExogSpec`, `Leaderboard`, `ForecastFrame`, `ConformalBands`,
`ForecastWithIntervals`, `RunResult` and `Tool` — none of which existed anywhere.

### 4.0 Cross-cutting conventions

- **Model config.** All contracts set `model_config = ConfigDict(extra="forbid")`.
  `ForecastRequest` and `Manifest` are additionally `frozen=True`: a manifest-embedded request must
  not be mutable after the fact. Frozen models cannot use mutating `mode="after"` validators — see
  the FR-204 note in §4.2.
- **Missing metrics are `None`, never `NaN`.** FR-214 makes zero-denominator metrics routine.
  Measured (§05 spike): Pydantic serialises a `NaN` float to `null` without error, and that `null`
  then **fails re-validation** into a bare `float` with `float_type`. The break is on the return
  leg, not on serialisation — so `NaN` in a `float` field is a *silent* corruption that only
  surfaces on replay. Every metric field is `float | None`.
- **Integer-keyed dicts.** `dict[int, float]` (coverage by level) serialises to JSON with **string**
  keys. Measured: it round-trips correctly under **both lax and strict** validation on Pydantic
  2.13.4 — an earlier draft of this section warned against `strict=True` here, which was wrong.
  G0's "round-trips without loss" is defined as
  `T.model_validate(json.loads(m.model_dump_json())) == m`, and the round-trip test asserts that.
- **Timestamps** are ISO-8601 UTC strings at every contract boundary. Engine-internal code may hold
  `datetime`; nothing crossing a boundary does.
- **Canonical panel** (for `data_fingerprint`) is defined in §4.7. Reproducibility depends on it
  being defined, not merely named.

### 4.1 Enums and frequency — `schemas/enums.py`, `schemas/freq.py`

```python
Family = Literal["local", "global", "ensemble"]
InformationSet = Literal["own_series", "panel"]  # FR-207 — separate axis from Family
Scope = Literal["panel", "series"]
Quantity = Literal["point", "lo", "hi"]
IntermittencyClass = Literal["smooth", "intermittent", "erratic", "lumpy"]  # Syntetos-Boylan


class ExclusionReason(StrEnum):  # FR-105, one member per rejection rule
    DUPLICATE_TIMESTAMPS = "duplicate_timestamps"
    NON_MONOTONIC = "non_monotonic_timestamps"
    TOO_SHORT = "insufficient_observations"
    ALL_ZERO = "all_zero"
    ALL_CONSTANT = "all_constant"
    EXCESS_MISSING = "excess_missing"
    UNPARSEABLE_DS = "unparseable_timestamps"
    FREQ_MISMATCH = "frequency_mismatch"  # AC-101: mixed-frequency rejection
    MISSING_FUTURE_EXOG = "missing_future_exog"  # FR-111


# ModelName is NOT a closed Literal — see §5.2. It is a str validated against the registry,
# so that the commercial_ok gate is reachable and v2 model additions are not a schema change.
ModelName = Annotated[str, AfterValidator(validate_model_name)]


def normalise_freq(value: str) -> str:  # FR-112
    """Map legacy pandas aliases (M/Q/Y/H/T) to >=2.2 form (ME/QE/YE/h/min) and validate
    via pandas.tseries.frequencies.to_offset. Raises InvalidFrequency otherwise."""
```

### 4.2 Request — `schemas/request.py`

```python
class ExogSpec(BaseModel):  # FR-102 — was referenced, never defined
    name: str
    kind: Literal["historic", "future_known"]
    dtype: Literal["float", "int", "category", "bool"]
    fill: Literal["none", "zero", "ffill", "interpolate"] = "none"


class DataMapping(BaseModel):
    unique_id_col: str
    ds_col: str
    y_col: str
    exog: list[ExogSpec] = []  # replaces the two bare list[str] fields


class ForecastRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    h: Annotated[int, Field(ge=1, le=520)]
    freq: Annotated[str, AfterValidator(normalise_freq)]  # FR-112, FR-505
    season_length: int | None = None  # inferred in profiling if None
    models: list[ModelName] = Field(default_factory=default_model_set)  # FR-216
    levels: list[Annotated[int, Field(ge=1, le=99)]] = [80, 95]
    n_windows: Annotated[int, Field(ge=1, le=20)] = 3
    step_size: int | None = None  # resolved to h
    selection: Literal["pooled", "per_series", "clustered"] = "pooled"
    ensemble: Literal["none", "median", "trimmed_mean", "inverse_error", "best_k"] = "median"
    ensemble_prob_method: Literal["vincentization", "linear_pool"] = "vincentization"
    ensemble_metric: Literal["mase", "rmsse", "scaled_crps"] = "mase"  # FR-403
    ensemble_trim: Annotated[float, Field(ge=0.0, lt=0.5)] = 0.2  # FR-403
    best_k: Annotated[int, Field(ge=1)] = 3
    gap_fill: Literal["none", "zero", "interpolate"] = "none"
    conformal: bool = True
    seed: int = 42
    assumptions: list[str] = []  # informational only

    @field_validator("models")
    @classmethod
    def _seasonal_naive_always(cls, v: list[str]) -> list[str]:
        # FR-204 — not negotiable. A field_validator, not a mutating model_validator:
        # the original assigned to self.models inside mode="after", which recurses forever
        # under validate_assignment and raises outright under frozen=True (which we now set).
        # Sorted + deduped so that `models` order cannot influence best_k tie-breaks,
        # which would otherwise put NFR-02 byte-identity at the mercy of list order.
        return sorted({*v, "SeasonalNaive"})

    @model_validator(mode="after")
    def _ensemble_needs_folds(self) -> "ForecastRequest":
        # FR-405a — LOFO weight estimation is impossible with a single window.
        if self.ensemble in ("inverse_error", "best_k") and self.n_windows < 2:
            raise EnsembleNeedsFolds(self.ensemble, self.n_windows)
        return self
```

`ResolvedRequest` is a `ForecastRequest` with every inferable field filled (`season_length`,
`step_size`, expanded `models`, AutoARIMA mode per FR-201a). **The manifest stores the resolved
form**, not the user's — replay must not re-run inference (§4.7).

### 4.3 Profile — `schemas/profile.py`  *(FR-502; this is the LLM's only input)*

```python
class SeriesProfile(BaseModel):
    unique_id: str
    n_obs: int
    start: str
    end: str
    n_missing: int
    pct_missing: float
    zero_share: float
    intermittency: IntermittencyClass  # computed pre-gap-fill (FR-106)
    adi: float
    cv2: float  # Syntetos-Boylan inputs
    seasonality_strength: float | None
    trend_strength: float | None
    short_history: bool


class ValidationReport(BaseModel):
    n_series_in: int
    n_series_out: int
    excluded: dict[str, ExclusionReason]  # unique_id -> reason (FR-105, never silent)
    excluded_detail: dict[str, str]  # unique_id -> human sentence for the UI


class DataProfile(BaseModel):
    """Derived only. Contains NO observations. This is what crosses the LLM trust boundary."""

    data_id: str
    n_series: int
    freq_inferred: str
    freq_confidence: float
    ds_min: str
    ds_max: str
    ragged: bool  # do series end on different dates? (FR-206a)
    season_length_candidates: list[int]
    intermittent_share: float
    pct_missing_overall: float
    exog_available: list[ExogSpec]
    series: list[SeriesProfile]  # capped; large panels send an aggregate summary
    validation: ValidationReport
```

### 4.4 Results — `schemas/results.py`

```python
class FoldScore(BaseModel):
    fold_index: int  # G2 asserts on this; nothing carried it before
    cutoff: str  # panel-wide calendar date (FR-206)
    model: str
    unique_id: str | None
    n_train_rows: int  # per-family, for the AC-206 dropna audit
    metrics: dict[str, float | None]


class LeaderboardRow(BaseModel):
    scope: Scope
    unique_id: str | None
    model: str
    family: Family  # split from information_set per FR-207
    information_set: InformationSet
    mase: float | None
    rmsse: float | None
    mae: float | None
    rmse: float | None
    smape: float | None
    scaled_crps: float | None  # named for what utilsforecast computes
    coverage: dict[int, float]  # out-of-calibration, single figure (FR-303)
    n_folds: int
    n_series_scored: int  # FR-209
    n_series_common: int  # FR-215
    rank: int
    vs_baseline_pct: float | None
    selected: bool
    selection_biased: bool  # FR-408
    selected_lofo_score: float | None  # FR-408 unbiased companion


class Leaderboard(BaseModel):
    rows: list[LeaderboardRow]
    aggregation: Literal["mean", "median"] = "mean"  # FR-209, must be stated
    baseline_model: str = "SeasonalNaive"
    any_beat_baseline: bool  # FR-406


class ForecastRow(BaseModel):
    unique_id: str
    ds: str
    model: str
    quantity: Quantity
    level: int | None
    value: float


class ForecastFrame(BaseModel):
    rows: list[ForecastRow]
    levels: list[int]


class ConformalBands(BaseModel):
    level: int
    half_width: dict[str, float]  # unique_id -> additive half-width
    pooled_fallback: set[str]  # series that used panel residuals (§5.4)
    clipped: dict[str, float]  # unique_id -> clip rate (FR-307)
    calibrated_from_folds: list[int]  # FR-302 — which folds; empty set is a bug


class ModelTiming(BaseModel):
    """FR-217. The cost proxy for NFR-01. Never merged into LeaderboardRow — see NFR-02."""

    model: str
    fold_index: int | None  # None = the final refit on full history
    train_cpu_seconds: float  # summed across worker processes
    predict_cpu_seconds: float
    train_wall_seconds: float
    predict_wall_seconds: float
    n_series_fitted: int  # 1 for a global fit; N for a local family
    n_rows_trained: int


class RunTiming(BaseModel):
    per_model: list[ModelTiming]
    overhead_cpu_seconds: dict[str, float]  # ingest, profile, validate, folds, conformal,
    # ensemble, evaluate, artifacts, persist
    total_wall_seconds: float
    n_workers: int
    # Invariant, asserted in tests: sum(per_model) + sum(overhead) accounts for total CPU time
    # within tolerance. Attributing only train+predict silently understates the run and would
    # make the S3 ETA optimistic on exactly the large panels where it matters most.


class RunResult(BaseModel):
    job_id: str
    leaderboard: Leaderboard
    forecast: ForecastFrame
    fold_scores: list[FoldScore]
    bands: list[ConformalBands]
    timing: RunTiming
    artifacts: "ArtifactPack"
    manifest: "Manifest"
```

### 4.5 Manifest — `schemas/results.py`

```python
class Manifest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    job_id: str
    engine_version: str
    package_versions: dict[str, str]  # incl. numpy, pandas, coreforecast, lightgbm
    python_version: str
    request: ForecastRequest  # the RESOLVED form (§4.2)
    mapping: DataMapping
    data_id: str
    data_fingerprint: str  # sha256 of the canonical panel (§4.7)
    cutoffs: list[str]  # panel-wide calendar dates, one per fold
    excluded_series: dict[str, ExclusionReason]
    autoarima_mode: Literal["seasonal", "fourier"]  # FR-201a changes the fitted model
    ets_mode: Literal["seasonal", "mstl"]  # FR-201c likewise
    crps_quantiles: list[float]  # FR-208 — CRPS is grid-dependent
    ensemble_params: dict[str, float | int | str]  # FR-403, was unrecorded
    prompt_versions: dict[str, str]  # §9.5 required this and it was absent
    thread_config: dict[str, int | str]  # OMP/MKL/n_jobs/BLAS vendor — NFR-02 needs it
    previous_job_id: str | None  # lineage for v2 stability monitoring
    started_at: str
    finished_at: str
    seed: int
```

**Reproducibility rule:** `Manifest` must contain everything needed to reproduce the run
bit-for-bit. A golden test re-runs from a manifest and asserts leaderboard equality. Note what this
now includes and previously did not: the resolved request, the thread configuration (without which
"byte-identical" is not achievable — see NFR-02), the CRPS quantile grid, the ensemble parameters,
the AutoARIMA mode, and the prompt versions.

`previous_job_id` exists because the Build Plan requires Phase 1 persistence to support v2
forecast-stability monitoring "without a migration", and there was no key on which to join two runs
of the same panel.

### 4.6 Errors — `errors.py`

Every domain error subclasses `XLForecastError` and carries the offending `unique_id` and/or column
so the UI can name it (§4 error-presentation rule in the Functional Spec). No bare `except`.

```python
class XLForecastError(Exception):
    def __init__(
        self,
        message: str,
        *,
        unique_id: str | None = None,
        column: str | None = None,
        fix: str | None = None,
    ): ...
```

`fix` is mandatory for anything user-facing: the error-presentation rule requires every error to
state the remedy, not just the fault.

### 4.7 Canonical panel and fingerprint

`data_fingerprint` is the sha256 of the panel serialised as Parquet with: columns ordered
`unique_id, ds, y, *sorted(exog)`; rows sorted by `(unique_id, ds)`; `ds` as UTC microsecond
timestamps; `y` as float64; `unique_id` as UTF-8 string; no index; zstd level 3. The fingerprint is
taken **after** ingestion and gap filling, **before** exclusion — so that a change in
`season_length` inference (which moves the FR-105 threshold) is visible as a leaderboard change
rather than hidden inside the fingerprint. Without this definition NFR-02 is not testable.

## 5. Engine design

### 5.1 Fold generation — `engine/folds.py` **[critical]**

```python
def make_cutoffs(panel, h, n_windows, step_size, freq) -> list[Timestamp]
def split(panel, cutoff, h) -> tuple[Panel, Panel]        # (train, test) — panel-wide
```

Computed **once** per job, as **panel-wide calendar dates**. `engine/run.py` slices train/test with
`split()` and drives each family's `fit`/`predict` per fold. This is the single most important
correctness property in the system (FR-205, FR-206).

**We do not call the libraries' `cross_validation`.** Verified against statsforecast 2.1.1 and
mlforecast 1.1.0:

- `StatsForecast.cross_validation(h, df, n_windows, step_size, test_size, input_size, level,
  fitted, refit, prediction_intervals, id_col, time_col, target_col)`
- `MLForecast.cross_validation(df, n_windows, h, id_col, time_col, target_col, step_size,
  static_features, dropna, keep_last_n, refit, max_horizon, …)`

Neither takes a cutoff argument, so `make_cutoffs` has nothing to hand them. Both derive cutoffs
from each series' **own last timestamp** — `utilsforecast.processing.backtest_splits` computes
`max_dates = groupby(unique_id).ds.max()`, statsforecast walks `range(-test_size, -h+1, step_size)`
per series group. On a ragged panel that gives every series a different fold-1 date, and a global
model trained at one series' cutoff has seen other series' later observations. See FR-206a.

Two further asymmetries the libraries would introduce and our own loop avoids:
`MLForecast.cross_validation` defaults `dropna=True`, discarding the first `max_lag` rows of every
series, and honours `keep_last_n`; statsforecast trains on the full pre-cutoff history. Identical
cutoffs would therefore still mean different training samples (AC-206).

Mandatory test (`tests/unit/test_folds.py::test_identical_test_index_across_families`): on a
deliberately ragged panel, assert the per-fold `(unique_id, ds)` test index is element-wise equal
across local, global and ensemble, and that each fold's cutoff is one calendar date.
**Never skip or xfail this test.**

### 5.2 Model registry — `engine/registry.py`

Each entry declares: constructor, family (`local`/`global`), information set
(`own_series`/`panel`), whether it handles intermittent data, minimum observations required, whether
it supports exogenous variables, licence, and licence `commercial_ok: bool`. Job submission fails if
any requested model has `commercial_ok=False`.

**The registry — not a `Literal` — is the source of truth for `ModelName`** (§4.1). With a closed
`Literal` containing only Apache/MIT models, as originally specified, a non-commercial model could
never be named in a request and the `commercial_ok` gate was unreachable dead code whose test would
have been vacuous. Registry validation also means adding a v2 model is a registry row rather than a
schema change. The gate's test uses an injected fixture model with `commercial_ok=False`.

**Per-model minimum observations interact with comparability.** A model that declines a series is
scored on a different support than one that accepts it; FR-215 defines the common-support rule and
`n_series_common` makes it visible. The registry must never let a model quietly skip series.

**Local ML entries carry their own minimums and their own hyperparameters** (FR-203c). Lag
construction consumes `max_lag` rows per series, so a series at the FR-105 floor leaves roughly 91
usable training rows at weekly frequency — enough for `LocalLinear`, marginal for the boosters.
Registry entries for `LocalLGBM`/`LocalXGB` therefore set a higher `min_obs` and small-data
hyperparameters (low `min_child_samples`/`min_child_weight`, shallow trees, more regularisation);
reusing the global defaults yields near-constant predictions and a leaderboard row that looks like a
bug rather than a finding.

**Matched pairs are a registry-level invariant** (FR-203b): a `LocalX` and its `GlobalX` twin must
resolve to the same learner class, the same feature recipe and the same folds, differing only in
information set. A registry test asserts this for every pair, because the pair comparison is only
meaningful if nothing else drifted.

### 5.3 Execution order — `engine/run.py`

```
ingest → profile → validate → folds
  → for each family: our own fold loop over shared panel-wide cutoffs
  → conformal calibration from CV residuals
  → evaluate (point + probabilistic) via utilsforecast
  → ensemble (competes on the same folds)
  → select
  → refit selected model(s) on full history → final forecast
  → artifact pack
  → persist (Parquet + Manifest + RunTiming)
```

Note the ordering, in two respects.

**Profile precedes validate.** FR-105's length threshold is
`2 × season_length + h + (n_windows − 1) × step_size`, and `season_length` is *inferred during
profiling* when not supplied (FR-104). The original order `ingest → validate → profile` was
circular: validation needed a value profiling had not yet produced. See FR-105a.

**Ensembles are built and scored inside the CV loop**, not from final forecasts — and their weights
are estimated leave-one-fold-out (FR-405a). Scoring an ensemble on different folds than its members
is a listed failure mode (FS §6); estimating its weights on the folds it is scored on is the
subtler one, and it is the one that will actually happen if nobody is watching.

### 5.4 Conformal calibration — `engine/conformal.py`

Absolute residuals from CV folds → per-series (or pooled, if a series has too few residuals)
quantiles → additive interval half-widths at each level, **clipped to the series' observed support**
(FR-307). Expose:

```python
def calibrate(cv_residuals, levels, folds, min_residuals=20) -> ConformalBands
def apply(forecast, bands) -> ForecastFrame
def empirical_coverage(cv_forecasts, actuals, bands) -> dict[int, float]
```

**Cross-conformal (FR-302, ADR-006 amendment).** The band used when *scoring* fold `k` is calibrated
from folds ≠ `k` only; `ConformalBands.calibrated_from_folds` records which. The band *delivered*
with the final forecast uses all folds. `empirical_coverage` therefore evaluates each fold against a
band it did not contribute to, and averages. Calibrating and evaluating on the same residuals — the
original specification — reports nominal coverage by construction and cannot fail.

**Fallback chain, fully defined.** Series with fewer than `min_residuals` residuals fall back to
pooled panel residuals. If the *panel* also has fewer than `min_residuals` (a 5-series monthly panel
at `n_windows=1, h=3` yields 15), the terminal fallback is the model's native interval where it has
one, else a width of `NaN` with the level reported as unavailable — never a silently wrong number.
Every fallback and clip rate is recorded per series in `XLF_Diagnostics`.

### 5.5 Ensembling — `engine/ensemble.py`

Point: median, trimmed mean, inverse-CV-error weights, best-k mean.
Probabilistic: **vincentization** averages quantiles level-by-level; **linear pooling** mixes
predictive distributions then re-extracts quantiles. These give different objects — implement both,
default to vincentization, and record the choice in the manifest (FR-404).

**Weight estimation is leave-one-fold-out (FR-405a).** For `inverse_error` and `best_k`, the weights
or member set used when scoring fold `k` come only from folds ≠ `k`. `median` and `trimmed_mean` fit
nothing and are exempt. Which error is inverted, and which metric ranks `best_k`, is
`request.ensemble_metric` (default pooled MASE) — the original spec named neither, which alone made
ensembles irreproducible. `ensemble_trim` (default 0.2) is likewise now an explicit field.

Edge cases are defined behaviour, not implementation detail (FR-403a): `best_k` with fewer than `k`
members falls back to all and records it; `trimmed_mean` over fewer than 5 members degrades to
`median` and records it; a one-member ensemble is not formed.

### 5.6 Metrics — `engine/evaluate.py`

Wraps `utilsforecast.evaluation.evaluate(df, metrics, models, train_df, level, id_col, time_col,
target_col, cutoff_col, agg_fn)`. Three things the original spec assumed and the library does not
provide:

1. **There is no `crps`.** Use `scaled_crps(df, models: dict[str, list[str]], quantiles: np.ndarray,
   …)`, whose quantile grid comes from `levels` — at `[80, 95]` that is the 5 points
   `{0.025, 0.1, 0.5, 0.9, 0.975}`. The grid goes in the manifest (FR-208): two runs with different
   `levels` produce CRPS values that are not comparable, and a leaderboard that hides that is
   dishonest.
2. **`mase` and `rmsse` require `seasonality` and `train_df`**, and divide by the mean absolute
   seasonal difference of the *training window*. That is zero for a series constant within an early
   fold. `smape` divides by `|y| + |ŷ|`; `scaled_crps` divides by the sum of actuals — both zero on
   an all-zero intermittent window. FR-214 defines the policy: per-series value `None`, excluded
   from that metric's aggregate only, counted in `n_series_scored`, reason in diagnostics. One
   degenerate series must never `NaN` a leaderboard row, which is what naive `mean()` would do.
3. **Aggregation must be nan-safe and stated** (FR-209), and computed over the FR-215 common
   support when models have different support.

---

## 6. API surface

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/v1/data` | Upload panel (Arrow IPC or Parquet), **including any future-known exog rows** (FR-111: `y` null, `ds` beyond last observation). Returns `data_id` + `DataProfile` + validation report. Not subject to NFR-03 — see §10 |
| `POST` | `/v1/parse` | `{data_id, text}` → `ForecastRequest` + `assumptions` + `clarifying_question?`. Budget is NFR-04 (3 s), not NFR-03 |
| `POST` | `/v1/confirm` | `{data_id, request}` → `{confirmation_token}`. Minted only from an explicit user action in S3. Bound to `(data_id, request_hash)`, single-use, 30-minute expiry |
| `POST` | `/v1/jobs` | `{data_id, request, confirmation_token}` → `{job_id}`. Validates, checks quota/licence, **rejects 4xx without a valid token** (AC-503), enqueues |
| `GET` | `/v1/jobs/{id}` | Status, progress, partial leaderboard |
| `GET` | `/v1/jobs/{id}/stream` | SSE: progress + partial results |
| `GET` | `/v1/jobs/{id}/results` | Full result set as Arrow/Parquet |
| `GET` | `/v1/jobs/{id}/manifest` | Reproducibility manifest |
| `DELETE` | `/v1/jobs/{id}` | Cancel |
| `POST` | `/v1/jobs/{id}/explain` | `{scope: panel\|series\|point, unique_id?, ds?, question?}` → streamed prose |
| `GET` | `/v1/licence` | Licence status, quota remaining |

**[FIXED]** There is no synchronous forecast endpoint (ADR-005).

**Streaming and auth.** `EventSource` cannot set headers and cannot issue `POST`, so it can consume
neither an authenticated `/stream` nor `POST /v1/jobs/{id}/explain`. Both stream over `fetch` +
`ReadableStream` instead. Auth is an `HttpOnly; Secure; SameSite=Lax` session cookie — **not** a
token in a workbook custom property, which would travel inside the `.xlsx` to anyone the file is
sent to. Custom properties carry `job_id` and `data_id` only (hard rule 8, §7.3).

---

## 7. Excel add-in

### 7.1 Reading

`addin/app/ranges.py` reads in chunks of 50,000 cells with successive `context.sync()` calls,
converts to Arrow, and streams to `POST /v1/data`. Show progress during read — silence reads as a
freeze. This bounded chunk loop is *required*, and is what hard rule 9 permits: the rule forbids a
sync whose cost scales with cells, not one that scales with chunks.

**Budget honestly.** 500,000 rows × 5 columns is 2.5M cells — 50 chunks. Under native Office.js
those are 50 local syncs. **Under xlwings Server they are 50 HTTPS round trips to our server**,
because routing Office.js through Python is what xlwings Server *is*. The original "tens of seconds"
estimate holds only for the native case; the xlwings case is minutes and is the subject of the
ADR-002 Phase 5 spike. Do not write the sheet writers before that spike reports.

Hard cap: 500,000 rows (FR-107). Above that, refuse with a message pointing at file-based input.
Note this caps **input** only — output overflow is a separate check (FR-708, §7.2).

### 7.2 Writing

`addin/app/sheets.py` writes each output sheet as one or a few batched range assignments. Never
per-cell. Sheets are created if absent, cleared and rewritten if present (after confirmation).
The manifest goes to a hidden `XLF_Manifest` sheet as JSON.

**All five sheets are written as one transaction, manifest included** (FR-701, FR-703). Under the
original "four sheets" wording the manifest sat outside the overwrite set, so a re-run left a
manifest describing the *previous* run beside the new results — breaking FR-704 and hard rule 10 on
the second run of every workbook.

**Output size is prechecked before anything is written** (FR-708). `XLF_Forecast_Long` is
`series × h × models × (1 + 2·levels)` rows; a 2,000-series panel at `h=52` with 9 models and 2
levels is ≈4.7M rows against Excel's 1,048,576 limit. The input cap does not bound the output.
On overflow, refuse and offer the two named degradations rather than truncating.

Workbook-state edge cases (protected sheet, co-authoring, referring formulas, foreign manifest) are
FR-703a and are handled before the first write, not discovered during it.

### 7.3 Polling

Pane polls `GET /v1/jobs/{id}` every 2s, or consumes `/stream` via `fetch` + `ReadableStream`
(**not** `EventSource` — see §6). On completion, fetch results and write sheets. Job state must
survive a pane reload — persist the active `job_id` and `data_id` in the workbook's custom
properties so a reopened pane reattaches. **Never a credential:** custom properties are part of the
file and travel with it when the workbook is emailed or synced to SharePoint.

---

## 8. Artifact pack

Computed deterministically by `explain/pack.py` at the end of every job. This is the only thing the
explanation LLM may read.

**On hard rule 4 and observed values.** Several fields below *are* observed values —
`Analogue.same_period_last_year`, `PeakDrop.value`, `CalendarContext.exog_flags`. That is
deliberate and necessary: FR-604 and the P1 journey require citing last year's value at the same
calendar position, and no artifact pack can ground an explanation without individual numbers. Rule 4
prohibits **bulk panel data** — series, arrays, slices, or any path handing the model unbounded `y`
— not named, per-point, artifact-mediated scalars. `llm/redact.py` is the only path to a provider
request; it builds payloads exclusively from artifact objects and caps them at
`MAX_ARTIFACT_POINTS`. NFR-07 is stated in those terms.

`Attribution` is global-models-only and `Decomposition` local-models-only, so for any given series
exactly one of them exists (FR-601a/b). Narration must not be written as though both are available.

```python
class PeakDrop(BaseModel):
    unique_id: str
    ds: str
    direction: Literal["peak", "drop"]
    value: float
    seasonal_expectation: float
    deviation_pct: float
    z_score: float


class Attribution(BaseModel):  # global models only
    unique_id: str
    ds: str
    horizon_step: int  # 1 = attributions to observed lags
    recursive_path: bool  # True when step > 1: lag features are the model's own
    # predictions, not observations (FR-601a)
    contributions: dict[str, float]  # feature -> SHAP value
    base_value: float


class Decomposition(BaseModel):  # AutoETS only — NOT ARIMA (FR-601b)
    unique_id: str
    ds: str
    ets_form: str  # e.g. "AAA", "MAM", "ANN" — determines which fields exist
    level: float
    trend: float | None
    seasonal: float | None


class CalendarContext(BaseModel):
    unique_id: str
    ds: str
    seasonal_index: float
    period_rank: int  # e.g. "3rd highest of 12 months"
    holidays: list[str]
    exog_flags: dict[str, float]


class Analogue(BaseModel):
    unique_id: str
    ds: str
    same_period_last_year: float | None
    seasonal_naive_value: float
    historical_pctile: float


class UncertaintyContext(BaseModel):
    unique_id: str
    ds: str
    interval_width_80: float
    width_vs_series_mean: float
    empirical_coverage_80: float


class ArtifactPack(BaseModel):
    job_id: str
    peaks_drops: list[PeakDrop]
    attributions: list[Attribution]
    decompositions: list[Decomposition]
    calendar: list[CalendarContext]
    analogues: list[Analogue]
    uncertainty: list[UncertaintyContext]
    leaderboard_summary: dict
    series_flags: dict[str, dict]
```

**Peak/drop detection:** score each forecast point against its seasonal expectation (or a rolling
median of the forecast path where seasonality is weak); flag the top-k by absolute z-score. This
gives both the UI and the agent something specific to discuss instead of inviting speculation.

---

## 9. LLM integration

### 9.1 Provider interface

```python
class LLMProvider(Protocol):
    async def structured(self, prompt: str, schema: type[BaseModel], **kw) -> BaseModel: ...
    async def narrate(self, prompt: str, tools: list[Tool], **kw) -> AsyncIterator[str]: ...
```

Implementations: OpenAI, Azure OpenAI, Bedrock, generic OpenAI-compatible base URL (FR-806).
Configured per deployment; enterprise customers point at their own endpoint.

**All LLM calls originate server-side.** The task pane is a browser and its bundle is inspectable —
no API key ever reaches it.

### 9.2 Parsing (`llm/parse.py`)

Single constrained call. Input: `DataProfile` (~500 tokens) + user text. Output: `ForecastRequest`.
Small/fast model. No tool loop. Result is re-validated against the full schema and business rules —
LLM output is untrusted input (FR-505).

### 9.3 Narration (`llm/narrate.py`)

Tool-using agent over a fixed read-only tool set:

```
get_leaderboard(job_id, scope, unique_id?)
get_peaks_drops(job_id, unique_id?, top_k?)
get_decomposition(job_id, unique_id, ds)
get_attributions(job_id, unique_id, ds)
get_calendar_context(job_id, unique_id, ds)
get_analogues(job_id, unique_id, ds)
get_uncertainty(job_id, unique_id, ds)
get_series_flags(job_id, unique_id)
```

No code execution. No raw data access. No tool that fits, forecasts, or computes a metric.

### 9.4 Numeric guardrail (`llm/guardrail.py`) **[critical]**

**Templating is the primary mechanism, not the complement.** Key figures are rendered by the engine
from templates — `"{model} was selected (scaled CRPS {crps:.3f} vs {baseline_crps:.3f} for seasonal
naive)"` — and the LLM writes only the connective reasoning around them (FR-605a). The regex
guardrail is the backstop for the residue.

After generation, extract every numeral from the prose (integers, decimals, percentages, currency)
and verify each against values present in the artifacts consumed during that turn. Matching rules,
all of which the original 1e-3-relative specification got wrong:

- **Unit-aware.** A prose `78%` matches an artifact `0.78`. Coverage, shares and percentage
  deltas are stored as fractions and written as percentages; a matcher that does not know this
  rejects every correct sentence it sees.
- **Tolerance from displayed precision, not a fixed epsilon.** `"about 78%"` must match `0.7834`;
  `"78.3%"` must match `0.7834` but not `0.79`. A flat 1e-3 relative tolerance rejects ordinary
  rounded prose — the regenerate path would fire on nearly every generation and the strip path
  would degrade nearly every explanation.
- **Exact for counts, years and ordinals.** Series counts, `n_folds`, calendar years.
- **Allowlist**, matched without artifact backing: the requested `levels`, `h`, `n_windows`, series
  counts, and calendar years/months already present in the consumed artifacts. `"the 80% interval"`
  and `"3rd highest of 12 months"` are legitimate numerals whose second figure (`12`) is not any
  artifact *value*; without an allowlist they are false rejections.

Any unmatched numeral → reject and regenerate once; on second failure, strip the offending sentence
and log. **The strip path firing is a defect, not a success.** AC-605 therefore measures the
*pre-guardrail* generation — first-pass rejection rate below 5%, strip path zero — because
post-guardrail output has zero unmatched numerals by construction and would pass even if every
generation hallucinated.

Causal phrasing is linted separately (FR-606a); the guardrail checks numbers, not claims.

### 9.5 Prompt discipline

Prompts live in `llm/prompts/*.md`, versioned, with the version recorded in the manifest. The
narration system prompt must state explicitly: report attributions within the fitted model, never
causal claims about the world (FR-606).

---

## 10. Non-functional requirements

| ID | Requirement |
|---|---|
| NFR-01 | 200 series × 156 weekly observations × 13 models (FR-216) × 3 folds completes in **< 10 min wall-clock** on 8 vCPUs. **The cost proxy is measured total compute time — `train + predict` — never fit counts, series counts or model counts.** Reported per model (FR-217). Model count is not a unit of work: a global model is one fit per fold over the whole panel while a local model is one fit per series per fold, and `SeasonalNaive` and `AutoARIMA` differ by orders of magnitude at identical fit counts. **Measured in the Phase 0 spike (docs/05): 1.9 min ideal, 2.9 min with a 50% overhead allowance — roughly 3× headroom, so NFR-01 passes and FR-216 is frozen.** Cost is concentrated in three models: `LocalLGBM` 54%, Fourier `AutoARIMA` 26%, `AutoETS` 12%; all three global models together are 0.5%. Re-measured against real M5/VN1 panels at Phase 3 |
| NFR-02 | Timing is measured, so it is **not reproducible** and must never enter the leaderboard: `RunTiming` is a separate object (§4.4) written to `XLF_Diagnostics`, and `Leaderboard` carries no duration column. Putting FR-217's per-model times into `LeaderboardRow` would make byte-identity unachievable by construction. Given that: two identical `(data_fingerprint, ResolvedRequest, thread_config)` triples produce byte-identical leaderboards, where "canonical panel", "byte-identical" and `thread_config` are as defined in §4.7 and §4.5. Thread count is part of the key because float reductions reorder under it, LightGBM is only deterministic with `deterministic=true, force_row_wise=true` at a fixed thread count, and BLAS thread count moves the ARIMA path. Stating NFR-02 without it, alongside FR-211's configurable parallelism and NFR-01's 8-vCPU target, made the three requirements jointly unsatisfiable — and golden tests would have flaked on any new CI runner |
| NFR-03 | API p95 latency < 300 ms for `GET /v1/jobs/{id}`, `GET /v1/licence`, `GET /v1/jobs/{id}/manifest`, `DELETE /v1/jobs/{id}`. **Explicitly excluded:** `POST /v1/data` (ingests, profiles and validates up to 500k rows — budgeted at 60 s p95 for a full-cap panel, scaling linearly), `POST /v1/parse` (NFR-04), `POST /v1/jobs/{id}/explain` (NFR-05), `GET /v1/jobs/{id}/results`. "All non-job endpoints" contradicted both §6 and NFR-04 |
| NFR-04 | Parse call returns in < 3 s p95 |
| NFR-05 | First explanation token streams within 2 s |
| NFR-06 | Workers scale to zero when idle |
| NFR-07 | No bulk panel data in any LLM request payload. Enforced structurally: `llm/redact.py` is the only code path that constructs a provider request, it accepts `DataProfile` and `ArtifactPack` objects only, and it caps per-point values at `MAX_ARTIFACT_POINTS`. Tested by asserting no provider call can be made with a panel-shaped argument. Named per-point artifact values are permitted — see §8 and hard rule 4 |
| NFR-08 | Panel data at rest is encrypted and deleted after a configurable retention window (default 30 days) |
| NFR-09 | Structured JSON logging with `job_id` correlation throughout; OpenTelemetry traces |
| NFR-10 | Engine achieves ≥ 85% line coverage; `engine/folds.py`, `engine/conformal.py`, `llm/guardrail.py` at 100% |
| NFR-12 | `POST /v1/data` and the engine reject panels whose output would exceed Excel's row limit before compute is spent, not after (FR-708) |
| NFR-11 | `mypy --strict` passes on `src/xlforecast/engine/` and `src/xlforecast/schemas/` |

---

## 11. Testing strategy

**Unit.** Schema validation, fold generation, conformal maths, ensemble maths, selection rules,
guardrail extraction. Property-based tests (hypothesis) for fold generation across frequencies and
for conformal band monotonicity in level. Schema round-trip is a property test over every contract:
`T.model_validate(json.loads(m.model_dump_json())) == m` — which is the operative definition of
G0's "without loss", and which pins the `dict[int, float]` and `float | None` decisions in §4.0.

**Golden / reproducibility.** Fixed synthetic panels with committed expected leaderboards. Any diff
fails CI. Plus a manifest-replay test: re-run from a stored manifest, assert identity.

**Statistical correctness.** Synthetic panels with known DGPs:
- Pure seasonal → AutoETS/SeasonalNaive should win; a global LGBM should not beat them materially.
- **Seasonal** random walk (`y_t = y_{t−m} + ε_t`) → nothing beats SeasonalNaive; assert the
  "no model beat baseline" path fires (AC-406).
- **Pure** random walk (`y_t = y_{t−1} + ε_t`) → AutoARIMA *does* beat SeasonalNaive and the notice
  does **not** fire (AC-406a). The original spec used this DGP for the previous assertion, which
  would have failed against a correct engine: for `h ≤ m`, seasonal-naive error variance is ≈`m·σ²`
  against `h·σ²` for naive, and AutoARIMA selects ARIMA(0,1,0). Gate G1 depended on it.
- Known noise distribution → **out-of-calibration** conformal coverage within ±5pp of nominal, with
  the in-calibration figure computed as a control (AC-301).
- Ragged panel → per-fold test index identical across families, cutoffs are single calendar dates
  (AC-205).
- Degenerate metrics → a series constant within an early training fold, and an all-zero evaluation
  window, both yield `None` rather than `NaN`/`inf`, and neither poisons the panel aggregate
  (FR-214).

**Benchmark.** `benchmarks/` runs the M3 competition data and records leaderboards **and
per-model `RunTiming`** over time, so both accuracy and performance regressions are visible — a
change that quietly triples `AutoETS` train time should be as loud as one that moves its MASE.
Not in CI (too slow) — nightly or on demand.

The M3 harness runs in **comparability mode**: a single forecast origin with the last `horizon`
observations held out, matching the Monash archive's protocol. It therefore bypasses
`engine/run.py` and drives the adapters directly, because comparing our 3-fold cross-validated
average against their single holdout and calling the difference a result would be meaningless.
The fold machinery, conformal layer and ensembling are covered by G1 and G2 instead.

**LLM.** 100-case explanation regression suite measured **pre-guardrail**: first-pass rejection rate
< 5%, strip path fires zero times, post-guardrail unmatched numerals zero (AC-605). Includes a
labelled adversarial slice for the FR-606a causal-phrasing lint. Parser suite of ~60 phrasings
(EN + IT) with expected `ForecastRequest` fields, including legacy pandas frequency aliases
(`M`, `Q`, `H`), which the parser will emit because its training data is full of them (FR-112).
Use a recorded-response cache so CI is deterministic and free.

**Add-in.** Manual smoke checklist per release across Excel for Windows, Mac and web — Office.js
behaviour differs across hosts and this cannot be fully automated.

---

## 12. Security and compliance notes

- Add-in manifest must pass Microsoft Marketplace certification policies; budget 14–28 business days
  per submission and batch changes accordingly.
- Marketplace add-ins are free-to-download only — licensing and billing are built in-house
  (Stripe + a licence service). Do not architect around marketplace transactability.
- Provide a self-hosted deployment mode (Docker Compose + Helm chart) for customers who cannot let
  data leave their tenant. This is the answer to the enterprise objection, and it is also a pricing
  tier.
- Secrets via environment/Key Vault. No secrets in the add-in bundle, ever.
