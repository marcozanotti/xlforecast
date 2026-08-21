# 01 — Functional Specification

Requirement IDs are stable. Reference them in commits, tests and PRs (e.g. `feat(engine): FR-210`).
Priority uses MoSCoW: **M**ust / **S**hould / **C**ould / **W**on't-in-v1.

---

## 1. Personas

**P1 — Demand planner (primary).** Mid-market consumer goods or e-commerce. Owns 50–2,000 SKUs.
Lives in Excel. Understands seasonality and MAPE; does not know what CRPS is. Currently forecasts
with a moving average and manual overrides. Cannot get budget for a planning suite. Judges the tool
on whether it beats what they already do, and on whether they can explain the number in the Monday
meeting.

**P2 — FP&A analyst (secondary).** 5–50 series (revenue by region/product line). Monthly cadence.
Needs defensible numbers and a written rationale for a board pack. Cares more about explanation
quality than about accuracy at the third decimal.

**P3 — Analytics-capable user (tertiary).** Knows Python or R. Uses the CLI directly, treats the
add-in as a convenience for colleagues. Will read the methodology page and will find any dishonesty
in the leaderboard. Source of credibility and word-of-mouth — do not disappoint them.

---

## 2. Primary user journey (P1, happy path)

1. User has a sheet: `sku | week | units` for 300 SKUs.
2. Opens the task pane, selects the range, maps columns to `unique_id` / `ds` / `y`.
3. Types: *"forecast the next 3 months, weekly"*.
4. Pane shows a parsed configuration card and the assumptions it made. User confirms.
5. Job runs. Progress bar shows model-by-model completion.
6. Four sheets are written. Pane shows: winning model, its CRPS, the seasonal-naive CRPS, and
   whether the competition beat the baseline.
7. User selects a forecast cell that looks odd and clicks **Explain this cell**.
8. Pane returns a grounded paragraph citing the seasonal index, last year's value at the same
   calendar position, the interval width, and — **whichever applies to the model that won that
   series** — either its fitted component decomposition (local models) or its feature attributions
   (global models). These two are mutually exclusive by construction: `Decomposition` exists only
   for local models and `Attribution` only for global ones. The pane must not promise both.

---

## 3. Feature requirements

### 3.1 Data ingestion (FR-1xx)

| ID | Pri | Requirement |
|---|---|---|
| FR-101 | M | Accept long-format input with columns `unique_id` (str), `ds` (date/datetime), `y` (float). Column names are user-mappable. |
| FR-102 | M | Accept optional exogenous columns, classified by the user as *historic-only* or *future-known*. Classification is per column and carries dtype + fill policy (`ExogSpec`), not a bare name list. |
| FR-103 | M | Accept input from: (a) an Excel range/table, (b) a local CSV or Parquet file via the CLI. |
| FR-104 | M | Infer frequency per panel from `ds`; report the inference and allow override. |
| FR-105 | M | Reject and report, per series: duplicate timestamps, non-monotonic timestamps, all-zero or all-constant series, >50% missing, and insufficient length. **Length threshold is `2 × season_length + h + (n_windows − 1) × step_size`** — the earliest CV training window must itself satisfy `2 × season_length`. The naive `2 × season_length + h` admits series that pass ingestion and then vanish inside cross-validation, which violates §6. With weekly defaults (`season_length=52, h=13, n_windows=3, step_size=h`) the threshold is 143 observations, not 117. |
| FR-105a | M | Validation runs **after** profiling, not before. `season_length` is inferred during profiling when not supplied (FR-104), and FR-105's threshold depends on it — the pipeline order in TS §5.3 is `ingest → profile → validate → folds`. Series excluded during profiling (unparseable `ds`, empty) are reported with the same per-series reason mechanism. |
| FR-106 | M | Support gap filling for irregular panels: `none` \| `zero` \| `interpolate`, with the choice recorded in the run manifest. **Intermittency classification (FR-108) runs on the pre-fill series**, because `gap_fill=zero` manufactures intermittency and would otherwise route smooth-but-gappy series to Croston. Both the pre-fill class and the post-fill zero share are recorded in series flags. |
| FR-107 | M | Hard-cap grid ingestion at 500,000 rows; above that, refuse with a message directing the user to file-based input. |
| FR-108 | S | Detect and label intermittent series (Syntetos–Boylan classification) and route them to intermittent-appropriate models. |
| FR-109 | S | Auto-detect wide-format input and offer to unpivot it. |
| FR-110 | C | Read directly from a database connection string. |
| FR-111 | M | **Future values for future-known exogenous columns.** A future-known column is unusable without its values over the horizon. They are supplied as additional rows in the same panel — `unique_id`, `ds` beyond the last observation, `y` null, exog populated — following the Nixtla `X_df` convention. Ingestion splits them out. Reject with a named error if a column is classified future-known and any series lacks exactly `h` future rows, or if any future row carries a non-null `y`. |
| FR-112 | M | **Frequency normalisation.** `freq` is normalised through `pandas.tseries.frequencies.to_offset` and stored in canonical form. Legacy pandas aliases (`M`, `Q`, `Y`, `H`, `T`) are accepted on input, mapped to their pandas ≥2.2 equivalents (`ME`, `QE`, `YE`, `h`, `min`), and the normalised alias is what reaches the manifest. Unparseable aliases are rejected at the schema boundary (see FR-505). |

**AC-101:** Given a range with 300 series at a single frequency and 12 malformed series covering
every FR-105 rejection reason at least once, ingestion completes, the 12 are excluded each with its
own named reason, and the remaining 288 proceed.

*Amended:* the original wording required "mixed frequencies". The architecture does not support
them — FR-104 infers frequency **per panel**, `ForecastRequest.freq` is a single scalar, and one
cutoff set spans the panel. A genuinely mixed-frequency range is an ingestion **rejection** with a
named error, not a supported case. That rejection is now part of this AC.

### 3.2 Model competition (FR-2xx)

| ID | Pri | Requirement |
|---|---|---|
| FR-201 | M | Provide local statistical models: `SeasonalNaive`, `HistoricAverage`, `WindowAverage`, `AutoARIMA`, `AutoETS`, `AutoCES`, `DynamicOptimizedTheta`. |
| FR-201b | M | **`WindowAverage` runs at the seasonal frequency** — `WindowAverage(window_size=season_length)`, a trailing mean over the last full seasonal cycle. This is the incumbent method: persona P1 "currently forecasts with a moving average", so it is the thing the product must actually beat to be worth buying. It is a baseline, not a candidate — reported with the same prominence as `SeasonalNaive` in S5, and `vs_baseline_pct` is computed against both. Note this is *not* `SeasonalWindowAverage`, which averages the same seasonal position across previous cycles; if that is wanted it is a separate registry entry. |
| FR-201a | M | **AutoARIMA runs non-seasonal with Fourier regressors** whenever `season_length > 24`. Measured (docs/05): seasonal `m=52` costs 0.923 CPU s/series against 0.297 s for non-seasonal + Fourier — a real **3.1×** saving, and 26% of total run cost even in Fourier mode. It is also poor practice for long seasonal periods, and Fourier terms are the standard treatment. Honest framing: this is an optimisation, **not** a rescue of NFR-01 — the spike shows even full seasonal AutoARIMA lands at 4.4 min against a 10-min budget. Below the threshold, seasonal AutoARIMA runs normally. The choice is recorded in the manifest — it changes the fitted model and must not be invisible. |
| FR-201c | M | **`AutoETS` runs as MSTL whenever `season_length > 24`.** Discovered in Phase 1: `statsforecast/ets.py` allocates a **fixed 24-slot seasonal state buffer** and returns early when `m > 24`, so `AutoETS(season_length=52)` silently fits a *non-seasonal* model — verified to produce forecasts numerically identical to `AutoETS(season_length=1)`. It then scores badly on seasonal data and the user has no way to know why, which is precisely the misleading leaderboard row this product exists to prevent. Above the threshold, `AutoETS` is constructed as `MSTL(season_length=m, trend_forecaster=AutoETS(model="ZZN"))` — the statsforecast-recommended route for long periods — aliased back to `AutoETS` and recorded as `ets_mode` in the manifest, exactly parallel to FR-201a. Measured on a weekly seasonal panel: MASE **1.123 → 0.765**, i.e. from worse than `SeasonalNaive` to better than it. |
| FR-216 | M | **Default model set — 13 models.** Baselines: `SeasonalNaive` (mandatory, FR-204), `HistoricAverage`, `WindowAverage` at seasonal frequency (FR-201b). Local statistical: `AutoARIMA` (Fourier mode per FR-201a), `AutoETS`, `DynamicOptimizedTheta`. Intermittent: `CrostonClassic`. Local ML: `LocalLinear`, `LocalLGBM`, `LocalXGB`. Global ML: `GlobalLinear`, `GlobalLGBM`, `GlobalXGB`. Opt-in: `AutoCES`, `ADIDA`, `IMAPA`, `ZeroModel`. The schema default was previously the literal `[...]` — `Ellipsis`, undefined. |
| FR-217 | M | **Cost is measured time, per model.** Every model records `train` and `predict` time separately, per fold and for the final refit, and the run reports `train + predict` per model as its cost. Both **CPU seconds** (summed across worker processes — parallelism-invariant, so it is comparable across runs, machines and worker counts) and **wall seconds** (what NFR-01's 10-minute budget is measured in) are recorded. Fit counts, series counts and model counts are explanatory colour, never the cost proxy: a global model is one fit per fold across the panel while a local model is one fit per series per fold, and `SeasonalNaive` and `AutoARIMA` differ by orders of magnitude at identical fit counts. |
| FR-217c | M | **Recursive prediction is a first-class cost.** The Phase 0 spike found `LocalLGBM`'s `predict` cost is **2.2× its `train` cost** (0.425 s vs 0.191 s per series), because recursive forecasting over `h` is `h` sequential predict-and-refeature passes. `LocalLGBM` is consequently the most expensive model in the default set at 54% of total run cost. No fit-count, series-count or model-count proxy can see this. It is the primary optimisation target for Phase 3, and any caching or vectorisation of the recursive path must be measured against `predict` time specifically, not against total run time. |
| FR-217a | M | **Overhead is reported alongside, not folded in.** Ingestion, profiling, validation, fold slicing, conformal calibration, ensembling, evaluation, artifact-pack construction and persistence are none of them a model's `train` or `predict`, but they are real cost. They are reported as a separate labelled breakdown, and a test asserts per-model time plus overhead accounts for total CPU time within tolerance. Without this the numbers do not sum, and the S3 runtime estimate is optimistic on precisely the large panels where a bad estimate is most expensive. |
| FR-217b | M | Per-model timings are written to `XLF_Diagnostics` and returned in `RunResult.timing`. They are **excluded from the leaderboard**: measured durations are not reproducible, and a duration column in `XLF_Leaderboard` would make NFR-02 byte-identity unachievable by construction. |
| FR-216a | M | **RESOLVED — `GlobalXGB` is in the default set.** All three FR-203b matched pairs are complete: `{Linear, LGBM, XGB} × {own_series, panel}`. The default set is therefore symmetric by construction, and this is now an invariant rather than a preference: **no ML learner may be added to the default set without its counterpart in the other information set.** A registry test enforces it. The cost is negligible — a global model is ~4 fits per run against ~800 for a local one (FR-217 gives the measured figure). |
| FR-202 | M | Provide intermittent models: `CrostonClassic`, `ADIDA`, `IMAPA`, `ZeroModel`. |
| FR-203 | M | Provide **global** models via `mlforecast` fitted once across the panel: `GlobalLGBM`, `GlobalXGB`, `GlobalLinear`, in recursive mode. Recursive is the only mode in v1 (see FR-210). |
| FR-203a | M | Provide **local** counterparts of the same three learners — `LocalLGBM`, `LocalXGB`, `LocalLinear` — fitted independently per series with the **identical feature recipe** (lags, rolling windows, date features, target transform) as their global twins. Implemented by running `mlforecast` per `unique_id`, not via `statsforecast.SklearnModel`: that wrapper calls `model.fit(X, y)` with `X` = exogenous columns only, performs no lag or date feature engineering, and requires future exogenous values of shape `(h, n_x)` at predict time — on a panel without exogenous columns its feature matrix is empty. |
| FR-203b | M | **The three learners × two information sets form three matched pairs.** `LocalLGBM` vs `GlobalLGBM` differ in exactly one variable — the information set — with learner, features and folds held constant. This is what makes FR-207 a controlled comparison rather than an annotation, and it is the leaderboard's most defensible single claim: pooling either helps on this panel or it does not, and the user can see which. |
| FR-203c | M | Local ML models carry a **higher minimum-observation threshold** in the registry than local statistical models, since lag construction consumes `max_lag` rows per series: a series at the FR-105 floor of 143 weekly observations leaves ~91 training rows after 52-week lags. Where a local ML model declines a series, FR-215 common-support rules apply and `n_series_common` makes it visible. Gradient-boosting defaults are unusable at this size — `min_child_samples=20, num_leaves=31` on ~91 rows produces near-constant predictions and "no further splits with positive gain" — so local ML hyperparameters are a separate, conservative registry entry, not the global ones reused. |
| FR-204 | M | `SeasonalNaive` is **always** included and always reported, regardless of user configuration. It cannot be disabled. |
| FR-205 | M | All models are evaluated on **identical** CV folds — same cutoffs, same `h`, same `step_size`, same `n_windows`, and the same per-fold **test index**, i.e. the exact `(unique_id, ds)` set being scored. |
| FR-206 | M | Fold generation is computed once by `engine/folds.py` and shared by all model families. Cutoffs are **panel-wide calendar dates**, not per-series offsets. `engine/run.py` slices train/test itself and drives `fit`/`predict` per fold; it must not call `statsforecast.cross_validation` or `mlforecast.cross_validation`. Rationale below. |
| FR-206a | M | **Why we do not use the libraries' own CV.** Neither `StatsForecast.cross_validation` nor `MLForecast.cross_validation` accepts a cutoff array — verified against statsforecast 2.1.1 and mlforecast 1.1.0. Both derive cutoffs internally from each series' **own last timestamp** (`utilsforecast.processing.backtest_splits` uses `max_dates = groupby(unique_id).ds.max()`; statsforecast uses `range(-test_size, -h+1, step_size)` per series group). On a ragged panel — the normal case for SKU data with new and discontinued products — series A's fold-1 cutoff is a *different calendar date* from series B's, so a global model trained at A's cutoff has seen B's observations from after it. That is look-ahead leakage in exactly the family FR-207 wants compared fairly against local models, and no assertion about cutoff *equality* can detect it, because per-series cutoffs are identical across families by construction. |
| FR-206b | M | Series with no observations before a given cutoff are excluded from **that fold** for **every** model, and the exclusion is recorded per fold. A model may not be scored on a fold from which another model was excluded. |
| FR-207 | M | Record and expose, as **two separate fields**, each model's `family` (`local` \| `global` \| `ensemble`) and its `information_set` (`own_series` \| `panel`), so global/local comparisons are legible. These are not the same axis: a `LocalLGBM` is family `global`-style learner with `own_series` information; an ensemble of local models has `own_series`; an ensemble containing a global model has `panel`. The original single `Literal["local","global","ensemble"]` could not express any of this. With FR-203b's matched pairs the leaderboard can state the pooling effect directly, per learner, rather than leaving the reader to infer it across differently-specified models. |
| FR-208 | M | Metrics: MASE, RMSSE, MAE, RMSE, sMAPE (point); **scaled CRPS**, pinball loss and empirical coverage per level (probabilistic). `utilsforecast.losses` has no plain `crps` — what exists is `scaled_crps(df, models, quantiles, …)`, a quantile-grid approximation normalised by the sum of actuals. The leaderboard field is therefore named `scaled_crps`, and the **quantile grid is recorded in the manifest**, because a run with `levels=[80,95]` (a 5-point grid) is not CRPS-comparable with one at `levels=[50,80,95]`. |
| FR-209 | M | Report all metrics both per-series and aggregated across the panel; the aggregation function must be stated (default: mean of per-series scores). Aggregation is **NaN-safe** and every leaderboard row carries `n_series_scored` alongside the aggregate, so a metric averaged over 240 of 288 series can never be silently compared with one averaged over 288. |
| FR-214 | M | **Metric degeneracy policy.** MASE and RMSSE divide by the mean absolute (seasonal) difference of the training window; sMAPE divides by `\|y\| + \|ŷ\|`; scaled CRPS divides by the sum of actuals. Each of these is **zero** on data the panel will routinely contain — a series constant within an early training fold, or an all-zero evaluation window on an intermittent SKU. FR-105 excludes all-constant *series*, not all-constant *folds*, so these arise after validation has passed. Per-series metric values in this situation are `null`, never `inf` or `NaN`; the series is excluded from that metric's aggregate only; the count appears in `n_series_scored`; and the reason appears in `XLF_Diagnostics`. One degenerate series must never NaN an entire leaderboard row. |
| FR-215 | M | **Common support.** Where models are scored on different series subsets — per-model minimum-observation requirements in the registry, or intermittent routing under FR-108 — the panel aggregate is computed over the **intersection** of series scored by all ranked models, and that intersection size is reported as `n_series_common`. Per-model aggregates over their own full support are additionally available but are never used for ranking. `vs_baseline_pct` is always computed against `SeasonalNaive` on the same support as the model it describes. |
| FR-210 | ~~M~~ **W** | ~~Support direct multi-step mode for global models in addition to recursive.~~ **Deferred to v2.** This was a Must here while the Build Plan simultaneously listed "direct multi-step for global models" under *Deferred to v2* — a flat contradiction. Resolved in favour of deferral: direct mode multiplies fits by `h` inside the CV loop and NFR-01 has no room for it. Note the knock-on: FR-601 attributions are therefore attributions over a **recursive** feature path (see FR-601a). |
| FR-211 | S | Parallelise across models and series with a configurable worker count. |
| FR-212 | S | Support a `--fast` profile that subsets series for a quick smoke run before the full competition. |
| FR-213 | C | Feature engineering config for global models: lags, rolling windows, date features, target transforms. |

**AC-205:** For the same job spec on a deliberately **ragged** panel (series with three different
end dates), a test asserts that for every fold the `(unique_id, ds)` **test index** scored for a
local model, a global model and an ensemble are element-wise identical sets, and that every fold's
cutoff is a single calendar date shared by all series. This test must never be skipped.

*Amended:* the original AC asserted that "the cutoff arrays handed to
`statsforecast.cross_validation` and `mlforecast.cross_validation` are element-wise identical".
That test cannot be written — **neither function accepts a cutoff argument** (FR-206a), so nothing
is handed to either. It also tested the wrong property twice over: identical cutoffs do not imply
identical training data (`MLForecast.cross_validation` defaults `dropna=True`, discarding the first
`max_lag` rows of every series, while statsforecast trains on full history), and per-series cutoffs
are trivially equal across families while still leaking. The test index is the property that
actually makes the leaderboard comparable.

**AC-206:** A test asserts that the effective training row count per family per fold is recorded in
diagnostics, so a local-vs-global comparison can be audited for the `dropna`/`keep_last_n`
asymmetry rather than silently confounded by it.

### 3.3 Uncertainty quantification (FR-3xx)

| ID | Pri | Requirement |
|---|---|---|
| FR-301 | M | Produce conformalised prediction intervals for **every** model at user-specified levels (default 80, 95). |
| FR-302 | M | **Cross-conformal calibration.** The band applied when scoring fold *k* is calibrated **only from the residuals of folds ≠ k**. The band delivered with the final forecast uses all folds' residuals. No leakage from the final fit window in either case. |
| FR-303 | M | Report empirical coverage per level per model, measured **out of calibration** — fold *k*'s coverage is evaluated against the band built from folds ≠ *k*, then averaged over folds — alongside nominal coverage. Reported separately for intermittent and smooth series (FR-307). |
| FR-303a | M | **Why.** The original FR-302/FR-303 pair calibrated on the same folds it measured coverage on. An empirical quantile covers its own calibration sample at the nominal rate *by construction*, so the resulting coverage number is a tautology: it would report ≈80% for an 80% band no matter how badly the intervals were built, and AC-301, success criterion #4 and gate G2 would all pass against a broken implementation. Cross-conformal costs no additional model fits — only repeated quantile computation over residuals that already exist — so `n_windows` stays at its default of 3. |
| FR-304 | M | Where a model has native intervals (ARIMA, ETS), report them as a separate, clearly labelled column set. They are excluded from the ranked comparison. |
| FR-305 | S | Support quantile output at arbitrary levels, not just symmetric intervals. |
| FR-306 | C | Weighted/adaptive conformal for non-stationary series. |
| FR-307 | M | **Support-aware bands.** Additive symmetric half-widths are clipped to the series' observed support — a non-negative series gets `lo = max(lo, 0)`. Unclipped, an intermittent series' lower bound sits below zero, every `y = 0` falls trivially inside the band, and reported coverage approaches 100% while meaning nothing. Because clipping makes coverage non-comparable between intermittent and smooth series, FR-303 reports the two classes separately, and the clip rate per series is recorded in diagnostics. |

**AC-301:** On a synthetic panel with known noise distribution, **out-of-calibration**
cross-conformal coverage of the 80% interval falls within [0.75, 0.85], and the equivalent
*in-calibration* figure is computed alongside it and asserted to be measurably tighter to nominal.
The second assertion is the point: it is the control that proves the first number is not a
tautology. A mutation that reverts FR-302 to same-fold calibration must fail this AC.

**AC-307:** On an intermittent panel, coverage is reported separately for intermittent and smooth
series, and a test asserts that removing the FR-307 clip drives intermittent-series coverage above
0.97 at nominal 0.80 — demonstrating that the unclipped number was meaningless.

### 3.4 Selection and ensembling (FR-4xx)

| ID | Pri | Requirement |
|---|---|---|
| FR-401 | M | Selection strategies: `pooled` (one winner for the panel, default), `per_series` (argmin per series), `clustered` (winner per feature-based cluster). |
| FR-402 | M | Default selection is `pooled`. Rationale: `per_series` overfits CV with few windows. The UI must warn when `per_series` is chosen with `n_windows < 5`. |
| FR-403 | M | Ensembles: `median`, `trimmed_mean` (trim fraction `ensemble_trim`, default 0.2), `inverse_error` weighted, `best_k` mean (`best_k`, default 3). `inverse_error` inverts the **pooled MASE** and `best_k` ranks by it; both are configurable via `ensemble_metric` and both are recorded in the manifest — the original spec named neither, which alone made ensembles irreproducible. |
| FR-403a | M | Ensemble edge cases are defined, not undefined behaviour: `best_k` with fewer than `k` constituents falls back to all available and records the fallback; `trimmed_mean` with fewer than 5 constituents (where trim 0.2 removes less than one model per side) degrades to `median` and records it; an ensemble over a single surviving model is not formed and does not appear in the leaderboard. |
| FR-404 | M | Probabilistic ensembling must be explicit about method — **vincentization** (quantile averaging) or **linear pooling** (distribution mixture). Default: vincentization. The choice is recorded in the manifest and shown in the UI. |
| FR-405 | M | The ensemble competes in the leaderboard on the same folds as its constituents; it does not get a free pass. |
| FR-405a | M | **Ensemble weights are estimated leave-one-fold-out.** For `inverse_error` and `best_k`, the weights (or the member set) used when scoring fold *k* are computed **only from folds ≠ k**. `median` and `trimmed_mean` fit no parameters and are exempt. Without this, an ensemble whose weights are derived from the fold errors it is then scored on is *precisely* the free pass FR-405 claims to forbid — and it is a subtler failure than the one §6 originally listed, because the folds genuinely are identical. Requires `n_windows ≥ 2` whenever a parameter-fitting ensemble method is selected; validated at request time. |
| FR-406 | M | If no model beats `SeasonalNaive`, the system says so prominently and defaults the recommendation to `SeasonalNaive`. |
| FR-407 | S | Allow the user to pin or exclude specific models before the run. |
| FR-408 | M | **Selection bias must be reported, not just warned about.** When `selection != pooled`, the aggregate score of the *selected* model is an argmin over models on the same folds it is scored on, so it is optimistically biased — the winner's curse. The leaderboard reports, alongside it, a leave-one-fold-out selected-model aggregate (select on folds ≠ *k*, score on fold *k*), and the UI labels the naive figure as selection-biased. This is free: selection is an argmin over fold scores that already exist. Without it, success criterion #1 — "beats seasonal naive on the panel aggregate" — can be satisfied by selection bias alone, which is the exact dishonesty this product exists to avoid. |

**AC-406:** On a **seasonal** random-walk panel (`y_t = y_{t−m} + ε_t`), the pane displays a "no
model beat the baseline" notice and the recommended model is `SeasonalNaive`.

*Amended — the original test would have failed against a correct engine.* On a *pure* random walk
(`y_t = y_{t−1} + ε_t`) the optimal forecast is the last value. `SeasonalNaive` at season length
*m* has error variance ≈ `m·σ²` for `h ≤ m`, against `h·σ²` for a naive forecast, and `AutoARIMA`
will select ARIMA(0,1,0) and reproduce exactly that. At the weekly defaults (`h=13, m=52`)
AutoARIMA therefore beats `SeasonalNaive` by a wide margin, the notice correctly does not fire, and
gate G1 fails on a working implementation. The DGP was wrong, not the engine.

**AC-406a:** On a *pure* random-walk panel, `AutoARIMA` and any naive-equivalent model **do** beat
`SeasonalNaive`, and the "no model beat the baseline" notice does **not** fire. Retained as a
separate test because it is the more informative of the two — it catches a fold or scoring bug that
AC-406 would sail through.

### 3.5 Natural language job specification (FR-5xx)

| ID | Pri | Requirement |
|---|---|---|
| FR-501 | M | Free-text input parsed into a validated `ForecastRequest` in a single constrained LLM call. No agent loop. |
| FR-502 | M | The LLM receives a **data profile** (series count, date range, inferred frequency, seasonality candidates, intermittency share, missingness, available exog columns) — never raw observations. |
| FR-503 | M | The parsed config is rendered as an editable form and requires explicit user confirmation before any compute is spent. |
| FR-504 | M | Every inferred-but-unstated parameter appears in a visible `assumptions` list (e.g. "assumed calendar quarter = 13 weeks"). |
| FR-505 | M | LLM output is re-validated server-side against the same schema and business rules as a manually built request. Unknown model names (validated against `engine/registry.py`, not a hard-coded literal), out-of-range `h`, and impossible frequencies are rejected. Frequency rejection is only possible because of FR-112 — `freq` was previously an unvalidated bare `str`, so this requirement had no mechanism behind it. Note the parser will emit legacy aliases like `M` for "monthly" because that is what its training data contains; FR-112 normalises rather than rejects those. |
| FR-506 | M | If parsing confidence is low or the request is ambiguous, return a clarifying question instead of a config. |
| FR-507 | S | Support follow-up refinement ("make it 6 months instead") against the existing config. |
| FR-508 | C | Multilingual input (IT/EN at minimum). |

**AC-503:** `POST /v1/jobs` **rejects with 4xx** a request carrying a parse-derived config without a
valid, unexpired confirmation token bound to that `(data_id, request_hash)`. Tested by omitting the
token, by replaying a token from a different request, and by expiring one.

*Amended:* the original AC — "no job can be enqueued without a confirmation event in the audit log"
— verified the logging, not the gate. A system that enqueues first and writes the audit line
afterwards passes it. The enforcement path is what needs the test.

### 3.6 Explanation (FR-6xx)

| ID | Pri | Requirement |
|---|---|---|
| FR-601 | M | Generate a deterministic **artifact pack** per job: peak/drop detections, component decomposition, feature attributions, calendar attribution, historical analogues, uncertainty context. See Technical Spec §8. |
| FR-601a | M | **Attribution scope.** SHAP attributions are produced for global models only, and — because v1 global models are recursive (FR-203, FR-210) — the lag features at horizon step `t > 1` hold the model's **own earlier predictions**, not observations. Attributions are therefore labelled as attributions over the recursive feature path, and the narration prompt is forbidden from describing them as drivers of the observed data. Where a single-step attribution is wanted, `h = 1` attributions are computed separately and labelled as such. Presenting recursive-path SHAP as "the driving features" would be the soft dishonesty §6 exists to prevent. |
| FR-601b | M | **Decomposition scope.** Component decomposition is produced for `AutoETS` only, read from the fitted state matrix, and only for the components the auto-selected ETS form actually has — an `ANN` fit has neither trend nor seasonal, and those fields are `null` rather than zero. **ARIMA has no level/trend/seasonal decomposition and does not get one**; `Decomposition` is never emitted for an ARIMA winner. An STL/MSTL decomposition of the *history* is not a substitute and must not be presented as the model's components — FR-606 forbids exactly that conflation. Implementation note: the ETS state matrix is private API (`model_["states"]`), its column layout varies with the selected form, and it therefore ships with a contract test that fails loudly on a statsforecast upgrade. |
| FR-602 | M | The explanation agent accesses artifacts only through a fixed read-only tool set. It has no code execution and no raw data access. |
| FR-603 | M | Panel-level explanation is generated automatically on job completion. Per-series explanations are generated on demand only. |
| FR-604 | M | **Explain this cell**: from a selected cell in the results sheet, resolve `(unique_id, ds)` and return a grounded explanation of that point. |
| FR-604a | M | Defined behaviour for every other selection: a multi-cell selection explains the first resolvable point and says so; a header row, a cell outside the five output sheets, or a cell in `XLF_Leaderboard`/`XLF_Diagnostics` returns a named message explaining what to select instead; a chart selection is treated as no selection. No path returns a stack trace or a generic failure (§4 error presentation rule). |
| FR-605 | M | Every numeral in generated prose must match an artifact value, or be on the explicit allowlist, or the response is rejected and regenerated. Matching is **unit-aware** (a prose `78%` matches an artifact `0.78`) and the tolerance is derived from the **displayed precision** of the numeral rather than a fixed relative epsilon — `"about 78%"` must match `0.7834`. Counts, years and ordinals match exactly. Allowlisted without artifact backing: the requested `levels`, `h`, `n_windows`, series counts, and calendar years/months already present in the consumed artifacts. See TS §9.4. |
| FR-605a | M | **Templated figures are the default path.** Key numbers (winner, its score, the baseline score, the delta, coverage) are rendered from templates filled by the engine; the LLM writes only the connective prose around them. Free numerals are the exception, and the guardrail is the backstop for that exception — not the primary mechanism. A fixed 1e-3 relative tolerance over all free numerals, as originally specified, would reject ordinary rounded prose on nearly every generation and drive the strip path continuously. |
| FR-606 | M | Explanations must be phrased as model attributions, not causal claims. "The model's seasonal component peaks in March" — never "sales peak because of Easter" unless Easter was an explicit regressor. |
| FR-606a | M | FR-606 is enforced, not merely instructed. A system-prompt line is not a test. Generated prose passes a causal-construction lint (`because`, `due to`, `caused by`, `driven by`, `thanks to`, `as a result of` — flagged when the grammatical subject is a real-world entity rather than a named model component or artifact field), and the AC-605 suite carries a labelled adversarial slice of prompts that invite causal phrasing. Without this the requirement silently rots. |
| FR-607 | M | Explanations are cached by `(job_id, scope, unique_id, ds, question_hash, prompt_version)`. The original `(job_id, unique_id, ds)` key does not distinguish panel-scope explanations (FR-603, which have neither `unique_id` nor `ds`) from point explanations, and does not include the question at all — so under FR-608 two different questions about the same job would collide and be served each other's answers. `prompt_version` is in the key so a prompt change invalidates rather than silently reuses. |
| FR-608 | S | Free-text Q&A against a completed job ("which SKUs are most uncertain?") using the same tool set. |
| FR-609 | S | Write an explanation into the sheet as a cell comment. |
| FR-610 | C | Export the explanation and leaderboard as a one-page methodology PDF for board packs. |

**AC-605:** Across a regression suite of 100 explanation requests on varied panels, measured on the
**pre-guardrail** generation: the first-pass rejection rate is below 5%, and the sentence-strip path
(TS §9.4, second failure) fires **zero** times. The post-guardrail output additionally contains zero
unmatched numerals.

*Amended:* the original AC asserted only the last of those. But §9.4 *strips* offending sentences on
second failure, so post-guardrail output contains zero unmatched numerals **by construction** — the
AC would pass with a model that hallucinated in all 100 cases. It tested the sanitiser, not the
system. The pre-guardrail measurement is the one that can fail.

### 3.7 Output (FR-7xx)

| ID | Pri | Requirement |
|---|---|---|
| FR-701 | M | Write **five** sheets: `XLF_Forecast`, `XLF_Forecast_Long`, `XLF_Leaderboard`, `XLF_Diagnostics`, and the hidden `XLF_Manifest`. Layouts in §5. The manifest sheet was previously counted separately from "the four sheets", which left it outside the FR-703 overwrite set. |
| FR-702 | M | Write whole ranges in batched operations. Never cell-by-cell. |
| FR-703 | M | Re-running a job overwrites **all five** sheets after confirmation, as one transaction; it never appends silently. `XLF_Manifest` is explicitly in the overwrite set — under the original wording a re-run left a manifest describing the *previous* run beside new results, violating both FR-704 and hard rule 10. |
| FR-703a | M | Defined behaviour when the workbook is not in the expected state: a sheet renamed or deleted is recreated; a **protected** sheet aborts the whole write with a named error rather than partially writing; formulas elsewhere referencing an output sheet trigger a warning listing the referring cells before overwrite; a workbook in co-authoring mode is refused with an explanation. A second job writing into a workbook that already holds a different job's manifest requires explicit confirmation naming both job ids. |
| FR-704 | M | Write a `XLF_Manifest` block (hidden sheet or named range) containing the full job spec, engine version, model versions, random seeds, and timestamp — enough to reproduce the run exactly. |
| FR-705 | S | Insert a native Excel line chart of history + forecast + interval band for a selected series. |
| FR-706 | S | CLI writes the same four tables as CSV/Parquet plus a JSON manifest. |
| FR-707 | C | Export to `.xlsx` directly from the CLI without Excel installed. |
| FR-708 | M | **Output size precheck.** Before writing, compute the row count of every output sheet. `XLF_Forecast_Long` is `series × h × models × (1 + 2·levels)` — a 2,000-series panel at `h=52` with 9 models and 2 levels is ≈4.7M rows against Excel's 1,048,576 limit. FR-107 caps the *input*; nothing capped the output. If any sheet would overflow, refuse before writing anything and offer two named degradations: winner-only long sheet, or full results written to Parquet with the workbook holding a link and a summary. Never truncate silently. |

### 3.8 Reliability, licensing, operations (FR-8xx)

| ID | Pri | Requirement |
|---|---|---|
| FR-801 | M | Jobs survive worker restarts by **resuming from the last completed fold**, not by restarting from scratch. Per-fold, per-model results are checkpointed to object storage as they complete — which is the same mechanism the S4 partial leaderboard already requires. "State is durable" was undefined between resume and restart, and G4 tests it. |
| FR-802 | M | Cancel a running job from the pane, with cancellation effective within one fold-model unit of work. The engine is CPU-bound compiled code (`coreforecast`, LightGBM), so cancellation **cannot** work by cancelling an asyncio task: engine work runs in a `ProcessPoolExecutor` and cancellation terminates the process, then marks the job cancelled and retains completed checkpoints. See TS §2. |
| FR-803 | M | Per-user quota on concurrent jobs and monthly compute-minutes, where a compute-minute is **wall-clock seconds × worker processes**, metered per fold-model unit and accumulated on completion of each. A job that exhausts quota mid-run completes the fold in flight, then stops and is marked `quota_exhausted` with partial results retained and downloadable. |
| FR-804 | M | Licence check on job submission (not on add-in load), with a **72-hour** offline grace period. Grace state lives server-side keyed to the licence, never in the workbook or the browser — hard rule 8 forbids client-side credential state, and a workbook custom property travels with the file. |
| FR-805 | M | Structured audit log: who ran what spec, when, on how many series, with which engine version. |
| FR-806 | S | Bring-your-own LLM endpoint (Azure OpenAI / Bedrock / OpenAI-compatible base URL). |
| FR-807 | S | Self-hosted deployment mode for enterprise customers where no data leaves the tenant. |

---

## 4. Task pane UI specification

Single pane, five states. Keep it boring and dense; the users are Excel users.

**S1 — Data.** Range picker; column mapping dropdowns for `unique_id`/`ds`/`y`; exogenous column
classifier; live validation summary ("288 of 300 series valid — 12 excluded, see details").

**S2 — Ask.** Free-text box with three example prompts. Below it, an **Advanced** disclosure holding
the full manual form (horizon, frequency, season length, model checkboxes, levels, CV windows,
selection strategy, ensemble method). The manual form is the source of truth; natural language just
fills it in.

**S3 — Confirm.** Read-only summary card of the resolved config, the `assumptions` list, an estimated
runtime, and a **Run** button. Estimated runtime is computed from series count × models × folds
against a calibration table — it must exist, because users will not wait for something with no ETA.
Pressing **Run** mints the FR-505/AC-503 confirmation token bound to `(data_id, request_hash)`;
without that token the job cannot be enqueued.

The calibration table is built from **measured per-model `train + predict` CPU seconds** (FR-217)
accumulated over previous runs, plus the FR-217a overhead term — not from a model count, which is
not a unit of work. It is keyed on the FR-201a AutoARIMA mode (seasonal and Fourier differ by
roughly an order of magnitude), on series count, on series length and on `n_windows`. CPU seconds
are the right basis precisely because they are parallelism-invariant: the estimate divides by the
worker count at the end rather than being contaminated by it.

**S4 — Running.** Progress by model and fold. Elapsed time. Cancel button. Partial leaderboard
streams in as models finish, so the user sees value before the job ends.

**S5 — Results.** Winner, its score, the baseline score, the delta, and a plain-sentence verdict.
Coverage check ("80% intervals covered 78% of out-of-calibration points — well calibrated"); the
number shown must be the FR-303 out-of-calibration figure, never the in-calibration one, which is
≈nominal by construction and would make this message meaningless. Where `selection != pooled`, the
winner's score is shown with its FR-408 selection-bias label. Buttons:
**Explain results**, **Explain this cell**, **Re-run**, **Export manifest**.

**Error presentation rule:** every error names the series or column at fault and states the fix.
No stack traces, no "an error occurred".

---

## 5. Output sheet layouts

### `XLF_Forecast` (wide — for charting)
```
unique_id | ds | y_hat | {y_hat_lo_L | y_hat_hi_L for each L in levels} | model
```
One row per series per horizon step. `model` is the selected model for that series.

**Interval columns are generated per requested level, not hard-coded to 80/95.** `levels` is
user-configurable (FR-301) and FR-305 anticipates arbitrary quantiles; a fixed
`lo_80/hi_80/lo_95/hi_95` layout has no defined output for `levels=[50,80,95]`. Column order follows
`sorted(levels)` so the layout is deterministic for the golden tests.

### `XLF_Forecast_Long` (tidy — for pivots)
```
unique_id | ds | model | quantity | level | value
```
where `quantity ∈ {point, lo, hi}`. Every model's forecast, not just the winner. This is the sheet
power users pivot against.

### `XLF_Leaderboard`
```
scope | unique_id | model | family | information_set | mase | rmsse | mae | rmse | smape |
scaled_crps | {coverage_L for each L in levels} | rank | vs_baseline_pct | n_folds |
n_series_scored | n_series_common | selected | selection_biased
```
`scope ∈ {panel, series}`. `vs_baseline_pct` is relative to `SeasonalNaive` on the same support —
negative is better.

Changes from the original layout, all of them requirement-driven:
- `family` and `information_set` are separate columns (FR-207) — `local|global|ensemble` alone
  could not answer the own-series-vs-panel question the requirement exists to make legible.
- `mae`, `rmse`, `smape` are present. FR-208 requires them and `LeaderboardRow` carries them; the
  original layout silently dropped all three.
- `crps` → `scaled_crps`, naming what `utilsforecast` actually computes (FR-208).
- Coverage columns are generated per level, matching `XLF_Forecast`.
- `n_series_scored` / `n_series_common` make FR-215 visible rather than implicit.
- `selection_biased` flags the FR-408 case so a per-series winner's score is never read as an
  unbiased estimate of future accuracy.

### `XLF_Diagnostics`
Four stacked blocks with header rows:
1. **Run summary** — job id, engine version, runtime, series in/out, exclusion reasons.
2. **Fold detail** — per model per `fold_index` scores, with the fold's cutoff date and its
   effective training row count per family, for auditing stability across cutoffs and for the
   AC-206 `dropna` asymmetry check. `fold_index` is an explicit column: gate G2 asserts fold
   identity between an ensemble and its members, and there was previously no fold identifier
   anywhere in the schema to assert on.
3. **Calibration** — nominal vs. out-of-calibration empirical coverage per level per model
   (FR-303), split by intermittency class, plus the per-series band-clip rate (FR-307) and a flag
   for series that fell back to pooled panel residuals (TS §5.4).
4. **Series flags** — intermittency class (pre-gap-fill, FR-106), post-fill zero share, seasonality
   strength, trend strength, short-history flag, and per-metric `null` reasons from FR-214.
5. **Timing** — per model: `train` and `predict` CPU seconds and wall seconds, per fold and for the
   final refit, with `n_series_fitted` and `n_rows_trained`; then the FR-217a overhead breakdown and
   the run total. This is the NFR-01 cost proxy and the S3 calibration input, and it is the block a
   user consults when a run was slower than its estimate.

---

## 6. Out-of-scope behaviours to actively prevent

These are failure modes, not missing features. Guard against them in code and tests.

- Reporting an ensemble score computed on different folds than its members.
- Estimating ensemble weights or `best_k` membership on the same folds the ensemble is scored on
  (FR-405a). This is the subtler and likelier version of the failure above: the folds are identical,
  which is exactly what makes it easy to miss.
- Calibrating conformal bands on the residuals that coverage is then measured against (FR-303a).
- Reporting a `per_series` selected-model score without labelling it as selection-biased (FR-408).
- Comparing panel aggregates computed over different series subsets (FR-215).
- Selecting per-series winners on 1–2 CV windows without a warning.
- Presenting native ARIMA intervals and conformal intervals in the same ranked table.
- Letting the LLM emit a forecast value, a metric, or a model choice.
- Writing results without a manifest.
- Silently dropping series without telling the user which and why.
