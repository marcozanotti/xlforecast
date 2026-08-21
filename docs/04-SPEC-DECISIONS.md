# 04 — Spec Decisions and Amendment Log

Resolutions of conflicts, gaps and errors found in the first-pass specification set, applied to
`00`–`03` on **2026-08-20**. Requirement IDs are stable, so amendments are made in place and logged
here; new IDs are appended rather than renumbered.

Read this before touching `schemas/` or `engine/` — several of these decisions are encoded directly
in the contract layer and cannot be deferred to the phase that consumes them.

Original documents are preserved for diffing at
`<scratchpad>/docs-original/` for the duration of this session; commit them to git if you want them
permanently.

---

## 1. Owner decisions

Four questions were escalated because they trade money, schedule or runtime budget against
correctness. All four were answered by the owner.

| # | Question | Decision | Consequence |
|---|---|---|---|
| D-1 | Fold control: neither Nixtla library accepts a cutoff array, and both derive per-series cutoffs that leak global models across ragged panel ends | **Own fold loop with panel-wide calendar cutoffs.** `engine/folds.py` slices train/test; `run.py` drives `fit`/`predict` per fold | Phase 1 grows from 3 to 4 weeks. FR-206/206a, AC-205/206. Buys the only real correctness guarantee in the system |
| D-2 | Conformal calibration and coverage measurement share folds, making the coverage claim a tautology | **Cross-conformal, `n_windows` stays at 3.** Fold *k*'s band comes from folds ≠ *k* | Near-zero extra compute — no additional model fits, only re-quantiling. NFR-01 unaffected. FR-302/303/303a, AC-301 |
| D-3 | AutoARIMA at `season_length=52` is 800 order searches per run and threatens NFR-01 | **Non-seasonal AutoARIMA + Fourier regressors when `season_length > 24`** | Standard practice for long seasonal periods, so this is a methodological improvement rather than a speed hack. AutoARIMA stays in the default 8. FR-201a, FR-216 |
| D-4 | ADR-002 (xlwings PRO) — two of its three stated benefits are delivered by the architecture anyway | **Provisional; spike in Phase 5 before writing sheet writers.** Measure a 250k-cell read against native Office.js; drop the licence if >2× slower | ADR-002 rationale corrected now; the choice itself is deferred to evidence |

---

## 2. Decisions taken without escalation

Made under stated reasoning; overrule any of them if you disagree.

**Default model set (FR-216) — owner-specified, 13 models.** Baselines `SeasonalNaive`,
`HistoricAverage`, `WindowAverage(season_length)`; local statistical `AutoARIMA` (Fourier),
`AutoETS`, `DynamicOptimizedTheta`; intermittent `CrostonClassic`; local ML `LocalLinear`,
`LocalLGBM`, `LocalXGB`; global ML `GlobalLinear`, `GlobalLGBM`, `GlobalXGB`. Opt-in: `AutoCES`,
`ADIDA`, `IMAPA`, `ZeroModel`. The schema default was previously the literal `[...]` —
`Ellipsis`, undefined.

Two consequences worth stating plainly:

- **`WindowAverage` at the seasonal frequency is the incumbent baseline**, not a filler model. P1
  "currently forecasts with a moving average", so it is the thing the product must beat to justify
  its price. `vs_baseline_pct` is reported against it as well as against `SeasonalNaive` (FR-201b).
- **The three ML learners × two information sets are three matched pairs** (FR-203b), differing in
  exactly one variable. That upgrades FR-207 from an annotation on the leaderboard to a controlled
  experiment, and it is the most defensible claim the leaderboard makes.

Resolved (FR-216a): `GlobalXGB` is **in**, so all three matched pairs are complete and the default
set is symmetric across information sets. This is now a registry invariant — no ML learner enters
the default set without its counterpart — rather than a judgement to be re-made each time.
**The whole set is frozen against the Phase 0 spike, not before it**: NFR-01's model count moves
8 → 13 and its 10-minute target is unvalidated at that size.

**`ModelName` is registry-validated, not a closed `Literal`.** The `commercial_ok` gate was
unreachable: a `Literal` containing only Apache/MIT models cannot express a prohibited one, so the
gate was dead code and its test would have been vacuous. Registry validation also makes a v2 model
addition a registry row rather than a schema change.

**`family` and `information_set` are separate fields.** FR-207 exists to make own-series-vs-panel
comparisons legible; `Literal["local","global","ensemble"]` conflates family with information set
and cannot express that an ensemble of local models is still own-series.

**Metrics are `float | None`, never `NaN`.** FR-214. Zero denominators are routine (a series
constant within an early training fold; an all-zero evaluation window on an intermittent SKU), and
`NaN` is not representable in JSON, so it would break the G0 round-trip gate as well as poisoning
any naive `mean()`.

**NFR-01's cost proxy is measured `train + predict` time, per model (FR-217).** Owner-specified.
Fit counts, series counts and model counts are explanatory only: a global model is one fit per fold
across the panel against one fit per series per fold for a local model, and `SeasonalNaive` and
`AutoARIMA` differ by orders of magnitude at identical fit counts. Two consequences that had to be
resolved rather than assumed:

- **CPU seconds *and* wall seconds are both recorded.** CPU seconds summed across workers are
  parallelism-invariant and therefore the comparable cost figure across runs, machines and worker
  counts; wall seconds are what NFR-01's 10-minute budget actually means. Reporting only one of them
  makes either the budget or the comparison meaningless.
- **Timings must stay out of the leaderboard** (FR-217b). Measured durations are not reproducible,
  so a duration column in `XLF_Leaderboard` would make NFR-02 byte-identity unachievable by
  construction. `RunTiming` is a separate object written to `XLF_Diagnostics` block 5.

Overhead — ingest, profile, validate, folds, conformal, ensemble, evaluate, artifacts, persist — is
reported as its own breakdown (FR-217a) rather than folded into any model, with a test asserting the
parts account for the total. Attributing only train and predict would understate every run and make
the S3 estimate optimistic on exactly the large panels where a bad estimate costs the most.

**Interval and coverage columns are generated per level.** The sheet layouts hard-coded 80/95 while
FR-301 makes `levels` user-configurable; `levels=[50,80,95]` had no defined output at all.

**Direct multi-step for global models (FR-210) demoted M → W.** It was a Must in the Functional Spec
and simultaneously listed under *Deferred to v2* in the Build Plan. Resolved toward deferral:
direct mode multiplies fits by `h` inside the CV loop and NFR-01 has no room. Knock-on: v1
attributions are over a recursive feature path and must be labelled as such (FR-601a).

**Manifest stores the *resolved* request** plus `thread_config`, `crps_quantiles`,
`ensemble_params`, `autoarima_mode`, `prompt_versions`, `data_id` and `previous_job_id`. Replay must
not re-run inference, byte-identity is not achievable without the thread configuration, and the v2
stability-monitoring note requires a lineage key that did not exist.

**Auth is a session cookie; workbook custom properties hold `job_id`/`data_id` only.** Hard rule 8
banned `localStorage` and pointed at custom properties instead — but a custom property is *part of
the file* and travels with it when the workbook is emailed or synced. The original rule pushed
toward a worse leak than the one it prohibited.

**Decomposition is `AutoETS`-only; ARIMA does not get one.** ARIMA has no level/trend/seasonal
decomposition in any form, and an STL of the *history* is not the model's decomposition — FR-606
forbids presenting it as one. The ETS state matrix is private API (`model_["states"]`, layout
varying by selected form), so it ships with a contract test that fails loudly on upgrade.

---

## 3. Verified against the libraries, not from memory

Checked by downloading and reading the wheels (`statsforecast 2.1.1`, `mlforecast 1.1.0`,
`utilsforecast 0.2.16`), because several original claims were version-stale.

| Claim in the original spec | Reality |
|---|---|
| "`statsforecast` depends on `numba` → `llvmlite`" (ADR-003, risk register, Phase 0) | **False since statsforecast 2.0.** Deps are `cloudpickle, coreforecast>=0.0.17, numpy, pandas<3.0, scipy, statsmodels>=0.14.5, tqdm, fugue>=0.9.4, utilsforecast>=0.1.4, threadpoolctl>=3`. Hot loops are compiled C++ in `coreforecast`. ADR-003's conclusion survives on different grounds; the Phase 0 "known friction point" does not exist |
| Cutoff arrays are "handed to" `statsforecast.cross_validation` / `mlforecast.cross_validation` (AC-205) | **Neither function has a cutoff parameter.** Both derive cutoffs internally from per-series max dates (`utilsforecast.processing.backtest_splits`; statsforecast's `range(-test_size, -h+1, step_size)` per series group). The AC could not be written |
| Metric `crps` (FR-208) | **Does not exist in `utilsforecast.losses`.** Available: `scaled_crps` (quantile-grid, normalised by the sum of actuals), `mqloss`, `scaled_mqloss`, `quantile_loss`, `coverage`, `calibration` |
| `mase`/`rmsse` are safe to average | Both require `seasonality` + `train_df` and divide by a training-window quantity that is **zero** on data the panel will routinely contain |
| ETS components are available for decomposition | `AutoETS` exposes `model_["fitted"]`, `model_["actual_residuals"]`, `model_["n_params"]`. A `states` matrix exists in `statsforecast/ets.py` but is private and its layout varies with the selected form. ARIMA has nothing |
| `ZeroModel` and the other 9 named models | **All present**, verified against `statsforecast/models.py` |
| Cloud Run scale-to-zero for the worker (NFR-06) | Cloud Run scales on inbound requests; a Redis-polling worker has none. Container Apps + KEDA does scale to zero; on GCP it needs Cloud Run Jobs or Pub/Sub push |
| `manylinux` floor | `coreforecast`/`statsforecast` publish `manylinux_2_27`/`2_28` only — **Alpine/musl will not work** |

---

## 4. New requirement IDs

| ID | Subject |
|---|---|
| FR-105a | Profile before validate — the original pipeline order was circular |
| FR-111 | Future-known exogenous values had no data path at all |
| FR-112 | Frequency normalisation (pandas ≥2.2 alias renames) |
| FR-201a | AutoARIMA seasonal policy |
| FR-206a/b | Why the libraries' CV is not used; per-fold series exclusion |
| FR-214 | Metric degeneracy policy |
| FR-215 | Common support and `n_series_scored` |
| FR-201b | `WindowAverage` at seasonal frequency as incumbent baseline |
| FR-203a/b/c | Local ML models, matched pairs, small-data thresholds |
| FR-216 | Default model set |
| FR-216a | Matched-pair symmetry invariant for the default set |
| FR-217/a/b | Measured per-model `train + predict` as the cost proxy; overhead accounting; leaderboard exclusion |
| FR-303a | Why calibration and coverage folds must differ |
| FR-307 | Support-aware (clipped) bands |
| FR-403a | Ensemble edge cases |
| FR-405a | Leave-one-fold-out ensemble weights |
| FR-408 | Selection-bias reporting |
| FR-601a/b | Attribution and decomposition scope |
| FR-604a | "Explain this cell" for non-forecast selections |
| FR-605a | Templating as the primary guardrail mechanism |
| FR-606a | Causal-phrasing lint |
| FR-703a | Workbook-state edge cases |
| FR-708 | Output-size precheck |
| AC-206 | Effective training rows per family per fold |
| AC-307 | Clip removal inflates intermittent coverage |
| AC-406a | Pure random walk — AutoARIMA *should* win |
| NFR-12 | Output overflow rejected before compute |

---

## 5. Acceptance criteria that were rewritten because they could not fail

Recorded separately because they are the most dangerous class of defect in the original set: each
would have passed against a broken implementation.

- **AC-301 / G2 / success criterion #4** — coverage measured on the calibration sample is nominal by
  construction. Now measured out-of-calibration, with the in-calibration figure retained as a
  *control* that must be tighter. A mutation reverting FR-302 must fail this AC.
- **AC-605 / G6** — measured post-guardrail, where the strip path guarantees zero unmatched numerals.
  Now measured pre-guardrail: first-pass rejection < 5%, strip path fires zero times.
- **AC-503 / G6** — asserted a confirmation entry in the audit log, which a system that enqueues
  first and logs second satisfies. Now asserts `POST /v1/jobs` rejects with 4xx.
- **AC-406 / G1** — would have *failed* against a correct engine, not passed against a broken one:
  on a pure random walk AutoARIMA selects ARIMA(0,1,0) and beats `SeasonalNaive` at `h < m`. The DGP
  is now a seasonal random walk, and the pure case is retained as AC-406a asserting the opposite.
- **G3** — "consistent with your published work" has no metric, tolerance or artifact. Now a
  committed baseline file with named tolerances, required before Phase 3 opens.
- **AC-205** — asserted a property of arguments that are never passed. Now asserts equality of the
  per-fold `(unique_id, ds)` test index on a deliberately ragged panel.

---

## 6. Still open

- ~~**NFR-01 feasibility**~~ — **RESOLVED by the Phase 0 spike, see `05-PHASE0-SPIKE.md`.**
  2.9 min projected against a 10-min budget at 13 models; **FR-216 is frozen**. The spike also
  corrected two claims in TS §4.0 (integer dict keys are fine under strict validation; the NaN
  failure is on re-validation, not serialisation) and one in FR-201a (Fourier mode is a 3.1×
  optimisation, not an NFR-01 rescue), and found `LocalLGBM`'s recursive `predict` costs 2.2× its
  `train` — now FR-217c.
- **ADR-002** is provisional pending the Phase 5 spike.
- **Image size** vs Phase 4 cold start is unmeasured; `fugue` is a mandatory transitive dependency of
  statsforecast and is not small.

---

## 7. Found during Phase 1 implementation

**FR-201c — `AutoETS` is silently non-seasonal above `m=24`.** `statsforecast/ets.py` allocates a
fixed 24-slot seasonal state buffer (`s = np.zeros(24)`) and returns early on
`if m > 24 and season != Component.Nothing`. So `AutoETS(season_length=52)` produces forecasts
*numerically identical* to `AutoETS(season_length=1)` — verified — while appearing in the
leaderboard as a seasonal model that simply performed poorly.

That is the exact failure mode this project exists to prevent: not a wrong number, but a number
whose provenance is invisible. A user comparing AutoETS against SeasonalNaive on weekly data would
conclude ETS is unsuited to their series, when in fact it was never given the chance to model the
seasonality they asked it to model.

Resolved the same way FR-201a resolves the equivalent AutoARIMA problem: above the threshold,
route through `MSTL` with an ETS trend forecaster, alias it back to `AutoETS`, and record
`ets_mode` in the manifest so the substitution is auditable rather than hidden. Measured on a
weekly seasonal panel: MASE **1.123 → 0.765**.

Worth noting for Phase 3: this was invisible to unit tests and only surfaced when a full
leaderboard was run on seasonal data and AutoETS placed implausibly low. Benchmark validation is
supposed to catch exactly this class of defect, which is an argument for running the M3
harness earlier rather than treating G3 as a formality.

**Related, and still open:** the Build Plan's statistical-correctness test says "Pure seasonal →
AutoETS/SeasonalNaive should win". With FR-201c that now holds. Without it, it would have failed —
another instance of a specified test that would have failed against a correct engine, alongside
AC-406.

---

## 8. Phase 2 — conformal, and an overreach of mine

**Owner decisions (2026-08-21).**

| # | Question | Decision |
|---|---|---|
| D-5 | Asymmetric bands balance the tails on smooth series but under-cover intermittent ones at 0.693 | **Symmetric + clipped everywhere.** Safer on persona P1's data, and one procedure keeps probabilistic scores commensurable per ADR-006 |
| D-6 | FR-303's coverage-by-intermittency-class split shows nothing (0.807 vs 0.809) | **Split the tails instead**, which differ sharply. Coverage becomes a single panel figure |

**AC-301's control behaves differently than the spec review predicted — and the difference
strengthens the argument.** I assumed the in-calibration figure would sit *closer* to nominal.
Measured, it sits systematically *above*: 0.844 against an honest 0.809 at level 80, and 1.000
against 0.953 at level 95. A conformal quantile with the finite-sample correction is conservative
on its own calibration sample. So the control cannot come out *low* — meaning it would report a
comfortable number for an interval that was far too narrow, which is a sharper indictment than
"it is roughly nominal by construction". The test asserts `control ≥ nominal` and
`control > honest`.

**FR-307 was wrong, and I wrote it.** The claim that clipping was needed because unclipped
coverage "approaches 100%" is false: clipping cannot change coverage on non-negative data, because
no observation lies below zero to be excluded. Measured 80.7% either way. AC-307 required removing
the clip to push coverage above 0.97 — an acceptance criterion asserting an impossible effect,
which fails permanently and protects nothing. Clipping earns its place on **sharpness** (−21.6%
width) and interpretability instead.

**Then FR-307b overreached in the other direction.** Having found the one-sided miscoverage, I
wrote that it means a model "is not well calibrated". Building the alternative disproved that: the
asymmetric band that balances the tails is *worse* on the data in question. The imbalance is a
property of wrapping a continuous interval around a point mass at zero, not a fault in the band.
FR-307c now says so.

The pattern across FR-201c, the `infer_freq` bug, and this one: **each was invisible to inspection
and surfaced only by running the thing.** The two in this section are worse than that — they were
defects in specification text I had written confidently a few sessions earlier, and the second was
introduced *while fixing the first*. Measuring the alternative before asserting a conclusion is the
cheap step that catches this, and it is the step I skipped.

---

## 9. Phase 4/5 — owner decisions (2026-08-21)

| # | Decision | Consequence |
|---|---|---|
| D-7 | **The project buys no licences** | ADR-002 resolves to **native Office.js**; xlwings PRO is out and its Phase 5 spike is cancelled. `TimeGPT` is struck from the v2 model register on the same grounds — it is a paid commercial API |
| D-8 | **Drop the five-demand-planner validation checkpoint** | Removed from the build plan's validation checkpoints. The post-G3 methodology review remains |

**D-7 costs less than it might look.** Two of xlwings' three stated benefits had already been
shown false: *"Python source stays server-side"* is delivered by the FastAPI architecture
regardless, and cross-host support is a property of Office.js, not of xlwings. That left one
real benefit — writing the add-in layer in Python — against a paid licence, a small vendor on
the critical path of our only distribution channel, and a non-obvious cost: xlwings Server
routes every range operation through our server, so the chunked read in TS §7.1 would have
been N HTTPS round trips carrying spreadsheet data rather than N local `context.sync()` calls.

The cost decision and the technical evidence point the same way, which is a comfortable place
to land. Native Office.js also removes the debugging indirection that gate G5 exists to
exercise: we now debug Office.js directly rather than at one remove, with better documentation.

The Office.js surface is small and bounded — a chunked range reader, five batched sheet
writers, a custom-properties accessor, a selection handler — and the pane is already
HTML + Alpine.js + Bootstrap, so browser JavaScript was being written either way.
