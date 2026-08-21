# 05 — Phase 0 Spike Results

Measured 2026-08-20. Build Plan Phase 0 step (a)–(c). **This memo freezes FR-216 and validates
NFR-01.** Harness: `<scratchpad>/spike/spike_timing.py`.

**Method.** Synthetic weekly panel, 156 observations, `m = 52`, `h = 13`. Every model timed with
`train` and `predict` separated, in CPU seconds (`getrusage`, all threads) and wall seconds
(FR-217). All BLAS/OMP thread counts pinned to 1, so CPU seconds are parallelism-invariant and
extrapolate honestly; the 8-vCPU wall figure is a division at the end, not a contaminated
measurement. Local models scale ×`n_series`; global models were measured directly at `n = 200`.

**Environment.** CPython 3.11.16 (via `uv python install` — the system `/usr/bin/python3.11` is
`3.11.0rc1`, a release candidate, and is not used). Host: 12 cores, glibc 2.35.

---

## 1. Resolved stack — G0 import gate passes

```
statsforecast 2.1.1   mlforecast 1.1.0    utilsforecast 0.2.16   coreforecast (compiled)
lightgbm 4.7.0        xgboost 3.2.0       scikit-learn 1.9.0
pydantic 2.13.4       polars 1.43.2       pyarrow 25.0.1         pandas 2.3.3   numpy 2.4.6
```

**`numba present: False | llvmlite present: False`** — confirmed by inspecting the *resolved*
environment, not merely package metadata. ADR-003's corrected reasoning is now empirically
verified, and the Phase 0 "known friction point" is definitively a non-issue at
`statsforecast >= 2.0`.

## 2. Cost per model — the NFR-01 projection

200 series × 4 fits (3 CV folds + final refit) × 13 models, CPU seconds:

| Model | CPU s / run | Share |
|---|---:|---:|
| **LocalLGBM** | **493.2** | **53.9%** |
| AutoARIMA (Fourier, k=3) | 237.4 | 25.9% |
| AutoETS | 111.8 | 12.2% |
| LocalXGB | 37.9 | 4.1% |
| LocalLinear | 27.6 | 3.0% |
| GlobalLGBM | 3.3 | 0.4% |
| DynamicOptimizedTheta | 2.2 | 0.2% |
| GlobalXGB | 1.0 | 0.1% |
| SeasonalNaive / HistoricAverage / WindowAverage / CrostonClassic / GlobalLinear | 0.2 each | ~0% |
| **TOTAL** | **915.3** | |

**Wall on 8 vCPUs: 1.9 min ideal, 2.9 min with 50% overhead allowance. NFR-01 budget is 10 min.**

**Verdict: NFR-01 passes at 13 models with roughly 3× headroom. FR-216 is frozen as specified.**

## 3. Four findings that change the specification

**(a) `LocalLGBM` is the most expensive model in the set — and its cost is in `predict`, not
`train`.** Per series: train 0.191 s, **predict 0.425 s (2.2×)**. Recursive forecasting over
`h = 13` is thirteen sequential predict-and-refeature passes, so prediction dominates. This is
invisible to any fit-count, series-count or model-count proxy, and it was found only because
FR-217 requires the two to be reported separately. It is the single best optimisation target in
the engine and the first thing Phase 3 should profile.

**(b) Seasonal AutoARIMA is expensive but was never a threat to NFR-01.** Seasonal `m=52` costs
0.923 s/series against 0.297 s for non-seasonal + Fourier — **3.1×** — but even the seasonal
variant lands at 4.4 min against a 10-min budget. FR-201a's cost argument holds directionally; its
claim to be *rescuing* NFR-01 does not, and has been corrected. Fourier mode stays, justified on
methodology (correct practice at `m > 24`) and on a real 3.1× saving.

**(c) Global models are effectively free and flat in panel size.** GlobalLGBM costs 0.82 s at
28,600 rows versus 0.84 s at 1,430 — a 20× row increase for no measurable cost, because tree
building at these sizes is dominated by fixed overhead and prediction is one vectorised recursive
pass over the panel rather than per-series. All three globals together are 4.4 s per run, 0.5% of
total. FR-216a's "nearly free" claim for `GlobalXGB` is confirmed at **1.0 s CPU per run**.

**(d) FR-203c's arithmetic is confirmed.** Local ML sees **~91 training rows per series** after
52-week lag construction, exactly as specified. The conservative small-data hyperparameters
(`min_child_samples=5, num_leaves=7`) are load-bearing, not decorative.

## 4. Pydantic findings — two spec corrections

| Check | Result |
|---|---|
| `dict[int, float]` JSON round-trip | Serialises to `{"80":0.78}`; **round-trips under both lax *and* strict** validation. TS §4.0's warning against `strict=True` was **wrong** and has been removed |
| `float` field holding `NaN` | Serialises to `null` (not a JSON error), then **fails re-validation** into a bare `float` with `float_type`. So the break is on the *return* leg, not serialisation. TS §4.0's reasoning corrected; the `float \| None` conclusion stands |
| `frozen=True` + mutating `mode="after"` validator | **Raises** `Instance is frozen`. D12 confirmed empirically — the original FR-204 validator could not have worked once `Manifest` embeds a frozen request |
| `field_validator` returning `sorted({*v, "SeasonalNaive"})` | Works, and is idempotent across round-trips |
| `set[str]` round-trip | Clean |

## 5. Caveats

- Synthetic panel, one machine, ideal linear scaling assumed across series. Real M3 panels have
  ragged ends, intermittency and longer histories; Phase 3 re-measures against NFR-01 for real.
- The 50% overhead allowance is an assumption, not a measurement — FR-217a's accounting exists
  precisely so that it stops being one after Phase 1.
- Parallel efficiency on 8 vCPUs is assumed ideal. Local models are embarrassingly parallel across
  series, so this is reasonable, but it is untested.

---

## 6. Addendum — found while building, not while spiking

**The numba/llvmlite friction is real. It is in `shap`, not `statsforecast`.**

`uv sync --all-extras` failed outright:

```
RuntimeError: Cannot install on Python version 3.11.16; only versions >=3.6,<3.10 are supported.
hint: `llvmlite` (v0.36.0) was included because `xlforecast[explain]` depends on `shap`
```

`shap` declares `numba` and `llvmlite` with **no lower bound** on non-macOS platforms, so the
resolver backtracks to `llvmlite 0.36` — which predates cp311 wheels and then tries to build LLVM
from source. Floors `llvmlite>=0.43` and `numba>=0.60` in the `explain` extra fix it; both have
cp311 wheels (earliest are 0.40 and 0.57 respectively).

This is worth stating plainly because §1 of this memo, written an hour earlier, reported "numba
present: False" and treated the risk as retired. Both statements are true and neither is the whole
picture:

| | numba present? |
|---|---|
| Core install (engine only) | **No** — verified: `statsforecast`, `mlforecast`, `utilsforecast`, `fugue`, `triad` declare no hard dependency on it, and the engine imports fine without it |
| Full install (`--all-extras`) | **Yes** — `shap` requires it, and `fugue` then imports it opportunistically |

So: ADR-003's corrected reasoning stands (the engine is numba-free, which is what its WASM
argument and the worker image size depend on), *and* the Phase 0 risk register entry should never
have been retired — only relocated. Both documents now say so. The regression guard is
`tests/integration/test_imports.py::test_engine_declares_no_dependency_on_numba`, which asserts
against package **metadata** rather than `sys.modules`, because the latter answers a different
question depending on which extras happen to be installed.

**One schema bug, caught by the G0 property test.** Pydantic does not run field validators over a
`default_factory` result, so `ForecastRequest()` carried `DEFAULT_MODELS` in declaration order
while a round-tripped one came back sorted — two objects for the same job that compare unequal.
That is precisely the NFR-02 byte-identity hazard the sorting was introduced to prevent, and it
would have been invisible until a golden test failed in Phase 1. Fixed with `validate_default=True`
on both `models` and `levels`.

---

## 7. Gate G0 — signed off

| G0 clause | Status |
|---|---|
| CI green | ✅ lint, format, `mypy --strict` on `schemas/`, 127 tests, 100% coverage on `schemas/` |
| `import statsforecast, mlforecast, utilsforecast` succeeds in the container | ✅ built on `python:3.11-slim-bookworm`, reported `G0 import gate OK - 2.1.1 1.1.0 0.2.16` |
| Schemas round-trip through JSON without loss | ✅ property test over all 24 contracts, including `None` metrics and integer-keyed coverage dicts |
| AutoARIMA spike recorded, FR-216 frozen against it | ✅ §2 above |

**Docker was not installed on the development machine.** It requires root, and `sudo` needs a
password that cannot be supplied non-interactively; rootless Docker is also unavailable, because
`newuidmap`/`newgidmap` are absent and installing them likewise needs root. The container clause
was therefore verified in **CI**, on a clean GitHub runner — which is the stronger test anyway,
since it has none of this machine's accumulated state. To build locally:
`sudo apt install docker.io && sudo usermod -aG docker $USER` (then re-login).

**A third instance of the same mistake, worth naming.** The G0 gate baked into the Dockerfile
asserted `find_spec('numba') is None` and failed the build — the image installs `--all-extras`,
so shap legitimately brings numba with it. That is the same error as the pytest guard in §6 and
the retired risk-register entry in §1: **asserting the absence of a package rather than the
absence of a declared dependency**. Presence depends on which extras are installed and answers a
different question each time; only metadata answers the question ADR-003 actually asks. All three
now assert against metadata.

---

## 8. Image size — the last open Phase 0 risk, now measured

Built locally on Docker 29.1.3 (buildx installed per-user at `~/.docker/cli-plugins/`, no root
required). The G0 gate baked into the Dockerfile passed in the build, and the running image was
verified: user `xlf` uid 10001, glibc 2.36, `OMP/MKL/OPENBLAS_NUM_THREADS=1`, `libgomp` present.

**2.92 GB → 2.10 GB (−28%).** Site-packages breakdown of the original:

| Package | MB | |
|---|---:|---|
| **nvidia** | **454** | **removed** — see below |
| xgboost | 228 | |
| polars runtime | 206 | |
| llvmlite | 173 | via shap (§6) |
| pyarrow | 156 | |
| scipy | 113 | |
| pandas | 79 | |
| statsmodels / sklearn / numpy | 151 | |

`xgboost` requires `nvidia-nccl-cu12` **unconditionally on Linux** — 454 MB of CUDA libraries for
multi-GPU distributed training, 25% of site-packages. This project is CPU-only by design (neural
forecasting is a non-goal; the deploy target is scale-to-zero), and NCCL loads lazily on GPU paths
only. Excluded via a `[tool.uv] override-dependencies` entry with an impossible marker; verified
that CPU XGBoost still trains and predicts, and that all 127 tests pass.

This matters for NFR-06: workers scale to zero, so every gigabyte is cold-start latency on a spiky
workload. 2.1 GB is still large and the remaining candidates are known — `llvmlite`+`numba`
(173+35 MB) exist only for shap, so a worker image that omits the `explain` extra would shed them,
at the cost of needing a second image. Deferred to Phase 4, where cold start is actually measurable
rather than hypothetical.

**Phase 0 risk register: all items now closed or measured.**

---

## 9. Gate G1 — signed off

| G1 clause | Status |
|---|---|
| Runs end-to-end on a CSV, emits four tables + manifest | ✅ `uv run xlforecast run panel.csv` |
| `RunTiming` per model, parts accounting for the total | ✅ FR-217/217a, asserted |
| Identical **test index** across families on a ragged panel | ✅ AC-205, with a control proving the libraries lack the property |
| Effective training rows per family per fold recorded | ✅ AC-206 — the gap is exactly `support × max_lag` |
| Byte-identical leaderboard twice, incl. shuffled input rows | ✅ NFR-02 under a recorded `thread_config` |
| Seasonal random walk → nothing beats `SeasonalNaive` | ✅ AC-406 |
| Pure random walk → `AutoARIMA` beats it | ✅ AC-406a |
| Fold-constant series → `None`, never `NaN` | ✅ FR-214 |

**284 tests. Coverage: engine 95%, ingest 92%, schemas 99%, `folds.py` 100%.** NFR-10's floors
are enforced in CI rather than merely stated.

### Two bugs the tests found, both in code that already "worked"

**`infer_freq` distrusted monthly data.** Confidence was the share of consecutive gaps equal to
the modal gap. Month-ends are 28, 29, 30 or 31 days apart, so a *perfectly regular* monthly panel
scored **0.56** — the ingest layer would have spent its life suspicious of the most common
business frequency there is. Confidence now measures alignment to the inferred calendar grid,
which also sharpens the semantics: gaps are FR-106's problem and do not reduce confidence, while
off-grid timestamps do and are rejected as `FREQ_MISMATCH`.

**Unparseable dates raised a polars `ComputeError`.** `str.to_datetime` raises rather than nulling
when it cannot infer a format at all, so a bad date column produced a library traceback instead of
the named, remediable error FS §4 requires. Translated at the boundary.

Neither surfaced in the end-to-end run. Both were found by writing tests for modules that were
already exercised indirectly — which is the argument for item 3 of this list having been worth
doing rather than declaring Phase 1 finished at the first green CLI run.

---

## 10. Gate G2 — signed off

| G2 clause | Status |
|---|---|
| Out-of-calibration coverage within ±5pp of nominal | ✅ asserted per model per level on a known-noise panel |
| In-calibration control proves the figure is not a tautology | ✅ AC-301, see below |
| Clip removal demonstrably changes the interval | ✅ AC-307 as rewritten — width, not coverage |
| Ensemble scored on the same folds as its members | ✅ `FoldScore.fold_index` equality |
| Ensemble weights demonstrably leave-one-fold-out | ✅ FR-405a — perturbing fold *k* does not move fold *k*'s weights |

**370 tests. `conformal.py` and `folds.py` at 100%; `engine/` at 97%.** Both 100% floors are
enforced in CI.

### The control behaves differently than the review predicted, and better

I expected the in-calibration figure to sit *closer* to nominal. It sits systematically
**above** it — 0.844 against an honest 0.809 at level 80, 1.000 against 0.953 at level 95 —
because a conformal quantile with the finite-sample correction is conservative on its own
calibration sample. That is a sharper indictment than the one the review made: the control
cannot come out *low*, so it would report a comfortable number for an interval that was far
too narrow.

### A configuration interaction worth knowing about (FR-302a)

A series has `n_windows × h` residuals, so per-series calibration engages only when that
product clears `min_residuals` (default 20). At the NFR-01 defaults (3 windows, h=13) it does,
at 39. At h=6 it does not, and every series falls back to the pooled panel residuals.

That blunts the AC-301 control: dropping one fold from a large pooled set barely moves the
quantile, so the two figures converge — measured gap **+0.090** under per-series calibration
against **+0.007** under pooled. The control stays directionally correct either way, but a
reader should be able to tell a strong result from a structurally weak one, so
`CalibrationRow.n_pooled_fallback` now reports which regime produced the number.

Cross-conformal also needs *more* residuals than in-calibration, since each fold's band is
built from a strictly smaller pool. `min_residuals` therefore bites harder here than its value
suggests.


---

## 11. Phase 2 completion — probabilistic scoring and delivered intervals

The two items left open at G2 sign-off are done.

**`scaled_crps` is a leaderboard column.** Scored cross-conformally, so the probabilistic
figure is held to the same held-out discipline as coverage. The quantile grid travels in
`Manifest.crps_quantiles` — `[80, 95]` gives a five-point grid, and a run at `[50, 80, 95]`
gives seven, so the two are **not CRPS-comparable**. A leaderboard that hid that would be
quietly dishonest.

Confirmed in passing: CRPS and MASE rank the models identically. That is the expected
consequence of ADR-006 — every band is `point ± q(that model's own residuals)`, so the
probabilistic ranking largely tracks the point ranking. CRPS earns its place by pricing
interval *width*, not by reordering. The methodology page must say so rather than let a
reader discover it.

**The delivered forecast is banded.** `XLF_Forecast` now carries `y_hat_lo_L`/`y_hat_hi_L`
per requested level, generated from `sorted(levels)` rather than hard-coded to 80/95 — the
original layout could not express `levels=[50, 80, 95]` at all. Delivered bands are
calibrated from every fold, unlike the scoring bands which each hold one out.

**Two gaps found while wiring, both structural rather than arithmetic:**

- `ConformalBands` had no `model` field. Half-widths are keyed by series, so two models'
  bands at the same level were indistinguishable and a lookup returned whichever came first.
  Silent, and wrong in a way no test on a single model would have caught.
- `select()` ran but its result never reached the leaderboard rows. `selected` drives which
  model's forecast is written to `XLF_Forecast`, so the sheet fell back to the top-ranked
  model — right under `pooled`, wrong under `per_series`.

**387 tests. `conformal.py`, `ensemble.py`, `folds.py`, `registry.py` at 100%; engine 96%.**

---

## 12. Gate G3 — signed off

Benchmarked against the **Monash Time Series Forecasting Archive**'s published M3 baselines
(Godahewa et al. 2021), not against our own prior work. Baselines and tolerances were committed
before the phase opened, as the gate requires.

### Results — mean MASE, single origin, 3,003 series

| dataset | model | ours | published | delta |
|---|---|---:|---:|---:|
| M3 Yearly (645, h=6) | AutoETS | 2.695 | 2.860 | −5.8% |
| | AutoARIMA | 2.882 | 3.417 | **−15.7%** ⚑ |
| | DynOptTheta | 2.576 | 2.774 | −7.1% |
| M3 Quarterly (756, h=8) | AutoETS | 1.143 | 1.170 | −2.3% |
| | AutoARIMA | 1.191 | 1.240 | −3.9% |
| | DynOptTheta | 1.126 | 1.117 | +0.8% |
| M3 Monthly (1428, h=18) | AutoETS | 0.863 | 0.865 | −0.2% |
| | AutoARIMA | 0.876 | 0.873 | +0.3% |
| | DynOptTheta | 0.850 | 0.864 | −1.6% |

**9 of 9 comparisons pass. Zero failures. One flagged as better-than-published.**

M3 Other (174 series) was run for regression tracking but is excluded from the gate: the paper
reports no baselines for it — the string "M3 Other" does not appear — and the series are undated,
declaring neither `start_timestamp` nor `@frequency`.

### On the flagged case

`AutoARIMA` beats the archive by 15.7% on M3 Yearly, which trips the flag. Three things say it is
real rather than a leak:

1. **Monthly agrees to 0.3%.** The same harness, split and metric reproduce the published numbers
   almost exactly on the largest dataset. A structural leak would not switch itself off.
2. **The archive's own ARIMA is anomalous on yearly** — 3.417, *worse* than its SES at 3.167.
   `auto.arima` is known to struggle on short yearly series (14–41 training points), so a
   reimplementation beating it comfortably is unsurprising.
3. **The split was verified directly**: train + test reconstructs every series exactly, every test
   block is exactly `h`, and no test observation appears in any training window.

### What makes the comparison trustworthy

Two properties are asserted as tests rather than assumed, because either would silently invalidate
every number above:

- **Our MASE is the archive's MASE.** `utilsforecast.losses.mase` matches the paper's equation 2
  to 1e-12 across seasonalities 1, 4 and 12, with the denominator taken over the training series
  only. Had it differed, the deltas would have been measuring the metric, not the engine.
- **The split is the archive's split.** Single origin, last `horizon` observations held out.

### Measured cost on real data

M3 Monthly, 1,428 series, one fit each: **943 s wall, AutoARIMA 91% of it** at 0.60 CPU s/series
at `m=12`. For context, the Phase 0 synthetic spike measured 0.923 s/series for seasonal AutoARIMA
at `m=52` and 0.297 s/series in FR-201a's Fourier mode. Real seasonal data at `m=12` sits between
them, which is consistent.

This does **not** re-validate NFR-01: that target is weekly data at `m=52` with 3 CV folds, where
FR-201a's Fourier mode applies, and this run is monthly at `m=12` with a single origin. It is a
real-data sanity check on the Phase 0 projection, not a replacement for it.

---

## 13. Phase 3 profiling — the optimisation we did not take

`AutoARIMA` is **91% of run cost** on M3 monthly (0.60 CPU s/series at m=12). Four levers were
measured on 250 real M3 monthly series, against **both** speed and accuracy:

| configuration | CPU | speedup | MASE | vs baseline |
|---|---:|---:|---:|---:|
| baseline (current) | 151.2 s | 1.00× | 1.0416 | — |
| `nmodels=20` | 113.3 s | 1.33× | 1.0436 | +0.20% |
| `approximation=True` | 51.3 s | **2.95×** | 1.0759 | **+3.30%** |
| `approximation=True, nmodels=20` | 39.9 s | 3.79× | 1.0594 | +1.71% |
| `max_p/max_q=3` | 148.4 s | 1.02× | 1.0415 | −0.01% |

**None were adopted.** `approximation=True` — which is what R's `auto.arima` does by default on
longer series — buys nearly 3× for 3.3% worse MASE. That is a real trade, and there is no reason
to make it: the Phase 0 projection puts a full NFR-01 run at 2.9 minutes against a 10-minute
budget, so we are not against the limit. Trading accuracy for speed we do not need, in a product
whose thesis is methodological rigour, is the wrong instinct.

The measurements are recorded so the lever exists when the budget does bind — larger panels,
more folds, or higher-frequency data.

**What was adopted instead: parallelism (FR-211).** The adapter had `n_jobs` hard-coded to 1.
Measured on 200 M3 monthly series:

| `n_jobs` | wall | speedup | forecasts identical to serial |
|---|---:|---:|---|
| 1 | 165.4 s | 1.00× | (reference) |
| 2 | 122.8 s | 1.35× | yes |
| 4 | 60.9 s | **2.72×** | yes |
| 8 | 69.9 s | 2.37× | yes |

`n_jobs=4` gives essentially the same speedup as `approximation=True` **at zero accuracy cost**,
because statsforecast parallelises across series and each series is fitted independently. That
property is now a test rather than an assumption, and the worker count is recorded in
`Manifest.thread_config` regardless.

Note that 8 workers is *slower* than 4 here — process-spawn overhead against a 200-series
workload — so the default stays 1 and the choice is the caller's (FR-211).
