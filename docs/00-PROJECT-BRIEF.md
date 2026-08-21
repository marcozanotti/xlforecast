# 00 — Project Brief

**Working name:** `xlforecast` (placeholder — rename before public release)
**Owner:** Marco Zanotti
**Status:** Pre-development
**Audience for this document set:** Claude Code (implementation agent) + owner

---

## 1. What we are building

An automated time series forecasting system with two front ends:

1. **A headless Python engine + CLI** (`xlforecast`) that takes a long-format panel and runs a
   reproducible model competition: local models, global models, cross-validation, calibrated
   probabilistic intervals, ensembling, and model selection.
2. **A Microsoft Excel add-in** that drives that engine from the spreadsheet, with a natural
   language layer for job specification and result explanation.

The engine is the product. The add-in is a distribution channel.

---

## 2. Why it might win

Existing Excel forecasting tools split into two groups:

- Free/native (Forecast Sheet, `FORECAST.ETS`, Copilot + Python) — single series, no competition,
  no calibration, no reproducibility.
- Enterprise planning suites (Forecast Pro, Blue Yonder, RELEX, SAP IBP) — $2k–$200k/yr, months of
  implementation, sold to procurement not to the analyst.

Nothing in between does an **honest, reproducible, multi-series model competition with calibrated
uncertainty** and reports its own accuracy against a baseline. That is the wedge.

**The differentiator is methodological rigour, not model count.** Every design decision below
protects that. If a feature would make the leaderboard misleading, it does not ship.

---

## 3. Non-goals (explicit — do not build these)

| Not building | Why |
|---|---|
| Inventory optimisation, replenishment, S&OP workflow | Different product, different buyer |
| Hierarchical reconciliation | Deferred to v2 — see Build Plan Phase 6 |
| Neural forecasting (NHITS, TFT, DeepAR) | Compute cost vs. perceptible accuracy gain is bad at MVP |
| Time series foundation models (Chronos, Moirai, TiRex, TimesFM) | Deferred to v2. GPU infra + per-model licence review. See §6 |
| A code interpreter for the LLM | Unauditable, non-reproducible. Contradicts the entire value prop |
| Letting the LLM select the forecasting model | Destroys reproducibility. CV selects; the LLM narrates |
| Multi-tenant SaaS billing, teams, SSO | Phase 7+, only after paid validation |
| Google Sheets support | Later, if at all |
| Real-time / streaming forecasts | Out of scope permanently |

---

## 4. Architectural decisions already made

These are settled. Do not relitigate them during implementation; raise an issue if evidence changes.

**ADR-001 — The engine is headless first, Excel second.**
The full competition runs as a Python package with a CLI, validated on the M3 competition data
before any add-in code exists. If the add-in turns out to be the wrong channel, the engine survives.

**ADR-002 — Excel integration via native Office.js. RESOLVED 2026-08-21: no xlwings PRO.**

The owner will not buy a licence, which settles it: xlwings Server requires a paid PRO
developer licence, so it is out and the Phase 5 spike is unnecessary. The add-in talks to
Office.js directly.

This costs less than it might appear, because the evidence had already eroded the case for
xlwings — see the struck-through rationale below. The Office.js surface we need is small and
bounded: a chunked range reader, five batched sheet writers, a custom-properties accessor and
a selection handler. That is a few hundred lines of JavaScript against a stable, free,
exhaustively documented API, and the task pane is already HTML + Alpine.js + Bootstrap, so we
were writing browser JS either way.

~~Two of the three rationales originally given for xlwings were already wrong.~~ *"Python
source stays server-side"* is delivered by the FastAPI architecture regardless — the engine is
behind an HTTP API either way, so no meaningful Python is client-side under either choice. And
*"works on Windows, macOS and Excel on the web"* is a property of Office.js, not of xlwings.
That left one real benefit — writing the add-in layer in Python — against a paid licence, a
small vendor on the critical path of our only distribution channel, and a cost that is not
obvious: xlwings Server routes **every** range operation through our server, so the chunked
read in TS §7.1 would have been N HTTPS round trips carrying spreadsheet data rather than N
local `context.sync()` calls. That works against the throughput constraint in §5 and puts grid
I/O on our infrastructure bill.

The cost decision and the technical evidence point the same way, which is a comfortable place
to end up. Native Office.js also removes the cross-host debugging indirection that gate G5
exists to exercise — we now debug Office.js directly rather than at one remove.

**ADR-003 — No client-side Python (Pyodide/WASM).**
The engine ships platform-specific compiled extensions with no Emscripten/Pyodide wheels:
`coreforecast` (C++), `statsforecast`'s bundled `_lib` shared object, and LightGBM/XGBoost for
`mlforecast`. None of these build for WASM today. Rules out xlwings Lite / xlwings Wasm for the
engine. Revisit only if the Nixtla stack ships Pyodide wheels.

*Corrected 2026-08-20.* This ADR previously argued from `statsforecast → numba → llvmlite`. That
chain no longer exists: **statsforecast 2.0 dropped numba entirely.** Verified dependency set of
2.1.1 is `cloudpickle, coreforecast>=0.0.17, numpy, pandas<3.0, scipy, statsmodels>=0.14.5, tqdm,
fugue>=0.9.4, utilsforecast>=0.1.4, threadpoolctl>=3` — no numba, no llvmlite. The conclusion
survives; the reasoning did not.

Note for completeness: `numba` *does* enter the project, via `shap` in the `explain` extra, and
Phase 0 had to floor it (see `05-PHASE0-SPIKE.md`). That does not disturb this ADR — the explain
layer is server-side and was never a candidate for the browser — but it does mean "we have no
numba dependency" is not a true sentence about the project as a whole, only about the engine.

**ADR-004 — Python in Excel is not a delivery target.**
It ships a fixed, curated Anaconda distribution. Arbitrary PyPI packages (`statsforecast`,
`mlforecast`) cannot be installed into it.

**ADR-005 — Long-running work is asynchronous, always.**
A competition over hundreds of series takes minutes to hours. Submit → job id → poll → write back.
No synchronous forecast endpoint exists, even for tiny jobs. One code path.

**ADR-006 — Conformal prediction is the unifying uncertainty layer.**
Parametric intervals (ARIMA/ETS) and empirical quantiles are not comparable. Every model in the
leaderboard is scored on conformalised intervals produced by the same procedure, so probabilistic
scores are commensurable. Native intervals may be reported *alongside* but never mixed into the
comparison.

**Amended 2026-08-20 — the procedure, not the principle.** Two defects in the original
specification made this ADR deliver the *appearance* of calibration rather than calibration:

1. *Circular coverage.* Calibrating bands on the CV residuals and then measuring coverage on those
   same residuals is tautological — an empirical quantile covers its own sample at the nominal rate
   by construction. Coverage must therefore be **cross-conformal**: the band applied to fold *k* is
   calibrated only from folds ≠ *k*, and reported coverage is averaged over out-of-calibration
   folds. Final delivered bands still use all folds' residuals. Costs no additional model fits.
2. *Symmetric additive bands on count data.* Absolute-residual half-widths put the lower bound
   below zero on intermittent demand — measured at 95.8% of points. Bands are clipped to the
   series' observed support, which removes 21.6% of interval width.
   *Amended again in Phase 2, after measurement:* clipping does **not** change coverage and cannot,
   since no observation lies below zero to be excluded. The real pathology is one-sided
   miscoverage — 0.00% lower tail against 15.62% upper on intermittent data — which the coverage
   figure hides entirely and which is now reported separately (FR-307a/b).

Known consequence to state on the methodology page: putting every model through one conformal
procedure makes the probabilistic ranking track the point ranking closely, since each band is
`point ± q(that model's own residuals)`. CRPS earns its place by pricing width, not by reordering
the leaderboard, and we should say so rather than let a reader discover it.

**ADR-007 — The LLM writes configs and prose. The engine writes numbers.**
No number reaches the user without passing through forecasting code. Enforced by an automated
numeric guardrail (see Technical Spec §9.4).

**ADR-008 — Excel is a control surface, not a data store.**
Grid ingestion is supported up to a hard cap. Beyond that, data comes from CSV/Parquet/database.
Designed in from day one, not retrofitted.

**ADR-009 — We do not depend on `timecopilot`.**
It is pre-alpha (v0.0.x), two maintainers, and its agent performs model selection — which
contradicts ADR-007. We wrap models directly. TimeGPT may be added later as an API-only option.

---

## 5. Hard constraints

| Constraint | Value | Consequence |
|---|---|---|
| Excel grid rows | 1,048,576 | Grid ingestion capped at 500,000 rows (FR-107) |
| Excel grid rows, **output** | 1,048,576 | `XLF_Forecast_Long` holds every model's forecast: series × h × models × (1 + 2·levels). A 2,000-series panel at h=52 with 9 models and 2 levels is ≈4.7M rows — **it overflows**. The input cap does *not* imply output headroom. Output size is prechecked (FR-708) |
| Office.js range read throughput | ~50k cells per `context.sync()` before UX degrades | Chunked reads, mandatory |
| Add-in runtime | Sandboxed browser control | No local filesystem, no subprocess, no local Python |
| Marketplace add-in monetisation | Free-to-download only | Licensing/billing must be built in-house |
| Marketplace certification | ~14–28 business days per submission | Batch changes; do not plan weekly releases |
| Enterprise IT | Third-party add-ins are frequently blocked | Bring-your-own-LLM-endpoint and data-minimisation are requirements, not nice-to-haves |

---

## 6. Model licensing register

Maintain this table in the repo. Any model added to the registry must have a row before it ships.

| Model | Source | Licence | Commercial use | Status |
|---|---|---|---|---|
| AutoARIMA, AutoETS, AutoCES, Theta, Croston, ADIDA, IMAPA, SeasonalNaive, HistoricAverage, WindowAverage, ZeroModel | `statsforecast` | Apache-2.0 | Yes | **In MVP** |
| LightGBM / XGBoost / CatBoost via `mlforecast` — global **and** local (FR-203a) | Nixtla + vendors | Apache-2.0 (XGBoost, mlforecast) / MIT (LightGBM) | Yes | **In MVP** |
| `LinearRegression` (Global/LocalLinear) | scikit-learn | BSD-3-Clause | Yes | **In MVP** |
| Chronos / Chronos-Bolt | Amazon | Apache-2.0 | Yes | v2 candidate |
| TiRex-2 | NXAI | Apache-2.0 | Yes | v2 candidate |
| TiRex (v1) | NXAI | NXAI Community Licence | **Requires review** | Blocked pending legal check |
| Moirai 1.0-R | Salesforce | CC-BY-NC-4.0 | **No** | **Prohibited** |
| TimeGPT | Nixtla | Commercial API | Yes, **paid** | **Excluded.** The project buys no licences (2026-08-21), which rules out paid model APIs on the same grounds as xlwings PRO |

---

## 7. Glossary

- **Panel** — a set of time series in long format, keyed by `unique_id`.
- **Local model** — fitted independently per series. Includes both statistical models (ARIMA, ETS,
  Theta, seasonal naive, moving average) and machine-learning models fitted per series
  (`LocalLGBM`, `LocalXGB`, `LocalLinear`).
- **Global model** — one model fitted across all series in the panel (`GlobalLGBM` via `mlforecast`).
- **Information set** — what a model was allowed to see: `own_series` or the whole `panel`. Cuts
  across family: a `LocalLGBM` is an ML model with an own-series information set. The three
  learners × two information sets form three **matched pairs**, which is how the leaderboard
  isolates the effect of pooling (FR-203b).
- **Competition** — cross-validated evaluation of all enabled models on identical folds.
- **Leaderboard** — ranked competition results, always including a seasonal-naive baseline.
- **Selection** — the rule mapping leaderboard → chosen model(s). Per-series, pooled, or clustered.
- **Calibration** — conformal adjustment of interval width using CV residuals.
- **Artifact pack** — the deterministic set of diagnostic objects that the explanation LLM reads.
- **Job** — one asynchronous competition run, identified by a UUID.

---

## 8. Success criteria for v1

1. On the M3 competition data, the engine is competitive with the published baselines in the
   Monash Time Series Forecasting Archive, and beats seasonal naive on the panel-level aggregate.

   *Changed from M5/VN1 in Phase 3.* M5 is 42,840 hierarchical Walmart series and VN1 is similarly
   large; neither resembles what this product's users own. M3 is 3,003 series across four
   frequencies with short histories — 645 yearly series of 20–47 observations, 1,428 monthly of
   66–144 — which is much closer to P1's 50–2,000 SKUs and P2's 5–50 revenue lines. It also has
   **externally published baselines**, which makes the gate falsifiable by someone who is not us.
2. Two identical job specs on identical data, **run under the same recorded thread configuration**,
   produce byte-identical leaderboards. Thread counts and BLAS vendor are part of the manifest
   because float reductions reorder under them; see NFR-02.
3. A 200-series × 156-week competition with the 13-model default set (FR-216) and 3 CV windows
   completes in under 10 minutes on 8 vCPUs. **Projected at 2.9 min by the Phase 0 spike (docs/05)**;
   re-measured on real panels at Phase 3, which is the figure that counts.
4. Empirical coverage of the 80% and 95% conformal intervals is within ±5 percentage points of
   nominal, measured **out of calibration** (cross-conformal, ADR-006), reported separately for
   intermittent and smooth series.
5. Zero numeric hallucinations across a 100-case explanation regression suite.
6. A non-technical user can go from raw range to written forecast in under three minutes without
   reading documentation.
