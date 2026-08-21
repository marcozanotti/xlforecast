# 03 — Build Plan

Sequenced to de-risk in the right order: **correctness before infrastructure, infrastructure before
UI, UI before AI.** Each phase ends with a gate. Do not start a phase until the previous gate passes.

Effort assumes one experienced developer working with an AI coding agent, part-time
(~10–12 hrs/week). Adjust the wall-clock, not the ordering.

---

## Phase 0 — Scaffolding
**Effort:** ~1 week

- **Spike first, before any repo file.** (a) Resolve the pin set in a throwaway 3.11 venv and
  confirm the stack imports. (b) Time `AutoARIMA` at `season_length=52` over 25 weekly series × 156
  points × 4 fits and extrapolate to 200 series. **Also time one local-ML learner per series**
  (`LocalLGBM`, 800 fits on ~91 training rows each) and one global fit, since FR-216 is now 12
  models and NFR-01's 10-minute target is unvalidated at that size. **Measure `train` and `predict`
  separately, in CPU seconds and wall seconds** (FR-217) — that is the cost proxy from here on, and
  the spike output is the first row of the S3 calibration table. Together these decide whether
  FR-216 holds and validate FR-201a/FR-203c. (c) Confirm Pydantic v2 behaviour for
  `dict[int, float]` and `float | None` round-trips. Output is a numbers memo, not code.
- `docs/04-SPEC-DECISIONS.md` — the resolutions the schema layer encodes. Schemas cannot be written
  honestly before this exists.
- Repo, `pyproject.toml`, `uv` lockfile, Python 3.11 pinned.
- Docker base image with the full Nixtla stack building cleanly. **Base on
  `python:3.11-slim-bookworm` (glibc 2.36).** `coreforecast` and `statsforecast` publish
  `manylinux_2_27`/`manylinux_2_28` wheels only — **Alpine/musl will not work** and has no sdist
  path worth taking.
- ruff + mypy + pytest + pre-commit + GitHub Actions CI. Enable the `pydantic.mypy` plugin or
  `--strict` fails on generated `__init__`s (NFR-11).
- `errors.py` — the `XLForecastError` hierarchy (TS §4.6), before the schemas that raise through it.
- All Pydantic schemas from Technical Spec §4, complete, with unit tests.

**On the numba/llvmlite risk this phase was built around: it no longer exists.** statsforecast 2.0
dropped numba in favour of compiled `coreforecast`; cp311 manylinux wheels are published for every
compiled dependency. The one way to reintroduce the problem is pinning `statsforecast < 2.0`, so the
`>= 2.0` floor *is* the mitigation. The real Phase 0 risks are the glibc floor above, image size
(`fugue` is a mandatory transitive dep of statsforecast and drags in `triad`/`adagio`; `mlforecast`
now pulls `optuna` and `narwhals`), and the AutoARIMA budget.

**Gate G0:** CI green. `import statsforecast, mlforecast, utilsforecast` succeeds in the container.
Every contract in TS §4 satisfies `T.model_validate(json.loads(m.model_dump_json())) == m` under
hypothesis, including `None` metrics and integer-keyed coverage dicts. The AutoARIMA spike has a
recorded number and FR-216's default model set is frozen against it.

---

## Phase 1 — Engine core
**Effort:** ~4 weeks · **The most important phase in the project**

*(+1 week over the original estimate: `engine/run.py` now owns the fold loop rather than delegating
to the libraries' `cross_validation` — see FR-206a. That buys the only real correctness guarantee in
the system and removes the ragged-panel leakage. It is not optional scope.)*

- `ingest/`: readers, profiling, validation (FR-105/FR-105a — profile **before** validate), gap
  filling, future-known exog rows (FR-111), `DataProfile`.
- `engine/folds.py`: panel-wide calendar cutoffs + `split()`, with the identical-**test-index** test
  on a ragged panel (AC-205).
- `engine/registry.py` with licence metadata and the `commercial_ok` gate.
- `engine/local.py` (statsforecast) and `engine/ml.py` — one mlforecast adapter parameterised by
  information set, giving `GlobalX` and `LocalX` from an identical feature recipe (FR-203a/b), with
  the registry test asserting matched-pair equivalence.
- `engine/evaluate.py` over `utilsforecast`.
- `engine/run.py` orchestrator producing a `RunResult` + `Manifest`.
- Typer CLI: `xlforecast run data.csv --h 12 --freq W --models auto`.

**Gate G1:**
- Runs end-to-end on a CSV, emits four tables + manifest, and a `RunTiming` with per-model
  `train`/`predict` seconds whose sum plus overhead accounts for total CPU time (FR-217/217a).
- Identical-test-index test passes on a ragged panel (AC-205); effective training rows per family
  per fold are recorded (AC-206).
- Reproducibility: same input + spec + recorded `thread_config` → byte-identical leaderboard, twice.
- **Seasonal** random-walk panel correctly reports that nothing beat `SeasonalNaive` (AC-406);
  **pure** random-walk panel correctly reports that AutoARIMA beat it (AC-406a).
- A panel with a fold-constant series and an all-zero evaluation window produces a complete
  leaderboard carrying `None` metrics, not `NaN` (FR-214).

---

## Phase 2 — Uncertainty, ensembles, selection
**Effort:** ~2 weeks

- `engine/conformal.py` with **cross-conformal** calibration (FR-302), support clipping (FR-307),
  the full fallback chain, and out-of-calibration empirical coverage.
- `engine/ensemble.py` — point methods plus vincentization and linear pooling.
- `engine/select.py` — pooled / per_series / clustered, with the low-`n_windows` warning.
- Ensembles wired **inside** the CV loop.

**Gate G2:** **Out-of-calibration** coverage within ±5pp of nominal on a synthetic panel with known
noise, with the in-calibration control asserted to be tighter (AC-301) — the control is what proves
the headline number is not a tautology. Removing the FR-307 clip demonstrably inflates intermittent
coverage (AC-307). Ensemble scores are computed on the same folds as their members, asserted via
`FoldScore.fold_index` equality, **and** ensemble weights are demonstrably leave-one-fold-out
(FR-405a): perturbing fold `k`'s errors must not change the weights used to score fold `k`.

---

## Phase 3 — Benchmark validation
**Effort:** ~1.5 weeks · **This is the credibility gate**

- `benchmarks/` harness for M5 and VN1 subsets.
- Compare against known-good results from your own prior work.
- Profile and optimise the obvious hot paths, ranked by measured per-model `train + predict` CPU
  seconds (FR-217) rather than by intuition about which models look expensive.
- Write the public methodology page draft — if you cannot explain the leaderboard in writing,
  the design is wrong.

**Gate G3:** Results on M5/VN1 subsets fall within a **committed tolerance band of a committed
baseline file** — named metrics and explicit tolerances, checked into `benchmarks/baselines/` before
Phase 3 opens. "Consistent with your published work" is not a gate: it has no metric, no tolerance
and no artifact, so it cannot fail — and this is the one gate explicitly empowered to kill the
project. Runtime meets NFR-01 at the FR-216 default set.
**If the engine is not competitive here, stop and fix it — do not proceed to the add-in.**

---

## Phase 4 — API and workers
**Effort:** ~2 weeks

- FastAPI app, all endpoints from Technical Spec §6.
- Redis + `arq`, with engine work in a `ProcessPoolExecutor` — cancellation terminates the process
  (FR-802), because asyncio cancellation cannot interrupt compiled CPU-bound code.
- Per-fold checkpointing: resume-from-last-completed-fold (FR-801) and the S4 partial leaderboard
  are the same mechanism.
- `/v1/confirm` and the confirmation-token gate on `/v1/jobs` (AC-503).
- Object storage abstraction, Parquet persistence, retention policy.
- Auth, quota, licence check on submission.
- Deploy: API to Cloud Run or Container Apps. **Worker to Container Apps (KEDA Redis scaler) or
  Cloud Run Jobs** — a Redis-polling arq worker on plain Cloud Run has no inbound request to scale
  on and never reaches zero (TS §2). The two platforms are not interchangeable here.

**Gate G4:** A job submitted via HTTP survives a worker restart by **resuming from its last
completed fold**, not restarting. Cancellation works mid-run and is observed to terminate the worker
process. Progress updates arrive per model per fold. Cold start to job-accepted is < 5 s p95.
(The original "cold start to first token" is NFR-05, an explanation-stream property that does not
exist until Phase 6 — a gate cannot test a later phase's feature.)

---

## Phase 5 — Excel add-in
**Effort:** ~3 weeks

- **ADR-002 spike first:** wall-clock for a 250,000-cell read through xlwings Server vs native
  Office.js. If xlwings is >2× slower, drop the licence and write Office.js directly. No sheet
  writer is written before this reports.
- xlwings Server project, manifest, sideload dev loop.
- Chunked range reader with progress; 500k-row cap enforced.
- Batched sheet writers for all five sheets — manifest inside the overwrite transaction (FR-703) —
  plus the FR-708 output-size precheck and the FR-703a workbook-state cases.
- Task pane states S1–S5, manual configuration form first — natural language comes later.
- Job reattachment via workbook custom properties (`job_id`/`data_id` only, never a credential).
- Streaming via `fetch` + `ReadableStream`, session-cookie auth (`EventSource` can do neither).

**Gate G5:** Full round trip in Excel for Windows, Mac and web: select range → configure manually →
run → **five** sheets written correctly, manifest included and refreshed on re-run (FR-703). Output
overflow refuses cleanly rather than truncating (FR-708). Verified on all three hosts — this is a
human checklist (TS §11), not an automatable gate.

---

## Phase 6 — LLM layer
**Effort:** ~2.5 weeks

- Provider interface + at least two implementations.
- `llm/parse.py` with `DataProfile` grounding, confirmation card, `assumptions` list.
- `explain/` artifact pack, complete.
- `llm/tools.py` read-only tool set; `llm/narrate.py` streaming agent.
- `llm/redact.py` — the sole path to a provider request (NFR-07).
- `llm/guardrail.py`: templating-first, unit-aware, precision-derived tolerance, allowlist
  (TS §9.4), plus the FR-606a causal lint, with the 100-case suite measured **pre-guardrail**.
- **Explain this cell** in the pane.

**Gate G6:** Across the regression suite, first-pass rejection < 5% and the sentence-strip path fires
zero times, in addition to zero unmatched numerals in the output (AC-605) — the post-guardrail
figure alone is true by construction and cannot fail. The causal-phrasing lint passes its
adversarial slice. Parser suite passes on 60 phrasings including legacy frequency aliases. No LLM
path can enqueue a job without a valid confirmation token (AC-503).

---

## Phase 7 — Productisation
**Effort:** ~3 weeks

- Licensing service + Stripe, trial flow, quota enforcement.
- Marketplace submission package: manifest, screenshots, privacy policy, support page.
- Onboarding: sample workbook, three-minute walkthrough.
- Methodology page published.
- Self-hosted Docker Compose bundle for enterprise pilots.

**Gate G7:** Submitted to Microsoft Marketplace. Ten external users onboarded without hand-holding.

---

## Deferred to v2 (do not build now)

Hierarchical reconciliation (`hierarchicalforecast`) · time series foundation models (Chronos,
TiRex-2, TimeGPT-as-API) · neural models · **direct multi-step for global models (FR-210 — demoted
from Must, where it contradicted this very list)** · database
connectors · forecast-stability monitoring across reruns · Google Sheets.

**Note:** forecast-stability monitoring — "your forecast for SKU X moved 38% since last cycle, and
here is what drove it" — is the strongest v2 candidate. It is defensible, it survives Copilot
commoditising the point forecast, and it is squarely your research. Design the manifest and result
persistence in Phase 1 so that cross-run comparison is possible later without a migration.
`Manifest.previous_job_id` and `data_id` (TS §4.5) are the join keys — added in **Phase 0**, because
there was previously no key on which to relate two runs of the same panel, and retrofitting one is
exactly the migration this note exists to avoid.

---

## Risk register

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| numba/llvmlite build friction under `uv` on 3.11 — **REAL, and it is in `shap`, not `statsforecast`** | Hit in Phase 0 | Medium | `shap` declares `numba` and `llvmlite` with **no lower bound**, so the resolver backtracks to `llvmlite 0.36` (pre-cp311) and tries to build LLVM from source. Floors `llvmlite>=0.43, numba>=0.60` pinned in the `explain` extra; regression-tested. The engine itself is numba-free — verified: a core install contains neither package |
| glibc floor — `coreforecast`/`statsforecast` ship manylinux_2_27/2_28 only | Medium | High | `python:3.11-slim-bookworm`. Alpine/musl is a dead end; settle it in Phase 0 |
| AutoARIMA at `m=52` blows NFR-01 (800 fits per run) | High | High | FR-201a Fourier mode; measure in the Phase 0 spike before freezing FR-216 |
| Ragged-panel leakage into global models via per-series cutoffs | High | High | Own fold loop, panel-wide cutoffs (FR-206/206a); AC-205 tests the test index, not the cutoffs |
| Circular conformal calibration reports tautological coverage | High | High | Cross-conformal (FR-302) plus the in-calibration control in AC-301 |
| Image size vs Phase 4 cold start | **Measured at Phase 0: 2.92 GB → 2.10 GB** | Medium | `xgboost` pulls `nvidia-nccl-cu12` (454 MB of CUDA, for multi-GPU training) unconditionally on Linux; excluded via `[tool.uv] override-dependencies`, CPU XGBoost verified unaffected. Remaining candidate is `llvmlite`+`numba` (208 MB, shap-only) — a worker image without the `explain` extra would shed them. Revisit at Phase 4 where cold start is measurable |
| Office.js behaviour differs across Windows/Mac/web | High | Medium | Test all three from the first add-in commit, not at the end |
| Marketplace certification rejection | Medium | Medium | Read the certification policies before Phase 7; batch submissions |
| Enterprise IT blocks the add-in | High | High | Self-hosted mode + data minimisation, both designed in from the start |
| Grid ingestion too slow at scale | Medium | Medium | Cap at 500k rows; push large panels to file input |
| LLM cost exceeds compute cost | Medium | Medium | Cache explanations; panel-level by default, per-series on demand; small model for parsing |
| Per-series selection overfits and users blame the tool | Medium | High | Pooled default + explicit warning + coverage reporting |
| Scope creep into planning-suite territory | High | High | The non-goals table in the Project Brief is binding |

---

## Validation checkpoints (independent of the build)

These run in parallel with development and can kill the project early, which is the point.

- **After G1:** show the CLI leaderboard to five demand planners. Do they understand it? Does
  "nothing beat seasonal naive" read as honest or as failure?
- **After G3:** publish the benchmark methodology. Solicit criticism from the forecasting community.
- **Before G5:** landing page, try to collect ten pre-payments at €50/month. If you cannot find ten
  people who will pay before the add-in exists, no amount of engine quality fixes that.
