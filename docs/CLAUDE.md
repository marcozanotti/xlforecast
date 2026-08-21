# CLAUDE.md

Guidance for Claude Code working in this repository. Read `docs/00-PROJECT-BRIEF.md` before
starting any task.

---

## Project

`xlforecast` — a reproducible time series model-competition engine with an Excel add-in front end.
Full specifications live in `docs/`:

- `00-PROJECT-BRIEF.md` — scope, non-goals, architectural decisions, constraints
- `01-FUNCTIONAL-SPEC.md` — numbered requirements (FR-xxx) and acceptance criteria (AC-xxx)
- `02-TECHNICAL-SPEC.md` — architecture, contracts, engine internals, NFRs
- `03-BUILD-PLAN.md` — phase order and gates
- `04-SPEC-DECISIONS.md` — resolutions of spec conflicts; read it before touching schemas or `engine/`
- `05-PHASE0-SPIKE.md` — measured cost per model; FR-216/NFR-01 are frozen against it
- `06-METHODOLOGY.md` — the public methodology page. If a change makes this harder to write
  honestly, that is the signal to stop and reconsider the change
- `07-DEPLOYMENT.md` — topology, configuration and the constraints that bite late
- `08-ADDIN-CHECKLIST.md` — gate G5's host verification. Human-run; it cannot be automated
- `09-RUNNING-THE-ADDIN.md` — how to get the add-in loaded in Excel to run that checklist

---

## Commands

```bash
uv sync --all-extras            # install (Python)
cd addin && npm ci             # install (add-in)
cd addin && npx vitest run     # add-in tests
cd addin && npx tsc --noEmit   # add-in types
uv run pytest                   # all tests
uv run pytest tests/unit -x     # fast loop
uv run ruff check --fix .       # lint
uv run ruff format .            # format
uv run mypy src/xlforecast      # types (strict on engine/ and schemas/)
uv run xlforecast run <file>    # CLI
uv run uvicorn xlforecast.api.main:app --reload
uv run arq xlforecast.worker.tasks.WorkerSettings
docker compose up               # full local stack
```

---

## Hard rules — do not violate these

1. **`SeasonalNaive` is always in the model set.** It cannot be disabled by configuration, by the
   LLM, or by a code path. Enforced in `ForecastRequest` validation (FR-204).
2. **All models share identical CV folds.** `engine/folds.py` is the single source of truth: it
   computes one set of **panel-wide calendar cutoffs** and slices train/test itself. Never call
   `statsforecast.cross_validation` or `mlforecast.cross_validation` — neither accepts a cutoff
   array, and both derive per-series cutoffs from each series' own last timestamp, which leaks
   panel futures into global models on ragged panels. Drive `fit`/`predict` per fold instead. The
   test `test_folds.py::test_identical_test_index_across_families` must never be skipped or xfail.
3. **The LLM never produces a number, a metric, or a model choice.** It emits configs and prose only.
   Every numeral in generated prose passes `llm/guardrail.py` before reaching the user.
4. **No bulk panel data in any LLM request.** The LLM sees `DataProfile` and `ArtifactPack` only.
   Individual observed values *are* permitted when they arrive as named, per-point artifact fields
   (`Analogue.same_period_last_year`, `PeakDrop.value`, `CalendarContext.exog_flags`) — the artifact
   pack cannot do its job otherwise. What is forbidden is series, arrays, slices, or any path that
   hands the model unbounded `y`. Enforced by `llm/redact.py`: every payload is built from artifact
   objects and capped at `MAX_ARTIFACT_POINTS` per request. If you find yourself iterating a panel
   into a prompt, stop.
5. **Never mix conformal and native intervals in a ranked comparison.** Native intervals are a
   separate, labelled column set (FR-304).
6. **Ensembles are scored inside the CV loop** on the same folds as their members, never from final
   forecasts.
7. **No synchronous forecast endpoint.** Everything goes through the job queue, including trivial
   jobs. One code path (ADR-005).
8. **No client-side secrets in the add-in.** No `localStorage`, no `sessionStorage`, and **no
   credentials in workbook custom properties** — those travel inside the `.xlsx` when the file is
   emailed or synced. Auth is an `HttpOnly; Secure; SameSite=Lax` session cookie. Custom properties
   hold non-secret reattachment state only (`job_id`, `data_id`).
9. **Batch every Excel range operation.** Writes are one `context.sync()` per sheet, never
   per-cell and never per-row. Reads are chunked — a bounded loop of ~50k-cell syncs is required
   (TS §7.1), not forbidden; what is forbidden is a sync whose cost scales with cells rather than
   with chunks.
10. **Every run emits a `Manifest`** sufficient to reproduce it exactly. No manifest, no result.

---

## Conventions

- Python 3.11. Full type annotations. `mypy --strict` on `engine/` and `schemas/`.
- Pydantic v2 for every boundary: API, CLI, LLM output, worker payloads.
- Polars for ingestion and reshaping; convert to pandas only at the Nixtla API boundary, once.
- No bare `except`. Domain errors subclass `XLForecastError` and carry the offending
  `unique_id`/column so the UI can name it (FS §4, error presentation rule).
- Logging: `structlog`, JSON output, always bind `job_id`.
- Prompts live in `llm/prompts/*.md`, versioned, never inline in Python.
- Commit messages reference requirement IDs: `feat(engine): conformal calibration (FR-301, FR-302)`.

---

## Testing expectations

- New engine logic ships with unit tests in the same PR.
- Statistical behaviour is tested against synthetic panels with known DGPs, not just on shape.
- `engine/folds.py`, `engine/conformal.py` and `llm/guardrail.py` require 100% line coverage.
- Metric degeneracy is a first-class test case, not an edge case: every metric test includes a
  series that is constant within a training fold (MASE/RMSSE zero denominator) and an evaluation
  window that is all-zero (scaled-CRPS zero denominator). See FR-214.
- LLM tests use recorded responses so CI is deterministic and costs nothing.
- Do not weaken a failing correctness test to make CI green. Fix the code or escalate.

---

## When you are unsure

- If a change would make the leaderboard easier to produce but less honest, **do not make it** —
  raise it instead. Methodological rigour is the product.
- If a requested feature appears in the non-goals table in `00-PROJECT-BRIEF.md`, say so and stop.
- If a model needs adding, add its licence row to the register in `00-PROJECT-BRIEF.md` §6 first,
  and its registry entry in `engine/registry.py` — `ModelName` is validated against the registry,
  not a hard-coded `Literal`.
  Models with `commercial_ok = false` (e.g. Moirai 1.0-R, CC-BY-NC-4.0) must not ship.
- Prefer asking a clarifying question over inventing a requirement.
