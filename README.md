# xlforecast

[![CI](https://github.com/marcozanotti/xlforecast/actions/workflows/ci.yml/badge.svg)](https://github.com/marcozanotti/xlforecast/actions/workflows/ci.yml)

Reproducible time series model-competition engine with an Excel add-in front end.

The engine is the product; the add-in is a distribution channel. What differentiates it is
methodological rigour: a leaderboard that always carries a seasonal-naive baseline and the
user's incumbent moving average, that says so plainly when nothing beat them, and that can be
reproduced exactly from a manifest.

Specifications live in [`docs/`](docs/). Read `00-PROJECT-BRIEF.md` first, then `CLAUDE.md`.

## Status

**Phase 1 complete — gates G0 and G1 passed.** 284 tests; engine 95% covered, ingest 92%,
schemas 99%, `engine/folds.py` 100%. `uv run xlforecast run panel.csv --h 13 --freq W` runs a
full competition and writes four tables plus a reproducibility manifest.

**Phase 0 — gate G0 passed.** CI green (lint, strict types, 127 tests, 100% coverage
on `schemas/`), the Nixtla stack imports cleanly in the container, and every contract in
Technical Spec §4 round-trips through JSON without loss.

Measured cost per model is recorded in `docs/05-PHASE0-SPIKE.md`; FR-216 (13-model default set)
and NFR-01 are frozen against it at 2.9 min projected against a 10-minute budget.

Next: Phase 1 — `ingest/` and `engine/folds.py`. See `docs/03-BUILD-PLAN.md`.

## Development

```bash
uv sync --all-extras            # install
uv run pytest                   # all tests
uv run pytest tests/unit -x     # fast loop
uv run ruff check --fix .       # lint
uv run ruff format .            # format
uv run mypy src/xlforecast      # types (strict on engine/ and schemas/)
docker compose up               # full local stack
```
