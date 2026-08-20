# xlforecast

Reproducible time series model-competition engine with an Excel add-in front end.

The engine is the product; the add-in is a distribution channel. What differentiates it is
methodological rigour: a leaderboard that always carries a seasonal-naive baseline and the
user's incumbent moving average, that says so plainly when nothing beat them, and that can be
reproduced exactly from a manifest.

Specifications live in [`docs/`](docs/). Read `00-PROJECT-BRIEF.md` first, then `CLAUDE.md`.

## Status

Phase 0 (scaffolding) — see `docs/03-BUILD-PLAN.md`.

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
