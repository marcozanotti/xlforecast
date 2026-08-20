# Golden / reproducibility fixtures

Phase 1 onward. Fixed synthetic panels with committed expected leaderboards, plus the
manifest-replay test: re-run from a stored `Manifest`, assert leaderboard identity.

NFR-02 is defined relative to a recorded `thread_config`, so these fixtures must record theirs.
Float reductions reorder under thread count and LightGBM is only deterministic at a fixed one — a
golden test that ignores this will flake on a differently-sized CI runner, and the tempting fix
(loosening the tolerance) is exactly what CLAUDE.md forbids.
