"""Per-fold checkpointing (FR-801) and cancellation polling (FR-802).

FR-801 says jobs "survive worker restarts". The original spec never said what that means --
restart from scratch, or resume? Restarting a 40-minute competition because a worker was
rescheduled is not surviving it in any useful sense, so this resumes from the last completed
fold.

The same mechanism serves S4's partial leaderboard: a fold's scores are durable the moment it
finishes, so the pane can show a leaderboard built from completed folds while later ones are
still running. One mechanism, two requirements.
"""

from __future__ import annotations

import io
import json
from dataclasses import dataclass, field
from typing import Protocol

import polars as pl

from xlforecast.schemas.results import FoldScore
from xlforecast.storage.objects import ObjectNotFoundError, ObjectStore

__all__ = ["Checkpointer", "ProgressSink", "RunControl"]


class ProgressSink(Protocol):
    """Called as work completes. Per model per fold, because that is what a user can read."""

    def __call__(self, *, fold_index: int, models_done: int, current_model: str | None) -> None: ...


@dataclass(slots=True)
class Checkpointer:
    """Durable per-fold scores, keyed by job and fold."""

    job_id: str
    store: ObjectStore

    def _scores_key(self, fold_index: int) -> str:
        return f"jobs/{self.job_id}/folds/{fold_index:04d}.json"

    def _preds_key(self, fold_index: int) -> str:
        return f"jobs/{self.job_id}/folds/{fold_index:04d}.parquet"

    def save(self, fold_index: int, scores: list[FoldScore], predictions: pl.DataFrame) -> None:
        """Both halves, because a fold is only resumable if both survive.

        Scores alone are not enough: the conformal layer calibrates from *residuals*, which
        it derives from the fold's predictions. A checkpoint carrying only scores resumes
        without error and then produces a run with no prediction intervals -- a silent
        degradation, which is the worst kind.
        """
        buffer = io.BytesIO()
        predictions.write_parquet(buffer, compression="zstd")
        # Predictions first: `completed()` keys off the scores file, so writing that last
        # means a fold is never advertised as resumable before its predictions exist.
        self.store.put(self._preds_key(fold_index), buffer.getvalue())
        payload = "[" + ",".join(s.model_dump_json() for s in scores) + "]"
        self.store.put(self._scores_key(fold_index), payload.encode())

    def load(self, fold_index: int) -> tuple[list[FoldScore], pl.DataFrame] | None:
        try:
            raw = self.store.get(self._scores_key(fold_index))
            parquet = self.store.get(self._preds_key(fold_index))
        except ObjectNotFoundError:
            return None
        scores = [FoldScore.model_validate(row) for row in json.loads(raw)]
        return scores, pl.read_parquet(io.BytesIO(parquet))

    def completed(self) -> set[int]:
        """Folds with *both* halves present. A half-written fold is simply recomputed."""
        prefix = f"jobs/{self.job_id}/folds/"
        keys = set(self.store.list_prefix(prefix))
        done = set()
        for key in keys:
            if not key.endswith(".json"):
                continue
            index = int(key.removeprefix(prefix).removesuffix(".json"))
            if self._preds_key(index) in keys:
                done.add(index)
        return done

    def clear(self) -> None:
        for key in self.store.list_prefix(f"jobs/{self.job_id}/folds/"):
            self.store.delete(key)


@dataclass(slots=True)
class RunControl:
    """The hooks `engine/run.py` needs to be resumable, cancellable and observable.

    All three are optional: the CLI passes none of them and behaves exactly as before, which
    keeps the engine usable without any of this machinery (ADR-001 -- the engine is the
    product, the service is a wrapper).
    """

    checkpointer: Checkpointer | None = None
    #: Polled between folds. Returning True stops the run cleanly, retaining completed folds.
    #: It cannot interrupt a fold in flight -- that is the worker's process kill (FR-802).
    should_stop: object | None = None
    progress: ProgressSink | None = None
    resume: bool = True
    #: Folds actually recovered from a checkpoint, for the run summary.
    resumed_folds: set[int] = field(default_factory=set)

    def stop_requested(self) -> bool:
        return bool(self.should_stop and self.should_stop())  # type: ignore[operator]

    def report(self, *, fold_index: int, models_done: int, current_model: str | None) -> None:
        if self.progress is not None:
            self.progress(
                fold_index=fold_index, models_done=models_done, current_model=current_model
            )
