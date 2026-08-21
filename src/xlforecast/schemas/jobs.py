"""Job state contracts (TS §6, FR-801/802/803).

Separate from `results.py` because these are *operational* state, not results: they change
while a job runs, they are not part of the reproducibility record, and none of them belongs
in a manifest.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from xlforecast.schemas.request import DataMapping, ForecastRequest
from xlforecast.schemas.results import Leaderboard

__all__ = ["JobProgress", "JobRecord", "JobStatus"]


class JobStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    #: FR-803 -- a job that exhausts quota mid-run finishes the fold in flight, then stops
    #: with partial results retained rather than being discarded.
    QUOTA_EXHAUSTED = "quota_exhausted"

    @property
    def terminal(self) -> bool:
        return self in {
            JobStatus.COMPLETED,
            JobStatus.FAILED,
            JobStatus.CANCELLED,
            JobStatus.QUOTA_EXHAUSTED,
        }


class JobProgress(BaseModel):
    """What the pane polls (TS §7.3) and what S4 renders.

    Progress is reported **per model per fold** because that is the granularity a user can
    interpret: "AutoARIMA, fold 2 of 3" means something, "47%" does not.
    """

    model_config = ConfigDict(extra="forbid")

    job_id: str
    status: JobStatus
    folds_total: int = Field(ge=0)
    folds_done: int = Field(ge=0)
    models_total: int = Field(ge=0)
    models_done_in_fold: int = Field(ge=0)
    current_model: str | None = None
    #: FR-S4: the partial leaderboard streams in as models finish, so the user sees value
    #: before the job ends. Recomputed from completed folds only -- never extrapolated.
    partial_leaderboard: Leaderboard | None = None
    message: str | None = None
    updated_at: str

    @property
    def fraction(self) -> float:
        total = self.folds_total * self.models_total
        if total == 0:
            return 0.0
        done = self.folds_done * self.models_total + self.models_done_in_fold
        return min(done / total, 1.0)


class JobRecord(BaseModel):
    """Durable job identity. Survives worker restarts (FR-801)."""

    model_config = ConfigDict(extra="forbid")

    job_id: str
    data_id: str
    request: ForecastRequest
    mapping: DataMapping
    owner: str
    status: JobStatus = JobStatus.QUEUED
    created_at: str
    started_at: str | None = None
    finished_at: str | None = None
    error: str | None = None
    #: Set when the engine ran to completion; the results live in the object store.
    result_key: str | None = None
    #: How many times this job has been picked up. arq is at-least-once, so a job can be
    #: redelivered after a worker dies -- FR-801 resumes from checkpoints rather than
    #: restarting, and this counts the redeliveries so a poison job can be capped.
    attempts: int = 0
