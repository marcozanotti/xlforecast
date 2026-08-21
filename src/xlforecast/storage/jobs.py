"""Durable job state (FR-801, FR-802).

A Protocol with an in-memory implementation. The Redis backend implements the same surface;
nothing above this layer knows which is in use, which is what lets the API and worker tests
run without a broker.

**Cancellation is a flag, not a signal.** FR-802 cannot be served by cancelling an asyncio
task -- the engine is CPU-bound compiled code and will not notice. So a cancel request sets a
flag here; the engine polls it between folds for a clean stop, and the worker kills the
process outright if a fold is mid-flight. Both paths are needed: polling alone cannot
interrupt a 40-second AutoARIMA fit, and killing alone loses the completed folds.
"""

from __future__ import annotations

from typing import Protocol

from xlforecast.errors import XLForecastError
from xlforecast.schemas.jobs import JobProgress, JobRecord

__all__ = ["InMemoryJobStore", "JobStore", "UnknownJobError"]


class UnknownJobError(XLForecastError):
    """No job with that id."""


class JobStore(Protocol):
    def create(self, record: JobRecord) -> None: ...
    def get(self, job_id: str) -> JobRecord: ...
    def update(self, record: JobRecord) -> None: ...
    def set_progress(self, progress: JobProgress) -> None: ...
    def progress(self, job_id: str) -> JobProgress | None: ...
    def request_cancel(self, job_id: str) -> None: ...
    def cancel_requested(self, job_id: str) -> bool: ...
    def list_for_owner(self, owner: str) -> list[JobRecord]: ...


class InMemoryJobStore:
    def __init__(self) -> None:
        self._records: dict[str, JobRecord] = {}
        self._progress: dict[str, JobProgress] = {}
        self._cancelled: set[str] = set()

    def create(self, record: JobRecord) -> None:
        self._records[record.job_id] = record

    def get(self, job_id: str) -> JobRecord:
        try:
            return self._records[job_id]
        except KeyError as exc:
            raise UnknownJobError(
                f"no job '{job_id}'", fix="Check the job id, or resubmit."
            ) from exc

    def update(self, record: JobRecord) -> None:
        self._records[record.job_id] = record

    def set_progress(self, progress: JobProgress) -> None:
        self._progress[progress.job_id] = progress

    def progress(self, job_id: str) -> JobProgress | None:
        return self._progress.get(job_id)

    def request_cancel(self, job_id: str) -> None:
        self.get(job_id)  # raises if unknown, so a typo is not silently a no-op
        self._cancelled.add(job_id)

    def cancel_requested(self, job_id: str) -> bool:
        return job_id in self._cancelled

    def list_for_owner(self, owner: str) -> list[JobRecord]:
        return [r for r in self._records.values() if r.owner == owner]

    def active_count(self, owner: str) -> int:
        """FR-803 -- concurrent-job quota."""
        return sum(1 for r in self.list_for_owner(owner) if not r.status.terminal)
