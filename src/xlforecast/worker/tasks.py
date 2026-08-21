"""arq task definitions (TS §2, §3).

`arq` is chosen over Celery for operational simplicity, **not** for being async-native --
that buys nothing here, since the engine is CPU-bound compiled code. The task is a thin async
shell around a synchronous subprocess (`worker/executor.py`), which is where the real work
and the real cancellation happen.

The task body is written so it can be driven without a broker: `run_job` takes its
dependencies explicitly and `WorkerSettings` supplies them from the environment. That keeps
G4's assertions testable without standing up Redis.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, ClassVar

from xlforecast.schemas.jobs import JobProgress, JobStatus
from xlforecast.storage.jobs import JobStore
from xlforecast.storage.objects import LocalObjectStore, ObjectStore
from xlforecast.worker.executor import ExecutionOutcome, execute_job, now

__all__ = ["MAX_ATTEMPTS", "WorkerSettings", "run_job"]

#: arq redelivers a job whose worker died. FR-801 makes that cheap by resuming from
#: checkpoints, but a job that kills its worker every time must eventually stop rather than
#: cycle forever.
MAX_ATTEMPTS = 3


@dataclass(slots=True)
class WorkerDeps:
    jobs: JobStore
    objects: ObjectStore


def run_job(job_id: str, *, jobs: JobStore, objects: ObjectStore) -> ExecutionOutcome:
    """Execute one job end to end, recording every state transition.

    Synchronous by design. The arq entry point below awaits it in a thread so the event loop
    stays responsive; putting the engine on the loop itself would block every other task for
    the length of a competition.
    """
    record = jobs.get(job_id)

    if record.attempts >= MAX_ATTEMPTS:
        failed = record.model_copy(
            update={
                "status": JobStatus.FAILED,
                "finished_at": now(),
                "error": f"gave up after {record.attempts} attempts.",
            }
        )
        jobs.update(failed)
        return ExecutionOutcome(status=JobStatus.FAILED, error=failed.error)

    record = record.model_copy(
        update={
            "status": JobStatus.RUNNING,
            "started_at": record.started_at or now(),
            "attempts": record.attempts + 1,
        }
    )
    jobs.update(record)
    jobs.set_progress(
        JobProgress(
            job_id=job_id,
            status=JobStatus.RUNNING,
            folds_total=record.request.n_windows,
            folds_done=0,
            models_total=len(record.request.models),
            models_done_in_fold=0,
            updated_at=now(),
        )
    )

    outcome = execute_job(record, objects, is_cancelled=lambda: jobs.cancel_requested(job_id))

    jobs.update(
        record.model_copy(
            update={
                "status": outcome.status,
                "finished_at": now(),
                "error": outcome.error,
                "result_key": outcome.result_key,
            }
        )
    )
    jobs.set_progress(
        JobProgress(
            job_id=job_id,
            status=outcome.status,
            folds_total=record.request.n_windows,
            # A cancelled or failed run reports what it completed rather than claiming the
            # whole horizon: FR-802 keeps finished folds, and pretending otherwise would
            # misrepresent what the user can still download.
            folds_done=_folds_completed(job_id, objects),
            models_total=len(record.request.models),
            models_done_in_fold=0,
            message=outcome.error or ("cancelled mid-fold" if outcome.forced else None),
            updated_at=now(),
        )
    )
    return outcome


def _folds_completed(job_id: str, objects: ObjectStore) -> int:
    from xlforecast.worker.checkpoint import Checkpointer

    return len(Checkpointer(job_id=job_id, store=objects).completed())


async def arq_run_job(ctx: dict[str, Any], job_id: str) -> str:
    """arq entry point. Offloads to a thread so the loop is not blocked for minutes."""
    import asyncio

    deps: WorkerDeps = ctx["deps"]
    outcome = await asyncio.to_thread(run_job, job_id, jobs=deps.jobs, objects=deps.objects)
    return outcome.status.value


async def startup(ctx: dict[str, Any]) -> None:  # pragma: no cover - wiring
    from xlforecast.storage.jobs import InMemoryJobStore

    root = os.environ.get("XLF_OBJECT_ROOT", "/var/lib/xlforecast")
    ctx["deps"] = WorkerDeps(jobs=InMemoryJobStore(), objects=LocalObjectStore(root))


class WorkerSettings:  # pragma: no cover - wiring
    """`uv run arq xlforecast.worker.tasks.WorkerSettings`."""

    functions: ClassVar[list[Any]] = [arq_run_job]
    on_startup = startup
    #: One job at a time per worker: each already saturates its cores through the engine's
    #: own `n_jobs`, and overcommitting would slow every job rather than finish any sooner.
    max_jobs = 1
    #: Generous, because a competition legitimately takes minutes. FR-801's resume is what
    #: makes a timeout survivable.
    job_timeout = 3600
    keep_result = 3600
