"""Running a competition in a killable subprocess (FR-802, FR-801).

**Why a subprocess at all.** The engine is CPU-bound compiled code -- `coreforecast`,
LightGBM, XGBoost. `arq.abort_job` cancels an asyncio task, which such code never observes;
moving it to a thread does not help either, since `CancelledError` is delivered at await
points and there are none inside a 40-second AutoARIMA fit. The only way to stop work already
in flight is to end the process running it.

So cancellation has two halves, and both are needed:

* **Cooperative**, between folds: the child polls a cancel marker and stops cleanly, keeping
  every completed fold. This is what happens almost always, because folds are short relative
  to a human noticing they made a mistake.
* **Forced**, mid-fold: the parent terminates the child after a grace period. Completed folds
  still survive, because they were checkpointed as they finished.

Cancellation is signalled through the object store rather than shared memory, because parent
and child do not share an address space and, in production, do not share a machine either.
"""

from __future__ import annotations

import multiprocessing as mp
import time
from dataclasses import dataclass
from datetime import UTC, datetime

import polars as pl

from xlforecast.engine.run import run_from_frame
from xlforecast.schemas.jobs import JobRecord, JobStatus
from xlforecast.storage.objects import ObjectStore
from xlforecast.worker.checkpoint import Checkpointer, RunControl

__all__ = ["CANCEL_GRACE_SECONDS", "ExecutionOutcome", "cancel_marker", "execute_job"]

#: How long a cooperative stop is given before the process is terminated. One fold of a large
#: panel can exceed this, which is exactly why the forced path exists.
CANCEL_GRACE_SECONDS = 5.0

_POLL_INTERVAL = 0.05


def cancel_marker(job_id: str) -> str:
    return f"jobs/{job_id}/cancel"


def data_key(data_id: str) -> str:
    return f"data/{data_id}.parquet"


def result_key(job_id: str) -> str:
    return f"jobs/{job_id}/result.json"


def manifest_key(job_id: str) -> str:
    return f"jobs/{job_id}/manifest.json"


@dataclass(slots=True)
class ExecutionOutcome:
    status: JobStatus
    error: str | None = None
    result_key: str | None = None
    forced: bool = False
    resumed_folds: int = 0


def _child(record_json: str, store: ObjectStore) -> None:
    """Entry point in the subprocess. Loads the panel, runs, persists.

    Nothing is returned across the process boundary: results go to the object store, which is
    also what makes them survive the process being killed.
    """
    import io

    record = JobRecord.model_validate_json(record_json)
    panel = pl.read_parquet(io.BytesIO(store.get(data_key(record.data_id))))

    control = RunControl(
        checkpointer=Checkpointer(job_id=record.job_id, store=store),
        should_stop=lambda: store.exists(cancel_marker(record.job_id)),
    )
    result = run_from_frame(
        panel,
        request=record.request,
        mapping=record.mapping,
        job_id=record.job_id,
        control=control,
    )
    store.put(result_key(record.job_id), result.model_dump_json().encode())
    store.put(manifest_key(record.job_id), result.manifest.model_dump_json(indent=2).encode())


def execute_job(
    record: JobRecord,
    store: ObjectStore,
    *,
    is_cancelled: object,
    grace_seconds: float = CANCEL_GRACE_SECONDS,
    poll_interval: float = _POLL_INTERVAL,
) -> ExecutionOutcome:
    """Run one job in a subprocess, honouring cancellation.

    `is_cancelled` is polled by the *parent*; the child polls the object-store marker. Both
    exist because the parent can kill and the child can stop cleanly, and neither alone is
    sufficient.
    """
    # "spawn" rather than "fork": the parent may hold BLAS thread pools and an event loop,
    # and forking those into a child that then calls into compiled code is a well-known way
    # to deadlock.
    context = mp.get_context("spawn")
    process = context.Process(target=_child, args=(record.model_dump_json(), store), daemon=True)
    process.start()

    requested_at: float | None = None
    forced = False
    while process.is_alive():
        if requested_at is None and is_cancelled():  # type: ignore[operator]
            # Tell the child to stop between folds, then start the clock on forcing it.
            store.put(cancel_marker(record.job_id), b"1")
            requested_at = time.monotonic()
        elif requested_at is not None and time.monotonic() - requested_at > grace_seconds:
            process.terminate()
            forced = True
            break
        time.sleep(poll_interval)

    process.join(timeout=grace_seconds)
    if process.is_alive():  # pragma: no cover - terminate ignored, escalate
        process.kill()
        process.join()
        forced = True

    cancelled = requested_at is not None
    if cancelled:
        return ExecutionOutcome(status=JobStatus.CANCELLED, forced=forced)
    if process.exitcode != 0:
        return ExecutionOutcome(
            status=JobStatus.FAILED,
            error=f"the worker process exited with code {process.exitcode}.",
        )
    if not store.exists(result_key(record.job_id)):
        return ExecutionOutcome(
            status=JobStatus.FAILED, error="the run finished without producing a result."
        )
    return ExecutionOutcome(status=JobStatus.COMPLETED, result_key=result_key(record.job_id))


def now() -> str:
    return datetime.now(UTC).isoformat()
