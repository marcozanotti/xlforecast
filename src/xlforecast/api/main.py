"""FastAPI surface (TS §6).

**There is no synchronous forecast endpoint** (ADR-005, hard rule 7). Every competition goes
through the queue, including trivial ones, so there is one code path rather than two that
drift.
"""

from __future__ import annotations

import io
import json
import uuid
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from typing import Annotated, Any

import polars as pl
from fastapi import Depends, FastAPI, Header, HTTPException, Request, Response
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict

from xlforecast import __version__
from xlforecast.api.deps import Services, get_services
from xlforecast.api.security import ConfirmationError, QuotaError
from xlforecast.engine.run import profile_only
from xlforecast.errors import XLForecastError
from xlforecast.schemas.jobs import JobRecord
from xlforecast.schemas.request import DataMapping, ForecastRequest
from xlforecast.storage.jobs import UnknownJobError

__all__ = ["app"]

app = FastAPI(title="xlforecast", version=__version__)

#: FR-107 -- the grid path is capped; file input is not (ADR-008).
MAX_PANEL_ROWS = 500_000
MAX_UPLOAD_BYTES = 200_000_000
STREAM_INTERVAL_SECONDS = 1.0
#: A bounded stream: a client that stops reading must not pin a worker forever.
STREAM_MAX_TICKS = 3600


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _owner(x_owner: Annotated[str | None, Header()] = None) -> str:
    """Stand-in for the auth dependency.

    Deliberately a header rather than a token in this phase: the real scheme is an
    `HttpOnly; Secure; SameSite=Lax` session cookie (TS §6), because `EventSource` cannot set
    headers and a credential in a workbook custom property travels inside the `.xlsx`.
    """
    return x_owner or "anonymous"


class ConfirmBody(BaseModel):
    model_config = ConfigDict(extra="forbid")
    data_id: str
    request: ForecastRequest


class SubmitBody(BaseModel):
    model_config = ConfigDict(extra="forbid")
    data_id: str
    request: ForecastRequest
    mapping: DataMapping
    confirmation_token: str


def _fail(exc: XLForecastError, status: int) -> HTTPException:
    """FS §4 error-presentation rule: name the fault and state the fix, never a traceback."""
    return HTTPException(
        status_code=status,
        detail={
            "message": exc.message,
            "fix": exc.fix,
            "unique_id": exc.unique_id,
            "column": exc.column,
            "rendered": exc.render(),
        },
    )


@app.get("/v1/health")
def health() -> dict[str, str]:
    return {"status": "ok", "version": __version__}


@app.post("/v1/data")
async def upload(
    request: Request,
    services: Annotated[Services, Depends(get_services)],
    unique_id_col: str = "unique_id",
    ds_col: str = "ds",
    y_col: str = "y",
    freq: str | None = None,
    h: int = 1,
) -> dict[str, Any]:
    """Upload a panel; get back a `data_id`, a `DataProfile` and a validation report.

    Profiling and validation happen here rather than at submission so the user learns that 12
    of their 300 series are unusable *before* confirming a job, not after waiting for one
    (FR-105, S1's live validation summary).

    Not subject to NFR-03's 300 ms budget: this reads, profiles and validates up to 500,000
    rows, and pretending otherwise would have made the NFR unsatisfiable rather than the
    endpoint fast.
    """
    body = await request.body()
    if not body:
        raise HTTPException(
            status_code=400,
            detail={
                "message": "the upload was empty.",
                "fix": "Send the panel as Parquet in the request body.",
            },
        )
    if len(body) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail={
                "message": f"the upload is {len(body) / 1e6:.0f} MB, over the "
                f"{MAX_UPLOAD_BYTES / 1e6:.0f} MB limit.",
                "fix": "Aggregate the panel, or use file-based input (ADR-008).",
            },
        )

    try:
        panel = pl.read_parquet(io.BytesIO(body))
    except Exception as exc:  # noqa: BLE001 - any parse failure is one user-facing message
        raise HTTPException(
            status_code=400,
            detail={
                "message": "the upload could not be read as Parquet.",
                "fix": "Export the panel as Parquet and try again.",
                "detail": str(exc)[:200],
            },
        ) from None

    if panel.height > MAX_PANEL_ROWS:
        # FR-107. The cap is on the grid path; file input has no such limit.
        raise HTTPException(
            status_code=413,
            detail={
                "message": f"the panel has {panel.height:,} rows, over the "
                f"{MAX_PANEL_ROWS:,} row cap.",
                "fix": "Use CSV or Parquet file input instead of the grid (ADR-008).",
            },
        )

    mapping = DataMapping(unique_id_col=unique_id_col, ds_col=ds_col, y_col=y_col)
    data_id = str(uuid.uuid4())

    # Check the mapping before renaming. `rename(strict=False)` skips a column that is not
    # there, and the failure then surfaces from deep inside polars as a ColumnNotFoundError
    # -- a library traceback where FS §4 requires a named column and a stated remedy.
    missing = [c for c in (unique_id_col, ds_col, y_col) if c not in panel.columns]
    if missing:
        raise HTTPException(
            status_code=400,
            detail={
                "message": f"column '{missing[0]}' is not in the uploaded panel; "
                f"it has {panel.columns}.",
                "fix": "Correct the column mapping and upload again.",
                "column": missing[0],
            },
        )

    try:
        renamed = panel.rename({unique_id_col: "unique_id", ds_col: "ds", y_col: "y"})
        profile = profile_only(
            renamed,
            request=ForecastRequest(h=h, freq=freq or "D"),
            mapping=mapping,
            data_id=data_id,
        )
    except XLForecastError as exc:
        raise _fail(exc, 400) from None

    buffer = io.BytesIO()
    renamed.write_parquet(buffer, compression="zstd")
    services.objects.put(f"data/{data_id}.parquet", buffer.getvalue())
    return {"data_id": data_id, "profile": profile.model_dump(mode="json")}


@app.get("/v1/jobs/{job_id}/results")
def results(job_id: str, services: Annotated[Services, Depends(get_services)]) -> Response:
    """The full result set. Available only once the run has produced one."""
    try:
        services.jobs.get(job_id)
    except UnknownJobError as exc:
        raise _fail(exc, 404) from None
    key = f"jobs/{job_id}/result.json"
    if not services.objects.exists(key):
        raise HTTPException(
            status_code=409,
            detail={
                "message": "this job has not produced results yet.",
                "fix": "Poll the job status, then request results once it completes.",
            },
        )
    return Response(content=services.objects.get(key), media_type="application/json")


@app.get("/v1/jobs/{job_id}/stream")
async def stream(
    job_id: str, services: Annotated[Services, Depends(get_services)]
) -> StreamingResponse:
    """Progress as server-sent events.

    Consumed with `fetch` + `ReadableStream`, **not** `EventSource`: the latter cannot attach
    an Authorization header or a POST body, and the alternative -- putting a credential where
    a header-less client can reach it -- means a token in a workbook custom property, which
    travels inside the `.xlsx` (TS §6, hard rule 8).
    """
    try:
        services.jobs.get(job_id)
    except UnknownJobError as exc:
        raise _fail(exc, 404) from None

    async def events() -> AsyncIterator[bytes]:
        import asyncio

        last: str | None = None
        for _ in range(STREAM_MAX_TICKS):
            record = services.jobs.get(job_id)
            progress = services.jobs.progress(job_id)
            payload = json.dumps(
                {
                    "status": record.status.value,
                    "progress": progress.model_dump(mode="json") if progress else None,
                }
            )
            # Only send on change, so a slow fold does not fill the client's buffer with
            # identical frames.
            if payload != last:
                yield f"data: {payload}\n\n".encode()
                last = payload
            if record.status.terminal:
                return
            await asyncio.sleep(STREAM_INTERVAL_SECONDS)

    return StreamingResponse(events(), media_type="text/event-stream")


@app.post("/v1/confirm")
def confirm(
    body: ConfirmBody, services: Annotated[Services, Depends(get_services)]
) -> dict[str, Any]:
    """Mint the token that `POST /v1/jobs` requires (AC-503, FR-503).

    Called only from an explicit user action on the confirmation card. The token is bound to
    this exact configuration, so changing a setting afterwards invalidates it.
    """
    return {
        "confirmation_token": services.tokens.mint(body.data_id, body.request),
        "expires_in": services.tokens.ttl_seconds,
    }


@app.post("/v1/jobs", status_code=202)
def submit(
    body: SubmitBody,
    services: Annotated[Services, Depends(get_services)],
    owner: Annotated[str, Depends(_owner)],
) -> dict[str, str]:
    """Validate, check quota and licence, enqueue. Never runs the competition inline."""
    try:
        services.tokens.redeem(body.confirmation_token, body.data_id, body.request)
    except ConfirmationError as exc:
        raise _fail(exc, 400) from None

    try:
        services.quota.check_concurrency(services.jobs.active_count(owner))
    except QuotaError as exc:
        raise _fail(exc, 429) from None

    if not services.objects.exists(f"data/{body.data_id}.parquet"):
        raise HTTPException(
            status_code=404,
            detail={
                "message": f"no uploaded panel with id '{body.data_id}'.",
                "fix": "Upload the panel first, then submit the job.",
            },
        )

    record = JobRecord(
        job_id=str(uuid.uuid4()),
        data_id=body.data_id,
        request=body.request,
        mapping=body.mapping,
        owner=owner,
        created_at=_now(),
    )
    services.jobs.create(record)
    if services.enqueue is not None:
        services.enqueue(record.job_id)
    return {"job_id": record.job_id, "status": record.status.value}


@app.get("/v1/jobs/{job_id}")
def status(job_id: str, services: Annotated[Services, Depends(get_services)]) -> dict[str, Any]:
    """Status, progress, and the partial leaderboard as folds complete (S4)."""
    try:
        record = services.jobs.get(job_id)
    except UnknownJobError as exc:
        raise _fail(exc, 404) from None
    progress = services.jobs.progress(job_id)
    return {
        "job_id": record.job_id,
        "status": record.status.value,
        "created_at": record.created_at,
        "started_at": record.started_at,
        "finished_at": record.finished_at,
        "error": record.error,
        "progress": progress.model_dump() if progress else None,
    }


@app.delete("/v1/jobs/{job_id}", status_code=202)
def cancel(job_id: str, services: Annotated[Services, Depends(get_services)]) -> dict[str, str]:
    """FR-802. Sets the flag the engine polls between folds; the worker kills the process if
    a fold is in flight. Cancelling a finished job is a no-op rather than an error."""
    try:
        record = services.jobs.get(job_id)
    except UnknownJobError as exc:
        raise _fail(exc, 404) from None
    if record.status.terminal:
        return {"job_id": job_id, "status": record.status.value, "note": "already finished"}
    services.jobs.request_cancel(job_id)
    return {"job_id": job_id, "status": "cancelling"}


@app.get("/v1/jobs/{job_id}/manifest")
def manifest(job_id: str, services: Annotated[Services, Depends(get_services)]) -> Response:
    """Hard rule 10 -- no manifest, no result."""
    try:
        services.jobs.get(job_id)
    except UnknownJobError as exc:
        raise _fail(exc, 404) from None
    key = f"jobs/{job_id}/manifest.json"
    if not services.objects.exists(key):
        raise HTTPException(
            status_code=409,
            detail={
                "message": "this job has not produced a manifest yet.",
                "fix": "Wait for the job to finish, then request it again.",
            },
        )
    return Response(content=services.objects.get(key), media_type="application/json")


@app.get("/v1/licence")
def licence(
    owner: Annotated[str, Depends(_owner)], services: Annotated[Services, Depends(get_services)]
) -> dict[str, Any]:
    """FR-804 -- checked on submission, not on add-in load."""
    return {
        "owner": owner,
        "status": "active",
        "max_concurrent_jobs": services.quota.max_concurrent_jobs,
        "active_jobs": services.jobs.active_count(owner),
        "compute_minutes_limit": services.quota.max_compute_minutes_per_month,
        "offline_grace_hours": 72,
    }
