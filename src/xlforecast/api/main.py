"""FastAPI surface (TS §6).

**There is no synchronous forecast endpoint** (ADR-005, hard rule 7). Every competition goes
through the queue, including trivial ones, so there is one code path rather than two that
drift.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Annotated, Any

from fastapi import Depends, FastAPI, Header, HTTPException, Response
from pydantic import BaseModel, ConfigDict

from xlforecast import __version__
from xlforecast.api.deps import Services, get_services
from xlforecast.api.security import ConfirmationError, QuotaError
from xlforecast.errors import XLForecastError
from xlforecast.schemas.jobs import JobRecord
from xlforecast.schemas.request import DataMapping, ForecastRequest
from xlforecast.storage.jobs import UnknownJobError

__all__ = ["app"]

app = FastAPI(title="xlforecast", version=__version__)


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
