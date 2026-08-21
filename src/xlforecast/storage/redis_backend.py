"""Redis-backed job state and token replay (TS §2).

The production backends for the two Protocols that were in-memory through Phase 4. Same
surface, so nothing above this layer changes; what changes is that state survives a process
restart and is shared between API instances.

That second property is the one that mattered. The in-process replay set was correct for a
single API instance and quietly wrong for several: two instances behind a load balancer would
each accept the same confirmation token once, so a single confirmation could enqueue two jobs.
A replay check that does not work across instances is worse than none, because it reads as
protection.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from xlforecast.errors import XLForecastError
from xlforecast.schemas.jobs import JobProgress, JobRecord
from xlforecast.storage.jobs import UnknownJobError

__all__ = ["RedisJobStore", "RedisReplayStore"]


def _key(job_id: str) -> str:
    return f"xlf:job:{job_id}"


def _progress_key(job_id: str) -> str:
    return f"xlf:progress:{job_id}"


def _cancel_key(job_id: str) -> str:
    return f"xlf:cancel:{job_id}"


def _owner_key(owner: str) -> str:
    return f"xlf:owner:{owner}"


@dataclass(slots=True)
class RedisJobStore:
    """`JobStore` over Redis.

    `ttl_seconds` bounds how long finished jobs linger. It is not the retention policy --
    NFR-08 governs *panel data*, which lives in the object store; this is only the
    operational record, and losing it after a fortnight costs nothing.
    """

    client: Any
    ttl_seconds: int = 14 * 24 * 3600

    def create(self, record: JobRecord) -> None:
        pipe = self.client.pipeline()
        pipe.set(_key(record.job_id), record.model_dump_json(), ex=self.ttl_seconds)
        pipe.sadd(_owner_key(record.owner), record.job_id)
        pipe.expire(_owner_key(record.owner), self.ttl_seconds)
        pipe.execute()

    def get(self, job_id: str) -> JobRecord:
        raw = self.client.get(_key(job_id))
        if raw is None:
            raise UnknownJobError(f"no job '{job_id}'", fix="Check the job id, or resubmit.")
        return JobRecord.model_validate_json(raw)

    def update(self, record: JobRecord) -> None:
        self.client.set(_key(record.job_id), record.model_dump_json(), ex=self.ttl_seconds)

    def set_progress(self, progress: JobProgress) -> None:
        self.client.set(
            _progress_key(progress.job_id), progress.model_dump_json(), ex=self.ttl_seconds
        )

    def progress(self, job_id: str) -> JobProgress | None:
        raw = self.client.get(_progress_key(job_id))
        return JobProgress.model_validate_json(raw) if raw else None

    def request_cancel(self, job_id: str) -> None:
        self.get(job_id)  # raises if unknown, so a typo is not silently a no-op
        self.client.set(_cancel_key(job_id), "1", ex=self.ttl_seconds)

    def cancel_requested(self, job_id: str) -> bool:
        return bool(self.client.exists(_cancel_key(job_id)))

    def list_for_owner(self, owner: str) -> list[JobRecord]:
        ids = self.client.smembers(_owner_key(owner)) or set()
        out = []
        for raw_id in ids:
            job_id = raw_id.decode() if isinstance(raw_id, bytes) else str(raw_id)
            try:
                out.append(self.get(job_id))
            except UnknownJobError:
                # Expired out from under the owner index. Harmless: the index is a hint, and
                # a stale member must not fail a quota check.
                self.client.srem(_owner_key(owner), job_id)
        return out

    def active_count(self, owner: str) -> int:
        return sum(1 for r in self.list_for_owner(owner) if not r.status.terminal)


class ReplayError(XLForecastError):
    """The token has already been redeemed."""


@dataclass(slots=True)
class RedisReplayStore:
    """Single-use enforcement for confirmation tokens, shared across API instances.

    `SET NX` is the whole mechanism: the first caller to claim a signature wins, atomically,
    whichever instance it reached. The TTL matches the token's own lifetime, because a
    signature that can no longer be redeemed does not need remembering.
    """

    client: Any
    ttl_seconds: int = 1800

    def claim(self, signature: str) -> bool:
        """True if this caller claimed the signature first."""
        return bool(self.client.set(f"xlf:token:{signature}", "1", nx=True, ex=self.ttl_seconds))
