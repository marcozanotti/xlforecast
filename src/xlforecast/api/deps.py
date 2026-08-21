"""Wiring for the API (TS §6).

Assembled in one place so tests can substitute in-memory stores for Redis and object storage
without touching route code.

Two deployment shapes are supported, and the difference matters:

* **With `REDIS_URL`** -- the real one. Job state, the token replay set and the queue all live
  in Redis, so the API and the worker see the same jobs and several API instances agree.
* **Without it** -- a development fallback that runs each job in a background thread of the
  API process. It uses the *same* subprocess executor, so cancellation, checkpointing and
  resume behave identically; what it does not give you is durability across an API restart or
  more than one API instance. It exists so the add-in can be exercised without standing up a
  broker, and it says so loudly rather than looking like production.
"""

from __future__ import annotations

import contextlib
import os
import threading
from dataclasses import dataclass, field
from typing import Any

from xlforecast.api.security import Quota, TokenService
from xlforecast.storage.jobs import InMemoryJobStore
from xlforecast.storage.objects import LocalObjectStore, MemoryObjectStore, ObjectStore

__all__ = ["Services", "get_services", "set_services"]


@dataclass
class Services:
    jobs: Any = field(default_factory=InMemoryJobStore)
    objects: ObjectStore = field(default_factory=MemoryObjectStore)
    tokens: TokenService = field(
        default_factory=lambda: TokenService(
            # No default in production: an unset signing key must fail loudly rather than
            # silently accept forged confirmations (TS §12 -- secrets via env/Key Vault).
            secret=os.environ.get("XLF_TOKEN_SECRET", "dev-only-insecure").encode()
        )
    )
    quota: Quota = field(default_factory=Quota)
    #: Called with a job id once the job is accepted. `None` means nothing runs it, which is
    #: right for unit tests and wrong for anything else.
    enqueue: Any = None
    #: True when jobs run in-process. Surfaced by `/v1/health` so a deployment cannot quietly
    #: be in development mode without anyone noticing.
    inline: bool = False

    @classmethod
    def from_env(cls) -> Services:
        root = os.environ.get("XLF_OBJECT_ROOT")
        objects: ObjectStore = LocalObjectStore(root) if root else MemoryObjectStore()
        secret = os.environ.get("XLF_TOKEN_SECRET", "dev-only-insecure").encode()
        redis_url = os.environ.get("REDIS_URL")

        if redis_url:
            import redis as redis_sync

            from xlforecast.storage.redis_backend import RedisJobStore, RedisReplayStore

            client: Any = redis_sync.from_url(redis_url)  # type: ignore[no-untyped-call]
            return cls(
                jobs=RedisJobStore(client=client),
                objects=objects,
                tokens=TokenService(secret=secret, replay=RedisReplayStore(client=client)),
                enqueue=_arq_enqueue(redis_url),
                inline=False,
            )

        memory_jobs = InMemoryJobStore()
        return cls(
            jobs=memory_jobs,
            objects=objects,
            tokens=TokenService(secret=secret),
            enqueue=_inline_enqueue(memory_jobs, objects),
            inline=True,
        )


def _arq_enqueue(redis_url: str) -> Any:
    """Hand the job to an arq worker over Redis."""

    async def enqueue(job_id: str) -> None:
        from arq import create_pool
        from arq.connections import RedisSettings

        pool = await create_pool(RedisSettings.from_dsn(redis_url))
        try:
            await pool.enqueue_job("arq_run_job", job_id)
        finally:
            await pool.aclose()

    return enqueue


def _inline_enqueue(jobs: Any, objects: ObjectStore) -> Any:
    """Development fallback: run the job in a background thread of this process.

    Deliberately the same `run_job` the arq worker calls, so it spawns the same subprocess and
    honours the same cancellation and checkpointing. The difference is durability, not
    behaviour -- which is what makes it useful for exercising the add-in and useless as a
    deployment.
    """

    async def enqueue(job_id: str) -> None:
        from xlforecast.worker.tasks import run_job

        def target() -> None:
            # The failure is already recorded on the job record by `run_job`; re-raising here
            # would only kill an anonymous thread nobody is watching.
            with contextlib.suppress(Exception):
                run_job(job_id, jobs=jobs, objects=objects)

        threading.Thread(target=target, name=f"xlf-job-{job_id}", daemon=True).start()

    return enqueue


_services: Services | None = None


def get_services() -> Services:
    global _services
    if _services is None:
        _services = Services.from_env()
    return _services


def set_services(services: Services) -> None:
    """Test seam. Production builds this once, lazily, from the environment."""
    global _services
    _services = services
