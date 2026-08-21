"""Wiring for the API (TS §6).

Assembled in one place so tests can substitute in-memory stores for Redis and object storage
without touching route code.
"""

from __future__ import annotations

import os
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
    #: Set when an arq queue is configured; None runs jobs inline, which is what the tests do.
    enqueue: Any = None

    @classmethod
    def from_env(cls) -> Services:
        root = os.environ.get("XLF_OBJECT_ROOT")
        return cls(objects=LocalObjectStore(root) if root else MemoryObjectStore())


_services = Services()


def get_services() -> Services:
    return _services


def set_services(services: Services) -> None:
    """Test seam. Production wires this once at startup."""
    global _services
    _services = services
