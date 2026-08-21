"""Object store and job state abstractions (TS §3).

Both are Protocols with in-memory and filesystem implementations, so the API, worker and
their tests run without Redis or a cloud account. The production backends implement the same
surface.
"""

from xlforecast.storage.jobs import InMemoryJobStore, JobStore
from xlforecast.storage.objects import LocalObjectStore, MemoryObjectStore, ObjectStore

__all__ = [
    "InMemoryJobStore",
    "JobStore",
    "LocalObjectStore",
    "MemoryObjectStore",
    "ObjectStore",
]
