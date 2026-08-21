"""Measured cost (FR-217).

The cost proxy for NFR-01 is measured `train + predict` time, never fit counts, series
counts or model counts. CPU seconds are summed across threads via `getrusage`, which makes
them parallelism-invariant and therefore comparable across runs, machines and worker counts;
wall seconds are what NFR-01's 10-minute budget actually means. Both are recorded.
"""

from __future__ import annotations

import resource
import time
from dataclasses import dataclass
from types import TracebackType

__all__ = ["Measured", "measure"]


def _cpu_seconds() -> float:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return usage.ru_utime + usage.ru_stime


@dataclass
class Measured:
    cpu: float = 0.0
    wall: float = 0.0


class measure:  # noqa: N801 - reads as a verb at call sites: `with measure() as train:`
    """Context manager yielding a `Measured` populated on exit.

    Usage::

        with measure() as train:
            model.fit(...)
    """

    __slots__ = ("_cpu0", "_wall0", "result")

    def __init__(self) -> None:
        self.result = Measured()

    def __enter__(self) -> Measured:
        self._wall0 = time.perf_counter()
        self._cpu0 = _cpu_seconds()
        return self.result

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.result.wall = time.perf_counter() - self._wall0
        self.result.cpu = _cpu_seconds() - self._cpu0
