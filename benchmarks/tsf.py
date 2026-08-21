"""Reader for the Monash archive's `.tsf` format.

Three properties of these files that a naive reader gets wrong:

* **CRLF line endings and non-UTF-8 bytes.** The M3 files are latin-1 (the Makridakis
  citation contains an en-dash), so `open()` with the default encoding raises, and `grep`
  reports them as binary.
* **The attribute list varies between datasets.** `m3_other` declares only `series_name` --
  no `start_timestamp` and no `@frequency` -- so a reader that assumes `id:start:values`
  fails on it. The `@attribute` declarations say how many prefix fields each row carries.
* **Series are ragged and may carry `?` for missing values.**
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

__all__ = ["TsfDataset", "read_tsf"]


@dataclass(frozen=True, slots=True)
class TsfDataset:
    name: str
    horizon: int
    frequency: str | None
    series: dict[str, np.ndarray]
    starts: dict[str, str]

    @property
    def n_series(self) -> int:
        return len(self.series)

    def split(self) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        """Monash protocol: the last `horizon` observations are the test set.

        A single forecast origin, not cross-validation -- which is the whole reason the
        benchmark harness runs the engine in a dedicated comparability mode.
        """
        train = {k: v[: -self.horizon] for k, v in self.series.items()}
        test = {k: v[-self.horizon :] for k, v in self.series.items()}
        return train, test


def read_tsf(path: str | Path) -> TsfDataset:
    path = Path(path)
    attributes: list[str] = []
    horizon: int | None = None
    frequency: str | None = None
    series: dict[str, np.ndarray] = {}
    starts: dict[str, str] = {}
    in_data = False

    with path.open(encoding="latin-1") as handle:
        for raw in handle:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("@"):
                head, _, rest = line[1:].partition(" ")
                key = head.lower()
                if key == "data":
                    in_data = True
                elif key == "attribute":
                    attributes.append(rest.split()[0])
                elif key == "horizon":
                    horizon = int(rest)
                elif key == "frequency":
                    frequency = rest.strip() or None
                continue
            if not in_data:
                continue

            parts = line.split(":")
            prefix, values = parts[: len(attributes)], parts[len(attributes)]
            name = prefix[0]
            if "start_timestamp" in attributes:
                starts[name] = prefix[attributes.index("start_timestamp")]
            series[name] = np.array(
                [np.nan if v in ("", "?") else float(v) for v in values.split(",")],
                dtype=float,
            )

    if horizon is None:
        raise ValueError(f"{path.name} declares no @horizon")
    return TsfDataset(
        name=path.stem, horizon=horizon, frequency=frequency, series=series, starts=starts
    )
