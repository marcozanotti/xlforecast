"""FR-211 / NFR-02 -- parallelism must buy wall-clock without touching the numbers.

Measured in Phase 3 on 200 M3 monthly series: n_jobs=4 is 2.72x faster than n_jobs=1 with
bit-identical forecasts. That is the difference between this lever and the approximation
lever, which buys 2.95x at the cost of 3.3% worse MASE.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl
import pytest

from xlforecast.engine.run import run_from_frame
from xlforecast.schemas.request import DataMapping, ForecastRequest

MAPPING = DataMapping(unique_id_col="unique_id", ds_col="ds", y_col="y")


def panel(n_series: int = 4, n_obs: int = 90) -> pl.DataFrame:
    rng = np.random.default_rng(17)
    frames = []
    for i in range(n_series):
        dates = pd.date_range("2016-01-31", periods=n_obs, freq="ME")
        t = np.arange(n_obs)
        y = 300 + 50 * np.sin(2 * np.pi * t / 12) + rng.normal(0, 8, n_obs)
        frames.append(
            pl.DataFrame({"unique_id": [f"S{i}"] * n_obs, "ds": list(dates), "y": y.tolist()})
        )
    return pl.concat(frames)


REQUEST = ForecastRequest(
    h=6, freq="ME", n_windows=2, models=["SeasonalNaive", "AutoETS"], ensemble="none"
)


@pytest.mark.slow
def test_worker_count_does_not_change_the_leaderboard():
    """statsforecast parallelises across series and each series is fitted independently, so
    the worker count is a scheduling decision rather than a modelling one. If this ever
    fails, NFR-02 is broken and the cause is not float ordering."""
    serial = run_from_frame(panel(), request=REQUEST, mapping=MAPPING, job_id="a", n_jobs=1)
    parallel = run_from_frame(panel(), request=REQUEST, mapping=MAPPING, job_id="b", n_jobs=2)
    assert serial.leaderboard.model_dump_json() == parallel.leaderboard.model_dump_json()


@pytest.mark.slow
def test_worker_count_does_not_change_the_forecast():
    serial = run_from_frame(panel(), request=REQUEST, mapping=MAPPING, job_id="a", n_jobs=1)
    parallel = run_from_frame(panel(), request=REQUEST, mapping=MAPPING, job_id="b", n_jobs=2)
    assert [r.model_dump() for r in serial.forecast.rows] == [
        r.model_dump() for r in parallel.forecast.rows
    ]


def test_the_worker_count_is_recorded():
    """NFR-02 defines byte-identity relative to a recorded configuration. n_jobs turns out
    not to affect the numbers, but it is recorded rather than assumed harmless."""
    result = run_from_frame(panel(2, 60), request=REQUEST, mapping=MAPPING, job_id="c", n_jobs=2)
    assert result.timing.n_workers == 2
    assert result.manifest.thread_config["n_jobs"] == "2"
