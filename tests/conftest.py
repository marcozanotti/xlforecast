from __future__ import annotations

import pytest

from xlforecast.schemas import (
    ArtifactPack,
    DataMapping,
    ForecastRequest,
    Leaderboard,
    Manifest,
    ResolvedRequest,
    RunTiming,
)


@pytest.fixture
def request_weekly() -> ForecastRequest:
    return ForecastRequest(h=13, freq="W")


@pytest.fixture
def resolved_weekly(request_weekly: ForecastRequest) -> ResolvedRequest:
    return ResolvedRequest.from_request(request_weekly, season_length=52)


@pytest.fixture
def mapping() -> DataMapping:
    return DataMapping(unique_id_col="sku", ds_col="week", y_col="units")


@pytest.fixture
def manifest(resolved_weekly: ResolvedRequest, mapping: DataMapping) -> Manifest:
    return Manifest(
        job_id="job-1",
        engine_version="0.1.0",
        python_version="3.11.16",
        request=resolved_weekly,
        mapping=mapping,
        data_id="data-1",
        data_fingerprint="0" * 64,
        cutoffs=["2023-10-01T00:00:00Z", "2023-12-31T00:00:00Z", "2024-03-31T00:00:00Z"],
        autoarima_mode="fourier",
        crps_quantiles=[0.025, 0.1, 0.5, 0.9, 0.975],
        thread_config={"OMP_NUM_THREADS": "1"},
        started_at="2026-08-20T10:00:00Z",
        finished_at="2026-08-20T10:03:00Z",
        seed=42,
    )


@pytest.fixture
def empty_timing() -> RunTiming:
    return RunTiming(total_wall_seconds=1.0, n_workers=8)


@pytest.fixture
def empty_pack() -> ArtifactPack:
    return ArtifactPack(job_id="job-1")


@pytest.fixture
def empty_leaderboard() -> Leaderboard:
    return Leaderboard()
