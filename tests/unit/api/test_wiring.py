"""Services wiring (TS §6) -- and a regression test for a job that never ran.

A job submitted through the API used to sit at "queued" forever: `enqueue` defaulted to
`None`, and the API and worker each built their own `InMemoryJobStore`, so even once enqueued
the worker could not see it. Neither component was doing anything wrong from its own point of
view, so neither logged anything. These tests exist so that cannot come back quietly.
"""

from __future__ import annotations

import io
import time

import numpy as np
import pandas as pd
import polars as pl
import pytest
from fastapi.testclient import TestClient

from xlforecast.api.deps import Services, set_services
from xlforecast.api.main import app
from xlforecast.api.security import TokenService
from xlforecast.storage.objects import LocalObjectStore

REQUEST = {
    "h": 4,
    "freq": "ME",
    "n_windows": 2,
    "models": ["SeasonalNaive", "HistoricAverage"],
    "ensemble": "none",
    "conformal": False,
}
MAPPING = {"unique_id_col": "sku", "ds_col": "month", "y_col": "units", "exog": []}


def panel_parquet(n_series: int = 2, n_obs: int = 60) -> bytes:
    rng = np.random.default_rng(11)
    dates = pd.date_range("2018-01-31", periods=n_obs, freq="ME")
    frames = [
        pl.DataFrame(
            {
                "sku": [f"S{i}"] * n_obs,
                "month": list(dates),
                "units": (
                    200 + 20 * np.sin(2 * np.pi * np.arange(n_obs) / 12) + rng.normal(0, 4, n_obs)
                ).tolist(),
            }
        )
        for i in range(n_series)
    ]
    buffer = io.BytesIO()
    pl.concat(frames).write_parquet(buffer)
    return buffer.getvalue()


class TestFromEnv:
    def test_without_redis_it_runs_jobs_inline_and_says_so(self, tmp_path, monkeypatch):
        monkeypatch.delenv("REDIS_URL", raising=False)
        monkeypatch.setenv("XLF_OBJECT_ROOT", str(tmp_path))
        services = Services.from_env()
        assert services.inline is True
        # The fault that started this: an enqueue of None means nothing ever runs.
        assert services.enqueue is not None
        assert isinstance(services.objects, LocalObjectStore)

    def test_with_redis_it_uses_the_shared_stores(self, tmp_path, monkeypatch):
        pytest.importorskip("fakeredis")
        monkeypatch.setenv("REDIS_URL", "redis://localhost:6379")
        monkeypatch.setenv("XLF_OBJECT_ROOT", str(tmp_path))
        services = Services.from_env()
        assert services.inline is False
        assert services.enqueue is not None
        # The replay set must be shared, or two API instances each accept the same token once.
        assert services.tokens.replay is not None

    def test_the_health_endpoint_reports_the_mode(self, tmp_path, monkeypatch):
        """So a deployment cannot quietly be running jobs in the API process."""
        monkeypatch.delenv("REDIS_URL", raising=False)
        monkeypatch.setenv("XLF_OBJECT_ROOT", str(tmp_path))
        set_services(Services.from_env())
        body = TestClient(app).get("/v1/health").json()
        assert body["mode"] == "inline (development)"

    def test_a_queued_deployment_reports_queued(self, tmp_path):
        set_services(Services(objects=LocalObjectStore(tmp_path), inline=False))
        assert TestClient(app).get("/v1/health").json()["mode"] == "queued"

    def test_the_signing_key_comes_from_the_environment(self, tmp_path, monkeypatch):
        monkeypatch.delenv("REDIS_URL", raising=False)
        monkeypatch.setenv("XLF_TOKEN_SECRET", "a-specific-key")
        assert Services.from_env().tokens.secret == b"a-specific-key"


@pytest.mark.slow
class TestSubmittedJobsActuallyRun:
    """The regression. Every assertion here failed before the wiring was fixed."""

    @pytest.fixture
    def client(self, tmp_path, monkeypatch):
        monkeypatch.delenv("REDIS_URL", raising=False)
        monkeypatch.setenv("XLF_OBJECT_ROOT", str(tmp_path))
        monkeypatch.setenv("XLF_TOKEN_SECRET", "test-key")
        services = Services.from_env()
        services.tokens = TokenService(secret=b"test-key")
        set_services(services)
        return TestClient(app)

    def submit(self, client: TestClient) -> str:
        upload = client.post(
            "/v1/data?unique_id_col=sku&ds_col=month&y_col=units&freq=ME&h=4",
            content=panel_parquet(),
            headers={"content-type": "application/octet-stream"},
        )
        assert upload.status_code == 200, upload.text
        data_id = upload.json()["data_id"]

        token = client.post("/v1/confirm", json={"data_id": data_id, "request": REQUEST}).json()[
            "confirmation_token"
        ]
        response = client.post(
            "/v1/jobs",
            json={
                "data_id": data_id,
                "request": REQUEST,
                "mapping": MAPPING,
                "confirmation_token": token,
            },
        )
        assert response.status_code == 202, response.text
        return str(response.json()["job_id"])

    def wait(self, client: TestClient, job_id: str, timeout: float = 180.0) -> dict:
        deadline = time.monotonic() + timeout
        status: dict = {}
        while time.monotonic() < deadline:
            status = client.get(f"/v1/jobs/{job_id}").json()
            if status["status"] in ("completed", "failed", "cancelled"):
                return status
            time.sleep(0.5)
        return status

    def test_a_submitted_job_leaves_the_queued_state(self, client):
        """It used to sit at 'queued' indefinitely, which is what makes this the whole test."""
        status = self.wait(client, self.submit(client))
        assert status["status"] != "queued"
        assert status["status"] == "completed", status.get("error")

    def test_it_produces_results_and_a_manifest(self, client):
        job_id = self.submit(client)
        assert self.wait(client, job_id)["status"] == "completed"

        results = client.get(f"/v1/jobs/{job_id}/results")
        assert results.status_code == 200
        panel = [r for r in results.json()["leaderboard"]["rows"] if r["scope"] == "panel"]
        assert {row["model"] for row in panel} == {"SeasonalNaive", "HistoricAverage"}

        # Hard rule 10: no manifest, no result.
        assert client.get(f"/v1/jobs/{job_id}/manifest").status_code == 200

    def test_progress_is_recorded_while_it_runs(self, client):
        job_id = self.submit(client)
        self.wait(client, job_id)
        progress = client.get(f"/v1/jobs/{job_id}").json()["progress"]
        assert progress is not None
        assert progress["folds_done"] == 2

    def test_the_inline_runner_uses_the_same_checkpoints(self, client, tmp_path):
        """Not a separate execution path -- the same subprocess executor, so FR-801 and
        FR-802 behave identically to the queued deployment."""
        job_id = self.submit(client)
        self.wait(client, job_id)
        checkpoints = list((tmp_path / "jobs" / job_id / "folds").glob("*.json"))
        assert len(checkpoints) == 2
