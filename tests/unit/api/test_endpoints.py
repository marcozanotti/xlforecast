"""TS §6 -- the HTTP surface, and the gates it is supposed to enforce."""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from xlforecast.api.deps import Services, set_services
from xlforecast.api.main import app
from xlforecast.api.security import Quota, TokenService
from xlforecast.schemas.request import DataMapping, ForecastRequest
from xlforecast.storage.jobs import InMemoryJobStore
from xlforecast.storage.objects import MemoryObjectStore

REQUEST = ForecastRequest(h=13, freq="W", n_windows=3, models=["SeasonalNaive", "AutoETS"])
MAPPING = DataMapping(unique_id_col="sku", ds_col="week", y_col="units")
DATA_ID = "data-1"


@pytest.fixture
def services():
    svc = Services(
        jobs=InMemoryJobStore(),
        objects=MemoryObjectStore(),
        tokens=TokenService(secret=b"test"),
        quota=Quota(max_concurrent_jobs=2),
    )
    svc.objects.put(f"data/{DATA_ID}.parquet", b"panel-bytes")
    set_services(svc)
    return svc


@pytest.fixture
def client(services):
    return TestClient(app)


def submit_body(token: str, **overrides):
    body = {
        "data_id": DATA_ID,
        "request": REQUEST.model_dump(mode="json"),
        "mapping": MAPPING.model_dump(mode="json"),
        "confirmation_token": token,
    }
    body.update(overrides)
    return body


def confirm(client, request=REQUEST) -> str:
    response = client.post(
        "/v1/confirm", json={"data_id": DATA_ID, "request": request.model_dump(mode="json")}
    )
    assert response.status_code == 200
    return response.json()["confirmation_token"]


class TestNoSynchronousForecast:
    """ADR-005 / hard rule 7 -- every competition goes through the queue, including trivial
    ones, so there is one code path rather than two that drift."""

    def test_submitting_returns_a_job_id_rather_than_a_result(self, client):
        response = client.post("/v1/jobs", json=submit_body(confirm(client)))
        assert response.status_code == 202
        assert "job_id" in response.json()
        assert "leaderboard" not in response.json()

    def test_there_is_no_forecast_endpoint(self, client):
        assert client.post("/v1/forecast", json={}).status_code == 404


class TestConfirmationGate:
    """AC-503 -- the gate rejects, rather than the audit log recording."""

    def test_a_job_without_a_token_is_rejected(self, client):
        response = client.post("/v1/jobs", json=submit_body(""))
        assert response.status_code == 400
        assert response.json()["detail"]["fix"]

    def test_a_token_for_a_different_configuration_is_rejected(self, client):
        token = confirm(client)
        altered = REQUEST.model_copy(update={"h": 52})
        response = client.post(
            "/v1/jobs", json=submit_body(token, request=altered.model_dump(mode="json"))
        )
        assert response.status_code == 400
        assert "confirmed" in response.json()["detail"]["message"]

    def test_a_token_cannot_be_replayed(self, client):
        token = confirm(client)
        assert client.post("/v1/jobs", json=submit_body(token)).status_code == 202
        assert client.post("/v1/jobs", json=submit_body(token)).status_code == 400


class TestQuota:
    """FR-803."""

    def test_concurrent_jobs_are_capped(self, client):
        for _ in range(2):
            assert client.post("/v1/jobs", json=submit_body(confirm(client))).status_code == 202
        response = client.post("/v1/jobs", json=submit_body(confirm(client)))
        assert response.status_code == 429
        assert response.json()["detail"]["fix"]

    def test_the_licence_endpoint_reports_remaining_capacity(self, client):
        client.post("/v1/jobs", json=submit_body(confirm(client)))
        body = client.get("/v1/licence").json()
        assert body["active_jobs"] == 1
        assert body["max_concurrent_jobs"] == 2


class TestJobLifecycle:
    def test_an_unknown_job_is_a_named_404(self, client):
        response = client.get("/v1/jobs/nope")
        assert response.status_code == 404
        assert response.json()["detail"]["fix"]

    def test_status_reports_the_queued_state(self, client):
        job_id = client.post("/v1/jobs", json=submit_body(confirm(client))).json()["job_id"]
        body = client.get(f"/v1/jobs/{job_id}").json()
        assert body["status"] == "queued"
        assert body["progress"] is None

    def test_cancelling_sets_the_flag_the_engine_polls(self, client, services):
        """FR-802 -- cancellation cannot be an asyncio cancel; the engine is compiled
        CPU-bound code and would not notice."""
        job_id = client.post("/v1/jobs", json=submit_body(confirm(client))).json()["job_id"]
        assert client.delete(f"/v1/jobs/{job_id}").status_code == 202
        assert services.jobs.cancel_requested(job_id)

    def test_cancelling_a_finished_job_is_a_no_op_not_an_error(self, client, services):
        from xlforecast.schemas.jobs import JobStatus

        job_id = client.post("/v1/jobs", json=submit_body(confirm(client))).json()["job_id"]
        record = services.jobs.get(job_id)
        services.jobs.update(record.model_copy(update={"status": JobStatus.COMPLETED}))
        response = client.delete(f"/v1/jobs/{job_id}")
        assert response.status_code == 202
        assert response.json()["note"] == "already finished"

    def test_submitting_against_an_unknown_dataset_is_rejected(self, client):
        token = confirm(client)
        body = submit_body(token, data_id="ghost")
        assert client.post("/v1/jobs", json=body).status_code in (400, 404)


class TestManifestEndpoint:
    """Hard rule 10 -- no manifest, no result."""

    def test_a_job_without_a_manifest_yet_says_so(self, client):
        job_id = client.post("/v1/jobs", json=submit_body(confirm(client))).json()["job_id"]
        response = client.get(f"/v1/jobs/{job_id}/manifest")
        assert response.status_code == 409
        assert response.json()["detail"]["fix"]

    def test_a_stored_manifest_is_returned_verbatim(self, client, services):
        job_id = client.post("/v1/jobs", json=submit_body(confirm(client))).json()["job_id"]
        services.objects.put(f"jobs/{job_id}/manifest.json", b'{"job_id":"x"}')
        response = client.get(f"/v1/jobs/{job_id}/manifest")
        assert response.status_code == 200
        assert response.json() == {"job_id": "x"}


class TestErrorPresentation:
    """FS §4 -- every error names the fault and states the fix. No stack traces."""

    def test_errors_carry_a_fix(self, client):
        for response in (
            client.post("/v1/jobs", json=submit_body("bad-token")),
            client.get("/v1/jobs/missing"),
        ):
            detail = response.json()["detail"]
            assert detail["fix"], detail

    def test_no_response_body_contains_a_traceback(self, client):
        response = client.post("/v1/jobs", json=submit_body("bad-token"))
        assert "Traceback" not in response.text


def test_health_reports_the_engine_version(client):
    body = client.get("/v1/health").json()
    assert body["status"] == "ok"
    assert body["version"]


class TestDataUpload:
    """FR-103 / FR-105 / FR-107 -- profiling and validation happen at upload, so the user
    learns which series are unusable before spending compute rather than after."""

    def _parquet(self, n_series: int = 4, n_obs: int = 60) -> bytes:
        import io

        import numpy as np
        import pandas as pd
        import polars as pl

        rng = np.random.default_rng(5)
        frames = []
        for i in range(n_series):
            dates = pd.date_range("2018-01-31", periods=n_obs, freq="ME")
            frames.append(
                pl.DataFrame(
                    {
                        "sku": [f"S{i}"] * n_obs,
                        "week": list(dates),
                        "units": (200 + rng.normal(0, 5, n_obs)).tolist(),
                    }
                )
            )
        buffer = io.BytesIO()
        pl.concat(frames).write_parquet(buffer)
        return buffer.getvalue()

    def _upload(self, client, payload: bytes):
        return client.post(
            "/v1/data?unique_id_col=sku&ds_col=week&y_col=units&freq=ME&h=6",
            content=payload,
            headers={"content-type": "application/octet-stream"},
        )

    def test_a_panel_uploads_and_returns_a_profile(self, client):
        response = self._upload(client, self._parquet())
        assert response.status_code == 200
        body = response.json()
        assert body["data_id"]
        assert body["profile"]["n_series"] == 4

    def test_the_profile_carries_the_validation_report(self, client):
        """S1's live validation summary: '288 of 300 series valid'."""
        body = self._upload(client, self._parquet()).json()
        report = body["profile"]["validation"]
        assert report["n_series_in"] == 4
        assert "excluded" in report

    def test_the_uploaded_panel_is_stored_for_the_job_to_load(self, client, services):
        data_id = self._upload(client, self._parquet()).json()["data_id"]
        assert services.objects.exists(f"data/{data_id}.parquet")

    def test_the_profile_contains_no_observations(self, client):
        """NFR-07 -- this object crosses the LLM trust boundary."""
        body = self._upload(client, self._parquet()).json()
        assert "200.0" not in json.dumps(body["profile"])

    def test_an_empty_upload_is_rejected_with_a_fix(self, client):
        response = client.post("/v1/data", content=b"")
        assert response.status_code == 400
        assert response.json()["detail"]["fix"]

    def test_a_non_parquet_upload_is_rejected_without_a_traceback(self, client):
        response = self._upload(client, b"this is not parquet")
        assert response.status_code == 400
        assert "Traceback" not in response.text
        assert response.json()["detail"]["fix"]

    def test_a_bad_column_mapping_names_the_column(self, client):
        response = client.post(
            "/v1/data?unique_id_col=sku&ds_col=week&y_col=revenue&freq=ME&h=6",
            content=self._parquet(),
            headers={"content-type": "application/octet-stream"},
        )
        assert response.status_code == 400
        detail = response.json()["detail"]
        assert detail["column"] == "revenue"
        assert detail["fix"]
        assert "Traceback" not in response.text


class TestResultsEndpoint:
    def test_results_before_completion_say_so(self, client):
        job_id = client.post("/v1/jobs", json=submit_body(confirm(client))).json()["job_id"]
        response = client.get(f"/v1/jobs/{job_id}/results")
        assert response.status_code == 409
        assert response.json()["detail"]["fix"]

    def test_stored_results_are_returned(self, client, services):
        job_id = client.post("/v1/jobs", json=submit_body(confirm(client))).json()["job_id"]
        services.objects.put(f"jobs/{job_id}/result.json", b'{"job_id":"x"}')
        assert client.get(f"/v1/jobs/{job_id}/results").json() == {"job_id": "x"}

    def test_results_for_an_unknown_job_are_a_named_404(self, client):
        assert client.get("/v1/jobs/ghost/results").status_code == 404


class TestProgressStream:
    """TS §7.3 -- consumed with fetch + ReadableStream, not EventSource."""

    def test_a_terminal_job_streams_once_and_closes(self, client, services):
        from xlforecast.schemas.jobs import JobStatus

        job_id = client.post("/v1/jobs", json=submit_body(confirm(client))).json()["job_id"]
        record = services.jobs.get(job_id)
        services.jobs.update(record.model_copy(update={"status": JobStatus.COMPLETED}))

        with client.stream("GET", f"/v1/jobs/{job_id}/stream") as response:
            assert response.status_code == 200
            assert response.headers["content-type"].startswith("text/event-stream")
            body = "".join(response.iter_text())
        assert body.count("data:") == 1
        assert "completed" in body

    def test_streaming_an_unknown_job_is_a_named_404(self, client):
        assert client.get("/v1/jobs/ghost/stream").status_code == 404
