"""FR-706 -- the CLI writes the same four tables as CSV plus a JSON manifest."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import polars as pl
import pytest
from typer.testing import CliRunner

from xlforecast.cli import app

runner = CliRunner()


@pytest.fixture(scope="module")
def panel_csv(tmp_path_factory):
    path = tmp_path_factory.mktemp("cli") / "panel.csv"
    rng = np.random.default_rng(3)
    dates = pd.date_range("2016-01-31", periods=80, freq="ME")
    frames = []
    for i in range(3):
        t = np.arange(80)
        y = 200 + 40 * np.sin(2 * np.pi * t / 12) + rng.normal(0, 6, 80)
        frames.append(
            pl.DataFrame({"sku": [f"S{i}"] * 80, "month": list(dates), "units": y.tolist()})
        )
    pl.concat(frames).write_csv(path)
    return path


@pytest.fixture(scope="module")
def result(panel_csv, tmp_path_factory):
    out = tmp_path_factory.mktemp("out")
    invocation = runner.invoke(
        app,
        [
            "run",
            str(panel_csv),
            "--h",
            "6",
            "--freq",
            "ME",
            "--unique-id-col",
            "sku",
            "--ds-col",
            "month",
            "--y-col",
            "units",
            "--models",
            "SeasonalNaive,WindowAverage,HistoricAverage",
            "--n-windows",
            "2",
            "--out",
            str(out),
        ],
    )
    return invocation, out


def test_exits_cleanly(result):
    invocation, _ = result
    assert invocation.exit_code == 0, invocation.output


def test_writes_four_tables_and_a_manifest(result):
    """Hard rule 10 -- no manifest, no result."""
    _, out = result
    for name in ("XLF_Forecast", "XLF_Forecast_Long", "XLF_Leaderboard", "XLF_Diagnostics"):
        assert (out / f"{name}.csv").exists(), name
    assert (out / "XLF_Manifest.json").exists()


def test_the_manifest_is_valid_json_carrying_the_reproducibility_fields(result):
    _, out = result
    manifest = json.loads((out / "XLF_Manifest.json").read_text())
    assert len(manifest["data_fingerprint"]) == 64
    assert manifest["cutoffs"]
    assert manifest["thread_config"]
    assert manifest["request"]["season_length"] == 12


def test_reports_cost_per_model(result):
    """FR-217 -- measured train + predict, reported per model."""
    invocation, out = result
    assert "CPU" in invocation.output
    timing = json.loads((out / "XLF_Timing.json").read_text())
    assert timing["per_model"]


def test_leaderboard_names_the_incumbent_comparison(result):
    invocation, _ = result
    assert "vs incumbent" in invocation.output


def test_bad_column_mapping_reports_a_fix_not_a_traceback(panel_csv, tmp_path):
    """FS §4 error-presentation rule."""
    invocation = runner.invoke(
        app,
        [
            "run",
            str(panel_csv),
            "--h",
            "6",
            "--freq",
            "ME",
            "--y-col",
            "revenue",
            "--out",
            str(tmp_path / "nope"),
        ],
    )
    assert invocation.exit_code == 1
    assert "Traceback" not in invocation.output
    assert "Fix:" in invocation.output


def test_version_command():
    invocation = runner.invoke(app, ["version"])
    assert invocation.exit_code == 0
    assert invocation.output.strip()
