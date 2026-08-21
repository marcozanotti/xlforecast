"""Gate G3's harness -- the `.tsf` reader, the split, and metric equivalence.

These run without the M3 files, which are downloaded from Zenodo rather than committed. What
they protect is the part of G3 that could silently invalidate every published comparison: if
our MASE is not the archive's MASE, or the split is not the archive's split, the numbers are
not comparable and the gate means nothing.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "benchmarks"))
from tsf import read_tsf

TSF_WITH_DATES = """# Dataset Information
# A fixture resembling m3_yearly.
@relation M3
@attribute series_name string
@attribute start_timestamp date
@frequency yearly
@horizon 3
@missing false
@equallength false
@data
T1:1975-01-01 00-00-00:1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0
T2:1980-01-01 00-00-00:10.0,20.0,30.0,40.0,50.0,60.0
"""

# m3_other declares only series_name -- no timestamp, no @frequency. A reader assuming
# `id:start:values` fails on it, which is why the attribute list drives the parse.
TSF_WITHOUT_DATES = """@relation M3
@attribute series_name string
@horizon 2
@missing false
@equallength false
@data
T1:3060.42,3021.19,3301.13,3287.03
T2:1.0,2.0,?,4.0
"""


@pytest.fixture
def dated(tmp_path):
    path = tmp_path / "dated.tsf"
    path.write_text(TSF_WITH_DATES, encoding="latin-1")
    return read_tsf(path)


@pytest.fixture
def undated(tmp_path):
    path = tmp_path / "undated.tsf"
    path.write_text(TSF_WITHOUT_DATES, encoding="latin-1")
    return read_tsf(path)


class TestTsfReader:
    def test_reads_metadata(self, dated):
        assert dated.horizon == 3
        assert dated.frequency == "yearly"
        assert dated.n_series == 2

    def test_reads_ragged_series(self, dated):
        assert len(dated.series["T1"]) == 8
        assert len(dated.series["T2"]) == 6

    def test_captures_start_timestamps_when_declared(self, dated):
        assert dated.starts["T1"].startswith("1975-01-01")

    def test_handles_a_dataset_with_no_timestamps(self, undated):
        """m3_other's actual shape. The attribute list, not a fixed layout, drives parsing."""
        assert undated.n_series == 2
        assert undated.frequency is None
        assert undated.starts == {}

    def test_missing_values_become_nan(self, undated):
        assert np.isnan(undated.series["T2"][2])

    def test_latin1_bytes_do_not_break_the_reader(self, tmp_path):
        """The real files carry an en-dash in the Makridakis citation, so a UTF-8 read
        raises and grep calls them binary."""
        path = tmp_path / "latin.tsf"
        path.write_bytes(
            b"# Makridakis 16 (4), 451\xe2\x80\x93476.\n@attribute series_name string\n"
            b"@horizon 2\n@data\nT1:1.0,2.0,3.0,4.0\n"
        )
        assert read_tsf(path).n_series == 1

    def test_a_file_without_a_horizon_is_rejected(self, tmp_path):
        path = tmp_path / "nohorizon.tsf"
        path.write_text("@attribute series_name string\n@data\nT1:1.0,2.0\n", encoding="latin-1")
        with pytest.raises(ValueError, match="horizon"):
            read_tsf(path)


class TestMonashSplit:
    """The archive evaluates at a single origin: the last `horizon` observations."""

    def test_test_block_is_exactly_the_horizon(self, dated):
        _, test = dated.split()
        assert all(len(v) == dated.horizon for v in test.values())

    def test_train_and_test_reconstruct_the_original(self, dated):
        train, test = dated.split()
        for name, values in dated.series.items():
            assert np.array_equal(np.concatenate([train[name], test[name]]), values)

    def test_no_test_observation_reaches_the_training_series(self, dated):
        train, test = dated.split()
        for name in dated.series:
            assert len(train[name]) == len(dated.series[name]) - dated.horizon
            assert train[name][-1] != test[name][0] or len(set(dated.series[name])) == 1


class TestMetricEquivalence:
    """The assertion that makes every published comparison meaningful.

    If our MASE is not the archive's MASE, the G3 numbers are not comparable and the gate
    reports nothing. The archive's formula (paper eq. 2) is::

        MASE = [ (1/h) * sum_{k=M+1..M+h} |F_k - Y_k| ]
             / [ (1/(M-S)) * sum_{k=S+1..M} |Y_k - Y_{k-S}| ]

    with M the number of TRAINING points and the denominator over the training series only.
    """

    @pytest.mark.parametrize("seasonality", [1, 4, 12])
    def test_utilsforecast_mase_matches_the_published_formula(self, seasonality):
        from utilsforecast.losses import mase

        rng = np.random.default_rng(seasonality)
        train_values = 100 + np.cumsum(rng.normal(0, 5, 80))
        horizon = 8
        actual = 100 + np.cumsum(rng.normal(0, 5, horizon))
        forecast = actual + rng.normal(0, 3, horizon)

        denominator = np.mean(np.abs(train_values[seasonality:] - train_values[:-seasonality]))
        expected = np.mean(np.abs(forecast - actual)) / denominator

        test_df = pd.DataFrame(
            {"unique_id": ["a"] * horizon, "ds": pd.RangeIndex(horizon), "y": actual, "m": forecast}
        )
        train_df = pd.DataFrame(
            {
                "unique_id": ["a"] * len(train_values),
                "ds": pd.RangeIndex(len(train_values)),
                "y": train_values,
            }
        )
        observed = mase(test_df, models=["m"], seasonality=seasonality, train_df=train_df)
        assert observed["m"].iloc[0] == pytest.approx(expected, rel=1e-12)

    def test_the_denominator_ignores_the_test_window(self):
        """Stated separately because using the full series would be an easy, invisible
        mistake: the number would still look plausible."""
        from utilsforecast.losses import mase

        train_values = np.arange(1.0, 41.0)
        actual = np.array([41.0, 42.0, 43.0])
        forecast = np.array([41.0, 42.0, 43.0])
        test_df = pd.DataFrame(
            {"unique_id": ["a"] * 3, "ds": pd.RangeIndex(3), "y": actual, "m": forecast}
        )
        train_df = pd.DataFrame(
            {"unique_id": ["a"] * 40, "ds": pd.RangeIndex(40), "y": train_values}
        )
        assert mase(test_df, models=["m"], seasonality=1, train_df=train_df)["m"].iloc[0] == 0.0


def test_the_committed_baselines_are_well_formed():
    """G3 requires the baselines to exist BEFORE Phase 3 opens, with named metrics and an
    explicit tolerance -- 'consistent with your published work' is not a gate."""
    import json

    spec = json.loads(
        (Path(__file__).resolve().parents[2] / "benchmarks/baselines/monash_m3.json").read_text()
    )
    assert spec["source"]["paper"]
    assert spec["gate_g3"]["tolerance_pct"] > 0
    for key in ("m3_yearly", "m3_quarterly", "m3_monthly"):
        entry = spec["datasets"][key]
        assert entry["mean_mase"]["ETS"] > 0
        assert entry["horizon"] > 0
        assert entry["seasonality"] in (1, 4, 12)
    # m3_other has no published baseline and must be excluded from the gate rather than
    # silently compared against nothing.
    assert spec["datasets"]["m3_other"]["mean_mase"] is None
    assert "m3_other" in spec["gate_g3"]["excluded"]
