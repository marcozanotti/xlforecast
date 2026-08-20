"""Small accessors, covered because they are load-bearing for later phases."""

from __future__ import annotations

from xlforecast.schemas import ForecastRequest, ResolvedRequest, ValidationReport
from xlforecast.schemas.registry import is_known, spec_for


def test_registry_lookup_helpers():
    assert is_known("AutoETS")
    assert not is_known("Prophet")
    assert spec_for("GlobalLGBM").information_set == "panel"


def test_validation_report_counts_exclusions():
    report = ValidationReport(
        n_series_in=300,
        n_series_out=288,
        excluded={f"S{i}": "all_zero" for i in range(12)},
    )
    assert report.n_excluded == 12
    assert report.n_series_in - report.n_series_out == report.n_excluded


def test_cutoff_count_matches_requested_windows():
    req = ResolvedRequest.from_request(ForecastRequest(h=13, freq="W"), season_length=52)
    assert req.cv_cutoff_count() == req.n_windows == 3


def test_fit_count_is_explanatory_only_not_a_cost_proxy():
    """FR-217 makes measured train + predict the cost proxy. Fit counts are kept only to
    explain *why* model count is a bad proxy: a local model is one fit per series per fold,
    a global model is one fit per fold, and the Phase 0 spike measured LocalLGBM at 493 CPU
    seconds against GlobalLGBM's 3.3 for the same nominal 'one model'."""
    req = ResolvedRequest.from_request(ForecastRequest(h=13, freq="W"), season_length=52)
    assert req.total_fits_per_local_model(200) == 200 * 4
    assert "cost" not in ResolvedRequest.total_fits_per_local_model.__doc__.split(".")[0].lower()
