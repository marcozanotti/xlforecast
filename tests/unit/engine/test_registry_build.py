"""TS §5.2 -- constructing every registered model, and FR-201a/201c's mode switches."""

from __future__ import annotations

import pytest

from xlforecast.engine.registry import build_local, build_ml_plan, is_ml, spec
from xlforecast.errors import InvalidModelNameError
from xlforecast.schemas.registry import MODEL_REGISTRY

STATISTICAL = [n for n in MODEL_REGISTRY if not is_ml(n)]
ML = [n for n in MODEL_REGISTRY if is_ml(n)]


@pytest.mark.parametrize("name", STATISTICAL)
def test_every_statistical_model_constructs(name):
    assert build_local(name, season_length=12) is not None


@pytest.mark.parametrize("name", ML)
def test_every_ml_model_constructs(name):
    plan = build_ml_plan(name, freq="W-SUN", season_length=52, seed=42)
    assert plan.name == name
    assert plan.estimator is not None


def test_registry_and_builders_agree_on_the_model_set():
    """Neither list may drift from the other: a model in the registry that no builder can
    construct would pass request validation and then fail at run time."""
    assert set(STATISTICAL) | set(ML) == set(MODEL_REGISTRY)


class TestSeasonalModeSwitches:
    """FR-201a and FR-201c -- both change the fitted model, so both are recorded in the
    manifest rather than applied silently."""

    def test_autoarima_drops_seasonality_above_the_threshold(self):
        assert build_local("AutoARIMA", season_length=52).season_length == 1
        assert build_local("AutoARIMA", season_length=12).season_length == 12

    def test_autoets_becomes_mstl_above_the_threshold(self):
        """statsforecast's ETS carries a fixed 24-slot seasonal buffer and returns early
        above it, so AutoETS(52) silently fits a non-seasonal model."""
        from statsforecast.models import MSTL, AutoETS

        assert isinstance(build_local("AutoETS", season_length=52), MSTL)
        assert isinstance(build_local("AutoETS", season_length=24), AutoETS)

    def test_the_mstl_substitution_keeps_the_model_name(self):
        """Aliased back so the leaderboard row still reads AutoETS; the substitution is
        auditable via the manifest's ets_mode, not hidden behind a different label."""
        assert build_local("AutoETS", season_length=52).alias == "AutoETS"

    def test_window_average_uses_the_seasonal_period(self):
        """FR-201b -- at the seasonal frequency this is the incumbent moving average."""
        assert build_local("WindowAverage", season_length=52).window_size == 52


class TestErrors:
    def test_unknown_model_is_rejected_with_the_known_list(self):
        with pytest.raises(InvalidModelNameError) as exc:
            spec("Prophet")
        assert "AutoETS" in (exc.value.fix or "")

    def test_ml_name_rejected_by_the_statistical_builder(self):
        with pytest.raises(InvalidModelNameError):
            build_local("GlobalLGBM", season_length=12)
