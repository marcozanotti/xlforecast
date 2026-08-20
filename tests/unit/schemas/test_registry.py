"""TS §5.2 -- registry metadata, the commercial_ok gate, and FR-203b matched pairs."""

from __future__ import annotations

import pytest

from xlforecast.errors import ProhibitedModelError
from xlforecast.schemas.registry import DEFAULT_MODELS, MODEL_REGISTRY, ModelSpec
from xlforecast.schemas.request import _validate_model_name


def test_every_default_model_is_registered():
    assert all(m in MODEL_REGISTRY for m in DEFAULT_MODELS)


def test_no_prohibited_model_ships():
    """CLAUDE.md: models with commercial_ok=False must not ship (e.g. Moirai, CC-BY-NC-4.0)."""
    assert all(spec.commercial_ok for spec in MODEL_REGISTRY.values())


def test_commercial_ok_gate_is_reachable():
    """With a closed Literal this check was dead code -- a prohibited model could never be
    named, so the gate could not fire and its test would have been vacuous. Registry
    validation makes it reachable, which this fixture model demonstrates."""
    MODEL_REGISTRY["_TestProhibited"] = ModelSpec(
        name="_TestProhibited",
        family="local",
        information_set="own_series",
        handles_intermittent=False,
        supports_exog=False,
        licence="CC-BY-NC-4.0",
        commercial_ok=False,
    )
    try:
        with pytest.raises(ProhibitedModelError) as exc:
            _validate_model_name("_TestProhibited")
        assert "CC-BY-NC-4.0" in str(exc.value)
    finally:
        del MODEL_REGISTRY["_TestProhibited"]


@pytest.mark.parametrize(
    ("local", "glob"),
    [("LocalLinear", "GlobalLinear"), ("LocalLGBM", "GlobalLGBM"), ("LocalXGB", "GlobalXGB")],
)
def test_matched_pairs_differ_only_in_information_set(local, glob):
    """FR-203b -- the pair comparison is meaningful only if nothing else drifted."""
    a, b = MODEL_REGISTRY[local], MODEL_REGISTRY[glob]
    assert a.information_set == "own_series"
    assert b.information_set == "panel"
    assert a.licence == b.licence
    assert a.supports_exog == b.supports_exog


def test_local_ml_demands_more_history_than_local_statistical():
    """FR-203c -- lag construction consumes max_lag rows per series; the spike confirmed
    ~91 usable training rows at the FR-105 floor."""
    assert MODEL_REGISTRY["LocalLGBM"].min_obs(52) > MODEL_REGISTRY["AutoETS"].min_obs(52)


def test_local_ml_flagged_for_small_data_hyperparameters():
    """Global defaults (min_child_samples=20, num_leaves=31) on ~91 rows produce
    near-constant predictions -- a leaderboard row that reads as a bug, not a finding."""
    assert MODEL_REGISTRY["LocalLGBM"].small_data_params
    assert not MODEL_REGISTRY["GlobalLGBM"].small_data_params


def test_intermittent_models_are_flagged_for_fr108_routing():
    for name in ("CrostonClassic", "ADIDA", "IMAPA", "ZeroModel"):
        assert MODEL_REGISTRY[name].handles_intermittent
