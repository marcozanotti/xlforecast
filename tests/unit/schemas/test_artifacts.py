"""TS §8 -- artifact pack, and the hard-rule-4 trust boundary."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from xlforecast.schemas import (
    MAX_ARTIFACT_POINTS,
    Analogue,
    ArtifactPack,
    Attribution,
    Decomposition,
)


def test_attribution_records_whether_it_is_on_the_recursive_path():
    """FR-601a -- at horizon_step > 1 the lag features hold the model's own predictions, not
    observations. Narration must not call those 'the driving features'."""
    a = Attribution(
        unique_id="S1",
        ds="2024-03-01T00:00:00Z",
        horizon_step=12,
        recursive_path=True,
        contributions={"lag_1": 0.4},
        base_value=100.0,
    )
    assert a.recursive_path
    assert a.horizon_step > 1


def test_single_step_attribution_is_not_recursive():
    a = Attribution(
        unique_id="S1",
        ds="2024-01-07T00:00:00Z",
        horizon_step=1,
        recursive_path=False,
        contributions={"lag_1": 0.4},
        base_value=100.0,
    )
    assert not a.recursive_path


def test_decomposition_carries_its_ets_form_and_allows_absent_components():
    """FR-601b -- an ANN fit has neither trend nor seasonal, and those are None, not zero."""
    d = Decomposition(unique_id="S1", ds="2024-01-07T00:00:00Z", ets_form="ANN", level=100.0)
    assert d.trend is None
    assert d.seasonal is None


def test_calendar_context_carries_the_denominator_of_an_ordinal():
    """'3rd highest of 12 months' -- the 12 must exist as an artifact value, or the numeric
    guardrail rejects a legitimate sentence (TS §9.4 allowlist)."""
    from xlforecast.schemas import CalendarContext

    c = CalendarContext(
        unique_id="S1",
        ds="2024-03-01T00:00:00Z",
        seasonal_index=1.2,
        period_rank=3,
        period_total=12,
    )
    assert c.period_total == 12


def test_analogue_may_carry_an_observed_value():
    """Hard rule 4 permits named per-point observed values; it forbids bulk panel data.
    FR-604 and the P1 journey require exactly this field."""
    a = Analogue(
        unique_id="S1",
        ds="2024-03-01T00:00:00Z",
        same_period_last_year=412.0,
        seasonal_naive_value=400.0,
        historical_pctile=0.82,
    )
    assert a.same_period_last_year == 412.0


def test_analogue_tolerates_a_missing_prior_year():
    a = Analogue(
        unique_id="S1", ds="2024-03-01T00:00:00Z", seasonal_naive_value=400.0, historical_pctile=0.5
    )
    assert a.same_period_last_year is None


def test_point_count_supports_the_hard_rule_4_cap(empty_pack):
    assert empty_pack.point_count() == 0
    pack = ArtifactPack(
        job_id="j",
        analogues=[
            Analogue(
                unique_id=f"S{i}",
                ds="2024-01-07T00:00:00Z",
                seasonal_naive_value=1.0,
                historical_pctile=0.5,
            )
            for i in range(5)
        ],
    )
    assert pack.point_count() == 5
    assert MAX_ARTIFACT_POINTS > 0


def test_coverage_is_bounded_to_a_probability():
    from xlforecast.schemas import UncertaintyContext

    with pytest.raises(ValidationError):
        UncertaintyContext(
            unique_id="S1",
            ds="2024-01-07T00:00:00Z",
            level=80,
            interval_width=10.0,
            width_vs_series_mean=0.1,
            empirical_coverage=1.4,
        )
