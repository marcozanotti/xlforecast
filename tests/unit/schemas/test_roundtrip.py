"""Gate G0: 'schemas round-trip through JSON without loss'.

The operative definition is `T.model_validate(json.loads(m.model_dump_json())) == m`, applied
to every contract. This file is the gate; if it passes, G0's schema clause is met.

Two Phase 0 spike findings are pinned here so a future change cannot silently undo them:
integer-keyed dicts round-trip under strict validation as well as lax, and a NaN in a bare
float field survives serialisation but dies on re-validation.
"""

from __future__ import annotations

import json
import math

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from pydantic import BaseModel, ValidationError

from xlforecast.schemas import (
    Analogue,
    ArtifactPack,
    Attribution,
    CalendarContext,
    ConformalBands,
    DataMapping,
    DataProfile,
    Decomposition,
    ExogSpec,
    FoldScore,
    ForecastFrame,
    ForecastRequest,
    ForecastRow,
    Leaderboard,
    LeaderboardRow,
    Manifest,
    ModelTiming,
    PeakDrop,
    ResolvedRequest,
    RunResult,
    RunTiming,
    SeriesProfile,
    UncertaintyContext,
    ValidationReport,
)

ALL_CONTRACTS: list[type[BaseModel]] = [
    ExogSpec,
    DataMapping,
    ForecastRequest,
    ResolvedRequest,
    SeriesProfile,
    ValidationReport,
    DataProfile,
    FoldScore,
    LeaderboardRow,
    Leaderboard,
    ForecastRow,
    ForecastFrame,
    ConformalBands,
    ModelTiming,
    RunTiming,
    Manifest,
    RunResult,
    PeakDrop,
    Attribution,
    Decomposition,
    CalendarContext,
    Analogue,
    UncertaintyContext,
    ArtifactPack,
]


def assert_round_trips(model: BaseModel) -> None:
    """G0's definition, stated once."""
    revived = type(model).model_validate(json.loads(model.model_dump_json()))
    assert revived == model


def test_every_contract_is_covered_by_this_gate():
    """Guards against a contract being added to schemas/ and quietly skipping G0."""
    import xlforecast.schemas as pkg

    exported = {
        getattr(pkg, n)
        for n in pkg.__all__
        if isinstance(getattr(pkg, n), type) and issubclass(getattr(pkg, n), BaseModel)
    }
    assert exported == set(ALL_CONTRACTS), (
        f"not covered: {exported.symmetric_difference(ALL_CONTRACTS)}"
    )


class TestFixtureRoundTrips:
    def test_request(self, request_weekly):
        assert_round_trips(request_weekly)

    def test_resolved_request(self, resolved_weekly):
        assert_round_trips(resolved_weekly)

    def test_manifest(self, manifest):
        assert_round_trips(manifest)

    def test_full_run_result(self, manifest, empty_pack, empty_timing, empty_leaderboard):
        assert_round_trips(
            RunResult(
                job_id="job-1",
                leaderboard=empty_leaderboard,
                forecast=ForecastFrame(levels=[80, 95]),
                timing=empty_timing,
                artifacts=empty_pack,
                manifest=manifest,
            )
        )


class TestIntegerKeyedDicts:
    """`coverage: dict[int, float]`. An earlier draft of TS §4.0 warned this would fail under
    strict validation; the spike showed that is wrong, and the claim was removed."""

    def test_lax_round_trip(self):
        row = LeaderboardRow(
            scope="panel",
            model="AutoETS",
            family="local",
            information_set="own_series",
            coverage={80: 0.78, 95: 0.94},
            n_folds=3,
            n_series_scored=1,
            n_series_common=1,
            rank=1,
        )
        assert_round_trips(row)

    def test_json_keys_are_strings_on_the_wire(self):
        row = LeaderboardRow(
            scope="panel",
            model="AutoETS",
            family="local",
            information_set="own_series",
            coverage={80: 0.78},
            n_folds=1,
            n_series_scored=1,
            n_series_common=1,
            rank=1,
        )
        assert json.loads(row.model_dump_json())["coverage"] == {"80": 0.78}

    def test_strict_validation_also_accepts_them(self):
        class Cov(BaseModel):
            coverage: dict[int, float]

        m = Cov(coverage={80: 0.78})
        assert Cov.model_validate_json(m.model_dump_json(), strict=True) == m


class TestNaNIsNotRepresentable:
    """FR-214's justification, pinned as a test."""

    def test_nan_in_a_bare_float_dies_on_the_return_leg(self):
        class Bare(BaseModel):
            mase: float

        payload = Bare(mase=math.nan).model_dump_json()
        assert json.loads(payload) == {"mase": None}, "NaN serialises to null, not an error"
        with pytest.raises(ValidationError):
            Bare.model_validate_json(payload)

    def test_optional_float_survives(self):
        row = LeaderboardRow(
            scope="panel",
            model="AutoETS",
            family="local",
            information_set="own_series",
            mase=None,
            n_folds=1,
            n_series_scored=1,
            n_series_common=1,
            rank=1,
        )
        assert_round_trips(row)


@settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
@given(
    h=st.integers(min_value=1, max_value=520),
    n_windows=st.integers(min_value=2, max_value=20),
    levels=st.lists(st.integers(min_value=1, max_value=99), min_size=1, max_size=6),
    seed=st.integers(min_value=-(2**31), max_value=2**31),
    trim=st.floats(min_value=0.0, max_value=0.49, allow_nan=False, allow_infinity=False),
)
def test_request_round_trips_for_arbitrary_valid_configurations(h, n_windows, levels, seed, trim):
    assert_round_trips(
        ForecastRequest(
            h=h, freq="W", n_windows=n_windows, levels=levels, seed=seed, ensemble_trim=trim
        )
    )


@settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
@given(
    coverage=st.dictionaries(
        st.integers(min_value=1, max_value=99),
        st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        max_size=5,
    ),
    mase=st.one_of(
        st.none(), st.floats(min_value=0.0, max_value=1e6, allow_nan=False, allow_infinity=False)
    ),
    rank=st.integers(min_value=1, max_value=100),
)
def test_leaderboard_row_round_trips_including_none_metrics(coverage, mase, rank):
    assert_round_trips(
        LeaderboardRow(
            scope="panel",
            model="AutoETS",
            family="local",
            information_set="own_series",
            coverage=coverage,
            mase=mase,
            n_folds=3,
            n_series_scored=1,
            n_series_common=1,
            rank=rank,
        )
    )


def test_request_normalisation_is_idempotent_across_round_trips():
    """FR-204 prepending and FR-112 normalisation must both be fixed points, or a manifest
    replayed twice would not equal itself."""
    once = ForecastRequest(h=13, freq="W", models=["AutoETS"])
    twice = ForecastRequest.model_validate(json.loads(once.model_dump_json()))
    assert once == twice
    assert twice.freq == "W-SUN"
