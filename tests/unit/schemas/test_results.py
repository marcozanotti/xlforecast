"""FR-214, FR-217, FR-302, FR-408 and the NFR-02 timing exclusion."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from xlforecast.schemas import (
    ConformalBands,
    FoldScore,
    Leaderboard,
    LeaderboardRow,
    ModelTiming,
    RunTiming,
)


def _row(**kw) -> LeaderboardRow:
    base = {
        "scope": "panel",
        "model": "AutoETS",
        "family": "local",
        "information_set": "own_series",
        "n_folds": 3,
        "n_series_scored": 288,
        "n_series_common": 288,
        "rank": 1,
    }
    return LeaderboardRow(**{**base, **kw})


class TestMetricDegeneracy:
    """FR-214 — zero denominators are routine, not exceptional."""

    def test_metrics_accept_none(self):
        """A series constant within a training fold has a zero MASE denominator. The value
        is None, never NaN or inf."""
        assert _row(mase=None, scaled_crps=None).mase is None

    def test_none_metrics_survive_json_round_trip(self):
        """A NaN float serialises to null and then FAILS re-validation into a bare float
        (confirmed in the Phase 0 spike) -- a silent corruption surfacing only on replay.
        float | None is what makes the return leg work."""
        row = _row(mase=None)
        assert LeaderboardRow.model_validate_json(row.model_dump_json()).mase is None

    def test_n_series_scored_makes_partial_support_visible(self):
        """FR-209/FR-215 -- a metric averaged over 240 of 288 series must never be silently
        compared with one averaged over 288."""
        row = _row(n_series_scored=240, n_series_common=240)
        assert row.n_series_scored == 240


class TestLeaderboardCarriesNoTimings:
    """FR-217b / NFR-02 -- measured durations are not reproducible."""

    @pytest.mark.parametrize(
        "field", ["duration", "runtime", "train_seconds", "cpu_seconds", "elapsed"]
    )
    def test_no_duration_field_exists(self, field):
        assert field not in LeaderboardRow.model_fields

    def test_adding_one_is_rejected_by_extra_forbid(self):
        with pytest.raises(ValidationError):
            _row(train_seconds=1.0)


class TestSelectionBias:
    """FR-408 -- the winner's-curse figure must be labelled, not silently reported."""

    def test_selected_row_can_carry_its_bias_flag_and_lofo_companion(self):
        row = _row(selected=True, selection_biased=True, mase=0.81, selected_lofo_score=0.94)
        assert row.selection_biased
        assert row.selected_lofo_score > row.mase

    def test_defaults_are_unbiased_and_unselected(self):
        row = _row()
        assert not row.selected
        assert not row.selection_biased


class TestConformalBands:
    """FR-302 -- cross-conformal provenance must be recorded."""

    def test_bands_record_which_folds_calibrated_them(self):
        b = ConformalBands(level=80, calibrated_from_folds=[0, 2])
        assert b.calibrated_from_folds == [0, 2]

    def test_empty_provenance_is_representable_so_a_test_can_catch_it(self):
        """A band calibrated from every fold and then scored on one of them reports nominal
        coverage by construction -- the defect ADR-006 was amended to prevent. The field
        exists so that condition is assertable rather than invisible."""
        assert ConformalBands(level=80).calibrated_from_folds == []

    def test_pooled_fallback_set_round_trips(self):
        b = ConformalBands(level=80, pooled_fallback={"A", "B"})
        assert ConformalBands.model_validate_json(b.model_dump_json()).pooled_fallback == {"A", "B"}


class TestFoldIdentity:
    """G2 asserts fold identity between an ensemble and its members."""

    def test_fold_index_exists_to_assert_on(self):
        fs = FoldScore(
            fold_index=1, cutoff="2024-01-07T00:00:00Z", model="AutoETS", n_train_rows=143
        )
        assert fs.fold_index == 1

    def test_train_rows_recorded_for_the_dropna_asymmetry_audit(self):
        """AC-206 -- mlforecast drops max_lag rows per series, statsforecast does not, so
        identical cutoffs still mean different training samples."""
        local = FoldScore(
            fold_index=0, cutoff="2024-01-07T00:00:00Z", model="AutoETS", n_train_rows=143
        )
        glob = FoldScore(
            fold_index=0, cutoff="2024-01-07T00:00:00Z", model="GlobalLGBM", n_train_rows=91
        )
        assert local.n_train_rows != glob.n_train_rows


class TestRunTiming:
    """FR-217 -- measured train + predict is the cost proxy, reported per model."""

    def _timing(self) -> RunTiming:
        return RunTiming(
            per_model=[
                ModelTiming(
                    model="LocalLGBM",
                    fold_index=0,
                    train_cpu_seconds=1.91,
                    predict_cpu_seconds=4.25,
                    train_wall_seconds=1.91,
                    predict_wall_seconds=4.25,
                    n_series_fitted=10,
                    n_rows_trained=910,
                ),
                ModelTiming(
                    model="AutoETS",
                    fold_index=0,
                    train_cpu_seconds=1.40,
                    predict_cpu_seconds=0.01,
                    train_wall_seconds=1.40,
                    predict_wall_seconds=0.01,
                    n_series_fitted=10,
                    n_rows_trained=1430,
                ),
            ],
            overhead_cpu_seconds={"ingest": 0.5, "conformal": 0.2, "evaluate": 0.3},
            total_wall_seconds=9.0,
            n_workers=8,
        )

    def test_train_and_predict_are_separable(self):
        """The split is load-bearing: the spike found LocalLGBM spends 2.2x more in predict
        than train, because recursive forecasting over h is h sequential passes. No
        fit-count, series-count or model-count proxy can see that."""
        lgbm = self._timing().per_model[0]
        assert lgbm.predict_cpu_seconds > 2 * lgbm.train_cpu_seconds

    def test_parts_account_for_the_whole(self):
        """FR-217a -- attributing only train and predict understates every run."""
        t = self._timing()
        assert t.total_cpu_seconds == pytest.approx(t.model_cpu_seconds + t.overhead_total)
        assert t.overhead_total == pytest.approx(1.0)

    def test_cost_by_model_ranks_most_expensive_first(self):
        ranked = list(self._timing().cost_by_model())
        assert ranked[0] == "LocalLGBM"

    def test_negative_durations_rejected(self):
        with pytest.raises(ValidationError):
            ModelTiming(
                model="x",
                train_cpu_seconds=-1.0,
                predict_cpu_seconds=0.0,
                train_wall_seconds=0.0,
                predict_wall_seconds=0.0,
                n_series_fitted=1,
                n_rows_trained=1,
            )


def test_leaderboard_states_its_aggregation_and_baselines():
    """FR-209 -- the aggregation function must be stated. FR-201b -- WindowAverage is the
    incumbent baseline and is reported alongside SeasonalNaive."""
    lb = Leaderboard()
    assert lb.aggregation == "mean"
    assert "SeasonalNaive" in lb.baseline_models
    assert "WindowAverage" in lb.baseline_models
