"""FR-204, FR-216, FR-405a, FR-708 and the D12 frozen-validator finding."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from xlforecast.errors import EnsembleConfigError, InvalidModelNameError
from xlforecast.schemas import DEFAULT_MODELS, DataMapping, ExogSpec, ForecastRequest
from xlforecast.schemas.request import ResolvedRequest


class TestSeasonalNaiveAlwaysIncluded:
    """FR-204 / hard rule 1 — cannot be disabled by configuration, LLM, or code path."""

    def test_added_when_absent(self):
        assert "SeasonalNaive" in ForecastRequest(h=4, freq="ME", models=["AutoETS"]).models

    def test_cannot_be_removed_by_omission(self):
        assert "SeasonalNaive" in ForecastRequest(h=4, freq="ME", models=[]).models

    def test_not_duplicated_when_present(self):
        models = ForecastRequest(h=4, freq="ME", models=["SeasonalNaive", "AutoETS"]).models
        assert models.count("SeasonalNaive") == 1

    def test_request_is_frozen_so_models_cannot_be_reassigned_later(self):
        """The original spec mutated self.models in a mode='after' validator. Under
        frozen=True that raises; under validate_assignment it recurses forever. Hence a
        field_validator (D12, confirmed in the Phase 0 spike)."""
        req = ForecastRequest(h=4, freq="ME")
        with pytest.raises(ValidationError):
            req.models = ["AutoETS"]  # type: ignore[misc]

    def test_models_are_sorted_so_order_cannot_affect_reproducibility(self):
        """List order must not influence best_k tie-breaking, or NFR-02 byte-identity
        depends on how the caller happened to order an argument."""
        a = ForecastRequest(h=4, freq="ME", models=["AutoETS", "AutoARIMA"]).models
        b = ForecastRequest(h=4, freq="ME", models=["AutoARIMA", "AutoETS"]).models
        assert a == b == sorted(a)


class TestModelRegistryValidation:
    """TS §5.2 — the registry, not a closed Literal, is the source of truth."""

    def test_unknown_model_rejected_with_the_known_list(self):
        with pytest.raises(InvalidModelNameError) as exc:
            ForecastRequest(h=4, freq="ME", models=["Prophet"])
        assert exc.value.fix is not None
        assert "AutoETS" in exc.value.fix

    def test_default_set_is_the_thirteen_from_fr216(self):
        req = ForecastRequest(h=13, freq="W")
        assert sorted(req.models) == sorted(DEFAULT_MODELS)
        assert len(req.models) == 13

    @pytest.mark.parametrize(
        ("local", "glob"),
        [("LocalLinear", "GlobalLinear"), ("LocalLGBM", "GlobalLGBM"), ("LocalXGB", "GlobalXGB")],
    )
    def test_matched_pairs_are_both_in_the_default_set(self, local, glob):
        """FR-203b/FR-216a — the pair comparison is only a controlled experiment if both
        halves actually run by default."""
        assert local in DEFAULT_MODELS
        assert glob in DEFAULT_MODELS


class TestEnsembleValidation:
    """FR-405a — weights estimated leave-one-fold-out need at least two folds."""

    @pytest.mark.parametrize("method", ["inverse_error", "best_k"])
    def test_parameter_fitting_ensembles_require_two_windows(self, method):
        with pytest.raises(EnsembleConfigError):
            ForecastRequest(h=4, freq="ME", ensemble=method, n_windows=1)

    @pytest.mark.parametrize("method", ["median", "trimmed_mean", "none"])
    def test_parameterless_ensembles_are_exempt(self, method):
        assert ForecastRequest(h=4, freq="ME", ensemble=method, n_windows=1).n_windows == 1


class TestLengthThreshold:
    """FR-105 — the naive 2*m + h admits series that then vanish inside cross-validation."""

    def test_threshold_accounts_for_the_earliest_training_window(self):
        req = ForecastRequest(h=13, freq="W", n_windows=3)
        assert req.min_observations(52) == 2 * 52 + 13 + 2 * 13 == 143

    def test_naive_threshold_would_have_been_too_short(self):
        req = ForecastRequest(h=13, freq="W", n_windows=3)
        assert req.min_observations(52) > 2 * 52 + 13


class TestAutoArimaMode:
    """FR-201a — Fourier above m=24. Measured 3.1x cheaper (docs/05)."""

    @pytest.mark.parametrize(
        ("m", "mode"),
        [(4, "seasonal"), (12, "seasonal"), (24, "seasonal"), (25, "fourier"), (52, "fourier")],
    )
    def test_threshold(self, m, mode):
        assert ForecastRequest(h=4, freq="ME").autoarima_mode(m) == mode

    def test_resolved_request_rejects_a_contradictory_mode(self):
        with pytest.raises(ValidationError):
            ResolvedRequest(h=13, freq="W", season_length=52, step_size=13, autoarima="seasonal")


class TestOutputSizePrecheck:
    """FR-708 / NFR-12 — FR-107 caps the input; nothing capped the output."""

    def test_large_panel_overflows_excel(self):
        req = ResolvedRequest.from_request(ForecastRequest(h=13, freq="W"), season_length=52)
        assert not req.fits_in_excel(2000)

    def test_typical_panel_fits(self):
        req = ResolvedRequest.from_request(ForecastRequest(h=13, freq="W"), season_length=52)
        assert req.fits_in_excel(300)

    def test_row_formula_matches_the_spec(self):
        req = ResolvedRequest.from_request(
            ForecastRequest(h=10, freq="W", models=["SeasonalNaive"], levels=[80]),
            season_length=52,
        )
        assert req.expected_output_rows(5) == 5 * 10 * 1 * (1 + 2 * 1)


class TestLevelsAndMapping:
    def test_levels_are_sorted_and_deduped_for_deterministic_columns(self):
        assert ForecastRequest(h=4, freq="ME", levels=[95, 80, 95]).levels == [80, 95]

    def test_empty_levels_rejected(self):
        with pytest.raises(ValidationError):
            ForecastRequest(h=4, freq="ME", levels=[])

    def test_duplicate_exog_columns_rejected(self):
        with pytest.raises(ValidationError):
            DataMapping(
                unique_id_col="a",
                ds_col="b",
                y_col="c",
                exog=[
                    ExogSpec(name="promo", kind="historic"),
                    ExogSpec(name="promo", kind="future_known"),
                ],
            )

    def test_future_known_columns_are_identifiable_for_fr111(self):
        m = DataMapping(
            unique_id_col="a",
            ds_col="b",
            y_col="c",
            exog=[
                ExogSpec(name="promo", kind="future_known"),
                ExogSpec(name="temp", kind="historic"),
            ],
        )
        assert m.future_known == ["promo"]


def test_extra_fields_are_forbidden_at_every_boundary():
    with pytest.raises(ValidationError):
        ForecastRequest(h=4, freq="ME", typo_field=1)  # type: ignore[call-arg]
