"""FR-403/403a/404/405/405a -- and gate G2's ensemble clauses."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from xlforecast.engine.ensemble import (
    MIN_MEMBERS_FOR_TRIM,
    EnsemblePlan,
    combine_point,
    combine_quantiles,
    ensemble_name,
    lofo_weights,
    member_errors,
)
from xlforecast.errors import EnsembleConfigError
from xlforecast.schemas.results import FoldScore

MEMBERS = ("A", "B", "C")


def scores(values: dict[int, dict[str, float]], metric: str = "mase") -> list[FoldScore]:
    """values[fold][model] -> metric. One series, for clarity."""
    out = []
    for fold, per_model in values.items():
        for model, value in per_model.items():
            out.append(
                FoldScore(
                    fold_index=fold,
                    cutoff="2024-01-31T00:00:00",
                    model=model,
                    unique_id="S0",
                    n_train_rows=100,
                    metrics={metric: value},
                )
            )
    return out


def predictions(per_model: dict[str, list[float]]) -> pl.DataFrame:
    n = len(next(iter(per_model.values())))
    frames = []
    for model, values in per_model.items():
        frames.append(
            pl.DataFrame(
                {
                    "unique_id": ["S0"] * n,
                    "ds": pl.datetime_range(
                        pl.datetime(2024, 1, 1), pl.datetime(2024, 1, n), "1d", eager=True
                    ),
                    "model": [model] * n,
                    "y_hat": values,
                }
            )
        )
    return pl.concat(frames)


class TestLeaveOneFoldOutWeights:
    """FR-405a and gate G2's second clause."""

    def test_weights_for_fold_k_ignore_fold_k(self):
        """The direct statement of the requirement."""
        data = scores({0: {"A": 1.0, "B": 2.0}, 1: {"A": 2.0, "B": 1.0}})
        plan = EnsemblePlan(method="inverse_error", members=("A", "B"))
        w0, _ = lofo_weights(data, plan=plan, exclude_fold=0)
        # Fold 0 excluded, so only fold 1 remains, where B is the better model.
        assert w0["B"] > w0["A"]

    def test_perturbing_fold_k_does_not_change_the_weights_used_to_score_fold_k(self):
        """Gate G2 states this as the test. It is the sharpest form: if fold k's own errors
        leaked into its weights, changing them would move the weights."""
        plan = EnsemblePlan(method="inverse_error", members=("A", "B"))
        original = scores({0: {"A": 1.0, "B": 4.0}, 1: {"A": 3.0, "B": 1.0}})
        perturbed = scores({0: {"A": 99.0, "B": 0.01}, 1: {"A": 3.0, "B": 1.0}})
        assert (
            lofo_weights(original, plan=plan, exclude_fold=0)[0]
            == lofo_weights(perturbed, plan=plan, exclude_fold=0)[0]
        )

    def test_the_delivered_weights_may_use_every_fold(self):
        plan = EnsemblePlan(method="inverse_error", members=("A", "B"))
        data = scores({0: {"A": 1.0, "B": 4.0}, 1: {"A": 3.0, "B": 1.0}})
        full, _ = lofo_weights(data, plan=plan, exclude_fold=None)
        held_out, _ = lofo_weights(data, plan=plan, exclude_fold=0)
        assert full != held_out

    def test_parameterless_methods_are_exempt(self):
        """`median` and `trimmed_mean` fit nothing, so there is nothing to leak."""
        data = scores({0: {"A": 1.0, "B": 4.0}, 1: {"A": 3.0, "B": 1.0}})
        for method in ("median", "trimmed_mean"):
            plan = EnsemblePlan(method=method, members=("A", "B"))
            weights, _ = lofo_weights(data, plan=plan, exclude_fold=0)
            assert weights == {"A": 0.5, "B": 0.5}

    def test_best_k_membership_is_also_held_out(self):
        """Selecting the members is itself a fitted parameter."""
        plan = EnsemblePlan(method="best_k", members=MEMBERS, best_k=1)
        data = scores({0: {"A": 0.1, "B": 9.0, "C": 9.0}, 1: {"A": 9.0, "B": 0.1, "C": 9.0}})
        assert set(lofo_weights(data, plan=plan, exclude_fold=0)[0]) == {"B"}
        assert set(lofo_weights(data, plan=plan, exclude_fold=1)[0]) == {"A"}


class TestMemberErrors:
    def test_degenerate_metrics_are_skipped_not_counted_as_zero(self):
        """FR-214 -- a None metric must not make a model look perfect on the series it
        failed to score."""
        data = scores({0: {"A": 2.0}, 1: {"A": 4.0}})
        data.append(
            FoldScore(
                fold_index=2,
                cutoff="c",
                model="A",
                unique_id="S0",
                n_train_rows=1,
                metrics={"mase": None},
            )
        )
        assert member_errors(data, members=("A",), metric="mase")["A"] == 3.0

    def test_a_model_with_no_usable_scores_is_absent_rather_than_infinite(self):
        data = scores({0: {"A": 1.0}})
        assert "B" not in member_errors(data, members=("A", "B"), metric="mase")


class TestEdgeCases:
    """FR-403a -- defined behaviour, not implementation detail."""

    def test_best_k_larger_than_the_member_count_falls_back_and_records_it(self):
        plan = EnsemblePlan(method="best_k", members=("A", "B"), best_k=5)
        data = scores({0: {"A": 1.0, "B": 2.0}, 1: {"A": 1.0, "B": 2.0}})
        weights, fallbacks = lofo_weights(data, plan=plan, exclude_fold=0)
        assert len(weights) == 2
        assert any("best_k" in f for f in fallbacks)

    def test_trimmed_mean_degrades_to_median_below_five_members(self):
        """With 0.2 trim and four members, floor(4*0.2)=0 -- the trim removes nothing and
        'trimmed mean' would be a plain mean wearing a different name."""
        plan = EnsemblePlan(method="trimmed_mean", members=("A", "B", "C"))
        combined, fallbacks = combine_point(
            predictions({"A": [1.0], "B": [2.0], "C": [30.0]}),
            plan=plan,
            weights=dict.fromkeys(MEMBERS, 1 / 3),
        )
        assert any("median" in f for f in fallbacks)
        assert combined.get_column("y_hat")[0] == 2.0  # median, not the mean of 11.0

    def test_trim_actually_trims_once_there_are_enough_members(self):
        members = tuple("ABCDE")
        plan = EnsemblePlan(method="trimmed_mean", members=members)
        combined, fallbacks = combine_point(
            predictions({"A": [1.0], "B": [2.0], "C": [3.0], "D": [4.0], "E": [100.0]}),
            plan=plan,
            weights=dict.fromkeys(members, 0.2),
        )
        assert not fallbacks
        assert combined.get_column("y_hat")[0] == 3.0  # 1 and 100 trimmed

    def test_a_single_member_ensemble_is_refused(self):
        with pytest.raises(EnsembleConfigError):
            EnsemblePlan(method="median", members=("A",))

    def test_an_ensemble_whose_members_did_not_run_is_refused(self):
        plan = EnsemblePlan(method="median", members=("A", "B", "C"))
        with pytest.raises(EnsembleConfigError) as exc:
            combine_point(
                predictions({"A": [1.0]}), plan=plan, weights=dict.fromkeys(MEMBERS, 1 / 3)
            )
        assert exc.value.fix

    def test_zero_error_members_are_tied_rather_than_infinitely_weighted(self):
        plan = EnsemblePlan(method="inverse_error", members=("A", "B"))
        data = scores({0: {"A": 0.0, "B": 1.0}, 1: {"A": 0.0, "B": 1.0}})
        weights, fallbacks = lofo_weights(data, plan=plan, exclude_fold=None)
        assert weights == {"A": 1.0}
        assert any("zero-error" in f for f in fallbacks)


class TestPointCombination:
    def test_median_ignores_an_outlying_member(self):
        plan = EnsemblePlan(method="median", members=MEMBERS)
        combined, _ = combine_point(
            predictions({"A": [10.0], "B": [11.0], "C": [1000.0]}),
            plan=plan,
            weights=dict.fromkeys(MEMBERS, 1 / 3),
        )
        assert combined.get_column("y_hat")[0] == 11.0

    def test_inverse_error_leans_on_the_better_member(self):
        plan = EnsemblePlan(method="inverse_error", members=("A", "B"))
        combined, _ = combine_point(
            predictions({"A": [0.0], "B": [10.0]}),
            plan=plan,
            weights={"A": 0.9, "B": 0.1},
        )
        assert combined.get_column("y_hat")[0] == pytest.approx(1.0)

    def test_the_ensemble_is_named_for_its_method(self):
        plan = EnsemblePlan(method="median", members=MEMBERS)
        combined, _ = combine_point(
            predictions({"A": [1.0], "B": [2.0], "C": [3.0]}),
            plan=plan,
            weights=dict.fromkeys(MEMBERS, 1 / 3),
        )
        assert combined.get_column("model")[0] == ensemble_name("median") == "Ensemble[median]"

    def test_weights_that_miss_every_surviving_member_fall_back_to_equal(self):
        plan = EnsemblePlan(method="inverse_error", members=("A", "B"))
        combined, _ = combine_point(
            predictions({"A": [0.0], "B": [10.0]}), plan=plan, weights={"Z": 1.0}
        )
        assert combined.get_column("y_hat")[0] == pytest.approx(5.0)


class TestProbabilisticCombination:
    """FR-404 -- vincentization and linear pooling give different objects."""

    LEVELS = np.array([0.1, 0.5, 0.9])

    def test_vincentization_averages_the_quantiles(self):
        q = {"A": np.array([0.0, 10.0, 20.0]), "B": np.array([10.0, 20.0, 30.0])}
        out = combine_quantiles(
            q, self.LEVELS, weights={"A": 0.5, "B": 0.5}, method="vincentization"
        )
        assert out == pytest.approx([5.0, 15.0, 25.0])

    def test_linear_pooling_is_wider_when_members_disagree(self):
        """The substantive difference: pooling represents disagreement as uncertainty,
        where vincentization averages it away."""
        q = {"A": np.array([0.0, 5.0, 10.0]), "B": np.array([100.0, 105.0, 110.0])}
        w = {"A": 0.5, "B": 0.5}
        vinc = combine_quantiles(q, self.LEVELS, weights=w, method="vincentization")
        pool = combine_quantiles(q, self.LEVELS, weights=w, method="linear_pool")
        assert (pool[-1] - pool[0]) > (vinc[-1] - vinc[0])

    def test_the_two_methods_agree_when_members_agree(self):
        q = {"A": np.array([0.0, 10.0, 20.0]), "B": np.array([0.0, 10.0, 20.0])}
        w = {"A": 0.5, "B": 0.5}
        vinc = combine_quantiles(q, self.LEVELS, weights=w, method="vincentization")
        pool = combine_quantiles(q, self.LEVELS, weights=w, method="linear_pool")
        assert pool == pytest.approx(vinc, abs=0.5)

    def test_both_methods_return_monotone_quantiles(self):
        q = {"A": np.array([1.0, 4.0, 9.0]), "B": np.array([2.0, 3.0, 20.0])}
        w = {"A": 0.5, "B": 0.5}
        for method in ("vincentization", "linear_pool"):
            out = combine_quantiles(q, self.LEVELS, weights=w, method=method)
            assert list(out) == sorted(out), method

    def test_zero_weight_members_are_excluded_from_the_mixture(self):
        q = {"A": np.array([0.0, 1.0, 2.0]), "B": np.array([100.0, 101.0, 102.0])}
        out = combine_quantiles(
            q, self.LEVELS, weights={"A": 1.0, "B": 0.0}, method="vincentization"
        )
        assert out == pytest.approx([0.0, 1.0, 2.0])

    def test_absent_weights_fall_back_to_equal_shares(self):
        q = {"A": np.array([0.0, 1.0, 2.0]), "B": np.array([2.0, 3.0, 4.0])}
        out = combine_quantiles(q, self.LEVELS, weights={}, method="vincentization")
        assert out == pytest.approx([1.0, 2.0, 3.0])


def test_min_members_constant_is_consistent_with_the_default_trim():
    """floor(n * 0.2) >= 1 requires n >= 5."""
    assert int(np.floor(MIN_MEMBERS_FOR_TRIM * 0.2)) == 1
    assert int(np.floor((MIN_MEMBERS_FOR_TRIM - 1) * 0.2)) == 0
