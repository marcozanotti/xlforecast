"""FR-401/402/406/408 -- selection, and the winner's curse."""

from __future__ import annotations

import pytest

from xlforecast.engine.select import LOW_WINDOW_THRESHOLD, select, selected_lofo_score
from xlforecast.schemas.results import FoldScore


def scores(spec: dict[int, dict[str, dict[str, float]]]) -> list[FoldScore]:
    """spec[fold][series][model] -> mase."""
    out = []
    for fold, per_series in spec.items():
        for uid, per_model in per_series.items():
            for model, value in per_model.items():
                out.append(
                    FoldScore(
                        fold_index=fold,
                        cutoff="2024-01-31T00:00:00",
                        model=model,
                        unique_id=uid,
                        n_train_rows=100,
                        metrics={"mase": value},
                    )
                )
    return out


TWO_FOLDS = scores(
    {
        0: {"S0": {"A": 1.0, "B": 2.0}, "S1": {"A": 3.0, "B": 1.0}},
        1: {"S0": {"A": 1.2, "B": 2.2}, "S1": {"A": 3.2, "B": 1.1}},
    }
)


class TestPooledSelection:
    def test_picks_one_winner_for_the_panel(self):
        """A wins S0 (1.0/1.2 against 2.0/2.2) but B wins S1 by a wider margin, so the
        pooled mean favours B: 1.575 against 2.10."""
        result = select(TWO_FOLDS, strategy="pooled")
        assert result.panel_winner == "B"
        assert set(result.per_series.values()) == {"B"}

    def test_pooled_and_per_series_can_disagree(self):
        """Which is the whole reason FR-401 offers both. Pooled gives every series B;
        per-series gives S0 to A, because B is worse there even though it wins the panel."""
        pooled = select(TWO_FOLDS, strategy="pooled")
        per_series = select(TWO_FOLDS, strategy="per_series", n_windows=5)
        assert pooled.per_series["S0"] == "B"
        assert per_series.per_series["S0"] == "A"

    def test_reports_an_unbiased_companion_score(self):
        """FR-408 -- the argmin's own score is not an unbiased estimate of its accuracy."""
        result = select(TWO_FOLDS, strategy="pooled")
        assert result.biased
        assert result.lofo_score is not None


class TestPerSeriesSelection:
    def test_picks_the_best_model_for_each_series(self):
        result = select(TWO_FOLDS, strategy="per_series", n_windows=5)
        assert result.per_series == {"S0": "A", "S1": "B"}

    def test_warns_below_five_windows(self):
        """FR-402."""
        result = select(TWO_FOLDS, strategy="per_series", n_windows=3)
        assert any("overfits" in w for w in result.warnings)

    def test_does_not_warn_at_or_above_the_threshold(self):
        result = select(TWO_FOLDS, strategy="per_series", n_windows=LOW_WINDOW_THRESHOLD)
        assert not any("overfits" in w for w in result.warnings)

    def test_is_flagged_as_biased(self):
        assert select(TWO_FOLDS, strategy="per_series", n_windows=5).biased


class TestWinnersCurse:
    """FR-408 -- the reason the companion score exists."""

    def test_lofo_selection_is_worse_than_the_naive_selected_score(self):
        """Constructed so that each model wins one fold by luck. Selecting on all folds and
        scoring on the same folds looks good; selecting on the other fold and scoring on
        this one reveals the luck for what it was."""
        noisy = scores(
            {
                0: {"S0": {"A": 0.1, "B": 5.0}},
                1: {"S0": {"A": 5.0, "B": 0.1}},
            }
        )
        naive_best = min(
            sum(s.metrics["mase"] for s in noisy if s.model == m) / 2 for m in ("A", "B")
        )
        lofo = selected_lofo_score(noisy, strategy="per_series", metric="mase")
        assert lofo is not None
        assert lofo > naive_best, (
            f"leave-one-fold-out {lofo:.2f} must expose the optimism in {naive_best:.2f}"
        )

    def test_a_genuinely_better_model_survives_the_holdout(self):
        """The contrast case: when the winner is really the winner, the two figures agree
        closely. Selection bias inflates luck, not skill."""
        consistent = scores(
            {
                0: {"S0": {"A": 0.5, "B": 3.0}},
                1: {"S0": {"A": 0.6, "B": 3.1}},
            }
        )
        lofo = selected_lofo_score(consistent, strategy="per_series", metric="mase")
        assert lofo == pytest.approx(0.55, abs=0.01)

    def test_one_fold_admits_no_unbiased_estimate(self):
        """With nothing to hold out, the honest answer is that there is no answer."""
        single = scores({0: {"S0": {"A": 1.0, "B": 2.0}}})
        assert selected_lofo_score(single, strategy="pooled", metric="mase") is None


class TestBaselineFallback:
    """FR-406 -- reporting this plainly is the product."""

    def test_recommends_the_baseline_when_nothing_beat_it(self):
        result = select(TWO_FOLDS, strategy="pooled", any_beat_baseline=False, baseline="A")
        assert result.panel_winner == "A"
        assert any("did not beat" in w or "No model beat" in w for w in result.warnings)

    def test_the_baseline_recommendation_is_not_flagged_as_biased(self):
        """No argmin was performed, so there is no winner's curse to declare."""
        result = select(TWO_FOLDS, strategy="pooled", any_beat_baseline=False, baseline="A")
        assert not result.biased


class TestDegenerateInput:
    def test_clustered_falls_back_to_pooled_and_says_so(self):
        result = select(TWO_FOLDS, strategy="clustered")
        assert result.strategy == "pooled"
        assert any("clustered" in w for w in result.warnings)

    def test_no_usable_scores_yields_no_winner_rather_than_a_wrong_one(self):
        none_metrics = [
            FoldScore(
                fold_index=0,
                cutoff="c",
                model="A",
                unique_id="S0",
                n_train_rows=10,
                metrics={"mase": None},
            )
        ]
        result = select(none_metrics, strategy="pooled")
        assert result.panel_winner is None
        assert result.per_series == {}

    def test_degenerate_metrics_do_not_make_a_model_look_perfect(self):
        """FR-214 -- a None must be skipped, not read as zero."""
        mixed = scores({0: {"S0": {"A": 2.0, "B": 1.0}}, 1: {"S0": {"A": 2.0, "B": 1.0}}})
        mixed.append(
            FoldScore(
                fold_index=0,
                cutoff="c",
                model="C",
                unique_id="S0",
                n_train_rows=10,
                metrics={"mase": None},
            )
        )
        assert select(mixed, strategy="pooled").panel_winner == "B"
