"""Orchestrator: spec -> RunResult (TS §5.3).

Execution order, with the two orderings that the original specification had wrong:

    ingest -> profile -> validate -> gap fill -> fingerprint -> folds
      -> per fold: local family and ML family on the SAME pre-sliced train/test
      -> evaluate (point; probabilistic arrives in Phase 2)
      -> refit on full history -> final forecast
      -> persist RunResult + Manifest + RunTiming

`profile` precedes `validate` because FR-105's length threshold depends on `season_length`,
which profiling infers (FR-105a); the original `ingest -> validate -> profile` was circular.
Intermittency is classified on the pre-fill frame because `gap_fill="zero"` manufactures
intermittency (FR-106).
"""

from __future__ import annotations

import math
import platform
import time
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Literal

import numpy as np
import polars as pl

from xlforecast import __version__
from xlforecast.engine import conformal as conformal_layer
from xlforecast.engine import local as local_family
from xlforecast.engine import ml as ml_family
from xlforecast.engine.ensemble import EnsemblePlan, combine_point, ensemble_name, lofo_weights
from xlforecast.engine.evaluate import (
    PROB_METRIC,
    build_leaderboard,
    score_fold,
    score_probabilistic,
)
from xlforecast.engine.folds import make_folds
from xlforecast.engine.registry import is_ml
from xlforecast.engine.select import Selection, select
from xlforecast.engine.timing import measure
from xlforecast.ingest.profile import profile_panel
from xlforecast.ingest.readers import gap_fill, read_panel, split_future_rows
from xlforecast.ingest.validate import validate_panel
from xlforecast.panel import DS, ID, canonical_sort, fingerprint, span
from xlforecast.schemas.artifacts import ArtifactPack
from xlforecast.schemas.profile import DataProfile
from xlforecast.schemas.request import DataMapping, ForecastRequest, ResolvedRequest
from xlforecast.schemas.results import (
    CalibrationRow,
    FoldScore,
    ForecastFrame,
    ForecastRow,
    Leaderboard,
    Manifest,
    ModelTiming,
    RunResult,
    RunTiming,
)

__all__ = ["THREAD_KEYS", "run_from_frame", "run_from_path"]

#: NFR-02 -- byte-identity is defined relative to a recorded thread configuration, because
#: float reductions reorder under thread count and LightGBM's histogram construction is only
#: deterministic at a fixed one. Recorded, not assumed.
THREAD_KEYS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")

_TRACKED = (
    "statsforecast",
    "mlforecast",
    "utilsforecast",
    "coreforecast",
    "lightgbm",
    "xgboost",
    "scikit-learn",
    "numpy",
    "pandas",
    "polars",
    "pyarrow",
)


@dataclass(slots=True)
class _Overhead:
    """FR-217a -- work that is nobody's train or predict, but is real cost."""

    totals: dict[str, float]

    def add(self, stage: str, cpu: float) -> None:
        self.totals[stage] = self.totals.get(stage, 0.0) + cpu


def _package_versions() -> dict[str, str]:
    out: dict[str, str] = {}
    for name in _TRACKED:
        try:
            out[name] = version(name)
        except PackageNotFoundError:  # pragma: no cover - optional extras
            continue
    return out


def _thread_config(n_jobs: int = 1) -> dict[str, str]:
    import os

    config = {k: os.environ.get(k, "unset") for k in THREAD_KEYS}
    config["n_jobs"] = str(n_jobs)
    config["blas"] = "unknown"
    try:
        import threadpoolctl

        pools = threadpoolctl.threadpool_info()
        if pools:
            config["blas"] = f"{pools[0].get('internal_api')}:{pools[0].get('num_threads')}"
    except (ImportError, AttributeError):  # pragma: no cover
        pass
    return config


def run_from_path(
    path: str | Path,
    *,
    request: ForecastRequest,
    mapping: DataMapping,
    job_id: str | None = None,
    n_jobs: int = 1,
) -> RunResult:
    raw = read_panel(path, mapping)
    return run_from_frame(raw, request=request, mapping=mapping, job_id=job_id, n_jobs=n_jobs)


def run_from_frame(
    raw: pl.DataFrame,
    *,
    request: ForecastRequest,
    mapping: DataMapping,
    job_id: str | None = None,
    n_jobs: int = 1,
) -> RunResult:
    """Run one competition. Every number in the result comes from forecasting code."""
    job_id = job_id or str(uuid.uuid4())
    data_id = job_id
    started = datetime.now(UTC).isoformat()
    wall0 = time.perf_counter()
    overhead = _Overhead(totals={})

    with measure() as t:
        history, _future = split_future_rows(raw, mapping, h=request.h)
        history = canonical_sort(history)
    overhead.add("ingest", t.cpu)

    # 1. Profile FIRST (FR-105a) -- validation's length threshold needs season_length.
    with measure() as t:
        prelim = profile_panel(
            history,
            data_id=data_id,
            mapping=mapping,
            freq=request.freq,
            season_length=request.season_length,
        )
        season_length = request.season_length or prelim.season_length_candidates[0]
        resolved = ResolvedRequest.from_request(request, season_length=season_length)
    overhead.add("profile", t.cpu)

    # 2. Validate, then drop the excluded series -- each with a named reason (FS §6).
    with measure() as t:
        report = validate_panel(
            history, request=request, profile=prelim, season_length=season_length
        )
        kept = history.filter(~pl.col(ID).is_in(list(report.excluded)))
    overhead.add("validate", t.cpu)

    # 3. Gap fill AFTER intermittency was classified on the pre-fill frame (FR-106).
    with measure() as t:
        filled = gap_fill(kept, freq=resolved.freq, method=resolved.gap_fill)
        panel = canonical_sort(filled)
        digest = fingerprint(panel)
        profile = profile_panel(
            panel,
            data_id=data_id,
            mapping=mapping,
            freq=resolved.freq,
            season_length=season_length,
            prefill=kept,
        ).model_copy(update={"validation": report})
    overhead.add("prepare", t.cpu)

    with measure() as t:
        folds = make_folds(
            panel,
            h=resolved.h,
            n_windows=resolved.n_windows,
            step_size=resolved.step_size,
            freq=resolved.freq,
        )
    overhead.add("folds", t.cpu)

    local_names = [m for m in resolved.models if not is_ml(m)]
    ml_names = [m for m in resolved.models if is_ml(m)]
    origin, _ = span(panel)

    plan = (
        EnsemblePlan(
            method=resolved.ensemble,
            members=tuple(resolved.models),
            trim=resolved.ensemble_trim,
            best_k=resolved.best_k,
            metric=resolved.ensemble_metric,
            prob_method=resolved.ensemble_prob_method,
        )
        if resolved.ensemble != "none" and len(resolved.models) >= 2
        else None
    )
    scored_models = [*resolved.models, *([ensemble_name(resolved.ensemble)] if plan else [])]

    timings: list[ModelTiming] = []
    fold_scores: list[FoldScore] = []
    fold_predictions: dict[int, pl.DataFrame] = {}
    ensemble_weights: dict[int | None, dict[str, float]] = {}
    ensemble_fallbacks: set[str] = set()
    for fold in folds:
        frames = []
        if local_names:
            preds, t_local = local_family.forecast_fold(
                local_names,
                fold,
                h=resolved.h,
                freq=resolved.freq,
                season_length=season_length,
                origin=origin,
                n_jobs=n_jobs,
            )
            frames.append(preds)
            timings.extend(t_local)
        if ml_names:
            preds, t_ml = ml_family.forecast_fold(
                ml_names,
                fold,
                h=resolved.h,
                freq=resolved.freq,
                season_length=season_length,
                seed=resolved.seed,
            )
            frames.append(preds)
            timings.extend(t_ml)

        combined = pl.concat(frames)

        # FR-405: the ensemble is built and scored INSIDE the CV loop, on this fold's
        # member predictions -- never assembled from final forecasts. FR-405a: the weights
        # it uses here were estimated without this fold.
        if plan is not None:
            with measure() as t:
                weights, fell_back = lofo_weights(fold_scores, plan=plan, exclude_fold=fold.index)
                ensemble_preds, more = combine_point(combined, plan=plan, weights=weights)
                combined = pl.concat([combined, ensemble_preds])
                ensemble_fallbacks.update(fell_back + more)
                ensemble_weights[fold.index] = weights
            overhead.add("ensemble", t.cpu)

        fold_predictions[fold.index] = combined
        with measure() as t:
            fold_scores.extend(
                score_fold(fold, combined, models=scored_models, season_length=season_length)
            )
        overhead.add("evaluate", t.cpu)

    # Conformal calibration precedes the leaderboard, because probabilistic scoring is an
    # input to it (FR-208) rather than an annotation on it.
    with measure() as t:
        residuals = conformal_layer.collect_residuals(folds, fold_predictions)
        support = conformal_layer.series_support(panel)
        grid = conformal_layer.quantile_levels(resolved.levels)
        calibration = _calibrate_all(
            residuals,
            models=scored_models,
            levels=resolved.levels,
            support=support,
            profile=profile,
        )
        bands = (
            [
                conformal_layer.calibrate(
                    residuals,
                    model=m,
                    level=level,
                    all_folds=frozenset(f.index for f in folds),
                )
                for m in scored_models
                for level in resolved.levels
            ]
            if resolved.conformal
            else []
        )
    overhead.add("conformal", t.cpu)

    # FR-208: scaled CRPS, scored cross-conformally so the probabilistic figure is held to
    # the same discipline as coverage.
    with measure() as t:
        if resolved.conformal:
            fold_scores = _attach_crps(
                fold_scores,
                residuals,
                models=scored_models,
                levels=resolved.levels,
                support=support,
                grid=grid,
            )
    overhead.add("evaluate", t.cpu)

    with measure() as t:
        leaderboard = build_leaderboard(fold_scores, models=scored_models)
    overhead.add("evaluate", t.cpu)

    with measure() as t:
        selection = select(
            fold_scores,
            strategy=resolved.selection,
            n_windows=resolved.n_windows,
            any_beat_baseline=leaderboard.any_beat_baseline,
        )
        leaderboard = _apply_selection(leaderboard, selection)
    overhead.add("select", t.cpu)

    # Refit on full history for the delivered forecast.
    final_frames = []
    if local_names:
        preds, t_local = local_family.forecast_full(
            local_names,
            panel,
            h=resolved.h,
            freq=resolved.freq,
            season_length=season_length,
            origin=origin,
            n_jobs=n_jobs,
        )
        final_frames.append(preds)
        timings.extend(t_local)
    if ml_names:
        preds, t_ml = ml_family.forecast_full(
            ml_names,
            panel,
            h=resolved.h,
            freq=resolved.freq,
            season_length=season_length,
            seed=resolved.seed,
        )
        final_frames.append(preds)
        timings.extend(t_ml)

    with measure() as t:
        final = pl.concat(final_frames)
        if plan is not None:
            weights, _ = lofo_weights(fold_scores, plan=plan, exclude_fold=None)
            ensemble_final, _ = combine_point(final, plan=plan, weights=weights)
            final = pl.concat([final, ensemble_final])
            ensemble_weights[None] = weights
        final = final.sort([ID, DS, "model"])

        rows: list[ForecastRow] = [
            ForecastRow(
                unique_id=r[ID],
                ds=str(r[DS]),
                model=r["model"],
                quantity="point",
                level=None,
                value=float(r["y_hat"]),
            )
            for r in final.iter_rows(named=True)
        ]
        # FR-301: the delivered band is calibrated from every fold, unlike the scoring bands
        # which each hold one out.
        if resolved.conformal:
            for model in scored_models:
                for level in resolved.levels:
                    band = next((b for b in bands if b.level == level and b.model == model), None)
                    if band is None:
                        continue
                    slice_ = final.filter(pl.col("model") == model)
                    if slice_.is_empty():
                        continue
                    banded, clip_rates = conformal_layer.apply_bands(slice_, band, support)
                    for r in banded.iter_rows(named=True):
                        if r["half_width"] is None or math.isnan(r["half_width"]):
                            continue
                        rows.append(
                            ForecastRow(
                                unique_id=r[ID],
                                ds=str(r[DS]),
                                model=r["model"],
                                quantity="lo",
                                level=level,
                                value=float(r["lo"]),
                            )
                        )
                        rows.append(
                            ForecastRow(
                                unique_id=r[ID],
                                ds=str(r[DS]),
                                model=r["model"],
                                quantity="hi",
                                level=level,
                                value=float(r["hi"]),
                            )
                        )
                    del clip_rates
        forecast = ForecastFrame(rows=rows, levels=resolved.levels)
    overhead.add("persist", t.cpu)

    manifest = Manifest(
        job_id=job_id,
        engine_version=__version__,
        package_versions=_package_versions(),
        python_version=platform.python_version(),
        request=resolved,
        mapping=mapping,
        data_id=data_id,
        data_fingerprint=digest,
        cutoffs=[str(f.cutoff) for f in folds],
        excluded_series=report.excluded,
        autoarima_mode=resolved.autoarima,
        ets_mode=resolved.ets,
        crps_quantiles=[float(q) for q in grid] if resolved.conformal else [],
        ensemble_params={
            "method": plan.method if plan else "none",
            "prob_method": resolved.ensemble_prob_method,
            "metric": resolved.ensemble_metric,
            "trim": resolved.ensemble_trim,
            "best_k": resolved.best_k,
            "fallbacks": "; ".join(sorted(ensemble_fallbacks)) or "none",
            "selection": selection.strategy,
            "selection_warnings": "; ".join(selection.warnings) or "none",
        },
        prompt_versions={},
        thread_config=_thread_config(n_jobs),
        previous_job_id=None,
        started_at=started,
        finished_at=datetime.now(UTC).isoformat(),
        seed=resolved.seed,
    )

    return RunResult(
        job_id=job_id,
        profile=profile,
        leaderboard=leaderboard,
        forecast=forecast,
        fold_scores=fold_scores,
        bands=bands,
        calibration=calibration,
        timing=RunTiming(
            per_model=timings,
            overhead_cpu_seconds=overhead.totals,
            total_wall_seconds=time.perf_counter() - wall0,
            n_workers=n_jobs,
        ),
        artifacts=_empty_pack(job_id),
        manifest=manifest,
    )


def _empty_pack(job_id: str) -> ArtifactPack:
    """Phase 6 fills this in; the field exists now so RunResult's shape is stable."""
    return ArtifactPack(job_id=job_id)


def profile_only(
    raw: pl.DataFrame, *, request: ForecastRequest, mapping: DataMapping, data_id: str
) -> DataProfile:
    """The `POST /v1/data` path (Phase 4): profile and validate without spending compute."""
    history, _ = split_future_rows(raw, mapping, h=request.h)
    history = canonical_sort(history)
    prelim = profile_panel(
        history,
        data_id=data_id,
        mapping=mapping,
        freq=request.freq,
        season_length=request.season_length,
    )
    season_length = request.season_length or prelim.season_length_candidates[0]
    report = validate_panel(history, request=request, profile=prelim, season_length=season_length)
    return prelim.model_copy(update={"validation": report})


def _calibrate_all(
    residuals: pl.DataFrame,
    *,
    models: list[str],
    levels: list[int],
    support: conformal_layer.Support,
    profile: DataProfile,
) -> list[CalibrationRow]:
    """Build XLF_Diagnostics block 3 (FR-303, FR-307a/b).

    Coverage is a single panel figure -- measurement showed splitting it by intermittency
    class carries no information (0.807 against 0.809). The **tails** are split by class,
    because that is where the classes actually differ (0.00/0.19 against 0.13/0.07), and the
    difference is invisible in the coverage number.
    """
    all_folds = frozenset(residuals.get_column("fold_index").unique().to_list())
    classes = {s.unique_id: s.intermittency for s in profile.series}
    intermittent = {uid for uid, k in classes.items() if k in ("intermittent", "lumpy")}
    smooth = set(classes) - intermittent

    rows: list[CalibrationRow] = []
    for model in models:
        for level in levels:
            nominal = level / 100.0
            bands = conformal_layer.calibrate(
                residuals, model=model, level=level, all_folds=all_folds
            )
            _, clip_rates = conformal_layer.apply_bands(
                residuals.filter(pl.col("model") == model), bands, support
            )
            rows.append(
                CalibrationRow(
                    model=model,
                    level=level,
                    scope="all",
                    nominal=nominal,
                    # FR-302a: when most series fall back to the pooled panel residuals, the
                    # AC-301 control loses its discriminating power -- dropping one fold from
                    # a large pool barely moves the quantile. Surfaced so a reader can tell
                    # a strong result from a structurally weak one.
                    n_pooled_fallback=len(bands.pooled_fallback),
                    mean_clip_rate=(
                        float(sum(clip_rates.values()) / len(clip_rates)) if clip_rates else None
                    ),
                    empirical=_finite(
                        conformal_layer.coverage(
                            residuals, model=model, level=level, support=support
                        )
                    ),
                    # The AC-301 control. Diagnostics only: it is conservative by
                    # construction and cannot report under-coverage, so it is evidence
                    # about the honest figure rather than a figure to show a user.
                    empirical_in_calibration=_finite(
                        conformal_layer.coverage(
                            residuals,
                            model=model,
                            level=level,
                            support=support,
                            in_calibration=True,
                        )
                    ),
                    mean_width=_finite(
                        conformal_layer.interval_width(
                            residuals, model=model, level=level, support=support
                        )
                    ),
                )
            )
            scopes: tuple[tuple[Literal["smooth", "intermittent"], set[str]], ...] = (
                ("smooth", smooth),
                ("intermittent", intermittent),
            )
            for scope, members in scopes:
                if not members:
                    continue
                subset = residuals.filter(pl.col(ID).is_in(list(members)))
                lower, upper = conformal_layer.tail_miscoverage(
                    subset, model=model, level=level, support=support
                )
                rows.append(
                    CalibrationRow(
                        model=model,
                        level=level,
                        scope=scope,
                        nominal=nominal,
                        lower_tail=_finite(lower),
                        upper_tail=_finite(upper),
                    )
                )
    return rows


def _finite(value: float) -> float | None:
    """FR-214 -- an unmeasurable figure is `None`, never `NaN`."""
    import math

    return None if value is None or math.isnan(value) or math.isinf(value) else float(value)


def _attach_crps(
    fold_scores: list[FoldScore],
    residuals: pl.DataFrame,
    *,
    models: list[str],
    levels: list[int],
    support: conformal_layer.Support,
    grid: np.ndarray,
) -> list[FoldScore]:
    """Merge scaled CRPS into the existing per-fold scores (FR-208).

    `FoldScore` is frozen, so this rebuilds rather than mutates -- which is the point: a
    score that could be edited after the fact is not a record of anything.
    """
    by_model: dict[str, dict[tuple[int, str], float | None]] = {}
    for model in models:
        quantiles, columns = conformal_layer.quantile_frame(
            residuals, model=model, levels=levels, support=support
        )
        if quantiles.is_empty():
            continue
        by_model[model] = score_probabilistic(quantiles, columns=columns, grid=grid)

    updated: list[FoldScore] = []
    for score in fold_scores:
        lookup = by_model.get(score.model)
        value = (
            lookup.get((score.fold_index, score.unique_id))
            if lookup and score.unique_id is not None
            else None
        )
        updated.append(score.model_copy(update={"metrics": {**score.metrics, PROB_METRIC: value}}))
    return updated


def _apply_selection(leaderboard: Leaderboard, selection: Selection) -> Leaderboard:
    """Mark the selected rows and attach FR-408's bias disclosure.

    `selected` drives which model's forecast is written to `XLF_Forecast`, so a leaderboard
    that computes a selection without recording it would silently fall back to the top-ranked
    model -- which is right under `pooled` and wrong under `per_series`.

    `selection_biased` and `selected_lofo_score` travel with the row because the selected
    model's own score is an argmin over the folds it is scored on. Reporting it unqualified
    would let success criterion #1 be satisfied by selection bias alone.
    """
    rows = []
    for row in leaderboard.rows:
        chosen = (
            selection.per_series.get(row.unique_id) == row.model
            if row.scope == "series" and row.unique_id is not None
            else row.model == selection.panel_winner
        )
        rows.append(
            row.model_copy(
                update={
                    "selected": chosen,
                    "selection_biased": chosen and selection.biased,
                    "selected_lofo_score": selection.lofo_score if chosen else None,
                }
            )
        )
    return leaderboard.model_copy(update={"rows": rows})
