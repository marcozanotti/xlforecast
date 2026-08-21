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

import platform
import time
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import polars as pl

from xlforecast import __version__
from xlforecast.engine import local as local_family
from xlforecast.engine import ml as ml_family
from xlforecast.engine.evaluate import build_leaderboard, score_fold
from xlforecast.engine.folds import make_folds
from xlforecast.engine.registry import is_ml
from xlforecast.engine.timing import measure
from xlforecast.ingest.profile import profile_panel
from xlforecast.ingest.readers import gap_fill, read_panel, split_future_rows
from xlforecast.ingest.validate import validate_panel
from xlforecast.panel import DS, ID, canonical_sort, fingerprint, span
from xlforecast.schemas.artifacts import ArtifactPack
from xlforecast.schemas.profile import DataProfile
from xlforecast.schemas.request import DataMapping, ForecastRequest, ResolvedRequest
from xlforecast.schemas.results import (
    ForecastFrame,
    ForecastRow,
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


def _thread_config() -> dict[str, str]:
    import os

    config = {k: os.environ.get(k, "unset") for k in THREAD_KEYS}
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
    path: str | Path, *, request: ForecastRequest, mapping: DataMapping, job_id: str | None = None
) -> RunResult:
    raw = read_panel(path, mapping)
    return run_from_frame(raw, request=request, mapping=mapping, job_id=job_id)


def run_from_frame(
    raw: pl.DataFrame,
    *,
    request: ForecastRequest,
    mapping: DataMapping,
    job_id: str | None = None,
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

    timings: list[ModelTiming] = []
    fold_scores = []
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

        with measure() as t:
            fold_scores.extend(
                score_fold(
                    fold, pl.concat(frames), models=resolved.models, season_length=season_length
                )
            )
        overhead.add("evaluate", t.cpu)

    with measure() as t:
        leaderboard = build_leaderboard(fold_scores, models=resolved.models)
    overhead.add("evaluate", t.cpu)

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
        final = pl.concat(final_frames).sort([ID, DS, "model"])
        forecast = ForecastFrame(
            rows=[
                ForecastRow(
                    unique_id=r[ID],
                    ds=str(r[DS]),
                    model=r["model"],
                    quantity="point",
                    level=None,
                    value=float(r["y_hat"]),
                )
                for r in final.iter_rows(named=True)
            ],
            levels=resolved.levels,
        )
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
        crps_quantiles=[],  # Phase 2
        ensemble_params={
            "method": resolved.ensemble,
            "metric": resolved.ensemble_metric,
            "trim": resolved.ensemble_trim,
            "best_k": resolved.best_k,
        },
        prompt_versions={},
        thread_config=_thread_config(),
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
        bands=[],  # Phase 2
        timing=RunTiming(
            per_model=timings,
            overhead_cpu_seconds=overhead.totals,
            total_wall_seconds=time.perf_counter() - wall0,
            n_workers=1,
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
