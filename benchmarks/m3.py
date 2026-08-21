"""M3 benchmark harness -- gate G3 (docs/03-BUILD-PLAN.md).

**Comparability mode.** The Monash archive evaluates on a *single* forecast origin: the last
`horizon` observations are the test set. Our engine is cross-validated by design, so this
harness deliberately bypasses `engine/run.py` and drives the model adapters directly at one
origin. Anything else would compare our 3-fold average against their single holdout and call
the difference a result.

That has a consequence worth stating: this mode exercises the model adapters and the metric
layer, not the fold machinery, the conformal layer or ensembling. Those are covered by G1 and
G2. G3 asks one question only -- are the models we fit competitive with published baselines
on data resembling our users' -- and it is the question that can stop the project.

Usage::

    uv run python benchmarks/m3.py --data-dir <dir> --dataset m3_yearly
    uv run python benchmarks/m3.py --data-dir <dir> --all --models SeasonalNaive,AutoETS
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

sys.path.insert(0, str(Path(__file__).parent))
from tsf import TsfDataset, read_tsf

from xlforecast.engine import local as local_family
from xlforecast.engine.timing import measure

BASELINES = Path(__file__).parent / "baselines" / "monash_m3.json"

#: M3 has no seasonality above 12, so FR-201a's Fourier switch (m > 24) never fires here and
#: FR-201c's MSTL substitution never fires either. Both stay covered by the unit suite.
FREQ = {"m3_yearly": "YE", "m3_quarterly": "QE", "m3_monthly": "ME", "m3_other": "QE"}

DEFAULT_MODELS = (
    "SeasonalNaive",
    "HistoricAverage",
    "AutoETS",
    "AutoARIMA",
    "DynamicOptimizedTheta",
)


def to_panel(series: dict[str, np.ndarray], *, freq: str, starts: dict[str, str]) -> pl.DataFrame:
    """Long-format panel with a synthetic calendar.

    The M3 series' absolute dates are immaterial to the models we fit here -- every
    seasonality is <= 12, so no date-feature or Fourier path activates -- but the engine
    requires a real calendar, so one is supplied. `m3_other` has no timestamps at all and is
    placed on an arbitrary quarterly grid with seasonality 1.
    """
    frames = []
    for name, values in series.items():
        raw = starts.get(name)
        start = pd.Timestamp(raw.replace("-", ":", 0)[:10]) if raw else pd.Timestamp("1975-01-01")
        dates = pd.date_range(start=start, periods=len(values), freq=freq)
        frames.append(
            pl.DataFrame(
                {"unique_id": [name] * len(values), "ds": list(dates), "y": values.tolist()}
            )
        )
    return pl.concat(frames).sort(["unique_id", "ds"])


def evaluate(
    dataset: TsfDataset, key: str, models: list[str], seasonality: int
) -> tuple[dict[str, dict[str, float]], dict[str, float]]:
    """Fit at one origin and score with the archive's own metric definitions."""
    from utilsforecast.evaluation import evaluate as uf_evaluate
    from utilsforecast.losses import mase, smape

    freq = FREQ[key]
    train_raw, test_raw = dataset.split()
    train = to_panel(train_raw, freq=freq, starts=dataset.starts)

    # Test timestamps continue each series' own calendar.
    test_frames = []
    for name, values in test_raw.items():
        last = train.filter(pl.col("unique_id") == name).get_column("ds").max()
        dates = pd.date_range(start=last, periods=len(values) + 1, freq=freq)[1:]
        test_frames.append(
            pl.DataFrame(
                {"unique_id": [name] * len(values), "ds": list(dates), "y": values.tolist()}
            )
        )
    test = pl.concat(test_frames)

    origin = train.get_column("ds").min()
    with measure() as timer:
        preds, timings = local_family.forecast_full(
            models, train, h=dataset.horizon, freq=freq, season_length=seasonality, origin=origin
        )

    wide = preds.pivot(on="model", index=["unique_id", "ds"], values="y_hat").join(
        test, on=["unique_id", "ds"], how="inner"
    )
    present = [m for m in models if m in wide.columns]
    frame = wide.select(["unique_id", "ds", "y", *present]).to_pandas()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        table = uf_evaluate(
            frame,
            metrics=[partial(mase, seasonality=seasonality), smape],
            models=present,
            train_df=train.select(["unique_id", "ds", "y"]).to_pandas(),
        )

    scores: dict[str, dict[str, float]] = {}
    for metric, group in table.groupby("metric"):
        for model in present:
            values = group[model].to_numpy(dtype=float)
            finite = values[np.isfinite(values)]
            scores.setdefault(model, {})[str(metric)] = (
                float(finite.mean()) if finite.size else float("nan")
            )
    # utilsforecast reports sMAPE as a fraction; the archive reports percent.
    for model in scores:
        if "smape" in scores[model]:
            scores[model]["smape"] *= 100.0

    cost = {t.model: t.train_cpu_seconds + t.predict_cpu_seconds for t in timings}
    cost["_total_wall"] = timer.wall
    return scores, cost


def compare(key: str, scores: dict[str, dict[str, float]], baselines: dict) -> list[dict]:
    """Map our models onto the published baselines and apply the G3 tolerance."""
    spec = baselines["datasets"][key]
    tolerance = baselines["gate_g3"]["tolerance_pct"]
    rows = []
    if not spec.get("mean_mase"):
        return rows
    for published, ours in baselines["model_map"].items():
        if published == "comment" or ours not in scores:
            continue
        reference = spec["mean_mase"].get(published)
        if reference is None:
            continue
        value = scores[ours]["mase"]
        delta = (value - reference) / reference * 100.0
        # Asymmetric: only being materially WORSE is a gate failure. Being materially better
        # is flagged instead -- it is either good news or the signature of a leak, and the
        # gate should not decide which on its own.
        rows.append(
            {
                "dataset": key,
                "published_model": published,
                "our_model": ours,
                "published_mase": reference,
                "our_mase": round(value, 4),
                "delta_pct": round(delta, 2),
                "verdict": (
                    "fail"
                    if delta > tolerance
                    else "better-than-published"
                    if delta < -tolerance
                    else "pass"
                ),
            }
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="M3 benchmark (gate G3)")
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--dataset", default=None, help="e.g. m3_yearly")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--models", default=",".join(DEFAULT_MODELS))
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    baselines = json.loads(BASELINES.read_text())
    keys = list(baselines["datasets"]) if args.all else [args.dataset]
    models = [m.strip() for m in args.models.split(",")]

    results, comparisons = {}, []
    for key in keys:
        suffix = key.replace("m3_", "")
        path = args.data_dir / f"m3_{suffix}_dataset.tsf"
        if not path.exists():
            print(f"  {key}: missing {path.name}, skipped")
            continue
        dataset = read_tsf(path)
        seasonality = baselines["datasets"][key]["seasonality"]
        started = time.perf_counter()
        scores, cost = evaluate(dataset, key, models, seasonality)
        results[key] = {"scores": scores, "cpu_seconds": cost}
        comparisons.extend(compare(key, scores, baselines))

        published = baselines["datasets"][key].get("mean_mase") or {}
        print(
            f"\n{key}  ({dataset.n_series} series, h={dataset.horizon}, S={seasonality}, "
            f"{time.perf_counter() - started:.0f}s)"
        )
        print(f"  {'model':<24}{'MASE':>9}{'sMAPE':>9}{'published':>11}{'delta':>9}")
        inverse = {v: k for k, v in baselines["model_map"].items() if k != "comment"}
        for model in models:
            if model not in scores:
                continue
            ref = published.get(inverse.get(model, ""), None)
            delta = f"{(scores[model]['mase'] - ref) / ref * 100:+.1f}%" if ref else "--"
            print(
                f"  {model:<24}{scores[model]['mase']:>9.3f}{scores[model]['smape']:>9.2f}"
                f"{(f'{ref:.3f}' if ref else '--'):>11}{delta:>9}"
            )

    if comparisons:
        failures = [c for c in comparisons if c["verdict"] == "fail"]
        flagged = [c for c in comparisons if c["verdict"] == "better-than-published"]
        print(
            f"\nG3: {len(comparisons) - len(failures)}/{len(comparisons)} comparisons pass "
            f"(tolerance {baselines['gate_g3']['tolerance_pct']}%, asymmetric)"
        )
        for c in failures:
            print(
                f"  FAIL  {c['dataset']:<15}{c['our_model']:<24}"
                f"{c['our_mase']:.3f} vs {c['published_mase']:.3f} ({c['delta_pct']:+.1f}% worse)"
            )
        for c in flagged:
            print(
                f"  CHECK {c['dataset']:<15}{c['our_model']:<24}"
                f"{c['our_mase']:.3f} vs {c['published_mase']:.3f} ({c['delta_pct']:+.1f}%) "
                "-- better than published; confirm no leakage"
            )
        if failures:
            return 1
    if args.out:
        args.out.write_text(json.dumps({"results": results, "comparisons": comparisons}, indent=2))
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
