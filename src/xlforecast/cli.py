"""Typer CLI (FR-706).

`xlforecast run data.csv --h 13 --freq W` writes the four tables plus a JSON manifest --
the same objects the add-in will write into sheets, so the engine is validated end to end
long before any Office.js code exists (ADR-001).
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import polars as pl
import typer

from xlforecast import __version__
from xlforecast.engine.run import run_from_path
from xlforecast.errors import XLForecastError
from xlforecast.schemas.request import DataMapping, ForecastRequest
from xlforecast.schemas.results import RunResult

app = typer.Typer(add_completion=False, help="Reproducible time series model competition.")


def _fmt(value: float | None) -> str:
    """`None` prints as `--`. FR-214 makes absent metrics routine, and a blank cell reads as
    a bug where an explicit dash reads as an answer."""
    return f"{value:.3f}" if value is not None else "--"


def _leaderboard_frame(result: RunResult) -> pl.DataFrame:
    rows = [r.model_dump() for r in result.leaderboard.rows]
    frame = pl.DataFrame(rows, infer_schema_length=None) if rows else pl.DataFrame()
    return frame.drop("coverage", "coverage_intermittent", strict=False)


def _write_outputs(result: RunResult, out: Path) -> dict[str, Path]:
    out.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}

    board = _leaderboard_frame(result)
    written["XLF_Leaderboard"] = out / "XLF_Leaderboard.csv"
    board.write_csv(written["XLF_Leaderboard"])

    long_rows = [r.model_dump() for r in result.forecast.rows]
    written["XLF_Forecast_Long"] = out / "XLF_Forecast_Long.csv"
    pl.DataFrame(long_rows, infer_schema_length=None).write_csv(written["XLF_Forecast_Long"])

    selected = {r.unique_id: r.model for r in result.leaderboard.rows if r.selected}
    winner = result.leaderboard.rows[0].model if result.leaderboard.rows else None
    wide = pl.DataFrame(long_rows, infer_schema_length=None)
    if not wide.is_empty():
        wide = (
            wide.filter(
                pl.col("model") == pl.col("unique_id").replace_strict(selected, default=winner)
            )
            .select(["unique_id", "ds", "value", "model"])
            .rename({"value": "y_hat"})
        )
    written["XLF_Forecast"] = out / "XLF_Forecast.csv"
    wide.write_csv(written["XLF_Forecast"])

    diagnostics = pl.DataFrame(
        [
            {
                "fold_index": f.fold_index,
                "cutoff": f.cutoff,
                "model": f.model,
                "unique_id": f.unique_id,
                "n_train_rows": f.n_train_rows,
                **f.metrics,
            }
            for f in result.fold_scores
        ],
        infer_schema_length=None,
    )
    written["XLF_Diagnostics"] = out / "XLF_Diagnostics.csv"
    diagnostics.write_csv(written["XLF_Diagnostics"])

    # Hard rule 10: no manifest, no result.
    written["XLF_Manifest"] = out / "XLF_Manifest.json"
    written["XLF_Manifest"].write_text(result.manifest.model_dump_json(indent=2))

    written["XLF_Timing"] = out / "XLF_Timing.json"
    written["XLF_Timing"].write_text(result.timing.model_dump_json(indent=2))
    return written


@app.command()
def run(
    path: Annotated[Path, typer.Argument(help="CSV or Parquet panel in long format.")],
    h: Annotated[int, typer.Option("--h", help="Forecast horizon.")] = 13,
    freq: Annotated[str, typer.Option("--freq", help="pandas offset alias, e.g. W or ME.")] = "W",
    unique_id_col: Annotated[str, typer.Option(help="Series id column.")] = "unique_id",
    ds_col: Annotated[str, typer.Option(help="Timestamp column.")] = "ds",
    y_col: Annotated[str, typer.Option(help="Target column.")] = "y",
    models: Annotated[str, typer.Option(help="Comma-separated, or 'auto' for FR-216.")] = "auto",
    n_windows: Annotated[int, typer.Option(help="CV windows.")] = 3,
    season_length: Annotated[int | None, typer.Option(help="Inferred when omitted.")] = None,
    out: Annotated[Path, typer.Option("--out", help="Output directory.")] = Path("xlf_out"),
) -> None:
    """Run a model competition and write four tables plus a manifest."""
    try:
        chosen = None if models == "auto" else [m.strip() for m in models.split(",")]
        request = (
            ForecastRequest(h=h, freq=freq, n_windows=n_windows, season_length=season_length)
            if chosen is None
            else ForecastRequest(
                h=h,
                freq=freq,
                n_windows=n_windows,
                season_length=season_length,
                models=chosen,
            )
        )
        mapping = DataMapping(unique_id_col=unique_id_col, ds_col=ds_col, y_col=y_col)
        result = run_from_path(path, request=request, mapping=mapping)
    except XLForecastError as exc:
        # FS §4 error-presentation rule: name the fault, state the fix, never a traceback.
        typer.secho(exc.render(), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from None

    written = _write_outputs(result, out)
    board = result.leaderboard
    panel_rows = [r for r in board.rows if r.scope == "panel"]

    typer.secho(f"\nxlforecast {__version__}  ·  job {result.job_id}", bold=True)
    typer.echo(
        f"{board.aggregation} over {panel_rows[0].n_series_common if panel_rows else 0} series"
        f"  ·  {len(result.manifest.cutoffs)} folds"
        f"  ·  excluded {len(result.manifest.excluded_series)}"
    )
    typer.echo(f"\n{'model':<24}{'MASE':>9}{'RMSSE':>9}{'vs base %':>11}{'vs incumbent %':>16}")
    for row in panel_rows:
        typer.echo(
            f"{row.model:<24}{_fmt(row.mase):>9}{_fmt(row.rmsse):>9}"
            f"{_fmt(row.vs_baseline_pct):>11}{_fmt(row.vs_incumbent_pct):>16}"
        )

    if not board.any_beat_baseline:
        # FR-406 -- said prominently, because saying it is the product.
        typer.secho(
            "\nNo model beat the seasonal-naive baseline. Recommending SeasonalNaive.",
            fg=typer.colors.YELLOW,
            bold=True,
        )

    cost = result.timing.cost_by_model()
    typer.echo(
        f"\nRuntime {result.timing.total_wall_seconds:.1f}s wall, "
        f"{result.timing.total_cpu_seconds:.1f}s CPU. Most expensive models:"
    )
    for model, seconds in list(cost.items())[:3]:
        share = seconds / result.timing.total_cpu_seconds * 100
        typer.echo(f"  {model:<24}{seconds:>8.2f}s CPU  ({share:.0f}%)")

    typer.echo("\nWritten:")
    for name, target in written.items():
        typer.echo(f"  {name:<20} {target}")


@app.command()
def version() -> None:
    """Print the engine version."""
    typer.echo(__version__)


if __name__ == "__main__":  # pragma: no cover
    app()
