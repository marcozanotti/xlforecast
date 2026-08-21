"""CSV / Parquet -> canonical panel (FR-101, FR-103, FR-106, FR-111)."""

from __future__ import annotations

from pathlib import Path

import polars as pl

from xlforecast.errors import IngestError
from xlforecast.panel import DS, ID, Y, canonical_sort
from xlforecast.schemas.enums import GapFill
from xlforecast.schemas.request import DataMapping

__all__ = ["FutureExogError", "gap_fill", "read_panel", "split_future_rows"]


class FutureExogError(IngestError):
    """Future-known exogenous rows are missing or malformed (FR-111)."""


def _read_raw(path: Path) -> pl.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        return pl.read_parquet(path)
    if suffix in {".csv", ".txt"}:
        return pl.read_csv(path, try_parse_dates=True)
    raise IngestError(
        f"cannot read '{path.name}': unsupported file type '{suffix}'.",
        fix="Supply a .csv or .parquet file.",
    )


def read_panel(path: str | Path, mapping: DataMapping) -> pl.DataFrame:
    """Read a file and rename the user's columns to the canonical three (FR-101).

    Exogenous columns keep their own names; only the core three are renamed, so nothing
    downstream has to thread a `DataMapping` through to find the target.
    """
    path = Path(path)
    if not path.exists():
        raise IngestError(f"file not found: {path}", fix="Check the path and try again.")

    raw = _read_raw(path)
    wanted = {mapping.unique_id_col: ID, mapping.ds_col: DS, mapping.y_col: Y}
    missing = [c for c in wanted if c not in raw.columns]
    if missing:
        raise IngestError(
            f"column(s) {missing} not found in {path.name}; it has {raw.columns}.",
            fix="Correct the column mapping.",
            column=missing[0],
        )

    exog = [e.name for e in mapping.exog]
    absent = [c for c in exog if c not in raw.columns]
    if absent:
        raise IngestError(
            f"exogenous column(s) {absent} not found in {path.name}.",
            fix="Remove them from the mapping or add them to the file.",
            column=absent[0],
        )

    frame = raw.select([*wanted, *exog]).rename(wanted)
    frame = frame.with_columns(
        pl.col(ID).cast(pl.Utf8),
        pl.col(DS).cast(pl.Datetime("us"), strict=False),
        pl.col(Y).cast(pl.Float64, strict=False),
    )
    if frame.get_column(DS).is_null().all():
        raise IngestError(
            f"no value in '{mapping.ds_col}' could be parsed as a date.",
            fix="Check the date format, or pre-parse the column.",
            column=mapping.ds_col,
        )
    # Deliberately NOT sorted. FR-105 requires reporting non-monotonic timestamps, and
    # sorting here would silently repair the very fault the requirement asks us to name.
    # `validate` checks ordering; `canonical_sort` is applied afterwards.
    return frame.select([ID, DS, Y, *sorted(exog)])


def split_future_rows(
    panel: pl.DataFrame, mapping: DataMapping, *, h: int
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Separate future-known exogenous rows from history (FR-111).

    A future-known column is unusable without its values over the horizon, and there was no
    path by which the user could supply them. The convention here follows Nixtla's `X_df`:
    extra rows carrying `unique_id`, a `ds` beyond the last observation, a null `y`, and the
    exogenous values populated.

    Returns `(history, future)`. `future` is empty when no column is future-known.
    """
    if not mapping.future_known:
        return panel, panel.clear()

    future = panel.filter(pl.col(Y).is_null())
    history = panel.filter(pl.col(Y).is_not_null())
    if future.is_empty():
        raise FutureExogError(
            f"columns {mapping.future_known} are marked future-known but the file carries no "
            "future rows for them.",
            fix=f"Append {h} rows per series with the future dates, a blank target, and the "
            "exogenous values filled in.",
        )

    counts = future.group_by(ID).len().rename({"len": "n"})
    wrong = counts.filter(pl.col("n") != h)
    if not wrong.is_empty():
        uid, n = wrong.row(0)
        raise FutureExogError(
            f"expected exactly {h} future rows per series, found {n}.",
            fix=f"Supply {h} future rows for every series with a future-known column.",
            unique_id=uid,
        )

    missing_series = set(history.get_column(ID).unique()) - set(future.get_column(ID).unique())
    if missing_series:
        raise FutureExogError(
            f"{len(missing_series)} series have no future rows.",
            fix=f"Supply {h} future rows for every series.",
            unique_id=sorted(missing_series)[0],
        )
    return history, future


def gap_fill(panel: pl.DataFrame, *, freq: str, method: GapFill) -> pl.DataFrame:
    """Fill calendar gaps per series (FR-106).

    Runs *after* intermittency classification, never before: `zero` manufactures
    intermittency, and classifying afterwards would route smooth-but-gappy series to Croston.
    The pipeline order in `run.py` enforces this; the docstring is here because the coupling
    is easy to miss when reading either function alone.
    """
    if method == "none":
        return panel

    filled = (
        panel.sort([ID, DS])
        .upsample(time_column=DS, every=_polars_every(freq), group_by=ID, maintain_order=True)
        .with_columns(pl.col(ID).forward_fill())
    )
    if method == "zero":
        filled = filled.with_columns(pl.col(Y).fill_null(0.0))
    else:
        filled = filled.with_columns(pl.col(Y).interpolate().over(ID))
    return canonical_sort(filled)


def _polars_every(freq: str) -> str:
    """Translate a pandas offset alias into a polars duration string."""
    import re

    import pandas as pd

    offset = pd.tseries.frequencies.to_offset(freq)
    n = getattr(offset, "n", 1)
    base = re.sub(r"^\d+", "", offset.freqstr).split("-")[0]
    table = {
        "W": "w",
        "D": "d",
        "B": "d",
        "ME": "mo",
        "MS": "mo",
        "QE": "3mo",
        "QS": "3mo",
        "YE": "1y",
        "YS": "1y",
        "h": "h",
        "min": "m",
        "s": "s",
    }
    if base not in table:
        raise IngestError(
            f"gap filling is not supported at frequency '{freq}'.",
            fix="Use gap_fill='none', or resample the panel before upload.",
            column="freq",
        )
    unit = table[base]
    leading = re.match(r"^(\d+)", unit)
    mult = int(leading.group(1)) if leading else 1
    suffix = re.sub(r"^\d+", "", unit)
    return f"{n * mult}{suffix}"
