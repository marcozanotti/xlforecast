"""The canonical panel: column names and the fingerprint definition (TS §4.7).

After mapping (FR-101) every panel has the same three columns regardless of what the user
called them. Fixing that here means nothing downstream has to thread a `DataMapping` through
to know which column holds the target.
"""

from __future__ import annotations

import hashlib
import io
from typing import Final

import pandas as pd
import polars as pl

__all__ = ["DS", "ID", "Y", "canonical_sort", "exog_columns", "fingerprint", "span"]

ID: Final = "unique_id"
DS: Final = "ds"
Y: Final = "y"

_CORE: Final = (ID, DS, Y)


def span(panel: pl.DataFrame) -> tuple[pd.Timestamp, pd.Timestamp]:
    """First and last timestamp, as pandas Timestamps.

    `Series.min()` is typed as a broad union in polars because it is generic over dtype;
    narrowing it once here keeps every caller from restating the same cast.
    """
    lo, hi = panel.select(pl.col(DS).min().alias("lo"), pl.col(DS).max().alias("hi")).row(0)
    return pd.Timestamp(lo), pd.Timestamp(hi)


def exog_columns(panel: pl.DataFrame) -> list[str]:
    """Non-core columns, in sorted order for determinism."""
    return sorted(c for c in panel.columns if c not in _CORE)


def canonical_sort(panel: pl.DataFrame) -> pl.DataFrame:
    """Canonical column and row order (TS §4.7).

    Row order is `(unique_id, ds)` and column order is `unique_id, ds, y, *sorted(exog)`.
    Both are load-bearing for NFR-02: a leaderboard is only byte-identical across runs if
    the panel it was computed from is byte-identical, and dataframe iteration order is not
    guaranteed to be stable across reads otherwise.
    """
    return panel.select([ID, DS, Y, *exog_columns(panel)]).sort([ID, DS])


def fingerprint(panel: pl.DataFrame) -> str:
    """sha256 of the canonical panel (TS §4.7).

    Taken after ingestion and gap filling but **before** exclusion, so that a change in
    `season_length` inference -- which moves the FR-105 threshold and therefore which series
    survive -- shows up as a leaderboard difference rather than being hidden inside the
    fingerprint.

    Serialised as Parquet with a fixed dtype layout so the digest depends on the data rather
    than on how the file happened to be read.
    """
    frame = canonical_sort(panel).with_columns(
        pl.col(ID).cast(pl.Utf8),
        pl.col(DS).cast(pl.Datetime("us")),
        pl.col(Y).cast(pl.Float64),
    )
    buf = io.BytesIO()
    frame.write_parquet(buf, compression="zstd", compression_level=3, statistics=False)
    return hashlib.sha256(buf.getvalue()).hexdigest()
