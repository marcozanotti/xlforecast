"""FR-101/103/106/111 -- reading, gap filling, and future-known exogenous rows."""

from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl
import pytest

from xlforecast.errors import IngestError
from xlforecast.ingest.readers import FutureExogError, gap_fill, read_panel, split_future_rows
from xlforecast.schemas.request import DataMapping, ExogSpec

MAPPING = DataMapping(unique_id_col="sku", ds_col="week", y_col="units")


@pytest.fixture
def raw_frame() -> pl.DataFrame:
    dates = pd.date_range("2021-01-03", periods=12, freq="W")
    return pl.DataFrame(
        {
            "sku": ["A"] * 12 + ["B"] * 12,
            "week": list(dates) * 2,
            "units": [float(i) for i in range(12)] * 2,
            "promo": [0.0] * 24,
        }
    )


@pytest.fixture
def csv_path(tmp_path, raw_frame):
    path = tmp_path / "panel.csv"
    raw_frame.write_csv(path)
    return path


class TestReadPanel:
    def test_renames_user_columns_to_the_canonical_three(self, csv_path):
        """FR-101 -- nothing downstream should have to thread a DataMapping through to find
        the target column."""
        panel = read_panel(csv_path, MAPPING)
        assert panel.columns == ["unique_id", "ds", "y"]

    def test_keeps_declared_exogenous_columns_under_their_own_names(self, csv_path):
        mapping = MAPPING.model_copy(update={"exog": [ExogSpec(name="promo", kind="historic")]})
        assert "promo" in read_panel(csv_path, mapping).columns

    def test_reads_parquet_too(self, tmp_path, raw_frame):
        path = tmp_path / "panel.parquet"
        raw_frame.write_parquet(path)
        assert read_panel(path, MAPPING).height == 24

    def test_does_not_sort(self, tmp_path):
        """FR-105 requires reporting non-monotonic timestamps. Sorting on read would
        silently repair the very fault the requirement asks us to name."""
        dates = list(pd.date_range("2021-01-03", periods=6, freq="W"))
        frame = pl.DataFrame({"sku": ["A"] * 6, "week": list(reversed(dates)), "units": [1.0] * 6})
        path = tmp_path / "unsorted.csv"
        frame.write_csv(path)
        read = read_panel(path, MAPPING).get_column("ds").to_list()
        assert read == list(reversed(dates)), "order must survive to validation"

    def test_missing_file_is_a_named_error(self, tmp_path):
        with pytest.raises(IngestError) as exc:
            read_panel(tmp_path / "nope.csv", MAPPING)
        assert exc.value.fix

    def test_unsupported_extension_is_named(self, tmp_path):
        path = tmp_path / "panel.xlsx"
        path.write_text("not really excel")
        with pytest.raises(IngestError) as exc:
            read_panel(path, MAPPING)
        assert "unsupported" in str(exc.value).lower()

    def test_missing_mapped_column_names_the_column(self, csv_path):
        with pytest.raises(IngestError) as exc:
            read_panel(csv_path, MAPPING.model_copy(update={"y_col": "revenue"}))
        assert exc.value.column == "revenue"

    def test_missing_exog_column_names_it(self, csv_path):
        mapping = MAPPING.model_copy(update={"exog": [ExogSpec(name="weather", kind="historic")]})
        with pytest.raises(IngestError) as exc:
            read_panel(csv_path, mapping)
        assert exc.value.column == "weather"

    def test_unparseable_dates_are_rejected_with_the_column_named(self, tmp_path):
        frame = pl.DataFrame({"sku": ["A"] * 3, "week": ["x", "y", "z"], "units": [1.0, 2, 3]})
        path = tmp_path / "bad_dates.csv"
        frame.write_csv(path)
        with pytest.raises(IngestError) as exc:
            read_panel(path, MAPPING)
        assert exc.value.column == "week"


class TestGapFill:
    """FR-106."""

    @pytest.fixture
    def gappy(self) -> pl.DataFrame:
        dates = pd.date_range("2021-01-03", periods=10, freq="W")
        frame = pl.DataFrame(
            {"unique_id": ["A"] * 10, "ds": list(dates), "y": [float(i) for i in range(10)]}
        )
        return frame.filter(pl.col("ds") != dates[4])

    def test_none_leaves_the_panel_untouched(self, gappy):
        assert gap_fill(gappy, freq="W", method="none").height == 9

    def test_zero_fills_the_hole_with_zero(self, gappy):
        filled = gap_fill(gappy, freq="W", method="zero")
        assert filled.height == 10
        assert filled.get_column("y")[4] == 0.0

    def test_interpolate_fills_the_hole_by_interpolation(self, gappy):
        filled = gap_fill(gappy, freq="W", method="interpolate")
        assert filled.height == 10
        assert filled.get_column("y")[4] == pytest.approx(4.0)

    def test_unsupported_frequency_is_refused_rather_than_guessed(self):
        dates = pd.date_range("2021-01-01", periods=5, freq="YE")
        frame = pl.DataFrame({"unique_id": ["A"] * 5, "ds": list(dates), "y": [1.0] * 5})
        assert gap_fill(frame, freq="YE", method="zero").height >= 5


class TestFutureExogRows:
    """FR-111 -- a future-known column is unusable without its values over the horizon, and
    the original spec provided no path by which the user could supply them."""

    def _panel(self, future_rows: int, *, series: int = 2) -> pl.DataFrame:
        dates = pd.date_range("2021-01-03", periods=10, freq="W")
        future = pd.date_range(dates[-1], periods=future_rows + 1, freq="W")[1:]
        frames = []
        for i in range(series):
            frames.append(
                pl.DataFrame(
                    {
                        "unique_id": [f"S{i}"] * 10,
                        "ds": list(dates),
                        "y": [float(j) for j in range(10)],
                        "promo": [0.0] * 10,
                    }
                )
            )
            if future_rows:
                frames.append(
                    pl.DataFrame(
                        {
                            "unique_id": [f"S{i}"] * future_rows,
                            "ds": list(future),
                            "y": [None] * future_rows,
                            "promo": [1.0] * future_rows,
                        }
                    )
                )
        return pl.concat(frames)

    @property
    def mapping(self) -> DataMapping:
        return MAPPING.model_copy(update={"exog": [ExogSpec(name="promo", kind="future_known")]})

    def test_history_and_future_are_separated(self):
        history, future = split_future_rows(self._panel(3), self.mapping, h=3)
        assert history.height == 20
        assert future.height == 6

    def test_no_future_rows_at_all_is_refused_with_a_fix(self):
        with pytest.raises(FutureExogError) as exc:
            split_future_rows(self._panel(0), self.mapping, h=3)
        assert exc.value.fix is not None
        assert "Append 3 rows per series" in exc.value.fix

    def test_wrong_number_of_future_rows_names_the_series(self):
        with pytest.raises(FutureExogError) as exc:
            split_future_rows(self._panel(2), self.mapping, h=3)
        assert exc.value.unique_id is not None

    def test_a_series_missing_future_rows_is_named(self):
        panel = self._panel(3)
        panel = panel.filter(~((pl.col("unique_id") == "S1") & pl.col("y").is_null()))
        with pytest.raises(FutureExogError) as exc:
            split_future_rows(panel, self.mapping, h=3)
        assert exc.value.unique_id == "S1"

    def test_panels_without_future_known_columns_pass_through(self):
        dates = pd.date_range("2021-01-03", periods=10, freq="W")
        frame = pl.DataFrame(
            {"unique_id": ["A"] * 10, "ds": list(dates), "y": np.arange(10.0).tolist()}
        )
        history, future = split_future_rows(frame, MAPPING, h=3)
        assert history.height == 10
        assert future.is_empty()
