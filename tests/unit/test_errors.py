"""FS §4 error-presentation rule: every error names the subject and states the fix."""

from __future__ import annotations

import pytest

from xlforecast.errors import IngestError, SeriesExcludedError, XLForecastError


def test_series_scoped_error_names_the_series():
    e = XLForecastError(
        "has 40 observations, needs 143.", fix="Shorten the horizon.", unique_id="SKU-17"
    )
    assert "SKU-17" in e.render()
    assert "Shorten the horizon." in e.render()


def test_column_scoped_error_names_the_column():
    assert "Column 'week'" in XLForecastError("is not parseable as a date.", column="week").render()


def test_render_is_usable_without_a_fix():
    assert (
        XLForecastError("something specific happened.").render() == "something specific happened."
    )


def test_hierarchy_allows_catching_all_domain_errors():
    with pytest.raises(XLForecastError):
        raise SeriesExcludedError("all zero.", unique_id="S1")
    assert issubclass(SeriesExcludedError, IngestError)
    assert issubclass(IngestError, XLForecastError)
