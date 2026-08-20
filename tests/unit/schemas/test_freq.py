"""FR-112 — frequency normalisation. Without this, FR-505's 'impossible frequencies are
rejected' had no mechanism behind it."""

from __future__ import annotations

import pytest

from xlforecast.errors import InvalidFrequencyError
from xlforecast.schemas.freq import default_season_length, normalise_freq


@pytest.mark.parametrize(
    ("legacy", "modern"),
    [
        ("M", "ME"),
        ("Q", "QE-DEC"),
        ("Y", "YE-DEC"),
        ("A", "YE-DEC"),
        ("H", "h"),
        ("T", "min"),
        ("S", "s"),
        ("BM", "BME"),
        ("2M", "2ME"),
        ("15T", "15min"),
    ],
)
def test_legacy_aliases_are_normalised_not_rejected(legacy, modern):
    """The NL parser emits legacy spellings because that is what its training data contains.
    FR-112's decision is to normalise, not reject."""
    assert normalise_freq(legacy) == modern


@pytest.mark.parametrize("alias", ["D", "W", "ME", "QE", "W-MON", "B", "h", "min"])
def test_modern_aliases_round_trip_idempotently(alias):
    once = normalise_freq(alias)
    assert normalise_freq(once) == once, "normalisation must be idempotent for the manifest"


@pytest.mark.parametrize("bad", ["", "   ", "ZZZ", "banana", "13", "W-NOTADAY"])
def test_invalid_frequencies_are_rejected_with_a_fix(bad):
    with pytest.raises(InvalidFrequencyError) as exc:
        normalise_freq(bad)
    assert exc.value.fix, "FS §4 error-presentation rule: every error states the remedy"


def test_lowercase_m_is_minutes_not_months():
    """'m' and 'M' are different offsets. Uppercasing before lookup would silently turn a
    monthly panel into a minutely one."""
    assert normalise_freq("min") == "min"
    assert normalise_freq("M") == "ME"


@pytest.mark.parametrize(
    ("alias", "season"), [("W", 52), ("D", 7), ("ME", 12), ("QE", 4), ("YE", 1), ("h", 24)]
)
def test_default_season_lengths(alias, season):
    assert default_season_length(alias) == season


def test_multiples_divide_the_seasonal_period():
    """2ME on a 12-month cycle is 6 steps. Returns 1 rather than silently rounding when the
    cycle does not divide evenly."""
    assert default_season_length("2ME") == 6
    assert default_season_length("5ME") == 1
