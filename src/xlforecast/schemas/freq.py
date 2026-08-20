"""Frequency normalisation and validation (FR-112, FR-505).

pandas 2.2 renamed a family of offset aliases and deprecated the old spellings; they are
removed in pandas 3. Both `statsforecast` and `mlforecast` pin `pandas<3`, so we are inside
the deprecation window right now and will break at the 3.0 boundary.

This matters more than it looks, because the NL parser will emit the *legacy* spellings:
"monthly" becomes `M` in every tutorial ever written, and that is what the model has read.
FR-112's decision is therefore to normalise rather than reject — accept `M`, store `ME`.

Before this module, `ForecastRequest.freq` was a bare `str` with a comment, which meant
FR-505's "impossible frequencies are rejected" had no mechanism behind it at all.
"""

from __future__ import annotations

import re

from pandas.tseries.frequencies import to_offset

from xlforecast.errors import InvalidFrequencyError

__all__ = ["LEGACY_ALIASES", "default_season_length", "normalise_freq"]

# pandas <2.2 spelling -> pandas >=2.2 spelling.
LEGACY_ALIASES: dict[str, str] = {
    "M": "ME",
    "SM": "SME",
    "BM": "BME",
    "CBM": "CBME",
    "Q": "QE",
    "BQ": "BQE",
    "Y": "YE",
    "A": "YE",
    "BA": "BYE",
    "BY": "BYE",
    "H": "h",
    "BH": "bh",
    "CBH": "cbh",
    "T": "min",
    "S": "s",
    "L": "ms",
    "U": "us",
    "N": "ns",
}

# Default seasonal period by base alias. Only a starting point: FR-104 infers season_length
# from the data and this is the fallback when inference is inconclusive.
_DEFAULT_SEASON: dict[str, int] = {
    "YE": 1,
    "QE": 4,
    "ME": 12,
    "SME": 24,
    "W": 52,
    "D": 7,
    "B": 5,
    "h": 24,
    "min": 60,
    "s": 60,
}

_ALIAS_RE = re.compile(r"^\s*([+-]?\d*)\s*([A-Za-z]+)((?:-[A-Za-z0-9]+)*)\s*$")


def _split(alias: str) -> tuple[str, str, str]:
    """Split `2W-MON` into ('2', 'W', '-MON')."""
    m = _ALIAS_RE.match(alias)
    if m is None:
        raise InvalidFrequencyError(
            f"'{alias}' is not a recognised frequency.",
            fix="Use a pandas offset alias such as D, W, ME (monthly), QE or YE.",
            column="freq",
        )
    return m.group(1), m.group(2), m.group(3)


def normalise_freq(alias: str) -> str:
    """Return the canonical pandas >=2.2 spelling of `alias`.

    Legacy aliases are mapped, not rejected. Anything pandas itself cannot parse raises
    `InvalidFrequencyError` — which is what gives FR-505 something to enforce.
    """
    if not isinstance(alias, str) or not alias.strip():
        raise InvalidFrequencyError(
            "Frequency is empty.",
            fix="Supply a frequency such as 'W' for weekly or 'ME' for month-end.",
            column="freq",
        )

    count, base, suffix = _split(alias)

    # Case-sensitive lookup first (M -> ME), then the already-modern lowercase spellings
    # (h, min, s, ms) which must not be uppercased into a different meaning: 'm' is not 'M'.
    canonical_base = LEGACY_ALIASES.get(base, base)

    candidate = f"{count}{canonical_base}{suffix}"
    try:
        offset = to_offset(candidate)
    except (ValueError, TypeError) as exc:
        raise InvalidFrequencyError(
            f"'{alias}' is not a valid pandas frequency.",
            fix="Use a pandas offset alias such as D, W, ME (monthly), QE or YE.",
            column="freq",
        ) from exc
    if offset is None:  # pragma: no cover - defensive; to_offset raises rather than returns None
        raise InvalidFrequencyError(
            f"'{alias}' is not a valid pandas frequency.",
            fix="Use a pandas offset alias such as D, W, ME (monthly), QE or YE.",
            column="freq",
        )
    return str(offset.freqstr)


def default_season_length(alias: str) -> int:
    """Fallback seasonal period for a normalised frequency (FR-104).

    A multiple divides the period: `2ME` on a 12-month cycle is 6 steps, not 12. Returns 1
    (non-seasonal) where the cycle does not divide evenly, rather than silently rounding.
    """
    normalised = normalise_freq(alias)
    count_str, base, _ = _split(normalised)
    count = int(count_str) if count_str not in ("", "+", "-") else 1
    base_period = _DEFAULT_SEASON.get(base, 1)
    if count <= 1:
        return base_period
    return base_period // count if base_period % count == 0 else 1
