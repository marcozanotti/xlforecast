"""Domain errors.

Every error names the series or column at fault and states the fix. That is a product
requirement, not a nicety: FS §4's error-presentation rule forbids "an error occurred" and
forbids stack traces reaching the user. `fix` is therefore mandatory on anything user-facing.

No bare `except` anywhere in the codebase (CLAUDE.md conventions).
"""

from __future__ import annotations

__all__ = [
    "ConformalCalibrationError",
    "EnsembleConfigError",
    "IngestError",
    "InvalidFrequencyError",
    "InvalidModelNameError",
    "OutputTooLargeError",
    "ProhibitedModelError",
    "SeriesExcludedError",
    "XLForecastError",
]


class XLForecastError(Exception):
    """Base for every domain error.

    Args:
        message: what went wrong, in plain language.
        fix: what the user should do about it. Required for user-facing errors.
        unique_id: the offending series, when the fault is series-scoped.
        column: the offending column, when the fault is column-scoped.
    """

    def __init__(
        self,
        message: str,
        *,
        fix: str | None = None,
        unique_id: str | None = None,
        column: str | None = None,
    ) -> None:
        self.message = message
        self.fix = fix
        self.unique_id = unique_id
        self.column = column
        super().__init__(self.render())

    def render(self) -> str:
        """Render for the UI: subject, fault, remedy — never a traceback."""
        subject = ""
        if self.unique_id is not None:
            subject = f"Series '{self.unique_id}': "
        elif self.column is not None:
            subject = f"Column '{self.column}': "
        tail = f" Fix: {self.fix}" if self.fix else ""
        return f"{subject}{self.message}{tail}"


class IngestError(XLForecastError):
    """Panel could not be read or mapped (FR-101..FR-112)."""


class SeriesExcludedError(IngestError):
    """A single series failed validation and is excluded (FR-105).

    Never raised to abort a run — collected into `ValidationReport.excluded` so the user is
    told which series went and why. Silently dropping a series is a listed failure mode (FS §6).
    """


class InvalidFrequencyError(IngestError):
    """`freq` is not a pandas offset alias (FR-112, FR-505)."""


class InvalidModelNameError(XLForecastError):
    """Model is not in the registry (FR-505)."""


class ProhibitedModelError(XLForecastError):
    """Model's licence forbids commercial use (`commercial_ok=False`, TS §5.2)."""


class EnsembleConfigError(XLForecastError):
    """Ensemble configuration cannot be satisfied (FR-403a, FR-405a)."""


class ConformalCalibrationError(XLForecastError):
    """Calibration cannot proceed and no fallback remains (TS §5.4)."""


class OutputTooLargeError(XLForecastError):
    """Projected output exceeds Excel's row limit (FR-708, NFR-12)."""
