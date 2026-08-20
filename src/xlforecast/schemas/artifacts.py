"""Artifact pack (TS §8) — the only thing the explanation LLM may read.

On hard rule 4 and observed values: several fields here *are* observations
(`Analogue.same_period_last_year`, `PeakDrop.value`, `CalendarContext.exog_flags`). That is
deliberate. FR-604 and the P1 journey require citing last year's value at the same calendar
position, and no artifact pack can ground an explanation without individual numbers. The rule
prohibits *bulk* panel data — series, arrays, slices, any path handing the model unbounded
`y` — not named, per-point, artifact-mediated scalars. `llm/redact.py` is the only path to a
provider request and caps payloads at `MAX_ARTIFACT_POINTS`.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    "MAX_ARTIFACT_POINTS",
    "Analogue",
    "ArtifactPack",
    "Attribution",
    "CalendarContext",
    "Decomposition",
    "PeakDrop",
    "UncertaintyContext",
]

#: Ceiling on per-point artifact values in a single provider request (hard rule 4, NFR-07).
MAX_ARTIFACT_POINTS = 200


class PeakDrop(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    unique_id: str
    ds: str
    direction: Literal["peak", "drop"]
    value: float
    seasonal_expectation: float
    deviation_pct: float
    z_score: float


class Attribution(BaseModel):
    """Global models only (FR-601a).

    v1 global models are recursive (FR-203, FR-210 deferred), so at `horizon_step > 1` the
    lag features hold the model's *own earlier predictions*, not observations. Narration must
    not describe these as drivers of observed data — that is the soft dishonesty FS §6 exists
    to prevent, and persona P3 will catch it.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    unique_id: str
    ds: str
    horizon_step: int = Field(ge=1)
    recursive_path: bool
    contributions: dict[str, float] = Field(default_factory=dict)
    base_value: float


class Decomposition(BaseModel):
    """AutoETS only — never ARIMA (FR-601b).

    ARIMA has no level/trend/seasonal decomposition in any form, and an STL of the *history*
    is not the model's decomposition; FR-606 forbids presenting it as one. `ets_form`
    determines which components exist: an `ANN` fit has neither trend nor seasonal, and those
    fields are `None` rather than zero.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    unique_id: str
    ds: str
    ets_form: str
    level: float
    trend: float | None = None
    seasonal: float | None = None


class CalendarContext(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    unique_id: str
    ds: str
    seasonal_index: float
    period_rank: int = Field(ge=1)
    period_total: int = Field(ge=1)  #: the "of 12" in "3rd highest of 12 months"
    holidays: list[str] = Field(default_factory=list)
    exog_flags: dict[str, float] = Field(default_factory=dict)


class Analogue(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    unique_id: str
    ds: str
    same_period_last_year: float | None = None
    seasonal_naive_value: float
    historical_pctile: float = Field(ge=0.0, le=1.0)


class UncertaintyContext(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    unique_id: str
    ds: str
    level: int = Field(ge=1, le=99)
    interval_width: float
    width_vs_series_mean: float
    #: Out-of-calibration coverage (FR-303). The in-calibration figure is ~nominal by
    #: construction and must never be surfaced to a user as evidence of calibration.
    empirical_coverage: float = Field(ge=0.0, le=1.0)


class ArtifactPack(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    job_id: str
    peaks_drops: list[PeakDrop] = Field(default_factory=list)
    attributions: list[Attribution] = Field(default_factory=list)
    decompositions: list[Decomposition] = Field(default_factory=list)
    calendar: list[CalendarContext] = Field(default_factory=list)
    analogues: list[Analogue] = Field(default_factory=list)
    uncertainty: list[UncertaintyContext] = Field(default_factory=list)
    leaderboard_summary: dict[str, float | int | str | None] = Field(default_factory=dict)
    series_flags: dict[str, dict[str, float | int | str | bool | None]] = Field(
        default_factory=dict
    )

    def point_count(self) -> int:
        """Number of per-point artifact values, for the hard-rule-4 cap."""
        return (
            len(self.peaks_drops)
            + len(self.attributions)
            + len(self.decompositions)
            + len(self.calendar)
            + len(self.analogues)
            + len(self.uncertainty)
        )
