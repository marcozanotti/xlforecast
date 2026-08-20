"""Profile contracts (TS §4.3) — the LLM's only input, and the trust boundary.

`DataProfile` was referenced by FR-502, TS §9.2, the repo layout and Phase 1, and defined
nowhere. It is the object that makes the enterprise story true: derived statistics cross to
the provider, observations do not.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from xlforecast.schemas.enums import ExclusionReason, IntermittencyClass
from xlforecast.schemas.request import ExogSpec

__all__ = ["DataProfile", "SeriesProfile", "ValidationReport"]


class SeriesProfile(BaseModel):
    """Per-series derived statistics. Contains no observations."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    unique_id: str
    n_obs: int = Field(ge=0)
    start: str
    end: str
    n_missing: int = Field(ge=0)
    pct_missing: float = Field(ge=0.0, le=1.0)
    zero_share: float = Field(ge=0.0, le=1.0)
    #: Computed on the PRE-gap-fill series (FR-106): `gap_fill="zero"` manufactures
    #: intermittency and would otherwise route smooth-but-gappy series to Croston.
    intermittency: IntermittencyClass
    adi: float | None = None
    cv2: float | None = None
    seasonality_strength: float | None = None
    trend_strength: float | None = None
    short_history: bool = False


class ValidationReport(BaseModel):
    """FR-105 outcome. A series is never dropped without a reason here (FS §6)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    n_series_in: int = Field(ge=0)
    n_series_out: int = Field(ge=0)
    excluded: dict[str, ExclusionReason] = Field(default_factory=dict)
    excluded_detail: dict[str, str] = Field(default_factory=dict)

    @property
    def n_excluded(self) -> int:
        return len(self.excluded)


class DataProfile(BaseModel):
    """Panel-level derived profile (FR-502).

    Everything here is a statistic. Nothing here is an observation. `llm/redact.py` may pass
    this object to a provider; it may not pass a panel.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    data_id: str
    n_series: int = Field(ge=0)
    n_rows: int = Field(ge=0)
    freq_inferred: str
    freq_confidence: float = Field(ge=0.0, le=1.0)
    ds_min: str
    ds_max: str
    #: Do series end on different dates? Drives the FR-206a fold decision, and is the single
    #: most consequential fact about a panel for leakage purposes.
    ragged: bool
    season_length_candidates: list[int] = Field(default_factory=list)
    intermittent_share: float = Field(ge=0.0, le=1.0)
    pct_missing_overall: float = Field(ge=0.0, le=1.0)
    exog_available: list[ExogSpec] = Field(default_factory=list)
    series: list[SeriesProfile] = Field(default_factory=list)
    validation: ValidationReport
