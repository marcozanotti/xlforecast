"""Retention sweep (NFR-08).

"Panel data at rest is encrypted and deleted after a configurable retention window (default
30 days)." Two things the requirement leaves out and this module settles:

* **What is deleted.** The uploaded panel and the per-fold checkpoints — the raw observations
  and everything derived directly from them. Results and manifests are **kept**, because the
  build plan's v2 forecast-stability feature compares a forecast against a previous cycle's,
  and deleting the manifests would make that impossible without a migration. Keeping a
  leaderboard is also not keeping customer data: it holds error metrics, not observations.
* **When.** A sweep, not a per-request check. Retention that only runs when someone happens
  to ask is not retention.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from xlforecast.storage.jobs import JobStore
from xlforecast.storage.objects import ObjectStore

__all__ = ["DEFAULT_RETENTION_DAYS", "RetentionPolicy", "SweepReport"]

DEFAULT_RETENTION_DAYS = 30


@dataclass(frozen=True, slots=True)
class SweepReport:
    panels_deleted: int = 0
    checkpoints_deleted: int = 0
    jobs_considered: int = 0
    #: Jobs skipped because they are still running. Deleting the panel out from under a live
    #: job would fail it in a way the user could not act on.
    jobs_skipped_active: int = 0


@dataclass(slots=True)
class RetentionPolicy:
    objects: ObjectStore
    jobs: JobStore
    retention_days: int = DEFAULT_RETENTION_DAYS

    def _expired(self, timestamp: str | None, now: datetime) -> bool:
        if not timestamp:
            return False
        try:
            created = datetime.fromisoformat(timestamp)
        except ValueError:  # pragma: no cover - defensive
            return False
        if created.tzinfo is None:
            created = created.replace(tzinfo=UTC)
        return now - created > timedelta(days=self.retention_days)

    def sweep(self, owner: str, *, now: datetime | None = None) -> SweepReport:
        """Delete expired panels and checkpoints for one owner.

        Returns what was removed rather than logging it, so the caller can record an audit
        line (FR-805) with real numbers instead of an assertion that something happened.
        """
        moment = now or datetime.now(UTC)
        panels = 0
        checkpoints = 0
        considered = 0
        skipped = 0

        for record in self.jobs.list_for_owner(owner):
            considered += 1
            if not record.status.terminal:
                # A live job still needs its panel; retention must not fail a running job.
                skipped += 1
                continue
            if not self._expired(record.finished_at or record.created_at, moment):
                continue

            panel_key = f"data/{record.data_id}.parquet"
            if self.objects.exists(panel_key):
                self.objects.delete(panel_key)
                panels += 1
            for key in self.objects.list_prefix(f"jobs/{record.job_id}/folds/"):
                self.objects.delete(key)
                checkpoints += 1

        return SweepReport(
            panels_deleted=panels,
            checkpoints_deleted=checkpoints,
            jobs_considered=considered,
            jobs_skipped_active=skipped,
        )

    def retains(self, key: str) -> bool:
        """Whether `key` survives a sweep. Results and manifests do; raw data does not."""
        return not (key.startswith("data/") or "/folds/" in key)
