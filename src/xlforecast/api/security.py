"""Confirmation tokens and the quota gate (AC-503, FR-803, FR-804).

**The confirmation token is the enforcement, not the audit line.** AC-503 originally read
"no job can be enqueued from natural language without a confirmation event in the audit log",
which a system that enqueues first and logs afterwards satisfies. The rewritten AC asserts
that `POST /v1/jobs` *rejects* a request lacking a valid token, which is what this module
provides.

The token is an HMAC over `(data_id, request_hash, expiry)`. It is single-use and bound to the
exact configuration the user saw, so confirming one config and submitting another fails --
which is the substance of FR-503, not the fact that a button was pressed.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import secrets
import time
from dataclasses import dataclass, field

from xlforecast.errors import XLForecastError
from xlforecast.schemas.request import ForecastRequest

__all__ = ["ConfirmationError", "QuotaError", "TokenService", "request_hash"]

TOKEN_TTL_SECONDS = 1800  # 30 minutes: long enough to read the S3 card, short enough to expire


class ConfirmationError(XLForecastError):
    """The confirmation token is missing, malformed, expired, replayed, or for another config."""


class QuotaError(XLForecastError):
    """FR-803 -- concurrent-job or compute-minute quota exhausted."""


def request_hash(data_id: str, request: ForecastRequest) -> str:
    """Stable digest of exactly what the user confirmed.

    Uses the model's JSON dump rather than `hash()`, because the token must survive a process
    restart and Python's string hashing is salted per process.
    """
    payload = json.dumps(
        {"data_id": data_id, "request": json.loads(request.model_dump_json())},
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode()).hexdigest()


@dataclass
class TokenService:
    """Mints and redeems confirmation tokens.

    `_spent` is in-process, which is right for a single API instance and wrong for several.
    The Redis-backed store replaces it at deploy time; the interface does not change. Noted
    rather than hidden, because a replay check that silently does not work across instances
    would be worse than none.
    """

    secret: bytes
    ttl_seconds: int = TOKEN_TTL_SECONDS
    _spent: set[str] = field(default_factory=set)

    def _sign(self, digest: str, expires_at: int, nonce: str) -> str:
        message = f"{digest}:{expires_at}:{nonce}".encode()
        signature = hmac.new(self.secret, message, hashlib.sha256).hexdigest()
        return f"{digest}.{expires_at}.{nonce}.{signature}"

    def mint(self, data_id: str, request: ForecastRequest) -> str:
        """Called only from an explicit user action in S3 (FR-503).

        The nonce makes each mint individually identifiable. Without it, confirming the same
        configuration twice inside the same second produces a byte-identical token, and the
        single-use check cannot tell the second confirmation from a replay of the first --
        which would break the legitimate re-run FR-703 exists to support.
        """
        return self._sign(
            request_hash(data_id, request),
            int(time.time()) + self.ttl_seconds,
            secrets.token_hex(8),
        )

    def redeem(self, token: str, data_id: str, request: ForecastRequest) -> None:
        """Raise unless `token` confirms exactly this configuration, now, for the first time."""
        try:
            digest, expiry_raw, nonce, signature = token.split(".")
            expires_at = int(expiry_raw)
        except (ValueError, AttributeError) as exc:
            raise ConfirmationError(
                "the confirmation token is malformed.",
                fix="Confirm the configuration again before running.",
            ) from exc

        expected = self._sign(digest, expires_at, nonce)
        # Constant-time, so a forged token cannot be refined byte by byte from timings.
        if not hmac.compare_digest(expected, token):
            raise ConfirmationError(
                "the confirmation token is not valid.",
                fix="Confirm the configuration again before running.",
            )
        if time.time() > expires_at:
            raise ConfirmationError(
                "the confirmation has expired.",
                fix="Review the configuration and confirm it again.",
            )
        if digest != request_hash(data_id, request):
            # The substance of FR-503: the user confirmed a different configuration from the
            # one being submitted.
            raise ConfirmationError(
                "this configuration is not the one that was confirmed.",
                fix="Review the changed settings and confirm again.",
            )
        if signature in self._spent:
            raise ConfirmationError(
                "this confirmation has already been used.",
                fix="Confirm again to run the same job a second time.",
            )
        self._spent.add(signature)


@dataclass(frozen=True, slots=True)
class Quota:
    """FR-803. A compute-minute is wall-clock seconds x worker processes, metered per
    fold-model unit -- the original requirement named neither the unit nor the measuring
    point."""

    max_concurrent_jobs: int = 3
    max_compute_minutes_per_month: float = 600.0

    def check_concurrency(self, active: int) -> None:
        if active >= self.max_concurrent_jobs:
            raise QuotaError(
                f"{active} jobs are already running, and the limit is {self.max_concurrent_jobs}.",
                fix="Wait for a running job to finish, or cancel one.",
            )

    def check_compute(self, used_minutes: float) -> None:
        if used_minutes >= self.max_compute_minutes_per_month:
            raise QuotaError(
                f"{used_minutes:.0f} of {self.max_compute_minutes_per_month:.0f} "
                "compute-minutes used this month.",
                fix="Wait for the quota to reset, or upgrade the plan.",
            )
