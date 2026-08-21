"""AC-503 / FR-803 -- the confirmation gate and the quota gate."""

from __future__ import annotations

import time

import pytest

from xlforecast.api.security import (
    ConfirmationError,
    Quota,
    QuotaError,
    TokenService,
    request_hash,
)
from xlforecast.schemas.request import ForecastRequest

DATA_ID = "data-1"
REQUEST = ForecastRequest(h=13, freq="W", n_windows=3)


@pytest.fixture
def tokens():
    return TokenService(secret=b"test-secret")


class TestConfirmationGate:
    """AC-503, rewritten: the gate is the enforcement, not the audit line. A system that
    enqueues first and logs a confirmation afterwards satisfied the original wording."""

    def test_a_minted_token_is_accepted_once(self, tokens):
        tokens.redeem(tokens.mint(DATA_ID, REQUEST), DATA_ID, REQUEST)

    def test_a_missing_token_is_rejected(self, tokens):
        with pytest.raises(ConfirmationError):
            tokens.redeem("", DATA_ID, REQUEST)

    def test_a_forged_token_is_rejected(self, tokens):
        digest = request_hash(DATA_ID, REQUEST)
        forged = f"{digest}.{int(time.time()) + 999}.abcdef0123456789.{'0' * 64}"
        with pytest.raises(ConfirmationError):
            tokens.redeem(forged, DATA_ID, REQUEST)

    def test_a_token_from_another_secret_is_rejected(self, tokens):
        other = TokenService(secret=b"different-secret")
        with pytest.raises(ConfirmationError):
            tokens.redeem(other.mint(DATA_ID, REQUEST), DATA_ID, REQUEST)

    def test_an_expired_token_is_rejected(self):
        expired = TokenService(secret=b"s", ttl_seconds=-1)
        with pytest.raises(ConfirmationError, match="expired"):
            expired.redeem(expired.mint(DATA_ID, REQUEST), DATA_ID, REQUEST)

    def test_a_token_cannot_be_replayed(self, tokens):
        """Otherwise one confirmation could enqueue the same job indefinitely."""
        token = tokens.mint(DATA_ID, REQUEST)
        tokens.redeem(token, DATA_ID, REQUEST)
        with pytest.raises(ConfirmationError, match="already been used"):
            tokens.redeem(token, DATA_ID, REQUEST)

    def test_a_token_does_not_confirm_a_different_configuration(self, tokens):
        """The substance of FR-503. Confirming h=13 and submitting h=52 is not a
        confirmation, however genuine the button press was."""
        token = tokens.mint(DATA_ID, REQUEST)
        altered = REQUEST.model_copy(update={"h": 52})
        with pytest.raises(ConfirmationError, match="not the one that was confirmed"):
            tokens.redeem(token, DATA_ID, altered)

    def test_a_token_does_not_confirm_a_different_dataset(self, tokens):
        token = tokens.mint(DATA_ID, REQUEST)
        with pytest.raises(ConfirmationError):
            tokens.redeem(token, "another-dataset", REQUEST)

    @pytest.mark.parametrize("bad", ["nonsense", "a.b", "a.b.c", "a.b.c.d.e", "a.notanint.c.d"])
    def test_malformed_tokens_are_rejected_without_crashing(self, tokens, bad):
        with pytest.raises(ConfirmationError):
            tokens.redeem(bad, DATA_ID, REQUEST)

    def test_every_rejection_states_a_remedy(self, tokens):
        """FS §4 error-presentation rule."""
        with pytest.raises(ConfirmationError) as exc:
            tokens.redeem("nonsense", DATA_ID, REQUEST)
        assert exc.value.fix


class TestRequestHash:
    def test_is_stable_across_processes(self):
        """The digest must survive a restart, so it cannot use Python's salted hash()."""
        assert request_hash(DATA_ID, REQUEST) == request_hash(DATA_ID, REQUEST)

    def test_changes_with_any_field(self):
        base = request_hash(DATA_ID, REQUEST)
        assert request_hash(DATA_ID, REQUEST.model_copy(update={"h": 14})) != base
        assert request_hash(DATA_ID, REQUEST.model_copy(update={"seed": 1})) != base

    def test_two_confirmations_of_the_same_config_are_distinct_tokens(self):
        """FR-703 supports re-running a job. Without a nonce, confirming the same
        configuration twice inside one second yields a byte-identical token, and the
        single-use check cannot distinguish the second confirmation from a replay."""
        service = TokenService(secret=b"s")
        first, second = service.mint(DATA_ID, REQUEST), service.mint(DATA_ID, REQUEST)
        assert first != second
        service.redeem(first, DATA_ID, REQUEST)
        service.redeem(second, DATA_ID, REQUEST)

    def test_is_insensitive_to_model_list_order(self):
        """FR-204 normalises `models` to a sorted, deduped list, so two requests that mean
        the same thing hash the same and one confirmation covers both."""
        a = ForecastRequest(h=4, freq="ME", models=["AutoETS", "AutoARIMA"])
        b = ForecastRequest(h=4, freq="ME", models=["AutoARIMA", "AutoETS"])
        assert request_hash(DATA_ID, a) == request_hash(DATA_ID, b)


class TestQuota:
    """FR-803 -- the original requirement named neither the unit nor the measuring point."""

    def test_concurrency_is_capped(self):
        quota = Quota(max_concurrent_jobs=2)
        quota.check_concurrency(1)
        with pytest.raises(QuotaError, match="already running"):
            quota.check_concurrency(2)

    def test_compute_minutes_are_capped(self):
        quota = Quota(max_compute_minutes_per_month=100.0)
        quota.check_compute(99.0)
        with pytest.raises(QuotaError, match="compute-minutes"):
            quota.check_compute(100.0)

    def test_quota_errors_state_a_remedy(self):
        with pytest.raises(QuotaError) as exc:
            Quota(max_concurrent_jobs=0).check_concurrency(0)
        assert exc.value.fix
