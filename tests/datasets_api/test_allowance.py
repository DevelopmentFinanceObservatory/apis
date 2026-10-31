"""`LiveAllowanceLedger.spend` counts each caller's writes per kind and UTC day, and refuses past
the limit with a wait to the next UTC midnight. `refund` and `spending` give a unit back."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta, timezone

import pytest

from datasets_api.allowance import (
    BUDGET_KEY,
    AllowanceExhausted,
    AllowanceKind,
    LiveAllowanceLedger,
    Spend,
    spending,
)
from dfo_platform.caller import Caller

from ._version_record_store import FakeVersionRecordDatasetStore

_NOON = datetime(2026, 6, 1, 12, tzinfo=UTC)
_ALICE = Caller(kind="address", key="ip:" + "a" * 32)
_BOB = Caller(kind="address", key="ip:" + "b" * 32)


@pytest.fixture
def ledger():
    store = FakeVersionRecordDatasetStore()
    try:
        yield LiveAllowanceLedger(
            store=store, limits={AllowanceKind.pin: 2, AllowanceKind.share: 1}
        )
    finally:
        store.close()


def test_a_caller_spends_up_to_its_limit_then_waits_for_utc_midnight(ledger):
    ledger.spend(_ALICE, AllowanceKind.pin, _NOON)
    ledger.spend(_ALICE, AllowanceKind.pin, _NOON)

    with pytest.raises(AllowanceExhausted) as refused:
        ledger.spend(_ALICE, AllowanceKind.pin, _NOON)

    assert refused.value.code == "allowance_exhausted"
    assert refused.value.retryable is False
    assert refused.value.headers == {"Retry-After": str(12 * 3600)}


def test_callers_kinds_and_days_count_apart(ledger):
    ledger.spend(_ALICE, AllowanceKind.share, _NOON)

    ledger.spend(_BOB, AllowanceKind.share, _NOON)
    ledger.spend(_ALICE, AllowanceKind.pin, _NOON)
    ledger.spend(_ALICE, AllowanceKind.share, _NOON + timedelta(hours=12))
    with pytest.raises(AllowanceExhausted):
        ledger.spend(_ALICE, AllowanceKind.share, _NOON + timedelta(hours=11, minutes=59))


def test_the_day_is_the_utc_day_whatever_the_clocks_zone(ledger):
    """23:30 at UTC-2 on 31 May is 01:30 UTC on 1 June, the day `_NOON` falls on."""
    late = datetime(2026, 5, 31, 23, 30, tzinfo=timezone(timedelta(hours=-2)))
    ledger.spend(_ALICE, AllowanceKind.share, late)

    with pytest.raises(AllowanceExhausted) as refused:
        ledger.spend(_ALICE, AllowanceKind.share, _NOON)

    assert refused.value.headers == {"Retry-After": str(12 * 3600)}


def test_a_zero_limit_refuses_the_first_spend():
    ledger = LiveAllowanceLedger(
        store=FakeVersionRecordDatasetStore(), limits={AllowanceKind.pin: 0}
    )

    with pytest.raises(AllowanceExhausted):
        ledger.spend(_ALICE, AllowanceKind.pin, _NOON)


_EXPORT = "exp_" + "a" * 32
_LEASE = _NOON + timedelta(minutes=2)
_UNITS = [Spend(BUDGET_KEY, AllowanceKind.pin_budget), Spend(_ALICE.key, AllowanceKind.pin)]


@pytest.fixture
def claims():
    store = FakeVersionRecordDatasetStore()
    try:
        yield LiveAllowanceLedger(
            store=store,
            limits={AllowanceKind.pin: 1, AllowanceKind.pin_budget: 5},
        )
    finally:
        store.close()


def _counts(ledger: LiveAllowanceLedger) -> dict[tuple[str, str], int]:
    with ledger._store.transaction() as tx:
        rows = tx.fetch_all("SELECT caller_key, kind, count FROM datasets.caller_allowance")
    return {(row["caller_key"], row["kind"]): row["count"] for row in rows}


def test_a_claim_runs_alone_until_its_lease_passes_and_then_passes_on_its_units(claims):
    claim = claims.claim(_EXPORT, now=_NOON, expires_at=_LEASE)
    held = claims.hold_claim(claim, _UNITS, _NOON)
    assert claims.name_claim_digest(held, "sha256:a") is True

    assert claims.claim(_EXPORT, now=_NOON, expires_at=_LEASE) is None
    taken = claims.claim(_EXPORT, now=_LEASE, expires_at=_LEASE + timedelta(minutes=2))

    assert (taken.spends, taken.spent_at, taken.digest) == (tuple(_UNITS), _NOON, "sha256:a")


def test_a_write_under_a_lost_lease_changes_nothing(claims):
    held = claims.hold_claim(claims.claim(_EXPORT, now=_NOON, expires_at=_LEASE), _UNITS, _NOON)
    claims.claim(_EXPORT, now=_LEASE, expires_at=_LEASE + timedelta(minutes=2))
    spent = _counts(claims)

    claims.close_claim(held, refund=True)
    claims.free_claim(held, _NOON)
    claims.refuse_claim(held, {"code": "over_ceiling"})

    assert claims.name_claim_digest(held, "sha256:a") is False
    assert claims.hold_claim(held, _UNITS[:1], _NOON) is None
    assert _counts(claims) == spent
    assert claims.read_claim(_EXPORT).refusal is None


def test_closing_a_claim_refunds_once_whoever_else_closes_it(claims):
    held = claims.hold_claim(claims.claim(_EXPORT, now=_NOON, expires_at=_LEASE), _UNITS, _NOON)

    claims.close_claim(held, refund=True)
    claims.spend(_ALICE, AllowanceKind.pin, _NOON)
    claims.close_claim(held, refund=True)

    assert _counts(claims) == {(_ALICE.key, "pin"): 1}


def test_a_held_unit_is_told_from_a_spent_one_until_its_lease_ends(claims):
    claims.hold_claim(claims.claim(_EXPORT, now=_NOON, expires_at=_LEASE), _UNITS, _NOON)

    assert claims.holds_unit(_UNITS[1], _NOON) is True
    assert claims.holds_unit(Spend(_BOB.key, AllowanceKind.pin), _NOON) is False
    assert claims.holds_unit(_UNITS[1], _LEASE) is False


def test_spends_together_take_every_unit_or_none():
    """A pin spends the shared budget and the caller's allowance together, so a caller past its
    allowance takes nothing from the budget every caller shares."""
    store = FakeVersionRecordDatasetStore()
    ledger = LiveAllowanceLedger(
        store=store, limits={AllowanceKind.open_pin_budget: 2, AllowanceKind.pin: 1}
    )
    budget = Spend(BUDGET_KEY, AllowanceKind.open_pin_budget)

    ledger.spend_together([budget, Spend(_ALICE.key, AllowanceKind.pin)], _NOON)
    with pytest.raises(AllowanceExhausted) as refused:
        ledger.spend_together([budget, Spend(_ALICE.key, AllowanceKind.pin)], _NOON)
    ledger.spend_together([budget, Spend(_BOB.key, AllowanceKind.pin)], _NOON)

    assert refused.value.kind is AllowanceKind.pin
    with pytest.raises(AllowanceExhausted) as spent_out:
        ledger.spend_together([budget], _NOON)
    assert spent_out.value.kind is AllowanceKind.open_pin_budget


def test_a_refund_gives_one_unit_back_and_a_spent_out_day_spends_again(ledger):
    ledger.spend(_ALICE, AllowanceKind.pin, _NOON)
    ledger.spend(_ALICE, AllowanceKind.pin, _NOON)

    ledger.refund(_ALICE, AllowanceKind.pin, _NOON)
    ledger.spend(_ALICE, AllowanceKind.pin, _NOON)
    ledger.refund(_ALICE, AllowanceKind.pin, _NOON)
    ledger.refund(_ALICE, AllowanceKind.pin, _NOON)
    ledger.refund(_ALICE, AllowanceKind.pin, _NOON)

    ledger.spend(_ALICE, AllowanceKind.pin, _NOON)
    ledger.spend(_ALICE, AllowanceKind.pin, _NOON)
    with pytest.raises(AllowanceExhausted):
        ledger.spend(_ALICE, AllowanceKind.pin, _NOON)


def test_spending_refunds_the_unit_when_the_write_fails(ledger):
    with pytest.raises(RuntimeError), spending(ledger, _ALICE, AllowanceKind.share, _NOON):
        raise RuntimeError("the write failed")
    with spending(ledger, _ALICE, AllowanceKind.share, _NOON):
        pass

    with pytest.raises(AllowanceExhausted):
        ledger.spend(_ALICE, AllowanceKind.share, _NOON)
