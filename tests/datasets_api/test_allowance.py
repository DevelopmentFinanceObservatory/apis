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
    assert refused.value.retryable is True
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
