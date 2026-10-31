"""`LiveAllowanceLedger.spend` counts each caller's writes per kind and UTC day, and refuses past
the limit with a wait to the next UTC midnight. `refund` and `spending` give a unit back."""

from __future__ import annotations

from contextlib import contextmanager
from datetime import UTC, datetime, timedelta, timezone

import pytest

from artifact_store import ArtifactNotFoundError, Tier
from artifact_store.errors import ArtifactWriteError
from datasets_api.allowance import (
    BUDGET_KEY,
    AllowanceExhausted,
    AllowanceKind,
    ClaimState,
    ExportClaim,
    LiveAllowanceLedger,
    Spend,
    spending,
)
from datasets_api.retention_admission import ExportClaimUnsettled, PreparingCopy
from dfo_platform.caller import Caller

from ._artifact_doubles import FakeArtifactStore
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


def _held(ledger: LiveAllowanceLedger) -> ExportClaim:
    return ledger.hold_claim(ledger.claim(_EXPORT, now=_NOON, expires_at=_LEASE), _UNITS, _NOON)


_LATER = _LEASE + timedelta(minutes=2)


def test_a_claim_with_no_copy_begun_refunds_as_it_is_taken_over(claims):
    held = _held(claims)

    assert claims.claim(_EXPORT, now=_NOON, expires_at=_LEASE) is None
    taken = claims.claim(_EXPORT, now=_LEASE, expires_at=_LATER)

    assert (taken.state, taken.spends) == (ClaimState.open, ())
    assert _counts(claims) == {}
    assert held.generation != taken.generation


@pytest.mark.parametrize(
    ("landed", "state"), [(None, ClaimState.copying), (True, ClaimState.landed)]
)
def test_a_claim_a_copy_began_on_keeps_its_units_through_a_takeover(claims, landed, state):
    copying = claims.begin_copy(_held(claims), "sha256:a", now=_NOON, expires_at=_LEASE)
    if landed:
        claims.settle_copy(copying, landed=True)

    taken = claims.claim(_EXPORT, now=_LEASE, expires_at=_LATER)

    assert (taken.state, taken.spends, taken.spent_at) == (state, tuple(_UNITS), _NOON)
    assert taken.digest == "sha256:a"
    assert _counts(claims) == {(unit.key, unit.kind): 1 for unit in _UNITS}


def test_every_write_on_an_old_read_changes_nothing(claims):
    held = _held(claims)
    current = claims.begin_copy(held, "sha256:a", now=_NOON, expires_at=_LEASE)
    spent = _counts(claims)

    claims.close_claim(held)
    claims.free_claim(held, _NOON)

    assert claims.begin_copy(held, "sha256:b", now=_NOON, expires_at=_LEASE) is None
    assert claims.hold_claim(held, _UNITS[:1], _NOON) is None
    assert claims.settle_copy(held, landed=False) is None
    assert _counts(claims) == spent
    assert claims.read_claim(_EXPORT) == current


def test_a_read_of_a_deleted_claim_stays_fenced_from_the_next_claim_of_its_export(claims):
    first = claims.claim(_EXPORT, now=_NOON, expires_at=_LEASE)
    claims.close_claim(first)
    second = claims.claim(_EXPORT, now=_NOON, expires_at=_LEASE)

    assert claims.hold_claim(first, _UNITS, _NOON) is None
    assert claims.read_claim(_EXPORT) == second


def test_a_copy_past_the_lease_does_not_begin(claims):
    held = _held(claims)

    assert claims.begin_copy(held, "sha256:a", now=_LEASE, expires_at=_LATER) is None
    assert claims.read_claim(_EXPORT) == held


def test_beginning_a_copy_gives_it_its_own_lease(claims):
    began = claims.begin_copy(_held(claims), "sha256:a", now=_NOON, expires_at=_LATER)

    assert claims.read_claim(_EXPORT).expires_at == began.expires_at == _LATER


def test_settling_a_copy_that_did_not_land_refunds_and_reopens_the_claim(claims):
    copying = claims.begin_copy(_held(claims), "sha256:a", now=_NOON, expires_at=_LEASE)

    opened = claims.settle_copy(copying, landed=False)

    assert (opened.state, opened.spends, opened.spent_at) == (ClaimState.open, (), None)
    assert opened.digest is None
    assert _counts(claims) == {}
    assert claims.hold_claim(opened, _UNITS, _NOON).state == ClaimState.held


def test_closing_a_claim_refunds_once_whoever_else_closes_it(claims):
    held = _held(claims)

    claims.close_claim(held)
    claims.spend(_ALICE, AllowanceKind.pin, _NOON)
    claims.close_claim(held)

    assert _counts(claims) == {(_ALICE.key, "pin"): 1}


def test_closing_a_landed_claim_keeps_the_units_that_paid_for_it(claims):
    landed = claims.settle_copy(
        claims.begin_copy(_held(claims), "sha256:a", now=_NOON, expires_at=_LEASE), landed=True
    )

    claims.close_claim(landed)

    assert claims.read_claim(_EXPORT) is None
    assert _counts(claims) == {(unit.key, unit.kind): 1 for unit in _UNITS}


def _exhausted(ledger: LiveAllowanceLedger, *, now: datetime = _NOON) -> AllowanceExhausted:
    with pytest.raises(AllowanceExhausted) as refused:
        ledger.spend(_ALICE, AllowanceKind.pin, now)
    return refused.value


@pytest.mark.parametrize(
    ("move", "held"),
    [
        pytest.param(lambda ledger, claim: claim, True, id="held"),
        pytest.param(
            lambda ledger, claim: ledger.begin_copy(claim, "d", now=_NOON, expires_at=_LEASE),
            True,
            id="copying",
        ),
        pytest.param(
            lambda ledger, claim: ledger.settle_copy(
                ledger.begin_copy(claim, "d", now=_NOON, expires_at=_LEASE), landed=True
            ),
            False,
            id="landed",
        ),
    ],
)
def test_a_limit_says_whether_a_running_claims_unit_can_come_back(claims, move, held):
    move(claims, _held(claims))

    assert _exhausted(claims).held is held


def test_a_stale_claims_unit_answers_held_until_it_settles(claims):
    held = _held(claims)

    assert _exhausted(claims, now=_LEASE).held is True
    assert claims.read_claim(_EXPORT) == held


def test_a_limit_names_only_the_held_units_of_its_own_key_and_day(claims):
    long_lease = _NOON + timedelta(days=2)
    claims.hold_claim(claims.claim(_EXPORT, now=_NOON, expires_at=long_lease), _UNITS, _NOON)
    claims.spend(_BOB, AllowanceKind.pin, _NOON)
    claims.spend(_ALICE, AllowanceKind.pin, _NOON + timedelta(days=1))

    with pytest.raises(AllowanceExhausted) as bob:
        claims.spend(_BOB, AllowanceKind.pin, _NOON)
    next_day = _exhausted(claims, now=_NOON + timedelta(days=1))

    assert (bob.value.held, next_day.held) == (False, False)


def test_spends_and_refunds_lock_rows_in_one_order_whatever_order_they_are_named(monkeypatch):
    store = FakeVersionRecordDatasetStore()
    ledger = LiveAllowanceLedger(store=store, limits=dict.fromkeys(AllowanceKind, 5))
    named = [
        Spend(_ALICE.key, AllowanceKind.pin),
        Spend(_BOB.key, AllowanceKind.pin),
        Spend(BUDGET_KEY, AllowanceKind.open_pin_budget),
        Spend(BUDGET_KEY, AllowanceKind.pin_budget),
    ]
    locked: list[tuple[str, ...]] = []

    class _Tx:
        def __init__(self, tx):
            self._tx = tx

        def __getattr__(self, name):
            return getattr(self._tx, name)

        def fetch_one(self, sql, params=()):
            if "datasets.caller_allowance" in sql:
                locked.append(tuple(params[:2]))
            return self._tx.fetch_one(sql, params)

    transaction = store.transaction

    @contextmanager
    def spied():
        with transaction() as tx:
            yield _Tx(tx)

    monkeypatch.setattr(store, "transaction", spied)

    ledger.spend_together(named, _NOON)
    spent = locked[:]
    locked.clear()
    ledger.refund_together(list(reversed(named)), _NOON)

    try:
        assert spent == locked
        assert [row[1] for row in spent[:2]] == ["pin_budget", "open_pin_budget"]
    finally:
        store.close()


def _preparing(
    ledger: LiveAllowanceLedger, claim: ExportClaim, artifacts: FakeArtifactStore
) -> PreparingCopy:
    return PreparingCopy(
        claim, allowances=ledger, artifacts=artifacts, caller=None, clock=lambda: _NOON
    )


class _Bucket(FakeArtifactStore):
    """A bucket whose copy raises `copy_raises`, after landing when `lands`, and whose pinned read
    fails when `read_fails`."""

    def __init__(self, *, lands=False, copy_raises=None, read_fails=False) -> None:
        super().__init__()
        self.lands = lands
        self.copy_raises = copy_raises
        self.read_fails = read_fails

    def copy(self, digest, *, from_tier, to_tier):
        if self.copy_raises is None or self.lands:
            copied = super().copy(digest, from_tier=from_tier, to_tier=to_tier)
        if self.copy_raises is not None:
            raise self.copy_raises
        return copied

    def exists(self, digest, *, tier):
        if self.read_fails:
            raise ArtifactWriteError(object_name=digest, digest=digest)
        return super().exists(digest, tier=tier)


_DIGEST = "sha256:a"
_ALL_UNITS = {(unit.key, unit.kind): 1 for unit in _UNITS}
_WRITE_FAILED = ArtifactWriteError(object_name=_DIGEST, digest=_DIGEST)


@pytest.mark.parametrize(
    ("bucket", "state", "units"),
    [
        pytest.param(_Bucket(lands=True), ClaimState.landed, _ALL_UNITS, id="present-keeps"),
        pytest.param(_Bucket(), ClaimState.open, {}, id="absent-refunds"),
    ],
)
def test_a_copy_an_earlier_attempt_began_settles_by_the_pinned_object(claims, bucket, state, units):
    copying = claims.begin_copy(_held(claims), _DIGEST, now=_NOON, expires_at=_LEASE)
    if bucket.lands:
        bucket._objects[(_DIGEST, Tier.pinned.value)] = b"bytes"

    _preparing(claims, copying, bucket).settle_earlier()

    assert claims.read_claim(_EXPORT).state == state
    assert _counts(claims) == units


@pytest.mark.parametrize(
    ("bucket", "state", "units"),
    [
        pytest.param(
            _Bucket(lands=True, copy_raises=_WRITE_FAILED),
            ClaimState.landed,
            _ALL_UNITS,
            id="landed-then-raised",
        ),
        pytest.param(_Bucket(copy_raises=_WRITE_FAILED), ClaimState.open, {}, id="never-landed"),
        pytest.param(
            _Bucket(copy_raises=_WRITE_FAILED, read_fails=True),
            ClaimState.copying,
            _ALL_UNITS,
            id="unknown",
        ),
    ],
)
def test_a_copy_that_raises_settles_by_the_pinned_object(claims, bucket, state, units):
    bucket._objects[(_DIGEST, Tier.working.value)] = b"bytes"

    with pytest.raises(ArtifactWriteError):
        _preparing(claims, _held(claims), bucket).copy(_DIGEST)

    assert claims.read_claim(_EXPORT).state == state
    assert _counts(claims) == units


def test_a_deduplicated_copy_refunds_but_a_paid_retrys_keeps_its_units(claims):
    """A retry finds the object its own units landed, which reads the same as another pin's."""
    bucket = _Bucket()
    bucket._objects[(_DIGEST, Tier.working.value)] = b"bytes"
    bucket._objects[(_DIGEST, Tier.pinned.value)] = b"bytes"
    held = _held(claims)
    landed = claims.settle_copy(
        claims.begin_copy(held, _DIGEST, now=_NOON, expires_at=_LEASE), landed=True
    )
    paid = claims.claim(_EXPORT, now=_LEASE, expires_at=_LATER)
    assert paid.generation != landed.generation

    _preparing(claims, paid, bucket).copy(_DIGEST)
    assert _counts(claims) == _ALL_UNITS

    other = claims.claim("exp_b", now=_NOON, expires_at=_LEASE)
    held = claims.hold_claim(other, _UNITS[:1], _NOON)
    _preparing(claims, held, bucket).copy(_DIGEST)
    assert claims.read_claim("exp_b").state == ClaimState.open
    assert _counts(claims) == _ALL_UNITS


def test_a_paid_claim_whose_working_bytes_have_gone_stays_paid(claims):
    bucket = _Bucket()
    bucket._objects[(_DIGEST, Tier.pinned.value)] = b"bytes"
    held = _held(claims)
    claims.settle_copy(claims.begin_copy(held, _DIGEST, now=_NOON, expires_at=_LEASE), landed=True)
    paid = claims.claim(_EXPORT, now=_LEASE, expires_at=_LATER)

    with pytest.raises(ArtifactNotFoundError):
        _preparing(claims, paid, bucket).copy(_DIGEST)

    assert claims.read_claim(_EXPORT).state == ClaimState.landed
    assert _counts(claims) == _ALL_UNITS


def test_a_copy_nobody_can_judge_refuses_before_the_render(claims):
    held = _held(claims)
    copying = claims.begin_copy(held, _DIGEST, now=_NOON, expires_at=_LEASE)

    with pytest.raises(ExportClaimUnsettled):
        _preparing(claims, copying, _Bucket(read_fails=True)).settle_earlier()

    assert claims.read_claim(_EXPORT) == copying


def test_a_prepare_that_ends_settles_its_claim_by_how_it_ended(claims):
    bucket = _Bucket()
    bucket._objects[(_DIGEST, Tier.working.value)] = b"bytes"

    with pytest.raises(ValueError), _preparing(claims, _held(claims), bucket):
        raise ValueError
    assert (claims.read_claim(_EXPORT), _counts(claims)) == (None, {})

    with _preparing(claims, _held(claims), bucket) as preparing:
        preparing.copy(_DIGEST)
    assert (claims.read_claim(_EXPORT), _counts(claims)) == (None, _ALL_UNITS)


def test_a_prepare_that_fails_past_its_copy_leaves_the_claim_to_its_retry(claims):
    bucket = _Bucket()
    bucket._objects[(_DIGEST, Tier.working.value)] = b"bytes"

    with pytest.raises(ValueError), _preparing(claims, _held(claims), bucket) as preparing:
        preparing.copy(_DIGEST)
        raise ValueError

    assert claims.read_claim(_EXPORT).state == ClaimState.landed
    assert claims.read_claim(_EXPORT).expires_at == _NOON
    assert _counts(claims) == _ALL_UNITS


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
