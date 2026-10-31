"""Admission of new pinned copies against the daily pin budgets.

Every new pinned copy a caller asks for spends here: `copy_to_pinned` for the pin door's named
artifact and each frame and derived root a share's create, replace or snapshot conversion pins
(`presentation_store.LiveFrameStore.pin`), and `PreparingCopy` for an export's file (`exports`).
The serving-frame copies a pin makes (`doors.pin`) and the canonical frames `maintenance` renders
again copy without it and spend nothing. A copy spends one unit of the day's pin budget
(`AllowanceKind.pin_budget`, `DATASETS_PIN_DAILY_MAX`) and, for an open caller, one unit of the
open callers' budget (`AllowanceKind.open_pin_budget`, `DATASETS_PIN_OPEN_DAILY_MAX`) and one of
the caller's `pin` allowance (`DATASETS_PIN_DAILY_ALLOWANCE`), all together before the copy
(`allowance.SharedAllowanceLedger.spend_together`), so a refusal by one spends none of the
others. A spent budget refuses `PinBudgetExhausted` or `OpenPinBudgetExhausted`, 503 with
`retryable: true` and `Retry-After` naming the reset at the next UTC midnight, and a spent
allowance `AllowanceExhausted`, 429. A unit on an export claim, held or behind a copy whose outcome
is unknown, can come back, so its limit refuses `PinUnitHeld`, 503 with `retryable: true` and a
`Retry-After` of the wait on the claim (`allowance.claim_retry_seconds`).

The units pay for the copy that stands. A copy that finds the pinned bytes already stored, made
by a concurrent pin, gives its units back, since that pin's units pay for it, and so does a copy
whose working bytes are gone. A copy that fails otherwise reads the pinned object: absent gives
the units back, and present keeps them, as does a read that fails too, since the bytes can land
before the copy raises. So one copy spends one set of units. A copy that stands keeps its units
through any later failure of the write that asked for it, since the pinned bytes stay. A caller
that skips the copy for an artifact already pinned spends nothing.

`PreparingCopy` runs one export prepare over its claim (`allowance.ExportClaim`, whose module
holds the state table). It spends the units onto the claim ahead of a render, so a refusal costs
that render nothing, and moves the claim to `copying` before every copy, so a prepare that lost
its claim copies nothing. The copy's own result settles it: a new object is `landed`, and a
deduplicated copy or absent bytes refund, unless the claim had already paid. The claim's lease is
static, with no thread renewing it, since Cloud Run throttles a prepare that outlives its request:
`definition_timeout_seconds` plus `PREPARE_LEASE_MARGIN`, as the worker's static lease does.
`begin_copy` writes `_COPY_LEASE`, so a running copy's claim stays live, and a render that outlives
its lease lands when nothing took the claim, since a takeover fails the generation fence. A retry
takes over a `copying` claim and settles it by the pinned object before any render, refusing
`ExportClaimUnsettled` if it cannot. A stale claim with no copy begun refunds as it is taken over,
so the next caller spends for itself. A stale unit answers `PinUnitHeld` until then or until
`sweep_export_claims`, the five-minute backstop, settles it. A crash holds a unit for at most the
lease plus the sweep interval: about 450 seconds with no copy begun, and 750 with one. A refund
lands on the UTC day that spent the unit, so after midnight it changes nothing a caller can see,
since that day's limits no longer apply.

Three causes can land a copy after its units came back, each at most one free object per digest:
a prepare paused between `begin_copy` and the copy, which `cpu_idle` makes ordinary, a client-side
`copy_blob` timeout with the server still copying, and a lost response retried with
`if_generation_match=0`, which reads as deduplicated. A pin landing first leaves two sets spent,
since an existence read cannot say who paid. A `copy_to_pinned` whose existence read also fails
keeps its units, with no sweep to settle them.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from types import MappingProxyType, TracebackType
from typing import Self, cast

from artifact_store import ArtifactNotFoundError, ArtifactStore, StoredArtifact, Tier
from dataset_store.clock import Clock
from datasets_api.allowance import (
    BUDGET_KEY,
    AllowanceExhausted,
    AllowanceKind,
    ClaimState,
    ExportClaim,
    SharedAllowanceLedger,
    Spend,
    claim_retry_seconds,
    seconds_to_reset,
    utc_day,
)
from dfo_platform.caller import Caller
from dfo_platform.problem import Refusal

PREPARE_LEASE_MARGIN = timedelta(seconds=30)
"""How long a prepare's claim outlives its `definition_timeout_seconds`, for the writes begun
before the deadline."""

_COPY_LEASE = timedelta(seconds=450)
"""How long a begun copy keeps its claim, above `LiveArtifactStore.copy`'s 420-second budget: a 30 s
inventory read, a `copy_blob` of 180 s (its last retry starts at 120 s and runs 60 s), a `get_blob`
of the same 180 s, and a 30 s inventory write."""

__all__ = [
    "PREPARE_LEASE_MARGIN",
    "ExportClaimUnsettled",
    "OpenPinBudgetExhausted",
    "PinBudgetExhausted",
    "PinUnitHeld",
    "PreparingCopy",
    "copy_to_pinned",
    "sweep_export_claims",
]

logger = logging.getLogger(__name__)


class PinBudgetExhausted(Refusal):
    """The day's `DATASETS_PIN_DAILY_MAX` new pinned copies are made. Maps to 503 with
    `retryable: true`, since the budget resets and a pin that raced to an existing copy gives its
    unit back, and `Retry-After` naming the reset at the next UTC midnight."""

    code = "service_unavailable"
    retryable = True
    detail = "today's budget of new pinned copies is spent; it resets at the next UTC midnight"

    def __init__(self, *, limit: int, now: datetime) -> None:
        self.limit = limit
        self.headers = MappingProxyType({"Retry-After": str(seconds_to_reset(now))})
        super().__init__(f"today's budget of {limit} new pinned copies is spent")


class OpenPinBudgetExhausted(Refusal):
    """The day's `DATASETS_PIN_OPEN_DAILY_MAX` new pinned copies for callers without the pin
    grant are made. Maps to 503 with `retryable: true` and `Retry-After` naming the reset, as
    `PinBudgetExhausted` does."""

    code = "service_unavailable"
    retryable = True
    detail = (
        "today's budget of new pinned copies for open callers is spent; it resets at the next "
        "UTC midnight"
    )

    def __init__(self, *, limit: int, now: datetime) -> None:
        self.limit = limit
        self.headers = MappingProxyType({"Retry-After": str(seconds_to_reset(now))})
        super().__init__(
            f"today's budget of {limit} new pinned copies for callers without the pin grant "
            "is spent"
        )


class _ClaimWait(Refusal):
    """A retryable refusal that lasts until an export claim settles, with `Retry-After` of the
    wait on it (`allowance.claim_retry_seconds`)."""

    retryable = True

    def __init__(self, *, retry_after: int) -> None:
        self.headers = MappingProxyType({"Retry-After": str(retry_after)})
        super().__init__()


class PinUnitHeld(_ClaimWait):
    """An export claim, of a running prepare or a dead one, holds the last unit of a pin limit,
    and may give it back. Maps to retryable `service_unavailable`, 503."""

    code = "service_unavailable"
    detail = "the last unit of a pin limit is held by an export that has not settled; retry"


class ExportClaimUnsettled(_ClaimWait):
    """An export prepare cannot go on over its claim: another writer moved it, or the pinned
    object that settles an earlier copy could not be read. Maps to retryable
    `service_unavailable`, 503."""

    code = "service_unavailable"
    detail = "this export's claim is not settled; retry"


_BUDGET_REFUSALS = {
    AllowanceKind.pin_budget: PinBudgetExhausted,
    AllowanceKind.open_pin_budget: OpenPinBudgetExhausted,
}


def _pin_spends(caller: Caller | None) -> list[Spend]:
    """The units one new pinned copy takes for `caller`, in the order they refuse."""
    spends = [Spend(BUDGET_KEY, AllowanceKind.pin_budget)]
    if caller is not None:
        spends += [
            Spend(BUDGET_KEY, AllowanceKind.open_pin_budget),
            Spend(caller.key, AllowanceKind.pin),
        ]
    return spends


def _spend_refusal(exhausted: AllowanceExhausted, *, now: datetime) -> Refusal:
    """The refusal of the spend `exhausted` stopped: `PinUnitHeld` when a claim holds the
    unit, else the copy's own for a day that spent it."""
    if exhausted.held_for is not None:
        return PinUnitHeld(retry_after=exhausted.held_for)
    budget = _BUDGET_REFUSALS.get(exhausted.kind)
    return exhausted if budget is None else budget(limit=exhausted.limit, now=now)


def _landed(artifacts: ArtifactStore, digest: str) -> bool | None:
    """Whether the pinned object of `digest` exists, `None` when the read fails."""
    try:
        return artifacts.exists(digest, tier=Tier.pinned)
    except Exception:
        logger.exception("Reading the pinned object %s failed", digest)
        return None


class PreparingCopy:
    """One export prepare's pinned copy, over `claim`, as a block that settles the claim as it
    ends. See the module docstring."""

    def __init__(
        self,
        claim: ExportClaim,
        *,
        allowances: SharedAllowanceLedger,
        artifacts: ArtifactStore,
        caller: Caller | None,
        clock: Clock,
    ) -> None:
        self._claim = claim
        self._allowances = allowances
        self._artifacts = artifacts
        self._caller = caller
        self._clock = clock

    def _unsettled(self) -> ExportClaimUnsettled:
        """The refusal of a prepare that cannot go on, asking the caller to wait out the row."""
        claim = self._allowances.read_claim(self._claim.export_id)
        retry_after = 1 if claim is None else claim_retry_seconds(claim.expires_at, self._clock())
        return ExportClaimUnsettled(retry_after=retry_after)

    def _settle(self, *, landed: bool) -> None:
        """Move the claim out of `copying`. A write that fails leaves it there for a retry or
        the sweep, so the prepare's own outcome stands."""
        try:
            settled = self._allowances.settle_copy(self._claim, landed=landed)
        except Exception:
            logger.exception("Settling the copy of %s failed", self._claim.export_id)
            return
        if settled is not None:
            self._claim = settled

    def _judge(self) -> None:
        """Settle a `copying` claim by the pinned object, or leave it when the read fails."""
        landed = _landed(self._artifacts, cast(str, self._claim.digest))
        if landed is not None:
            self._settle(landed=landed)

    def settle_earlier(self) -> None:
        """Settle a copy an earlier attempt began, before this prepare spends a render.

        Raises:
            ExportClaimUnsettled: The copy's outcome stays unknown.
        """
        if self._claim.state == ClaimState.copying:
            self._judge()
        if self._claim.state == ClaimState.copying:
            raise self._unsettled()

    def hold(self) -> None:
        """Spend the copy's units onto the claim, unless it stands behind a copy already.

        Raises:
            PinBudgetExhausted, OpenPinBudgetExhausted, AllowanceExhausted, PinUnitHeld: As
                `copy_to_pinned`.
            ExportClaimUnsettled: Another writer moved the claim.
        """
        if self._claim.state != ClaimState.open:
            return
        now = self._clock()
        try:
            held = self._allowances.hold_claim(self._claim, _pin_spends(self._caller), now)
        except AllowanceExhausted as exhausted:
            raise _spend_refusal(exhausted, now=now) from None
        if held is None:
            raise self._unsettled()
        self._claim = held

    def copy(self, artifact_digest: str) -> StoredArtifact:
        """Copy `artifact_digest` to the pinned tier on the held units, moving the claim to
        `copying` first and settling it by what the copy shows. A claim that paid for another
        digest reopens and spends for this one.

        Raises:
            ArtifactNotFoundError: The working bytes are gone.
            PinBudgetExhausted, OpenPinBudgetExhausted, AllowanceExhausted, PinUnitHeld: As
                `copy_to_pinned`, when a reopened claim spends.
            ExportClaimUnsettled: Another writer moved the claim. Nothing is copied.
        """
        if self._claim.state == ClaimState.landed and self._claim.digest != artifact_digest:
            reopened = self._allowances.reopen_claim(self._claim)
            if reopened is None:
                raise self._unsettled()
            self._claim = reopened
            self.hold()
        paid = self._claim.state == ClaimState.landed
        began = self._allowances.begin_copy(
            self._claim, artifact_digest, expires_at=self._clock() + _COPY_LEASE
        )
        if began is None:
            raise self._unsettled()
        self._claim = began
        try:
            copied = self._artifacts.copy(
                artifact_digest, from_tier=Tier.working, to_tier=Tier.pinned
            )
        except ArtifactNotFoundError:
            self._settle(landed=paid)
            raise
        except Exception:
            self._judge()
            raise
        self._settle(landed=paid or not copied.deduplicated)
        return copied

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        # The prepare's own outcome stands, so a write that fails is logged and the sweep
        # settles the claim.
        state = self._claim.state
        try:
            if state == ClaimState.copying or (exc is not None and state == ClaimState.landed):
                self._allowances.free_claim(self._claim, self._clock())
            else:
                self._allowances.close_claim(self._claim)
        except Exception:
            logger.exception("Settling the claim of %s failed", self._claim.export_id)


def copy_to_pinned(
    artifacts: ArtifactStore,
    artifact_digest: str,
    *,
    allowances: SharedAllowanceLedger | None,
    caller: Caller | None,
    now: datetime,
) -> StoredArtifact:
    """Copy `artifact_digest` from the working tier to the pinned tier, spending the units one
    new pinned copy takes (see the module docstring) before it, and return the pinned copy.
    `caller` names an open caller, and `None` a caller holding the pin grant or a service
    write, which spends the pin budget alone. `allowances=None` copies without spending, for an
    operator's run.

    Raises:
        PinBudgetExhausted: The day's pin budget is spent. Nothing is copied.
        OpenPinBudgetExhausted: The open callers' budget is spent. Nothing is copied.
        AllowanceExhausted: The caller's `pin` allowance is spent. Nothing is copied.
        PinUnitHeld: A running export prepare holds the last unit. Nothing is copied.
        DatasetsStoreUnavailable: The units could not be spent. Nothing is copied.
        ArtifactNotFoundError: The working bytes are gone. The units are given back.
        ArtifactWriteError, ArtifactInventoryError: The copy failed. The units are given back
            only when the pinned object is absent.
    """
    if allowances is None:
        return artifacts.copy(artifact_digest, from_tier=Tier.working, to_tier=Tier.pinned)
    spends = _pin_spends(caller)
    try:
        allowances.spend_together(spends, now)
    except AllowanceExhausted as exhausted:
        raise _spend_refusal(exhausted, now=now) from None
    try:
        copied = artifacts.copy(artifact_digest, from_tier=Tier.working, to_tier=Tier.pinned)
    except ArtifactNotFoundError:
        _refund(allowances, spends, now=now)
        raise
    except Exception:
        if _landed(artifacts, artifact_digest) is False:
            _refund(allowances, spends, now=now)
        raise
    if copied.deduplicated:
        _refund(allowances, spends, now=now)
    return copied


def _refund(allowances: SharedAllowanceLedger, spends: list[Spend], *, now: datetime) -> None:
    """Give back each of `spends` together. The copy's outcome stands either way, so a refund
    that fails is logged and leaves the units spent."""
    try:
        allowances.refund_together(spends, now)
    except Exception:
        logger.exception("Refunding the pin units %s failed", spends)


def _settle_stale_claim(
    claim: ExportClaim,
    *,
    allowances: SharedAllowanceLedger,
    artifacts: ArtifactStore,
    now: datetime,
) -> bool:
    """Move one stale claim on, and return whether it moved. A `copying` claim is judged by the
    pinned object, and a `landed` one stays through its UTC day so that a retry finds its paid
    copy: from the next day a retry has units of its own, which the pinned bytes give back."""
    if claim.state == ClaimState.copying and artifacts.exists(
        cast(str, claim.digest), tier=Tier.pinned
    ):
        return allowances.settle_copy(claim, landed=True) is not None
    if claim.state == ClaimState.landed and utc_day(cast(datetime, claim.spent_at)) >= utc_day(now):
        return False
    return allowances.close_claim(claim)


def sweep_export_claims(
    *, allowances: SharedAllowanceLedger, artifacts: ArtifactStore, now: datetime
) -> tuple[int, int]:
    """Settle every claim whose lease has passed at `now`, one by one, and return how many moved
    and how many failed. Each names its generation, so a second run or a writer that got there
    first leaves nothing to do."""
    settled = failed = 0
    for claim in allowances.expired_claims(now):
        try:
            settled += _settle_stale_claim(
                claim, allowances=allowances, artifacts=artifacts, now=now
            )
        except Exception:
            logger.exception("Settling the claim of %s failed", claim.export_id)
            failed += 1
    return settled, failed
