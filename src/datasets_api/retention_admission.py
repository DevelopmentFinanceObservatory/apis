"""Admission of new pinned copies against the daily pin budgets.

Every new pinned copy a caller asks for spends here, over a claim (`allowance.CopyClaim`, whose
module holds the state table): `copy_to_pinned` for the pin door's named artifact and each frame
and derived root a share's create, replace or snapshot conversion pins
(`presentation_store.LiveFrameStore.pin`), each under a claim of its own, and `PreparingCopy` for
an export's file (`exports`), under the claim keyed by its export id. The serving-frame copies a pin makes
(`doors.pin`) and the canonical frames `maintenance` renders again copy without it and spend
nothing. A copy spends one unit of the day's pin budget (`AllowanceKind.pin_budget`,
`DATASETS_PIN_DAILY_MAX`) and, for an open caller, one unit of the open callers' budget
(`AllowanceKind.open_pin_budget`, `DATASETS_PIN_OPEN_DAILY_MAX`) and one of the caller's `pin`
allowance (`DATASETS_PIN_DAILY_ALLOWANCE`), all together onto its claim before the copy, so a
refusal by one spends none of the others. A spent budget refuses `PinBudgetExhausted` or
`OpenPinBudgetExhausted`, 503 with `retryable: true` and `Retry-After` naming the reset at the
next UTC midnight, and a spent allowance `AllowanceExhausted`, 429. A unit on a claim, held or
behind a copy whose outcome is unknown, can come back, so its limit refuses `PinUnitHeld`, 503
with `retryable: true` and a `Retry-After` of `allowance.RUNNING_CLAIM_RETRY_SECONDS` while the
claim's lease runs, else `allowance.SWEEP_INTERVAL_SECONDS`.

The units pay for the copy that stands, and one mechanism settles every copy. The claim moves to
`copying` before the copy, and the copy's own result settles it: a new object is `landed`, and a
deduplicated copy, made by a concurrent pin whose units pay for it, or working bytes gone refund.
A copy that fails otherwise reads the pinned object with the short read a work permit allows:
absent refunds, present keeps the units, and a read that fails leaves the claim `copying` for
`sweep_copy_claims` to settle with the long read. A copy that stands keeps its units through any
later failure of the write that asked for it, since the pinned bytes stay. A caller that skips
the copy for an artifact already pinned spends nothing.

`PreparingCopy` spends an export's units onto its claim ahead of a render, so a refusal costs that
render nothing, and a prepare that lost its claim copies nothing. A claim paid already keeps its
units. The prepare's lease is static, with no thread renewing it, since Cloud Run throttles a
prepare that outlives its request: `definition_timeout_seconds` plus `PREPARE_LEASE_MARGIN`, as the
worker's static lease does. The copy reads its source location first. `begin_copy` refuses past
the lease, so a render that outlives it copies nothing, and writes the copy lease, the store's
`copy_lands_within` plus `_BEGIN_COPY_ALLOWANCE`. The copy starts only while the whole of
`copy_lands_within` is left on that lease, so the `begin_copy` transaction's own time sits outside
the copy's budget. Settling the copy writes a fresh claim lease on either outcome, so the owner
keeps the claim through its reference, shape and record writes. A retry takes over a `copying`
claim and settles it by the pinned object before any render, refusing `CopyClaimUnsettled` if it
cannot. A stale claim with no copy begun refunds as it is taken over, so the next caller spends
for itself. A stale unit answers `PinUnitHeld` until then or until `sweep_copy_claims`, the
five-minute backstop, settles it.

Counted from the spend, a crash holds a unit for at most the claim's lease plus the copy lease plus
the sweep interval. Under the default values (a 120-second definition timeout, `CopyBounds()` and
the five-minute schedule) that is 150 + 100 + 300 = 550 seconds for an export and 100 + 100 + 300
= 500 for a pin. A refund lands on the UTC day that spent the unit, so after midnight it changes
nothing a caller can see, since that day's limits no longer apply.

Four causes can land a copy after its units came back, each at most one free object per digest:
a prepare paused between its lease check and the copy, which `cpu_idle` makes ordinary, a copy
whose reply trickles past its lease, a client-side `copy_blob` timeout with the server still
copying, and a lost response retried with `if_generation_match=0`, which reads as deduplicated. A
pin landing first leaves two sets spent, since an existence read cannot say who paid.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from datetime import datetime, timedelta
from types import MappingProxyType, TracebackType
from typing import Self, cast
from uuid import uuid4

from artifact_store import (
    ArtifactNotFoundError,
    ArtifactStore,
    ExistsRead,
    StoredArtifact,
    Tier,
)
from dataset_store.clock import Clock
from datasets_api.allowance import (
    BUDGET_KEY,
    AllowanceExhausted,
    AllowanceKind,
    ClaimState,
    CopyClaim,
    SharedAllowanceLedger,
    Spend,
    seconds_to_reset,
    utc_day,
)
from dfo_platform.caller import Caller
from dfo_platform.problem import Refusal

PREPARE_LEASE_MARGIN = timedelta(seconds=30)
"""How long a prepare's claim outlives its `definition_timeout_seconds`, for the writes begun
before the deadline."""

_BEGIN_COPY_ALLOWANCE = timedelta(seconds=30)
"""How long the `begin_copy` transaction may take and still leave its copy the whole of the store's
`copy_lands_within`. A slower one copies nothing."""

SWEEP_BUDGET = timedelta(seconds=240)
"""How long one sweep starts claims, under the five-minute schedule, so runs never overlap and a
slow bucket holds up no refund past the next run."""

_UNSETTLED_RETRY_SECONDS = 30
"""`CopyClaimUnsettled`'s wait: the pinned bucket could not be read, and a quick retry would
only take a work permit to fail again."""


__all__ = [
    "PREPARE_LEASE_MARGIN",
    "CopyClaimUnsettled",
    "OpenPinBudgetExhausted",
    "PinBudgetExhausted",
    "PinUnitHeld",
    "PreparingCopy",
    "copy_to_pinned",
    "sweep_copy_claims",
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


class PinUnitHeld(Refusal):
    """A claim, of a running copy or a dead one, holds the last unit of a pin limit, and may give
    it back. Maps to retryable `service_unavailable`, 503, with `Retry-After` of
    `retry_after`."""

    code = "service_unavailable"
    retryable = True
    detail = "the last unit of a pin limit is held by a copy that has not settled; retry"

    def __init__(self, *, retry_after: int) -> None:
        self.headers = MappingProxyType({"Retry-After": str(retry_after)})
        super().__init__()


class CopyClaimUnsettled(Refusal):
    """A copy cannot go on over its claim: another writer moved it, its lease passed, or the
    pinned object that settles an earlier copy could not be read. Maps to retryable
    `service_unavailable`, 503, with `Retry-After` of `_UNSETTLED_RETRY_SECONDS`."""

    code = "service_unavailable"
    retryable = True
    detail = "the claim this copy runs under is not settled; retry"
    headers = MappingProxyType({"Retry-After": str(_UNSETTLED_RETRY_SECONDS)})


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
    """Whether the pinned object of `digest` exists, `None` when the read fails. The read is
    `ExistsRead.short`, since its caller holds a permit."""
    try:
        return artifacts.exists(digest, tier=Tier.pinned)
    except Exception:
        logger.exception("Reading the pinned object %s failed", digest)
        return None


class PreparingCopy:
    """One pinned copy over `claim`, an export's or a pin's own, as a block that settles the claim
    as it ends. `lease` is the claim's own, which a settled copy writes again. See the module
    docstring."""

    def __init__(
        self,
        claim: CopyClaim,
        *,
        allowances: SharedAllowanceLedger,
        artifacts: ArtifactStore,
        caller: Caller | None,
        clock: Clock,
        lease: timedelta,
    ) -> None:
        self._claim = claim
        self._allowances = allowances
        self._artifacts = artifacts
        self._caller = caller
        self._clock = clock
        self._lease = lease

    def _settle(self, *, landed: bool) -> None:
        """Move the claim out of `copying`. A write that fails leaves it there for a retry or
        the sweep, so the prepare's own outcome stands."""
        try:
            settled = self._allowances.settle_copy(
                self._claim, landed=landed, expires_at=self._clock() + self._lease
            )
        except Exception:
            logger.exception("Settling the copy of %s failed", self._claim.key)
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
            CopyClaimUnsettled: The copy's outcome stays unknown.
        """
        if self._claim.state == ClaimState.copying:
            self._judge()
        if self._claim.state == ClaimState.copying:
            raise CopyClaimUnsettled()

    def hold(self) -> None:
        """Spend the copy's units onto the claim, unless it stands behind a copy already.

        Raises:
            PinBudgetExhausted, OpenPinBudgetExhausted, AllowanceExhausted, PinUnitHeld: As
                `copy_to_pinned`.
            CopyClaimUnsettled: Another writer moved the claim.
        """
        if self._claim.state != ClaimState.open:
            return
        now = self._clock()
        try:
            held = self._allowances.hold_claim(self._claim, _pin_spends(self._caller), now)
        except AllowanceExhausted as exhausted:
            raise _spend_refusal(exhausted, now=now) from None
        if held is None:
            raise CopyClaimUnsettled()
        self._claim = held

    def copy(
        self, artifact_digest: str, *, source: tuple[str, str] | None = None
    ) -> StoredArtifact:
        """Copy `artifact_digest` to the pinned tier on the held units, moving the claim to
        `copying` first and settling it by what the copy shows. A claim that paid for another
        digest reopens and spends for this one. `source` is the working copy's bucket and object
        name when the caller holds them, else the copy reads them first.

        Raises:
            ArtifactNotFoundError: The working bytes are gone.
            PinBudgetExhausted, OpenPinBudgetExhausted, AllowanceExhausted, PinUnitHeld: As
                `copy_to_pinned`, when a reopened claim spends.
            CopyClaimUnsettled: Another writer moved the claim, its lease passed, or too little
                was left of it to copy. Nothing is copied.
        """
        if self._claim.state == ClaimState.landed and self._claim.digest != artifact_digest:
            reopened = self._allowances.reopen_claim(self._claim)
            if reopened is None:
                raise CopyClaimUnsettled()
            self._claim = reopened
            self.hold()
        paid = self._claim.state == ClaimState.landed
        if source is None:
            location = self._artifacts.location(artifact_digest, tier=Tier.working)
            if location is None:
                raise ArtifactNotFoundError(digest=artifact_digest, tier=Tier.working)
            source = (location.bucket, location.object_name)
        lands_within = self._artifacts.copy_lands_within
        now = self._clock()
        began = self._allowances.begin_copy(
            self._claim,
            artifact_digest,
            now=now,
            expires_at=now + lands_within + _BEGIN_COPY_ALLOWANCE,
        )
        if began is None:
            raise CopyClaimUnsettled()
        self._claim = began
        if self._clock() + lands_within > began.expires_at:
            self._settle(landed=paid)
            raise CopyClaimUnsettled()
        try:
            copied = self._artifacts.copy(
                artifact_digest, from_tier=Tier.working, to_tier=Tier.pinned, source=source
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
        # The caller's own outcome stands, so a write that fails is logged and the sweep
        # settles the claim.
        state = self._claim.state
        try:
            if state == ClaimState.copying or (exc is not None and state == ClaimState.landed):
                self._allowances.free_claim(self._claim, self._clock())
            else:
                self._allowances.close_claim(self._claim)
        except Exception:
            logger.exception("Settling the claim of %s failed", self._claim.key)


def copy_to_pinned(
    artifacts: ArtifactStore,
    artifact_digest: str,
    *,
    allowances: SharedAllowanceLedger | None,
    caller: Caller | None,
    clock: Clock,
) -> StoredArtifact:
    """Copy `artifact_digest` from the working tier to the pinned tier, spending the units one
    new pinned copy takes onto a claim of its own (see the module docstring), and return the
    pinned copy. `caller` names an open caller, and `None` a caller holding the pin grant or a
    service write, which spends the pin budget alone. `allowances=None` copies without spending,
    for an operator's run.

    Raises:
        PinBudgetExhausted: The day's pin budget is spent. Nothing is copied.
        OpenPinBudgetExhausted: The open callers' budget is spent. Nothing is copied.
        AllowanceExhausted: The caller's `pin` allowance is spent. Nothing is copied.
        PinUnitHeld: A claim holds the last unit. Nothing is copied.
        CopyClaimUnsettled: Too little lease was left to copy. Nothing is copied.
        DatasetsStoreUnavailable: The units could not be spent. Nothing is copied.
        ArtifactNotFoundError: The working bytes are gone. The units are given back.
        ArtifactWriteError, ArtifactInventoryError: The copy failed. The units are given back
            when the pinned object is absent, and the sweep settles a read that failed.
    """
    if allowances is None:
        return artifacts.copy(artifact_digest, from_tier=Tier.working, to_tier=Tier.pinned)
    lease = artifacts.copy_lands_within + _BEGIN_COPY_ALLOWANCE
    now = clock()
    claim = allowances.claim(f"pin_{uuid4().hex}", now=now, expires_at=now + lease)
    if claim is None:
        raise CopyClaimUnsettled()
    with PreparingCopy(
        claim,
        allowances=allowances,
        artifacts=artifacts,
        caller=caller,
        clock=clock,
        lease=lease,
    ) as copying:
        copying.hold()
        return copying.copy(artifact_digest)


def _settle_stale_claim(
    claim: CopyClaim,
    *,
    allowances: SharedAllowanceLedger,
    artifacts: ArtifactStore,
    now: datetime,
) -> bool:
    """Move one stale claim on, and return whether it moved. A `copying` claim is judged by the
    pinned object, and a `landed` one stays through its UTC day so that a retry finds its paid
    copy: from the next day a retry has units of its own, which the pinned bytes give back."""
    if claim.state == ClaimState.copying and artifacts.exists(
        cast(str, claim.digest), tier=Tier.pinned, read=ExistsRead.long
    ):
        return allowances.settle_copy(claim, landed=True, expires_at=claim.expires_at) is not None
    if claim.state == ClaimState.landed and utc_day(cast(datetime, claim.spent_at)) >= utc_day(now):
        return False
    return allowances.close_claim(claim)


def sweep_copy_claims(
    *,
    allowances: SharedAllowanceLedger,
    artifacts: ArtifactStore,
    now: datetime,
    monotonic: Callable[[], float] = time.monotonic,
) -> tuple[int, int, int]:
    """Settle every claim whose lease has passed at `now`, one by one, `held` ones first, and
    return how many moved, how many failed and how many were left for the next run once
    `SWEEP_BUDGET` ran out. Each names its generation, so a second run or a writer that got there
    first leaves nothing to do."""
    settled = failed = 0
    stop = monotonic() + SWEEP_BUDGET.total_seconds()
    claims = allowances.expired_claims(now)
    for started, claim in enumerate(claims):
        if monotonic() >= stop:
            return settled, failed, len(claims) - started
        try:
            settled += _settle_stale_claim(
                claim, allowances=allowances, artifacts=artifacts, now=now
            )
        except Exception:
            logger.exception("Settling the claim of %s failed", claim.key)
            failed += 1
    return settled, failed, 0
