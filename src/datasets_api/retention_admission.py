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
allowance `AllowanceExhausted`, 429. A unit a running export prepare holds is not spent yet, so
its limit refuses `PinUnitHeld`, 503 with `retryable: true` and a one-second `Retry-After`.

The units pay for the copy that stands. A copy that finds the pinned bytes already stored, made
by a concurrent pin, gives its units back, since that pin's units pay for it, and so does a copy
whose working bytes are gone, since only that proves nothing landed. Any other failure of a copy
leaves its outcome unknown, because the bytes can land before it raises, so the units stay, and
a retry that finds the bytes stored gives its own set back. A copy that stands keeps its units
through any later failure of the write that asked for it, since the pinned bytes stay. A caller
that skips the copy for an artifact already pinned spends nothing.

`PreparingCopy` runs one export prepare over its claim (`allowance.ExportClaim`). It spends the
units onto the claim ahead of a render, so a refusal costs that render nothing, and names the
digest on the claim before the copy. When the prepare ends, a claim with no digest gives its
units back and goes. So does a claim with a digest the prepare recorded, keeping them. A failed
prepare with a digest frees the claim and leaves it: the copy may stand, so the retry that takes
over inherits the paid units and spends nothing more. The sweep settles what no retry took
(`sweep_export_claims`), and a process that dies leaves its claim to the sweep too.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Collection, Sequence
from datetime import UTC, datetime
from types import MappingProxyType, TracebackType
from typing import Self

from artifact_store import ArtifactNotFoundError, ArtifactStore, StoredArtifact, Tier
from dataset_store.clock import Clock
from datasets_api.allowance import (
    BUDGET_KEY,
    AllowanceExhausted,
    AllowanceKind,
    ExportClaim,
    SharedAllowanceLedger,
    Spend,
    seconds_to_reset,
)
from datasets_api.backpressure import Backpressure
from datasets_api.codes_exports import ExportFileRefused
from dfo_platform.caller import Caller
from dfo_platform.problem import Refusal

__all__ = [
    "ExportClaimLost",
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


class PinUnitHeld(Backpressure):
    """A running export prepare holds the last unit of a pin limit, and may give it back. Maps to
    retryable `service_unavailable`, 503, with the one-second `Retry-After` of full capacity."""

    code = "service_unavailable"
    detail = "the last unit of a pin limit is held by a running export; retry"


class ExportClaimLost(Backpressure):
    """An export prepare outlived its claim's lease, so another may hold it. Maps to retryable
    `service_unavailable`, 503."""

    code = "service_unavailable"
    detail = "another prepare took this export over; retry"


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


def _spend_refusal(
    exhausted: AllowanceExhausted,
    *,
    allowances: SharedAllowanceLedger,
    spends: Sequence[Spend],
    now: datetime,
) -> Refusal:
    """The refusal of the spend `exhausted` stopped: `PinUnitHeld` when a live claim holds the
    unit, else the copy's own for a day that spent it."""
    unit = next(spend for spend in spends if spend.kind == exhausted.kind)
    if allowances.holds_unit(unit, now):
        return PinUnitHeld()
    budget = _BUDGET_REFUSALS.get(exhausted.kind)
    return exhausted if budget is None else budget(limit=exhausted.limit, now=now)


class PreparingCopy:
    """One export prepare's pinned copy, over `claim`, as a block. See the module docstring."""

    def __init__(
        self,
        claim: ExportClaim,
        *,
        allowances: SharedAllowanceLedger,
        caller: Caller | None,
        clock: Clock,
    ) -> None:
        self._claim = claim
        self._allowances = allowances
        self._caller = caller
        self._clock = clock
        self._digest = claim.digest

    def hold(self) -> None:
        """Spend the copy's units onto the claim, unless it holds them from an earlier attempt.

        Raises:
            PinBudgetExhausted, OpenPinBudgetExhausted, AllowanceExhausted, PinUnitHeld: As
                `copy_to_pinned`.
            ExportClaimLost: The claim's lease ended and another prepare may hold it.
        """
        if self._claim.spends:
            return
        spends, now = _pin_spends(self._caller), self._clock()
        try:
            held = self._allowances.hold_claim(self._claim, spends, now)
        except AllowanceExhausted as exhausted:
            refusal = _spend_refusal(exhausted, allowances=self._allowances, spends=spends, now=now)
            raise refusal from None
        if held is None:
            raise ExportClaimLost()
        self._claim = held

    def copy(self, artifacts: ArtifactStore, artifact_digest: str) -> StoredArtifact:
        """Copy `artifact_digest` to the pinned tier on the held units, naming it on the claim
        first. A deduplicated copy pays for nothing, unless an earlier attempt named the same
        digest, whose own copy may be the one found.

        Raises:
            ArtifactNotFoundError: The working bytes are gone.
            ExportClaimLost: The claim's lease ended and another prepare may hold it.
        """
        earlier = self._digest == artifact_digest
        if not earlier:
            if not self._allowances.name_claim_digest(self._claim, artifact_digest):
                raise ExportClaimLost()
            self._digest = artifact_digest
        try:
            copied = artifacts.copy(artifact_digest, from_tier=Tier.working, to_tier=Tier.pinned)
        except ArtifactNotFoundError:
            if not earlier:
                self._digest = None
            raise
        if copied.deduplicated and not earlier:
            self._digest = None
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
        try:
            decided = isinstance(exc, Refusal) and exc.decided_by_request and not exc.retryable
            if decided and self._digest is None:
                self._allowances.refuse_claim(self._claim, ExportFileRefused.facts_of(exc))
            elif exc is None or self._digest is None:
                self._allowances.close_claim(self._claim, refund=self._digest is None)
            else:
                self._allowances.free_claim(self._claim, self._clock())
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
        ArtifactWriteError, ArtifactInventoryError: The copy's outcome is unknown. The units stay.
    """
    if allowances is None:
        return artifacts.copy(artifact_digest, from_tier=Tier.working, to_tier=Tier.pinned)
    spends = _pin_spends(caller)
    try:
        allowances.spend_together(spends, now)
    except AllowanceExhausted as exhausted:
        raise _spend_refusal(exhausted, allowances=allowances, spends=spends, now=now) from None
    try:
        copied = artifacts.copy(artifact_digest, from_tier=Tier.working, to_tier=Tier.pinned)
    except ArtifactNotFoundError:
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


def sweep_export_claims(
    *, allowances: SharedAllowanceLedger, pinned: Callable[[], Collection[str]], now: datetime
) -> int:
    """Settle every claim whose lease has passed at `now`, and return how many went. A claim
    whose digest has no object in `pinned`, the raw pinned bucket since an inventory row can be
    what failed, holds no copy, so its units come back. A claim behind a standing copy keeps
    them, and stays through their UTC day so that a retry finds its paid copy: from the next
    day a retry has units of its own, which the pinned bytes give back. A second run finds
    nothing new to do, and a claim a prepare took over meanwhile keeps its new lease."""
    claims = allowances.expired_claims(now)
    objects = pinned() if any(claim.digest for claim in claims) else ()
    settled = 0
    for claim in claims:
        stands = claim.digest in objects
        if stands and claim.spent_at is not None and _same_day(claim.spent_at, now):
            continue
        allowances.close_claim(claim, refund=not stands)
        settled += 1
    return settled


def _same_day(first: datetime, second: datetime) -> bool:
    return first.astimezone(UTC).date() == second.astimezone(UTC).date()
