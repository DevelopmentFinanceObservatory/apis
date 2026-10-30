"""Admission of new pinned copies against the daily pin budgets.

Every new pinned copy a caller asks for goes through `copy_to_pinned`: the pin door's named
artifact, an export's file (`exports`), and each frame and derived root a share's create, replace
or snapshot conversion pins (`presentation_store.LiveFrameStore.pin`). The serving-frame copies a pin makes (`doors.pin`)
and the canonical frames `maintenance` renders again copy without it and spend nothing. A copy spends one unit of the day's pin budget
(`AllowanceKind.pin_budget`, `DATASETS_PIN_DAILY_MAX`) and, for an open caller, one unit of the
open callers' budget (`AllowanceKind.open_pin_budget`, `DATASETS_PIN_OPEN_DAILY_MAX`) and one of
the caller's `pin` allowance (`DATASETS_PIN_DAILY_ALLOWANCE`), all together before the copy
(`allowance.SharedAllowanceLedger.spend_together`), so a refusal by one spends none of the
others. A spent budget refuses `PinBudgetExhausted` or `OpenPinBudgetExhausted`, 503 with
`retryable: true` and `Retry-After` naming the reset at the next UTC midnight, and a spent
allowance `AllowanceExhausted`, 429.

The units pay for the copy that stands. A copy that finds the pinned bytes already stored, made
by a concurrent pin, gives its units back, since that pin's units pay for it, and so does a copy
whose working bytes are gone, since nothing landed. A copy that stands keeps its units through
any later failure of the write that asked for it, since the pinned bytes stay. A caller that
skips the copy for an artifact already pinned spends nothing.
"""

from __future__ import annotations

import logging
from datetime import datetime
from types import MappingProxyType

from artifact_store import ArtifactNotFoundError, ArtifactStore, StoredArtifact, Tier
from datasets_api.allowance import (
    BUDGET_KEY,
    AllowanceExhausted,
    AllowanceKind,
    SharedAllowanceLedger,
    Spend,
    seconds_to_reset,
)
from dfo_platform.caller import Caller
from dfo_platform.problem import Refusal

__all__ = ["OpenPinBudgetExhausted", "PinBudgetExhausted", "copy_to_pinned"]

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


_BUDGET_REFUSALS = {
    AllowanceKind.pin_budget: PinBudgetExhausted,
    AllowanceKind.open_pin_budget: OpenPinBudgetExhausted,
}


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
        DatasetsStoreUnavailable: The units could not be spent. Nothing is copied.
        ArtifactNotFoundError: The working bytes are gone. The units are given back.
    """
    if allowances is None:
        return artifacts.copy(artifact_digest, from_tier=Tier.working, to_tier=Tier.pinned)
    spends = [Spend(BUDGET_KEY, AllowanceKind.pin_budget)]
    if caller is not None:
        spends += [
            Spend(BUDGET_KEY, AllowanceKind.open_pin_budget),
            Spend(caller.key, AllowanceKind.pin),
        ]
    try:
        allowances.spend_together(spends, now)
    except AllowanceExhausted as exhausted:
        budget = _BUDGET_REFUSALS.get(exhausted.kind)
        if budget is None:
            raise
        raise budget(limit=exhausted.limit, now=now) from None
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
