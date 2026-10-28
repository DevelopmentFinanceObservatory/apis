"""Daily allowances per caller, kept in `datasets.caller_allowance`.

A caller (`dfo_platform.caller.Caller`) holds one allowance of each caller kind per UTC day:
`share` (`DATASETS_SHARE_DAILY_ALLOWANCE`, default 10) for share writes, and
`pin` (`DATASETS_PIN_DAILY_ALLOWANCE`, default 50) for the pins of callers without the pin
grant. Two budgets every caller shares are kept the same way, under `BUDGET_KEY`: `pin_budget`
(`DATASETS_PIN_DAILY_MAX`, default 500), the new pinned copies the pin door makes, and
`open_pin_budget` (`DATASETS_PIN_OPEN_DAILY_MAX`, default 200), those it makes for callers
without the pin grant. `AllowanceLedger.spend` takes one unit, or refuses `AllowanceExhausted`
(429 `allowance_exhausted`) with `Retry-After` counting the seconds to the next UTC midnight,
when every allowance and budget resets. `SharedAllowanceLedger.spend_together` takes one unit
of each of several, all or none, and refuses naming the first spent out. A spent unit stays
spent when the write it admitted later fails.

`LiveAllowanceLedger` spends each unit with one upsert that adds one to the day's row only while
it is below the limit, so spends racing on one row cannot pass it. `spend_together` runs its
upserts in one transaction and rolls them all back when one finds its row at the limit.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
from enum import StrEnum
from types import MappingProxyType
from typing import Protocol

from dataset_store import DatasetStore
from datasets_api.config import DatasetsBuildSettings
from datasets_api.store_errors import DatasetsStoreUnavailable
from dfo_platform.caller import Caller
from dfo_platform.problem import Refusal
from dfo_platform.store_errors import extract_sqlstate, safe_diagnostics


class AllowanceKind(StrEnum):
    share = "share"
    pin = "pin"
    pin_budget = "pin_budget"
    open_pin_budget = "open_pin_budget"


BUDGET_KEY = "all"
"""The key a budget every caller shares is kept under, beside the `ip:` keys of callers."""


@dataclass(frozen=True)
class Spend:
    """One unit of the `kind` allowance or budget kept under `key`."""

    key: str
    kind: AllowanceKind


def seconds_to_reset(now: datetime) -> int:
    """Whole seconds from `now` to the next UTC midnight, at least 1."""
    midnight = datetime.combine(now.astimezone(UTC).date() + timedelta(days=1), time(), UTC)
    return max(1, math.ceil((midnight - now).total_seconds()))


class AllowanceExhausted(Refusal):
    """The caller has spent today's allowance of one kind. Maps to retryable 429 with
    `Retry-After` set to the next UTC midnight."""

    code = "allowance_exhausted"
    retryable = True

    def __init__(self, *, kind: AllowanceKind, limit: int, now: datetime) -> None:
        self.kind = kind
        self.limit = limit
        self.headers = MappingProxyType({"Retry-After": str(seconds_to_reset(now))})
        super().__init__(
            f"this caller has spent today's {kind} allowance of {limit}; it resets at the next "
            "UTC midnight"
        )


def allowance_limits(settings: DatasetsBuildSettings) -> Mapping[AllowanceKind, int]:
    """Each kind's daily limit from `settings`."""
    return {
        AllowanceKind.share: settings.share_daily_allowance,
        AllowanceKind.pin: settings.pin_daily_allowance,
        AllowanceKind.pin_budget: settings.pin_daily_max,
        AllowanceKind.open_pin_budget: settings.pin_open_daily_max,
    }


class AllowanceLedger(Protocol):
    def spend(self, caller: Caller, kind: AllowanceKind, now: datetime) -> None:
        """Spend one unit of `caller`'s `kind` allowance for the UTC day of `now`.

        Raises:
            AllowanceExhausted: The day's allowance is spent.
            DatasetsStoreUnavailable: The ledger could not be read or written.
        """
        ...


class SharedAllowanceLedger(AllowanceLedger, Protocol):
    def spend_together(self, spends: Sequence[Spend], now: datetime) -> None:
        """Spend one unit of each of `spends` for the UTC day of `now`, all or none.

        Raises:
            AllowanceExhausted: The first of `spends` whose day's limit is spent. Nothing is
                spent.
            DatasetsStoreUnavailable: The ledger could not be read or written.
        """
        ...


class _LimitReached(Exception):
    """Rolls back the spends of `spend_together` made before one found its row at the limit."""


def _utc_day(now: datetime) -> date:
    return now.astimezone(UTC).date()


class LiveAllowanceLedger:
    """`AllowanceLedger` over `datasets.caller_allowance`."""

    def __init__(self, *, store: DatasetStore, limits: Mapping[AllowanceKind, int]) -> None:
        self._store = store
        self._limits = dict(limits)

    def spend(self, caller: Caller, kind: AllowanceKind, now: datetime) -> None:
        self.spend_together([Spend(caller.key, kind)], now)

    def spend_together(self, spends: Sequence[Spend], now: datetime) -> None:
        for spend in spends:
            if self._limits[spend.kind] <= 0:
                raise AllowanceExhausted(kind=spend.kind, limit=self._limits[spend.kind], now=now)
        exhausted: AllowanceKind | None = None
        failure: str | None = None
        sqlstate: str | None = None
        try:
            with self._store.transaction() as tx:
                for spend in spends:
                    # A day's first spend inserts 1. Later ones add one while the count is below
                    # the limit, and at the limit the update is skipped and returns no row.
                    row = tx.fetch_one(
                        "INSERT INTO datasets.caller_allowance (caller_key, kind, day, count) "
                        "VALUES (%s, %s, %s, 1) "
                        "ON CONFLICT (caller_key, kind, day) DO UPDATE "
                        "SET count = datasets.caller_allowance.count + 1 "
                        "WHERE datasets.caller_allowance.count < %s "
                        "RETURNING count",
                        (spend.key, spend.kind.value, _utc_day(now), self._limits[spend.kind]),
                    )
                    if row is None:
                        exhausted = spend.kind
                        raise _LimitReached
        except _LimitReached:
            pass
        except Exception as exc:
            # Raise after leaving the handler so the wrapped error has no `__context__`.
            spent = ", ".join(f"{spend.kind} of {spend.key!r}" for spend in spends)
            failure = f"spending {spent} failed: {safe_diagnostics(exc)}"
            sqlstate = extract_sqlstate(exc)
        if failure is not None:
            raise DatasetsStoreUnavailable(failure, sqlstate=sqlstate)
        if exhausted is not None:
            raise AllowanceExhausted(kind=exhausted, limit=self._limits[exhausted], now=now)


__all__ = [
    "BUDGET_KEY",
    "AllowanceExhausted",
    "AllowanceKind",
    "AllowanceLedger",
    "LiveAllowanceLedger",
    "SharedAllowanceLedger",
    "Spend",
    "allowance_limits",
    "seconds_to_reset",
]
