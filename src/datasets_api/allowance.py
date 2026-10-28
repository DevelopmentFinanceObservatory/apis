"""Daily allowances per caller, kept in `datasets.caller_allowance`.

A caller (`dfo_platform.caller.Caller`) holds one allowance of each `AllowanceKind` per UTC day:
`share` (`DATASETS_SHARE_DAILY_ALLOWANCE`, default 10) for share writes, and
`pin` (`DATASETS_PIN_DAILY_ALLOWANCE`, default 50) for the pins of callers without the pin
grant. `AllowanceLedger.spend` takes one unit, or refuses `AllowanceExhausted` (429
`allowance_exhausted`) with `Retry-After` counting the seconds to the next UTC midnight, when
every allowance resets. A spent unit stays spent when the write it admitted later fails.

`LiveAllowanceLedger.spend` is one upsert that adds one to the day's row only while it is below
the limit, so spends racing on one caller cannot pass it.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
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
    }


class AllowanceLedger(Protocol):
    def spend(self, caller: Caller, kind: AllowanceKind, now: datetime) -> None:
        """Spend one unit of `caller`'s `kind` allowance for the UTC day of `now`.

        Raises:
            AllowanceExhausted: The day's allowance is spent.
            DatasetsStoreUnavailable: The ledger could not be read or written.
        """
        ...


def _utc_day(now: datetime) -> date:
    return now.astimezone(UTC).date()


class LiveAllowanceLedger:
    """`AllowanceLedger` over `datasets.caller_allowance`."""

    def __init__(self, *, store: DatasetStore, limits: Mapping[AllowanceKind, int]) -> None:
        self._store = store
        self._limits = dict(limits)

    def spend(self, caller: Caller, kind: AllowanceKind, now: datetime) -> None:
        limit = self._limits[kind]
        if limit <= 0:
            raise AllowanceExhausted(kind=kind, limit=limit, now=now)
        failure: str | None = None
        sqlstate: str | None = None
        row = None
        try:
            with self._store.transaction() as tx:
                # A day's first spend inserts 1. Later ones add one while the count is below the
                # limit, and at the limit the update is skipped and returns no row.
                row = tx.fetch_one(
                    "INSERT INTO datasets.caller_allowance (caller_key, kind, day, count) "
                    "VALUES (%s, %s, %s, 1) "
                    "ON CONFLICT (caller_key, kind, day) DO UPDATE "
                    "SET count = datasets.caller_allowance.count + 1 "
                    "WHERE datasets.caller_allowance.count < %s "
                    "RETURNING count",
                    (caller.key, kind.value, _utc_day(now), limit),
                )
        except Exception as exc:
            # Raise after leaving the handler so the wrapped error has no `__context__`.
            failure = (
                f"spending the {kind} allowance of caller {caller.key!r} failed: "
                f"{safe_diagnostics(exc)}"
            )
            sqlstate = extract_sqlstate(exc)
        if failure is not None:
            raise DatasetsStoreUnavailable(failure, sqlstate=sqlstate)
        if row is None:
            raise AllowanceExhausted(kind=kind, limit=limit, now=now)


__all__ = [
    "AllowanceExhausted",
    "AllowanceKind",
    "AllowanceLedger",
    "LiveAllowanceLedger",
    "allowance_limits",
    "seconds_to_reset",
]
