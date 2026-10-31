"""Daily allowances per caller, kept in `datasets.caller_allowance`.

A caller (`dfo_platform.caller.Caller`) holds one allowance of each caller kind per UTC day:
`share` (`DATASETS_SHARE_DAILY_ALLOWANCE`, default 10) for share writes, `card`
(`DATASETS_CARD_DAILY_ALLOWANCE`, default 500) for Presentation card creates and revisions, and
`pin` (`DATASETS_PIN_DAILY_ALLOWANCE`, default 100) for the pins of callers without the pin
grant. Two budgets every caller shares are kept the same way, under `BUDGET_KEY`: `pin_budget`
(`DATASETS_PIN_DAILY_MAX`, default 2,500), the new pinned copies the pin door and the share routes
make (`datasets_api.retention_admission`), and `open_pin_budget` (`DATASETS_PIN_OPEN_DAILY_MAX`,
default 2,000), those they make for callers without the pin grant. `AllowanceLedger.spend` takes one unit, or refuses `AllowanceExhausted`
(429 `allowance_exhausted`) with `Retry-After` counting the seconds to the next UTC midnight,
when every allowance and budget resets. `SharedAllowanceLedger.spend_together` takes one unit
of each of several, all or none, and refuses naming the first spent out. `AllowanceLedger.refund`
gives one unit back, `SharedAllowanceLedger.refund_together` gives back one of each of several,
all or none, and `spending` spends one around a write and refunds it when the write
fails, so a failed write costs its caller nothing. A pin's units stay spent when its copy
stands and a later step fails.

An export prepare claims one row of `datasets.export_prepare_claim` under its export id before
it spends a unit (`claim`), so one prepare runs across every instance. The row carries the units
the prepare holds (`hold_claim`, which spends them in the row's transaction) and the digest it
is about to copy (`name_claim_digest`), and a lease: a claim whose `expires_at` has passed is
taken over by the next prepare, units and digest included. Every write names the lease it holds,
so a prepare that outlived its lease changes nothing. `close_claim` deletes the row and gives
the units back in one transaction, so no unit returns twice. A unit a live claim holds counts
against its limit, and `holds_unit` tells it from a unit the day has spent.

`LiveAllowanceLedger` spends each unit with one upsert that adds one to the day's row only while
it is below the limit, so spends racing on one row cannot pass it. `spend_together` runs its
upserts in one transaction and rolls them all back when one finds its row at the limit. A
refund locks the day's row, then takes one from it or deletes the row it would leave at zero, so
a spend racing the refund waits for its commit. `refund_together` runs its refunds in one
transaction.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, replace
from datetime import UTC, date, datetime, time, timedelta
from enum import StrEnum
from types import MappingProxyType
from typing import Any, Protocol, cast

from dataset_store import DatasetStore, Transaction
from datasets_api.config import DatasetsBuildSettings
from datasets_api.store_errors import DatasetsStoreUnavailable
from dfo_platform.caller import Caller
from dfo_platform.problem import Refusal
from dfo_platform.store_errors import extract_sqlstate, safe_diagnostics

logger = logging.getLogger(__name__)


class AllowanceKind(StrEnum):
    share = "share"
    card = "card"
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


@dataclass(frozen=True, kw_only=True)
class ExportClaim:
    """One row of `datasets.export_prepare_claim`, read under the lease `expires_at`. `spends`
    are the units it holds, spent at `spent_at`, and `digest` the file a copy was begun for."""

    export_id: str
    expires_at: datetime
    spends: tuple[Spend, ...] = ()
    spent_at: datetime | None = None
    digest: str | None = None
    refusal: Mapping[str, Any] | None = None


def seconds_to_reset(now: datetime) -> int:
    """Whole seconds from `now` to the next UTC midnight, at least 1."""
    midnight = datetime.combine(now.astimezone(UTC).date() + timedelta(days=1), time(), UTC)
    return max(1, math.ceil((midnight - now).total_seconds()))


class AllowanceExhausted(Refusal):
    """The caller has spent today's allowance of one kind. Maps to 429 with `retryable: false`,
    since a client that waits for the reset holds a request open for hours, and `Retry-After`
    naming the reset at the next UTC midnight."""

    code = "allowance_exhausted"
    retryable = False

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
        AllowanceKind.card: settings.card_daily_allowance,
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

    def refund(self, caller: Caller, kind: AllowanceKind, now: datetime) -> None:
        """Give back one unit of `caller`'s `kind` allowance spent for the UTC day of `now`. A
        day with nothing spent stays at nothing.

        Raises:
            DatasetsStoreUnavailable: The ledger could not be written.
        """
        ...


@contextmanager
def spending(
    ledger: AllowanceLedger, caller: Caller, kind: AllowanceKind, now: datetime
) -> Iterator[None]:
    """Spend one unit of `caller`'s `kind` allowance for the block, and give it back when the
    block raises, so a retry of a failed write spends one unit in all. A refund that fails is
    logged, and the block's own error propagates.

    Raises:
        AllowanceExhausted: The day's allowance is spent. The block does not run.
        DatasetsStoreUnavailable: The spend could not be written.
    """
    ledger.spend(caller, kind, now)
    try:
        yield
    except BaseException:
        try:
            ledger.refund(caller, kind, now)
        except Exception:
            logger.exception("Refunding a %s allowance unit of %s failed", kind, caller.key)
        raise


class SharedAllowanceLedger(AllowanceLedger, Protocol):
    def spend_together(self, spends: Sequence[Spend], now: datetime) -> None:
        """Spend one unit of each of `spends` for the UTC day of `now`, all or none.

        Raises:
            AllowanceExhausted: The first of `spends` whose day's limit is spent. Nothing is
                spent.
            DatasetsStoreUnavailable: The ledger could not be read or written.
        """
        ...

    def refund_together(self, spends: Sequence[Spend], now: datetime) -> None:
        """Give back one unit of each of `spends` spent for the UTC day of `now`, all or none. A
        day with nothing spent stays at nothing.

        Raises:
            DatasetsStoreUnavailable: The ledger could not be written. Nothing is given back.
        """
        ...

    def holds_unit(self, spend: Spend, now: datetime) -> bool:
        """Whether a live export claim holds a unit of `spend` for the UTC day of `now`, so the
        unit can come back, in place of the day having spent it.

        Raises:
            DatasetsStoreUnavailable: The claims could not be read.
        """
        ...

    def read_claim(self, export_id: str) -> ExportClaim | None:
        """The claim `export_id` holds, or `None`.

        Raises:
            DatasetsStoreUnavailable: The claim could not be read.
        """
        ...

    def claim(self, export_id: str, *, now: datetime, expires_at: datetime) -> ExportClaim | None:
        """Take the claim of `export_id` until `expires_at`, inheriting the units and digest of
        one whose lease has passed at `now`, or return `None` while a lease runs.

        Raises:
            DatasetsStoreUnavailable: The claim could not be written.
        """
        ...

    def hold_claim(
        self, claim: ExportClaim, spends: Sequence[Spend], now: datetime
    ) -> ExportClaim | None:
        """Spend one unit of each of `spends` onto `claim`, all or none, and return the claim
        holding them, or `None` when its lease is lost, with nothing spent.

        Raises:
            AllowanceExhausted: The first of `spends` whose day's limit is spent.
            DatasetsStoreUnavailable: The units could not be written.
        """
        ...

    def name_claim_digest(self, claim: ExportClaim, digest: str) -> bool:
        """Write `digest` on `claim` and return whether its lease still held.

        Raises:
            DatasetsStoreUnavailable: The claim could not be written.
        """
        ...

    def free_claim(self, claim: ExportClaim, now: datetime) -> None:
        """End `claim`'s lease at `now` and keep the row, for a retry or the sweep to settle.

        Raises:
            DatasetsStoreUnavailable: The claim could not be written.
        """
        ...

    def refuse_claim(self, claim: ExportClaim, refusal: Mapping[str, Any]) -> None:
        """Give `claim`'s units back and leave `refusal` on it for the rest of its lease.

        Raises:
            DatasetsStoreUnavailable: The claim could not be written.
        """
        ...

    def close_claim(self, claim: ExportClaim, *, refund: bool) -> None:
        """Delete `claim`, giving its units back when `refund`, in one transaction.

        Raises:
            DatasetsStoreUnavailable: The claim could not be written.
        """
        ...

    def expired_claims(self, now: datetime) -> list[ExportClaim]:
        """Every claim whose lease has passed at `now`.

        Raises:
            DatasetsStoreUnavailable: The claims could not be read.
        """
        ...


class _LimitReached(Exception):
    """Rolls back the spends of a transaction made before one found its row at the limit."""

    def __init__(self, kind: AllowanceKind) -> None:
        self.kind = kind


class _ClaimLost(Exception):
    """Rolls back the spends of `hold_claim` when its claim's lease has been lost."""


def _utc_day(now: datetime) -> date:
    return now.astimezone(UTC).date()


class LiveAllowanceLedger:
    """`AllowanceLedger` over `datasets.caller_allowance`."""

    def __init__(self, *, store: DatasetStore, limits: Mapping[AllowanceKind, int]) -> None:
        self._store = store
        self._limits = dict(limits)

    def spend(self, caller: Caller, kind: AllowanceKind, now: datetime) -> None:
        self.spend_together([Spend(caller.key, kind)], now)

    def _refuse_zero_limits(self, spends: Sequence[Spend], now: datetime) -> None:
        for spend in spends:
            if self._limits[spend.kind] <= 0:
                raise AllowanceExhausted(kind=spend.kind, limit=self._limits[spend.kind], now=now)

    def _spend_in(self, tx: Transaction, spends: Sequence[Spend], now: datetime) -> None:
        for spend in spends:
            # A day's first spend inserts 1. Later ones add one while the count is below the
            # limit, and at the limit the update is skipped and returns no row.
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
                raise _LimitReached(spend.kind)

    def spend_together(self, spends: Sequence[Spend], now: datetime) -> None:
        self._refuse_zero_limits(spends, now)
        exhausted: AllowanceKind | None = None
        failure: str | None = None
        sqlstate: str | None = None
        try:
            with self._store.transaction() as tx:
                self._spend_in(tx, spends, now)
        except _LimitReached as reached:
            exhausted = reached.kind
        except Exception as exc:
            # Raise after leaving the handler so the wrapped error has no `__context__`.
            spent = ", ".join(f"{spend.kind} of {spend.key!r}" for spend in spends)
            failure = f"spending {spent} failed: {safe_diagnostics(exc)}"
            sqlstate = extract_sqlstate(exc)
        if failure is not None:
            raise DatasetsStoreUnavailable(failure, sqlstate=sqlstate)
        if exhausted is not None:
            raise AllowanceExhausted(kind=exhausted, limit=self._limits[exhausted], now=now)

    def refund(self, caller: Caller, kind: AllowanceKind, now: datetime) -> None:
        self.refund_together([Spend(caller.key, kind)], now)

    def refund_together(self, spends: Sequence[Spend], now: datetime) -> None:
        failure: str | None = None
        sqlstate: str | None = None
        try:
            with self._store.transaction() as tx:
                self._refund_in(tx, spends, now)
        except Exception as exc:
            refunded = ", ".join(f"{spend.kind} of {spend.key!r}" for spend in spends)
            failure = f"refunding {refunded} failed: {safe_diagnostics(exc)}"
            sqlstate = extract_sqlstate(exc)
        if failure is not None:
            raise DatasetsStoreUnavailable(failure, sqlstate=sqlstate)

    def _refund_in(self, tx: Transaction, spends: Sequence[Spend], now: datetime) -> None:
        for spend in spends:
            params = (spend.key, spend.kind.value, _utc_day(now))
            # The row holds a positive count, so the last unit deletes it. The lock holds the
            # count read here until commit, so a racing spend waits.
            row = tx.fetch_one(
                "SELECT count FROM datasets.caller_allowance "
                "WHERE caller_key = %s AND kind = %s AND day = %s FOR UPDATE",
                params,
            )
            if row is None:
                continue
            if row["count"] == 1:
                tx.execute(
                    "DELETE FROM datasets.caller_allowance "
                    "WHERE caller_key = %s AND kind = %s AND day = %s",
                    params,
                )
            else:
                tx.execute(
                    "UPDATE datasets.caller_allowance SET count = count - 1 "
                    "WHERE caller_key = %s AND kind = %s AND day = %s",
                    params,
                )

    @contextmanager
    def _claim_transaction(self, export_id: str) -> Iterator[Transaction]:
        """A transaction over `export_id`'s claim. A store failure raises after the handler
        exits, so the wrapped error has no `__context__`."""
        failure: str | None = None
        sqlstate: str | None = None
        try:
            with self._store.transaction() as tx:
                yield tx
        except (_LimitReached, _ClaimLost):
            raise
        except Exception as exc:
            failure = f"writing the claim of {export_id!r} failed: {safe_diagnostics(exc)}"
            sqlstate = extract_sqlstate(exc)
        if failure is not None:
            raise DatasetsStoreUnavailable(failure, sqlstate=sqlstate)

    def holds_unit(self, spend: Spend, now: datetime) -> bool:
        with self._claim_transaction(spend.key) as tx:
            rows = tx.fetch_all(
                "SELECT spends, spent_at FROM datasets.export_prepare_claim "
                "WHERE expires_at > %s AND spent_at IS NOT NULL",
                (now,),
            )
        return any(
            cast(Mapping[str, str], row["spends"]).get(spend.kind.value) == spend.key
            and _utc_day(cast(datetime, row["spent_at"])) == _utc_day(now)
            for row in rows
        )

    def read_claim(self, export_id: str) -> ExportClaim | None:
        with self._claim_transaction(export_id) as tx:
            row = tx.fetch_one(
                f"SELECT {_CLAIM_COLUMNS} FROM datasets.export_prepare_claim WHERE export_id = %s",
                (export_id,),
            )
        return None if row is None else _claim(row)

    def claim(self, export_id: str, *, now: datetime, expires_at: datetime) -> ExportClaim | None:
        with self._claim_transaction(export_id) as tx:
            row = tx.fetch_one(
                "INSERT INTO datasets.export_prepare_claim (export_id, expires_at) "
                "VALUES (%s, %s) "
                "ON CONFLICT (export_id) DO UPDATE "
                "SET expires_at = EXCLUDED.expires_at, refusal = NULL "
                "WHERE datasets.export_prepare_claim.expires_at <= %s "
                f"RETURNING {_CLAIM_COLUMNS}",
                (export_id, expires_at, now),
            )
        return None if row is None else _claim(row)

    def hold_claim(
        self, claim: ExportClaim, spends: Sequence[Spend], now: datetime
    ) -> ExportClaim | None:
        self._refuse_zero_limits(spends, now)
        try:
            with self._claim_transaction(claim.export_id) as tx:
                self._spend_in(tx, spends, now)
                row = tx.fetch_one(
                    "UPDATE datasets.export_prepare_claim SET spends = %s, spent_at = %s "
                    "WHERE export_id = %s AND expires_at = %s RETURNING export_id",
                    (
                        {spend.kind.value: spend.key for spend in spends},
                        now,
                        claim.export_id,
                        claim.expires_at,
                    ),
                )
                if row is None:
                    raise _ClaimLost
        except _LimitReached as reached:
            exhausted = AllowanceExhausted(
                kind=reached.kind, limit=self._limits[reached.kind], now=now
            )
        except _ClaimLost:
            return None
        else:
            return replace(claim, spends=tuple(spends), spent_at=now)
        raise exhausted

    def _update_claim(self, claim: ExportClaim, assignment: str, value: object) -> bool:
        with self._claim_transaction(claim.export_id) as tx:
            row = tx.fetch_one(
                f"UPDATE datasets.export_prepare_claim SET {assignment} "
                "WHERE export_id = %s AND expires_at = %s RETURNING export_id",
                (value, claim.export_id, claim.expires_at),
            )
        return row is not None

    def name_claim_digest(self, claim: ExportClaim, digest: str) -> bool:
        return self._update_claim(claim, "artifact_digest = %s", digest)

    def free_claim(self, claim: ExportClaim, now: datetime) -> None:
        self._update_claim(claim, "expires_at = %s", now)

    def refuse_claim(self, claim: ExportClaim, refusal: Mapping[str, Any]) -> None:
        with self._claim_transaction(claim.export_id) as tx:
            row = tx.fetch_one(
                "UPDATE datasets.export_prepare_claim "
                "SET refusal = %s, spends = %s, spent_at = NULL "
                "WHERE export_id = %s AND expires_at = %s RETURNING export_id",
                (dict(refusal), {}, claim.export_id, claim.expires_at),
            )
            if row is not None and claim.spent_at is not None:
                self._refund_in(tx, claim.spends, claim.spent_at)

    def close_claim(self, claim: ExportClaim, *, refund: bool) -> None:
        with self._claim_transaction(claim.export_id) as tx:
            row = tx.fetch_one(
                "DELETE FROM datasets.export_prepare_claim "
                "WHERE export_id = %s AND expires_at = %s RETURNING export_id",
                (claim.export_id, claim.expires_at),
            )
            if refund and row is not None and claim.spent_at is not None:
                self._refund_in(tx, claim.spends, claim.spent_at)

    def expired_claims(self, now: datetime) -> list[ExportClaim]:
        with self._claim_transaction("expired claims") as tx:
            rows = tx.fetch_all(
                f"SELECT {_CLAIM_COLUMNS} FROM datasets.export_prepare_claim "
                "WHERE expires_at <= %s",
                (now,),
            )
        return [_claim(row) for row in rows]


_CLAIM_COLUMNS = "export_id, expires_at, spends, spent_at, artifact_digest, refusal"


def _claim(row: Mapping[str, Any]) -> ExportClaim:
    return ExportClaim(
        export_id=row["export_id"],
        expires_at=row["expires_at"],
        spends=tuple(Spend(key, AllowanceKind(kind)) for kind, key in row["spends"].items()),
        spent_at=row["spent_at"],
        digest=row["artifact_digest"],
        refusal=row["refusal"],
    )


__all__ = [
    "BUDGET_KEY",
    "AllowanceExhausted",
    "AllowanceKind",
    "AllowanceLedger",
    "ExportClaim",
    "LiveAllowanceLedger",
    "SharedAllowanceLedger",
    "Spend",
    "allowance_limits",
    "seconds_to_reset",
    "spending",
]
