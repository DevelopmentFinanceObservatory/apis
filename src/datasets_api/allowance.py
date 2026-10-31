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

An export prepare runs under one row of `datasets.export_prepare_claim` under its export id, so
one prepare runs across every instance. The row is a state machine, and a unit on it is a unit
spent:

| State     | Units held | Means                                                     |
| :-------- | :--------- | :-------------------------------------------------------- |
| `open`    | none       | A prepare runs and has spent nothing.                     |
| `held`    | yes        | Units are spent ahead of a render. No copy has begun.     |
| `copying` | yes        | A copy of `digest` began on them. Its outcome is unknown. |
| `landed`  | yes        | This claim's copy of `digest` stands, so they paid for it.|
| `refused` | none       | The request's own file refused, for the prepare's waiters.|

| Transition    | From                      | To       | Units                 |
| :------------ | :------------------------ | :------- | :-------------------- |
| `claim`       | no row                    | `open`   | none                  |
| `claim`       | `refused`, stale `open`   | `open`   | none                  |
| `claim`       | stale `held`              | `open`   | refunded              |
| `claim`       | stale `copying`, `landed` | the same | kept, new lease       |
| `hold_claim`  | `open`                    | `held`   | spent                 |
| `begin_copy`  | `held`, `landed`          | `copying`| kept                  |
| `settle_copy` | `copying`, it landed      | `landed` | kept                  |
| `settle_copy` | `copying`, it did not     | `open`   | refunded              |
| `refuse_claim`| `open`, `held`            | `refused`| refunded              |
| `free_claim`  | any                       | the same | kept, lease ended     |
| `close_claim` | any                       | no row   | refunded from `held` and `copying` |

Each transition is one transaction that names the `generation` the writer read, every one of
them writes a new random value, so a value never comes back even after a row is deleted and
claimed again, and any refund reads the units from the row inside that transaction. A writer
acting on an old read, a holder that lost its lease or a sweep behind a hold, changes nothing and
refunds nothing. `close_claim` can also require the lease to be stale, so a sweep cannot close a
claim a heartbeat revived. `renew_claim` moves `expires_at` alone: a running prepare's heartbeat
keeps its lease without touching the fence, and only a stale lease lets another writer in.

A limit-reached spend reads the claims inside its own transaction, whose skipped upsert keeps the
day's row locked, so no refund can land between the failed spend and the verdict. A unit on a live
claim, or on any `copying` one, may come back, so it raises `AllowanceExhausted` with `held` set.
A unit on a stale `held` claim is settled and the spend tried once more, so no refusal stands on a
recoverable unit. Spends and refunds lock rows in one order (`_in_lock_order`), and a claim's row
before any of its units.

`LiveAllowanceLedger` spends each unit with one upsert that adds one to the day's row only while
it is below the limit, so spends racing on one row cannot pass it. `spend_together` runs its
upserts in one transaction and rolls them all back when one finds its row at the limit. A
refund locks the day's row, then takes one from it or deletes the row it would leave at zero, so
a spend racing the refund waits for its commit. `refund_together` runs its refunds in one
transaction.
"""

from __future__ import annotations

import contextlib
import logging
import math
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
from enum import StrEnum
from types import MappingProxyType
from typing import Any, Protocol
from uuid import uuid4

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


class ClaimState(StrEnum):
    open = "open"
    held = "held"
    copying = "copying"
    landed = "landed"
    refused = "refused"


@dataclass(frozen=True, kw_only=True)
class ExportClaim:
    """One read of a row of `datasets.export_prepare_claim`. See the module docstring."""

    export_id: str
    generation: str
    state: ClaimState
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
    naming the reset at the next UTC midnight. `held` says a live export claim holds a unit, so
    the limit can free again today."""

    code = "allowance_exhausted"
    retryable = False

    def __init__(
        self, *, kind: AllowanceKind, limit: int, now: datetime, held: bool = False
    ) -> None:
        self.kind = kind
        self.limit = limit
        self.held = held
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

    def read_claim(self, export_id: str) -> ExportClaim | None:
        """The claim `export_id` holds, or `None`.

        Raises:
            DatasetsStoreUnavailable: The claim could not be read.
        """
        ...

    def claim(self, export_id: str, *, now: datetime, expires_at: datetime) -> ExportClaim | None:
        """Take the claim of `export_id` until `expires_at`, or return `None` while a live
        prepare holds it. A stale claim passes on as the module docstring's table says.

        Raises:
            DatasetsStoreUnavailable: The claim could not be written.
        """
        ...

    def renew_claim(self, claim: ExportClaim, expires_at: datetime) -> None:
        """Move `claim`'s lease to `expires_at`, unless another writer moved its state.

        Raises:
            DatasetsStoreUnavailable: The claim could not be written.
        """
        ...

    def hold_claim(
        self, claim: ExportClaim, spends: Sequence[Spend], now: datetime
    ) -> ExportClaim | None:
        """Move `claim` from `open` to `held`, spending one unit of each of `spends` onto it, all
        or none. Return `None` when its generation is gone, with nothing spent.

        Raises:
            AllowanceExhausted: The first of `spends` whose day's limit is reached.
            DatasetsStoreUnavailable: The units could not be written.
        """
        ...

    def begin_copy(self, claim: ExportClaim, digest: str) -> ExportClaim | None:
        """Move `claim` to `copying` for `digest`, or return `None` when it is not this
        claim's to copy on any more.

        Raises:
            DatasetsStoreUnavailable: The claim could not be written.
        """
        ...

    def settle_copy(self, claim: ExportClaim, *, landed: bool) -> ExportClaim | None:
        """Move `claim` from `copying` to `landed`, or to `open` with its units refunded. Return
        `None` when its generation is gone.

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
        """Move `claim` to `refused`, refunding its units and leaving `refusal` for its waiters.

        Raises:
            DatasetsStoreUnavailable: The claim could not be written.
        """
        ...

    def close_claim(self, claim: ExportClaim, *, stale_at: datetime | None = None) -> None:
        """Delete `claim`, refunding the units of a `held` or `copying` row. With `stale_at`, only
        while its lease has passed at that time.

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
    """Rolls back a transaction's spends when one finds its row at the limit. `held` says a live
    claim holds a unit of `spend`, and `stale` that a stale `held` claim does."""

    def __init__(self, spend: Spend, *, held: bool, stale: bool) -> None:
        self.spend = spend
        self.held = held
        self.stale = stale


class _ClaimLost(Exception):
    """Rolls back a transition whose generation or state has gone."""


_LOCK_ORDER = {
    kind: rank
    for rank, kind in enumerate(
        (
            AllowanceKind.pin_budget,
            AllowanceKind.open_pin_budget,
            AllowanceKind.pin,
            AllowanceKind.share,
            AllowanceKind.card,
        )
    )
}


def _in_lock_order(spends: Iterable[Spend]) -> list[Spend]:
    """`spends` in the one order every spend and refund takes their rows' locks."""
    return sorted(spends, key=lambda spend: (_LOCK_ORDER[spend.kind], spend.key))


def utc_day(now: datetime) -> date:
    return now.astimezone(UTC).date()


class LiveAllowanceLedger:
    """`AllowanceLedger` over `datasets.caller_allowance`."""

    def __init__(self, *, store: DatasetStore, limits: Mapping[AllowanceKind, int]) -> None:
        self._store = store
        self._limits = dict(limits)

    def spend(self, caller: Caller, kind: AllowanceKind, now: datetime) -> None:
        self.spend_together([Spend(caller.key, kind)], now)

    @contextmanager
    def _transaction(self, doing: str) -> Iterator[Transaction]:
        """A transaction whose store failure raises `DatasetsStoreUnavailable` after the handler
        exits, so the wrapped error has no `__context__`."""
        failure: str | None = None
        sqlstate: str | None = None
        try:
            with self._store.transaction() as tx:
                yield tx
        except (_LimitReached, _ClaimLost):
            raise
        except Exception as exc:
            failure = f"{doing} failed: {safe_diagnostics(exc)}"
            sqlstate = extract_sqlstate(exc)
        if failure is not None:
            raise DatasetsStoreUnavailable(failure, sqlstate=sqlstate)

    def _spend_in(self, tx: Transaction, spends: Sequence[Spend], now: datetime) -> None:
        for spend in _in_lock_order(spends):
            # A day's first spend inserts 1. Later ones add one while the count is below the
            # limit, and at the limit the update is skipped, keeps the row locked and returns
            # no row.
            row = tx.fetch_one(
                "INSERT INTO datasets.caller_allowance (caller_key, kind, day, count) "
                "VALUES (%s, %s, %s, 1) "
                "ON CONFLICT (caller_key, kind, day) DO UPDATE "
                "SET count = datasets.caller_allowance.count + 1 "
                "WHERE datasets.caller_allowance.count < %s "
                "RETURNING count",
                (spend.key, spend.kind.value, utc_day(now), self._limits[spend.kind]),
            )
            if row is None:
                claims = self._claims_on(tx, spend, now)
                raise _LimitReached(
                    spend,
                    held=any(
                        claim["expires_at"] > now or claim["state"] == ClaimState.copying
                        for claim in claims
                    ),
                    stale=any(
                        claim["expires_at"] <= now and claim["state"] == ClaimState.held
                        for claim in claims
                    ),
                )

    def _claims_on(
        self, tx: Transaction, spend: Spend, now: datetime
    ) -> Sequence[Mapping[str, Any]]:
        """The lease and state of each claim whose unit of `spend` for `now`'s UTC day may
        come back."""
        midnight = datetime.combine(utc_day(now), time(), UTC)
        return tx.fetch_all(
            "SELECT expires_at, state FROM datasets.export_prepare_claim "
            f"WHERE {_HOLDS_UNIT} AND spent_at >= %s AND spent_at < %s AND state IN (%s, %s)",
            (
                spend.kind.value,
                spend.key,
                midnight,
                midnight + timedelta(days=1),
                ClaimState.held.value,
                ClaimState.copying.value,
            ),
        )

    def _settle_stale(self, spend: Spend, now: datetime) -> None:
        """Close each stale `held` claim on a unit of `spend`, which began no copy."""
        with self._transaction(f"settling stale claims on {spend.kind}") as tx:
            rows = tx.fetch_all(
                "DELETE FROM datasets.export_prepare_claim "
                f"WHERE {_HOLDS_UNIT} AND state = %s AND expires_at <= %s "
                "RETURNING spends, spent_at",
                (spend.kind.value, spend.key, ClaimState.held.value, now),
            )
            for row in rows:
                self._refund_row(tx, row)

    def _spending[T](
        self,
        spends: Sequence[Spend],
        now: datetime,
        *,
        doing: str,
        write: Callable[[Transaction], T],
    ) -> T:
        """Run `write`, which spends `spends`, in one transaction, and once more after settling
        a stale claim the limit found.

        Raises:
            AllowanceExhausted: The first of `spends` whose day's limit is reached.
            DatasetsStoreUnavailable: The units could not be written.
        """
        for spend in spends:
            if self._limits[spend.kind] <= 0:
                raise AllowanceExhausted(kind=spend.kind, limit=self._limits[spend.kind], now=now)
        settled = False
        while True:
            try:
                with self._transaction(doing) as tx:
                    return write(tx)
            except _LimitReached as reached:
                if settled or not reached.stale:
                    raise AllowanceExhausted(
                        kind=reached.spend.kind,
                        limit=self._limits[reached.spend.kind],
                        now=now,
                        held=reached.held,
                    ) from None
                self._settle_stale(reached.spend, now)
                settled = True

    def spend_together(self, spends: Sequence[Spend], now: datetime) -> None:
        spent = ", ".join(f"{spend.kind} of {spend.key!r}" for spend in spends)
        self._spending(
            spends,
            now,
            doing=f"spending {spent}",
            write=lambda tx: self._spend_in(tx, spends, now),
        )

    def refund(self, caller: Caller, kind: AllowanceKind, now: datetime) -> None:
        self.refund_together([Spend(caller.key, kind)], now)

    def refund_together(self, spends: Sequence[Spend], now: datetime) -> None:
        refunded = ", ".join(f"{spend.kind} of {spend.key!r}" for spend in spends)
        with self._transaction(f"refunding {refunded}") as tx:
            self._refund_in(tx, spends, now)

    def _refund_in(self, tx: Transaction, spends: Sequence[Spend], now: datetime) -> None:
        for spend in _in_lock_order(spends):
            params = (spend.key, spend.kind.value, utc_day(now))
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

    def _refund_row(self, tx: Transaction, row: Mapping[str, Any]) -> None:
        """Refund the units a claim's row held as this transaction read it."""
        if row["spent_at"] is not None:
            self._refund_in(tx, _spends(row["spends"]), row["spent_at"])

    def read_claim(self, export_id: str) -> ExportClaim | None:
        with self._transaction(f"reading the claim of {export_id!r}") as tx:
            row = tx.fetch_one(
                f"SELECT {_CLAIM_COLUMNS} FROM datasets.export_prepare_claim WHERE export_id = %s",
                (export_id,),
            )
        return None if row is None else _claim(row)

    def claim(self, export_id: str, *, now: datetime, expires_at: datetime) -> ExportClaim | None:
        with self._transaction(f"claiming {export_id!r}") as tx:
            ended = tx.fetch_one(
                "DELETE FROM datasets.export_prepare_claim WHERE export_id = %s "
                "AND (state = %s OR (state IN (%s, %s) AND expires_at <= %s)) "
                "RETURNING spends, spent_at",
                (
                    export_id,
                    ClaimState.refused.value,
                    ClaimState.open.value,
                    ClaimState.held.value,
                    now,
                ),
            )
            if ended is not None:
                self._refund_row(tx, ended)
            row = tx.fetch_one(
                "INSERT INTO datasets.export_prepare_claim (export_id, generation, expires_at) "
                "VALUES (%s, %s, %s) ON CONFLICT (export_id) DO NOTHING "
                f"RETURNING {_CLAIM_COLUMNS}",
                (export_id, uuid4().hex, expires_at),
            )
            if row is None:
                # A stale row left here began a copy, which only a settle can judge.
                row = tx.fetch_one(
                    "UPDATE datasets.export_prepare_claim "
                    "SET generation = %s, expires_at = %s "
                    "WHERE export_id = %s AND expires_at <= %s "
                    f"RETURNING {_CLAIM_COLUMNS}",
                    (uuid4().hex, expires_at, export_id, now),
                )
        return None if row is None else _claim(row)

    def renew_claim(self, claim: ExportClaim, expires_at: datetime) -> None:
        with self._transaction(f"renewing the claim of {claim.export_id!r}") as tx:
            tx.execute(
                "UPDATE datasets.export_prepare_claim SET expires_at = %s "
                "WHERE export_id = %s AND generation = %s",
                (expires_at, claim.export_id, claim.generation),
            )

    def _lock(self, tx: Transaction, claim: ExportClaim, *states: ClaimState) -> Mapping[str, Any]:
        """The row of `claim`, locked, while it stays at this generation and in `states`."""
        row = tx.fetch_one(
            "SELECT state, spends, spent_at FROM datasets.export_prepare_claim "
            "WHERE export_id = %s AND generation = %s FOR UPDATE",
            (claim.export_id, claim.generation),
        )
        if row is None or row["state"] not in states:
            raise _ClaimLost
        return row

    def _move(
        self, tx: Transaction, claim: ExportClaim, assignments: str, params: Sequence[object]
    ) -> ExportClaim:
        """Write `assignments` on `claim`'s row, moving its generation."""
        row = tx.fetch_one(
            "UPDATE datasets.export_prepare_claim "
            f"SET generation = %s, {assignments} "
            f"WHERE export_id = %s AND generation = %s RETURNING {_CLAIM_COLUMNS}",
            (uuid4().hex, *params, claim.export_id, claim.generation),
        )
        if row is None:
            raise _ClaimLost
        return _claim(row)

    def hold_claim(
        self, claim: ExportClaim, spends: Sequence[Spend], now: datetime
    ) -> ExportClaim | None:
        def write(tx: Transaction) -> ExportClaim:
            self._lock(tx, claim, ClaimState.open)
            self._spend_in(tx, spends, now)
            return self._move(
                tx,
                claim,
                "state = %s, spends = %s, spent_at = %s",
                (
                    ClaimState.held.value,
                    {spend.kind.value: spend.key for spend in spends},
                    now,
                ),
            )

        try:
            return self._spending(
                spends, now, doing=f"holding units on {claim.export_id!r}", write=write
            )
        except _ClaimLost:
            return None

    def begin_copy(self, claim: ExportClaim, digest: str) -> ExportClaim | None:
        try:
            with self._transaction(f"beginning the copy of {claim.export_id!r}") as tx:
                self._lock(tx, claim, ClaimState.held, ClaimState.landed)
                return self._move(
                    tx,
                    claim,
                    "state = %s, artifact_digest = %s",
                    (ClaimState.copying.value, digest),
                )
        except _ClaimLost:
            return None

    def settle_copy(self, claim: ExportClaim, *, landed: bool) -> ExportClaim | None:
        try:
            with self._transaction(f"settling the copy of {claim.export_id!r}") as tx:
                row = self._lock(tx, claim, ClaimState.copying)
                if landed:
                    return self._move(tx, claim, "state = %s", (ClaimState.landed.value,))
                self._refund_row(tx, row)
                return self._move(
                    tx,
                    claim,
                    "state = %s, spends = %s, spent_at = NULL, artifact_digest = NULL",
                    (ClaimState.open.value, {}),
                )
        except _ClaimLost:
            return None

    def free_claim(self, claim: ExportClaim, now: datetime) -> None:
        with (
            contextlib.suppress(_ClaimLost),
            self._transaction(f"freeing the claim of {claim.export_id!r}") as tx,
        ):
            self._move(tx, claim, "expires_at = %s", (now,))

    def refuse_claim(self, claim: ExportClaim, refusal: Mapping[str, Any]) -> None:
        with (
            contextlib.suppress(_ClaimLost),
            self._transaction(f"refusing the claim of {claim.export_id!r}") as tx,
        ):
            row = self._lock(tx, claim, ClaimState.open, ClaimState.held)
            self._refund_row(tx, row)
            self._move(
                tx,
                claim,
                "state = %s, spends = %s, spent_at = NULL, refusal = %s",
                (ClaimState.refused.value, {}, dict(refusal)),
            )

    def close_claim(self, claim: ExportClaim, *, stale_at: datetime | None = None) -> None:
        lease, params = ("", ()) if stale_at is None else (" AND expires_at <= %s", (stale_at,))
        with self._transaction(f"closing the claim of {claim.export_id!r}") as tx:
            row = tx.fetch_one(
                "DELETE FROM datasets.export_prepare_claim "
                f"WHERE export_id = %s AND generation = %s{lease} "
                "RETURNING state, spends, spent_at",
                (claim.export_id, claim.generation, *params),
            )
            if row is not None and row["state"] in (ClaimState.held, ClaimState.copying):
                self._refund_row(tx, row)

    def expired_claims(self, now: datetime) -> list[ExportClaim]:
        with self._transaction("reading the expired claims") as tx:
            rows = tx.fetch_all(
                f"SELECT {_CLAIM_COLUMNS} FROM datasets.export_prepare_claim "
                "WHERE expires_at <= %s",
                (now,),
            )
        return [_claim(row) for row in rows]


_CLAIM_COLUMNS = (
    "export_id, generation, state, expires_at, spends, spent_at, artifact_digest, refusal"
)

_HOLDS_UNIT = "spends ->> CAST(%s AS text) = %s"
"""Whether a claim's row holds a unit of a kind under a key."""


def _spends(spends: Mapping[str, str]) -> tuple[Spend, ...]:
    return tuple(Spend(key, AllowanceKind(kind)) for kind, key in spends.items())


def _claim(row: Mapping[str, Any]) -> ExportClaim:
    return ExportClaim(
        export_id=row["export_id"],
        generation=row["generation"],
        state=ClaimState(row["state"]),
        expires_at=row["expires_at"],
        spends=_spends(row["spends"]),
        spent_at=row["spent_at"],
        digest=row["artifact_digest"],
        refusal=row["refusal"],
    )


__all__ = [
    "BUDGET_KEY",
    "AllowanceExhausted",
    "AllowanceKind",
    "AllowanceLedger",
    "ClaimState",
    "ExportClaim",
    "LiveAllowanceLedger",
    "SharedAllowanceLedger",
    "Spend",
    "allowance_limits",
    "seconds_to_reset",
    "spending",
    "utc_day",
]
