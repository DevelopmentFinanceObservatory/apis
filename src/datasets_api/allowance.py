"""Daily allowances per caller, kept in `datasets.caller_allowance`.

A caller (`dfo_platform.caller.Caller`) holds one allowance of each caller kind per UTC day:
`share` (`DATASETS_SHARE_DAILY_ALLOWANCE`, default 10) for share writes, `card`
(`DATASETS_CARD_DAILY_ALLOWANCE`, default 500) for Presentation card creates and revisions, and
`pin` (`DATASETS_PIN_DAILY_ALLOWANCE`, default 100) for the pins of callers without the pin
grant. Two budgets every caller shares are kept the same way, under `BUDGET_KEY`: `pin_budget`
(`DATASETS_PIN_DAILY_MAX`, default 2,500), the new pinned copies the pin door and the share routes
make (`datasets_api.retention_admission`), and `open_pin_budget` (`DATASETS_PIN_OPEN_DAILY_MAX`,
default 2,000), those they make for callers without the pin grant. `AllowanceLedger.spend`
takes one unit, or refuses `AllowanceExhausted` (429 `allowance_exhausted`) with `Retry-After`
counting the seconds to the next UTC midnight, when every allowance and budget resets.
`AllowanceLedger.refund` gives one unit back, and `spending` spends one around a write and
refunds it when the write fails, so a failed write costs its caller nothing. A new pinned copy's
units are spent together onto a claim, all or none, refusing naming the first spent out, and stay
spent when its copy stands and a later step fails.

Every new pinned copy runs under one row of `datasets.copy_claim`: an export prepare's
under its export id, so one prepare runs across every instance, and a pin's under a key of its own
(`retention_admission.copy_to_pinned`). A unit on the row is a unit spent:

| State     | Units   | Means                                                      |
| :-------- | :------ | :--------------------------------------------------------- |
| `open`    | none    | A copy's caller holds the claim and has spent nothing.     |
| `held`    | spent   | Units are spent ahead of a render. No copy has begun.      |
| `copying` | spent   | A copy of `digest` began on them. Its outcome is unknown.  |
| `landed`  | paid    | The object of `digest` stands, so they stay spent.         |

| Transition       | From                                     | To        | Units                   |
| :--------------- | :--------------------------------------- | :-------- | :---------------------- |
| `claim`          | no row                                   | `open`    | none                    |
| `claim`          | stale `open`, stale `held`               | `open`    | refunded from `held`    |
| `claim`          | stale `copying`, stale `landed`          | the same  | kept                    |
| `begin_pin_copy` | no row                                   | `copying` | spent                   |
| `hold_claim`     | live `open`                              | `held`    | spent                   |
| `begin_copy`     | live `held`, live `landed` of the digest | `copying` | kept                    |
| `settle_copy`    | `copying`, landed                        | `landed`  | kept, claim's lease     |
| `settle_copy`    | `copying`, not landed                    | `open`    | refunded, claim's lease |
| `reopen_claim`   | `landed`                                 | `open`    | kept for the old digest |
| `free_claim`     | any                                      | the same  | kept, lease ended       |
| `close_claim`    | any                                      | no row    | refunded unless landed  |

Every write gives the row a new random `generation`, so a value never comes back, even after the
row is deleted and claimed again. `claim` matches a stale `expires_at`, and every other transition
names the generation the writer read, so a writer acting on an old read, a holder that lost its
claim or a sweep behind a hold, changes nothing and refunds nothing. `expires_at` moves only with
the generation, so the lease a writer read stays true while its generation stands, and any refund
reads the units from the row inside its transaction. A claim pays for one digest: a retry that
renders another reopens the claim and spends for itself. `landed` rests on an existence read,
which says the object stands and not who paid it, so a pin landing first leaves two sets spent.

A limit-reached spend reads the claims inside its own transaction, whose skipped upsert keeps the
day's row locked, so no refund can land between the failed spend and the verdict. A unit on a
`held` or `copying` claim of that day may come back, whether its lease runs or not, so it raises
`AllowanceExhausted` with `held_for` set: `RUNNING_CLAIM_RETRY_SECONDS` while a claim's lease
runs, else `SWEEP_INTERVAL_SECONDS`. A stale claim keeps its unit until a retry takes it over or the sweep
settles it. Spends and refunds lock rows in one order (`_in_lock_order`), and a claim's
row before any of its units.

`LiveAllowanceLedger` spends each unit with one upsert that adds one to the day's row only while
it is below the limit, so spends racing on one row cannot pass it. A claim's units are upserted in
the claim's own transaction, which rolls them all back when one finds its row at the limit. A
refund locks the day's row, then takes one from it or deletes the row it would leave at zero, so
a spend racing the refund waits for its commit. `hold_claim` and `begin_copy` read the time
inside their transaction after the claim's row lock, and `begin_pin_copy` as its first statement,
since that time names the day whose rows its spends lock, so the wait for a connection sits
outside the lease each writes and the lease check it makes.
"""

from __future__ import annotations

import contextlib
import logging
import math
from collections.abc import Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
from enum import StrEnum
from types import MappingProxyType
from typing import Any, Protocol, cast
from uuid import uuid4

from dataset_store import DatasetStore, Transaction
from dataset_store.clock import Clock
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


@dataclass(frozen=True, kw_only=True)
class CopyClaim:
    """One read of a row of `datasets.copy_claim`. See the module docstring."""

    key: str
    generation: str
    state: ClaimState
    expires_at: datetime
    spends: tuple[Spend, ...] = ()
    spent_at: datetime | None = None
    digest: str | None = None


def seconds_to_reset(now: datetime) -> int:
    """Whole seconds from `now` to the next UTC midnight, at least 1."""
    midnight = datetime.combine(now.astimezone(UTC).date() + timedelta(days=1), time(), UTC)
    return max(1, math.ceil((midnight - now).total_seconds()))


class AllowanceExhausted(Refusal):
    """The caller has spent today's allowance of one kind. Maps to 429 with `retryable: false`,
    since a client that waits for the reset holds a request open for hours, and `Retry-After`
    naming the reset at the next UTC midnight. `held_for` says a pin claim holds a unit, so
    the limit can free again today, and how many seconds to wait on it."""

    code = "allowance_exhausted"
    retryable = False

    def __init__(
        self, *, kind: AllowanceKind, limit: int, now: datetime, held_for: int | None = None
    ) -> None:
        self.kind = kind
        self.limit = limit
        self.now = now
        self.held_for = held_for
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
    def read_claim(self, key: str) -> CopyClaim | None:
        """The claim `key` holds, or `None`.

        Raises:
            DatasetsStoreUnavailable: The claim could not be read.
        """
        ...

    def claim(self, key: str, *, now: datetime, expires_at: datetime) -> CopyClaim | None:
        """Take the claim of `key` until `expires_at`, or return `None` while a live
        prepare holds it. A stale claim passes on as the module docstring's table says.

        Raises:
            DatasetsStoreUnavailable: The claim could not be written.
        """
        ...

    def hold_claim(
        self, claim: CopyClaim, spends: Sequence[Spend], *, clock: Clock
    ) -> CopyClaim | None:
        """Move `claim` from `open` to `held`, spending one unit of each of `spends` onto it, all
        or none, for the UTC day `clock` reads inside the transaction. Return `None` when its
        generation is gone or its lease has passed, with nothing spent.

        Raises:
            AllowanceExhausted: The first of `spends` whose day's limit is reached.
            DatasetsStoreUnavailable: The units could not be written.
        """
        ...

    def begin_copy(
        self, claim: CopyClaim, digest: str, *, clock: Clock, lease: timedelta
    ) -> CopyClaim | None:
        """Move `claim` to `copying` for `digest`, its lease `lease` on from the time `clock`
        reads inside the transaction once the row is locked, or return `None` when its
        generation is gone, its lease has passed at that time, or it paid for another digest.
        The wait for a connection and the row lock sit outside the new lease.

        Raises:
            DatasetsStoreUnavailable: The claim could not be written.
        """
        ...

    def settle_copy(
        self, claim: CopyClaim, *, landed: bool, expires_at: datetime
    ) -> CopyClaim | None:
        """Move `claim` from `copying` to `landed`, or to `open` with its units refunded, ending
        the copy's lease at `expires_at`, the prepare's own. Return `None` when its generation is
        gone.

        Raises:
            DatasetsStoreUnavailable: The claim could not be written.
        """
        ...

    def reopen_claim(self, claim: CopyClaim) -> CopyClaim | None:
        """Move `claim` from `landed` to `open`, leaving its units spent on the object they paid
        for. Return `None` when its generation is gone.

        Raises:
            DatasetsStoreUnavailable: The claim could not be written.
        """
        ...

    def free_claim(self, claim: CopyClaim, now: datetime) -> None:
        """End `claim`'s lease at `now` and keep the row, for a retry or the sweep to settle.

        Raises:
            DatasetsStoreUnavailable: The claim could not be written.
        """
        ...

    def begin_pin_copy(
        self, key: str, spends: Sequence[Spend], digest: str, *, clock: Clock, lease: timedelta
    ) -> CopyClaim:
        """Spend one unit of each of `spends`, all or none, and write a `copying` claim of them
        under `key` for `digest`, in one transaction. One read of `clock`, the transaction's
        first statement, names the units' day, `spent_at` and the start of the `lease`, so the
        wait for a connection sits outside it. A pin knows its digest before it spends, so its
        claim begins its copy.

        Raises:
            AllowanceExhausted: The first of `spends` whose day's limit is reached. Nothing is
                written.
            DatasetsStoreUnavailable: The units could not be written.
        """
        ...

    def close_claim(self, claim: CopyClaim, *, landed: bool = False) -> bool:
        """Delete `claim`, refunding the units of a `held` or `copying` row unless `landed` says
        its copy stands, and return whether its generation still stood.

        Raises:
            DatasetsStoreUnavailable: The claim could not be written.
        """
        ...

    def expired_claims(self, now: datetime) -> list[CopyClaim]:
        """Every claim whose lease has passed at `now`, each `copying` one after every other, so
        a slow bucket read cannot hold up a refund that needs none.

        Raises:
            DatasetsStoreUnavailable: The claims could not be read.
        """
        ...


class _LimitReached(Exception):
    """Rolls back a transaction's spends when one finds its row at the limit. `held_for` is
    `AllowanceExhausted`'s."""

    def __init__(self, spend: Spend, *, now: datetime, held_for: int | None) -> None:
        self.spend = spend
        self.now = now
        self.held_for = held_for


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


SWEEP_INTERVAL_SECONDS = 300
"""How often Cloud Scheduler runs `maintenance`'s `sweep-export-claims`, and so the wait on a
claim whose lease has passed."""

RUNNING_CLAIM_RETRY_SECONDS = 10
"""The wait on a claim whose lease runs. A prepare takes about 6 s, so a typical one settles
before the retry, and a long render is polled every 10 s."""


class LiveAllowanceLedger:
    """`AllowanceLedger` over `datasets.caller_allowance`."""

    def __init__(self, *, store: DatasetStore, limits: Mapping[AllowanceKind, int]) -> None:
        self._store = store
        self._limits = dict(limits)

    def spend(self, caller: Caller, kind: AllowanceKind, now: datetime) -> None:
        with self._spending(f"spending {kind} of {caller.key!r}") as tx:
            self._spend_in(tx, [Spend(caller.key, kind)], now)

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
            if self._limits[spend.kind] <= 0:
                raise _LimitReached(spend, now=now, held_for=None)
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
                raise _LimitReached(spend, now=now, held_for=self._held_for(tx, spend, now))

    def _held_for(self, tx: Transaction, spend: Spend, now: datetime) -> int | None:
        """The shortest wait on a claim holding a unit of `spend` for `now`'s UTC day, or `None`
        when none does."""
        midnight = datetime.combine(utc_day(now), time(), UTC)
        rows = tx.fetch_all(
            "SELECT expires_at FROM datasets.copy_claim "
            "WHERE spends ->> CAST(%s AS text) = %s AND spent_at >= %s AND spent_at < %s "
            "AND state IN (%s, %s)",
            (
                spend.kind.value,
                spend.key,
                midnight,
                midnight + timedelta(days=1),
                ClaimState.held.value,
                ClaimState.copying.value,
            ),
        )
        if not rows:
            return None
        if any(cast(datetime, row["expires_at"]) > now for row in rows):
            return RUNNING_CLAIM_RETRY_SECONDS
        return SWEEP_INTERVAL_SECONDS

    @contextmanager
    def _spending(self, doing: str) -> Iterator[Transaction]:
        """A transaction whose block spends units with `_spend_in`, all or none.

        Raises:
            AllowanceExhausted: The first unit whose day's limit is reached.
            DatasetsStoreUnavailable: The units could not be written.
        """
        try:
            with self._transaction(doing) as tx:
                yield tx
        except _LimitReached as reached:
            raise AllowanceExhausted(
                kind=reached.spend.kind,
                limit=self._limits[reached.spend.kind],
                now=reached.now,
                held_for=reached.held_for,
            ) from None

    def refund(self, caller: Caller, kind: AllowanceKind, now: datetime) -> None:
        with self._transaction(f"refunding {kind} of {caller.key!r}") as tx:
            self._refund_in(tx, [Spend(caller.key, kind)], now)

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

    def read_claim(self, key: str) -> CopyClaim | None:
        with self._transaction(f"reading the claim of {key!r}") as tx:
            row = tx.fetch_one(
                f"SELECT {_CLAIM_COLUMNS} FROM datasets.copy_claim WHERE claim_key = %s",
                (key,),
            )
        return None if row is None else _claim(row)

    def claim(self, key: str, *, now: datetime, expires_at: datetime) -> CopyClaim | None:
        with self._transaction(f"claiming {key!r}") as tx:
            ended = tx.fetch_one(
                "DELETE FROM datasets.copy_claim WHERE claim_key = %s "
                "AND state IN (%s, %s) AND expires_at <= %s RETURNING spends, spent_at",
                (key, ClaimState.open.value, ClaimState.held.value, now),
            )
            if ended is not None:
                self._refund_row(tx, ended)
            row = tx.fetch_one(
                "INSERT INTO datasets.copy_claim (claim_key, generation, expires_at) "
                "VALUES (%s, %s, %s) ON CONFLICT (claim_key) DO NOTHING "
                f"RETURNING {_CLAIM_COLUMNS}",
                (key, uuid4().hex, expires_at),
            )
            if row is None:
                # A stale row left here began a copy, which only a settle can judge.
                row = tx.fetch_one(
                    "UPDATE datasets.copy_claim "
                    "SET generation = %s, expires_at = %s "
                    "WHERE claim_key = %s AND expires_at <= %s "
                    f"RETURNING {_CLAIM_COLUMNS}",
                    (uuid4().hex, expires_at, key, now),
                )
        return None if row is None else _claim(row)

    def _lock(self, tx: Transaction, claim: CopyClaim, *states: ClaimState) -> Mapping[str, Any]:
        """The row of `claim`, locked, while it stays at this generation and in `states`."""
        row = tx.fetch_one(
            "SELECT state, expires_at, spends, spent_at, artifact_digest "
            "FROM datasets.copy_claim "
            "WHERE claim_key = %s AND generation = %s FOR UPDATE",
            (claim.key, claim.generation),
        )
        if row is None or row["state"] not in states:
            raise _ClaimLost
        return row

    def _move(
        self, tx: Transaction, claim: CopyClaim, assignments: str, params: Sequence[object]
    ) -> CopyClaim:
        """Write `assignments` on `claim`'s row, moving its generation."""
        row = tx.fetch_one(
            "UPDATE datasets.copy_claim "
            f"SET generation = %s, {assignments} "
            f"WHERE claim_key = %s AND generation = %s RETURNING {_CLAIM_COLUMNS}",
            (uuid4().hex, *params, claim.key, claim.generation),
        )
        if row is None:
            raise _ClaimLost
        return _claim(row)

    def hold_claim(
        self, claim: CopyClaim, spends: Sequence[Spend], *, clock: Clock
    ) -> CopyClaim | None:
        try:
            with self._spending(f"holding units on {claim.key!r}") as tx:
                row = self._lock(tx, claim, ClaimState.open)
                now = clock()
                if row["expires_at"] <= now:
                    raise _ClaimLost
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
        except _ClaimLost:
            return None

    def begin_copy(
        self, claim: CopyClaim, digest: str, *, clock: Clock, lease: timedelta
    ) -> CopyClaim | None:
        try:
            with self._transaction(f"beginning the copy of {claim.key!r}") as tx:
                row = self._lock(tx, claim, ClaimState.held, ClaimState.landed)
                now = clock()
                if row["expires_at"] <= now or (
                    row["state"] == ClaimState.landed and row["artifact_digest"] != digest
                ):
                    raise _ClaimLost
                return self._move(
                    tx,
                    claim,
                    "state = %s, artifact_digest = %s, expires_at = %s",
                    (ClaimState.copying.value, digest, now + lease),
                )
        except _ClaimLost:
            return None

    def begin_pin_copy(
        self, key: str, spends: Sequence[Spend], digest: str, *, clock: Clock, lease: timedelta
    ) -> CopyClaim:
        with self._spending(f"beginning the pin copy {key!r}") as tx:
            # One read names the day the spends count under, `spent_at` and the lease, so it
            # comes before the upserts that choose the day's rows.
            now = clock()
            self._spend_in(tx, spends, now)
            row = tx.fetch_one(
                "INSERT INTO datasets.copy_claim (claim_key, generation, state, expires_at, "
                "spends, spent_at, artifact_digest) VALUES (%s, %s, %s, %s, %s, %s, %s) "
                f"RETURNING {_CLAIM_COLUMNS}",
                (
                    key,
                    uuid4().hex,
                    ClaimState.copying.value,
                    now + lease,
                    {spend.kind.value: spend.key for spend in spends},
                    now,
                    digest,
                ),
            )
        return _claim(cast(Mapping[str, Any], row))

    def settle_copy(
        self, claim: CopyClaim, *, landed: bool, expires_at: datetime
    ) -> CopyClaim | None:
        try:
            with self._transaction(f"settling the copy of {claim.key!r}") as tx:
                row = self._lock(tx, claim, ClaimState.copying)
                if landed:
                    return self._move(
                        tx,
                        claim,
                        "state = %s, expires_at = %s",
                        (ClaimState.landed.value, expires_at),
                    )
                self._refund_row(tx, row)
                return self._move(
                    tx,
                    claim,
                    f"{_REOPENED}, expires_at = %s",
                    (ClaimState.open.value, {}, expires_at),
                )
        except _ClaimLost:
            return None

    def reopen_claim(self, claim: CopyClaim) -> CopyClaim | None:
        try:
            with self._transaction(f"reopening the claim of {claim.key!r}") as tx:
                self._lock(tx, claim, ClaimState.landed)
                return self._move(tx, claim, _REOPENED, (ClaimState.open.value, {}))
        except _ClaimLost:
            return None

    def free_claim(self, claim: CopyClaim, now: datetime) -> None:
        with (
            contextlib.suppress(_ClaimLost),
            self._transaction(f"freeing the claim of {claim.key!r}") as tx,
        ):
            self._move(tx, claim, "expires_at = %s", (now,))

    def close_claim(self, claim: CopyClaim, *, landed: bool = False) -> bool:
        with self._transaction(f"closing the claim of {claim.key!r}") as tx:
            row = tx.fetch_one(
                "DELETE FROM datasets.copy_claim "
                "WHERE claim_key = %s AND generation = %s RETURNING state, spends, spent_at",
                (claim.key, claim.generation),
            )
            if (
                row is not None
                and not landed
                and row["state"] in (ClaimState.held, ClaimState.copying)
            ):
                self._refund_row(tx, row)
        return row is not None

    def expired_claims(self, now: datetime) -> list[CopyClaim]:
        with self._transaction("reading the expired claims") as tx:
            rows = tx.fetch_all(
                f"SELECT {_CLAIM_COLUMNS} FROM datasets.copy_claim "
                "WHERE expires_at <= %s ORDER BY state = %s, expires_at",
                (now, ClaimState.copying.value),
            )
        return [_claim(row) for row in rows]


_REOPENED = "state = %s, spends = %s, spent_at = NULL, artifact_digest = NULL"

_CLAIM_COLUMNS = "claim_key, generation, state, expires_at, spends, spent_at, artifact_digest"


def _spends(spends: Mapping[str, str]) -> tuple[Spend, ...]:
    return tuple(Spend(key, AllowanceKind(kind)) for kind, key in spends.items())


def _claim(row: Mapping[str, Any]) -> CopyClaim:
    return CopyClaim(
        key=row["claim_key"],
        generation=row["generation"],
        state=ClaimState(row["state"]),
        expires_at=row["expires_at"],
        spends=_spends(row["spends"]),
        spent_at=row["spent_at"],
        digest=row["artifact_digest"],
    )


__all__ = [
    "BUDGET_KEY",
    "RUNNING_CLAIM_RETRY_SECONDS",
    "SWEEP_INTERVAL_SECONDS",
    "AllowanceExhausted",
    "AllowanceKind",
    "AllowanceLedger",
    "ClaimState",
    "CopyClaim",
    "LiveAllowanceLedger",
    "SharedAllowanceLedger",
    "Spend",
    "allowance_limits",
    "seconds_to_reset",
    "spending",
    "utc_day",
]
