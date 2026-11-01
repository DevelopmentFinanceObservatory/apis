"""Record the reads one region's resolution lost, so the response cache stores none of it.

A read that fails, runs past its deadline, or could not start inside a resolution often leaves
the answer standing on less than it would have read, with no warning to say so. So does a
decision that consulted a StatVar's default scope when that scope was a countries fallback
this request alone took (`note_unsettled_scope`, called by `place.ConsultedPresence`). Two
examples of a lost read are the dated
read the unit gate checks a sparse series with and the re-read that names the provenances a
no-data answer is withheld under. `recording_degraded_reads` binds a record for one region's
resolution (`pipeline.resolve_variable`, `resubmit.resolve_spec_resubmit`). Any site in its
extent notes what it lost with `note_degraded_read`, and the resolution marks its region
`RegionResult.degraded`. The record travels in the context, so the worker threads a
resolution starts with `asyncio.to_thread` or `submit_with_context` note into it. A record
bound inside another passes each note to the outer one as it is noted, unless it is detached,
so a task that outlives the record it started in, such as a trailing preview, still reaches
the records around it. A detached record keeps what it holds. The UN leg's reads
(`phase_intake._un_shapes`) bear on the answer only when the UN pass runs, so
`pipeline._fill_gap` notes them then, and a place-options scope's reads
(`place_options.build_place_options_events`) mark only its options, which the response bundle
does not hold.

The record spans every pass of a resolution, so a fact noted in a pass reaches the region
whichever pass answers. A reading the part may set aside records into a record of its own,
and the outcome that compares readings hands each record on with `carry` under one rule: its
lost reads always, and the scopes it consulted only when the outcome's rule compared that
reading's result. A party reading withdrawn for want of data hands on both
(`pipeline.resolve_variable`), and one withdrawn for another reason its lost reads alone. A
re-election stand-in hands on both when the choice compared it, the winner and the equivalents
ahead of it (`phase_place_materialise._reelect_zero_observation_sibling`), and every
alternate admission reads hands on both (`candidate_admission`).

A read on the request's critical path whose loss the answer survives runs inside
`fail_open_reads`, which bounds it by a budget and notes it when it is lost. A read that ran
out its budget was healthy but slow. It is noted as cut (`DegradedReads.cut`), which leaves
`degraded` and so everything the user sees unchanged: no warning, no degraded leg, no
incomplete trailing event. The answer stands on less than a later request may read, so
no cache stores it. `core.resolve_events` binds one record for the whole request, which every
record not detached passes its notes to, and the response cache and part memo store nothing
of a request whose record holds a cut read (`response_cache_flow.CacheScope`,
`PartMemoScope`). A cut read follows a lost read's rule: a detached record keeps it, and
`carry` always hands it on.

Only a read that was made and lost is noted. A fact left unknown because no read was made for
it is not: a one-shape bind menu skips the availability read over a named set's places
(`phase_rank_bind._phase_prune_and_rank`), so that shape's availability there is unknown,
and its region is not degraded.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Final

from dc_graph import GraphCapExceededError, GraphDeadlineExceededError
from dfo_platform.request_deadline import bounded_deadline, request_deadline
from search.engine.errors import UPSTREAM_READ_ERRORS, EngineInfraError

logger = logging.getLogger(__name__)


@dataclass
class DegradedReads:
    """The reads a resolution lost, by name, in the order they were noted, the StatVars whose
    unsettled default scope one of its decisions consulted, and the reads their own fail-open
    budget cut. `outer` is the record each note passes on to, `None` for a detached record
    and the outermost one."""

    reads: list[str] = field(default_factory=list)
    unsettled: set[str] = field(default_factory=set)
    cut: list[str] = field(default_factory=list)
    outer: DegradedReads | None = field(default=None, repr=False, compare=False)

    @property
    def degraded(self) -> bool:
        """Whether the answer stands on a lost read or an unsettled scope, which the user is
        told of (`RegionResult.degraded`). A cut read is not."""
        return bool(self.reads or self.unsettled)

    @property
    def storable(self) -> bool:
        """Whether a cache may store what the record covers, which holds when it is not
        `degraded` and no read was cut."""
        return not (self.degraded or self.cut)

    def merge_into(self, other: DegradedReads) -> None:
        """Add what this record holds to `other`."""
        other.reads.extend(self.reads)
        other.unsettled.update(self.unsettled)
        other.cut.extend(self.cut)


_RECORD: ContextVar[DegradedReads | None] = ContextVar("search_degraded_reads", default=None)


def _bound_chain() -> Iterator[DegradedReads]:
    """The bound record and each record it passes its notes to, innermost first."""
    record = _RECORD.get()
    while record is not None:
        yield record
        record = record.outer


@contextmanager
def recording_degraded_reads(
    *, detached: bool | None = None, into: DegradedReads | None = None
) -> Iterator[DegradedReads]:
    """Collect, for the body of the `with` block and the tasks and threads it starts, the
    reads it swallows. A record not `detached` passes each note on to the record bound around
    it as it is noted, so a note made after the block closed still reaches the outer record. A
    `detached` record keeps them. Given `into`, the block records into that record, which
    belongs to one reading recorded over several blocks, and is detached: its owner carries it
    (`carry`), so `into` with `detached=False` raises `TypeError`."""
    if into is not None and detached is False:
        raise TypeError("a record given as `into` is detached; its owner carries it")
    outer = None if detached else _RECORD.get()
    record = into if into is not None else DegradedReads(outer=outer)
    token = _RECORD.set(record)
    try:
        yield record
    finally:
        _RECORD.reset(token)


def note_degraded_read(read: str) -> None:
    """Note that `read` was lost or could not start, and the answer stands without it.
    Outside a record it notes nothing."""
    for record in _bound_chain():
        record.reads.append(read)


def _note_cut_read(read: str) -> None:
    """Note that `read` ran out its own fail-open budget, so no cache stores the answer,
    which the user is not told of. Outside a record it notes nothing."""
    for record in _bound_chain():
        record.cut.append(read)


def note_unsettled_scope(stat_var: str) -> None:
    """Note that a decision consulted `stat_var`'s default scope while it was a countries
    fallback this request alone took, which a later request may read over every place with
    data. Outside a record it notes nothing."""
    for record in _bound_chain():
        record.unsettled.add(stat_var)


def carry(record: DegradedReads, *, consulted: bool = True) -> None:
    """Note in the bound record what a detached `record` held: its lost and cut reads, and
    the scopes it consulted when `consulted`, as a reading the answer stands on does. A
    reading the answer passed over hands on its lost and cut reads alone. Outside a record it
    notes nothing."""
    for outer in _bound_chain():
        outer.reads.extend(record.reads)
        outer.cut.extend(record.cut)
        if consulted:
            outer.unsettled.update(record.unsettled)


LOST_READ_ERRORS: Final = (
    EngineInfraError,
    TimeoutError,
    GraphCapExceededError,
    *UPSTREAM_READ_ERRORS,
)
"""What a read that fails open loses to: an infrastructure failure of the graph or LLM client
(`EngineInfraError`, `GraphInfraError` among them), a deadline it crossed (a graph
`GraphDeadlineExceededError`, or a `TimeoutError` such as a single-flight `WaitTimeout`), a
response past its byte cap (`dc_graph.GraphCapExceededError`, which the engine's own cap error
extends), an SDMX failure, or an upstream answer the read could not use. A fault in the code,
such as a `ValueError`, `TypeError` or pydantic `ValidationError`, is none of these and
raises."""


@dataclass(frozen=True)
class ReadBudget:
    """A read's own time limit: `deadline`, set no later than `request_deadline`, the
    request's own (`None` when the request has none). A `fail_open_reads` block holds one,
    and a read that bounds itself by a budget of its own outside such a block, as
    `related_read` does, carries its own."""

    deadline: float
    request_deadline: float | None

    @classmethod
    def from_now(cls, budget_s: float, *, request_deadline: float | None) -> ReadBudget:
        """`budget_s` from now, or `request_deadline` when that comes first."""
        deadline = time.monotonic() + budget_s
        if request_deadline is not None:
            deadline = min(deadline, request_deadline)
        return cls(deadline=deadline, request_deadline=request_deadline)

    def ran_out(self, exc: BaseException) -> bool:
        """Whether `exc` is a deadline error (`TimeoutError`, `GraphDeadlineExceededError`)
        raised once this budget's deadline passed and before the request's own did, so the
        deadline that fired is the budget, earlier than the request's. The clock tells which
        deadline fired, and no message is read. A deadline error after the request's own
        deadline passed is no budget running out."""
        if not isinstance(exc, TimeoutError | GraphDeadlineExceededError):
            return False
        now = time.monotonic()
        return now >= self.deadline and (
            self.request_deadline is None or now < self.request_deadline
        )


_SCOPE: ContextVar[ReadBudget | None] = ContextVar("search_fail_open_scope", default=None)
"""The budget of the innermost `fail_open_reads` block, whose `request_deadline` is the one
bound outside every such block."""


def scope_budget_ran_out(exc: BaseException) -> bool:
    """Whether `exc` is the innermost `fail_open_reads` block's own budget running out
    (`ReadBudget.ran_out`). Outside a block it is not."""
    scope = _SCOPE.get()
    return scope is not None and scope.ran_out(exc)


def note_lost_read(read: str, exc: BaseException, *, budget: ReadBudget | None = None) -> None:
    """Note `read`, lost to `exc`, or cut when `exc` is `budget` running out, or with no
    `budget` the innermost `fail_open_reads` block's (`ReadBudget.ran_out`). A read that is
    healthy but slower than its budget leaves an answer served as if whole and stored by no
    cache, while a failed read, or one past the request's own deadline, leaves a degraded one.
    Every site that notes a read lost to an exception notes it here, so the rule holds
    wherever the read ran."""
    budget = budget if budget is not None else _SCOPE.get()
    if budget is not None and budget.ran_out(exc):
        _note_cut_read(read)
    else:
        note_degraded_read(read)


@contextmanager
def fail_open_reads(
    name: str, *, budget_s: float, about: Mapping[str, object] | None = None
) -> Iterator[float]:
    """Bound every read made in the block, and swallow the one it loses.

    The block's reads run under `budget_s` from entry, or the bound request deadline when
    that comes first (`dfo_platform.request_deadline`), so a stalled read on the request's
    critical path holds it for the budget at most. The bound is the ambient deadline the
    graph and SDMX clients narrow each read with (`bounded_deadline`), so it reaches reads
    whose method takes no deadline, the waits that narrow by it, and the reads of worker
    threads that copy the context (`asyncio.to_thread`, `submit_with_context`) once the
    block has started them. It only narrows: a scope inside another, or under a nearer
    request deadline, keeps the nearer bound. Yields the bound.

    A read lost to one of `LOST_READ_ERRORS` ends the block, and the code after the block
    runs, so a site sets its fallback before the block. The lost read is noted as `name`
    (`note_lost_read`) unless the block's own budget ran out first (`scope_budget_ran_out`):
    a deadline error raised once the block's deadline passed, while that deadline is earlier
    than the request's own and the request's has not passed. Such a result is dropped, logged
    and noted as cut, so the answer stands without it unwarned and no cache stores it. The
    warning logged for a lost read names what the read covered, `about` (the variables,
    places or groups it was to read), and carries the traceback unless the read ran past a
    deadline, whose exception says nothing more. Every other exception propagates. A read
    whose failure fails the request keeps the request's deadline and stays outside such a
    block, as `convertibility._dated_plan` does."""
    bound = time.monotonic() + budget_s
    ambient = bounded_deadline(None)
    deadline = bound if ambient is None else min(bound, ambient)
    outer = _SCOPE.get()
    scope = ReadBudget(
        deadline=deadline, request_deadline=ambient if outer is None else outer.request_deadline
    )
    token = _SCOPE.set(scope)
    try:
        with request_deadline(deadline):
            yield deadline
    except LOST_READ_ERRORS as exc:
        if scope.ran_out(exc):
            logger.info(
                "%s dropped for %s: its %.1f s budget ran out: %s: %s",
                name,
                dict(about or {}),
                budget_s,
                type(exc).__name__,
                exc,
            )
            _note_cut_read(name)
            return
        logger.warning(
            "%s lost for %s: %s: %s",
            name,
            dict(about or {}),
            type(exc).__name__,
            exc,
            exc_info=not isinstance(exc, TimeoutError | GraphDeadlineExceededError),
        )
        note_degraded_read(name)
    finally:
        _SCOPE.reset(token)
