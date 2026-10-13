"""Admission checks run before materialisation or network calls.

`bounded_materialise_slot` limits the whole frames a mint, a rebuild, a head render, a file
renewal and an observations read hold to the process's `MaterialisePermits`, and caps waiting
at both the configured wait and client deadline. A publish builds its frame outside the
permits. `MaterialisePermits.every_permit` holds every permit, so a coverage reload never
overlaps a permit holder. A request that took its coverage snapshot before its permit confirms
the snapshot (`coverage_current`) before it waits, while it waits and at admission, and starts
again on the served one when a reload replaced it (`under_current_coverage`). A request that
holds a permit therefore builds its frame beside the one index the service serves. A request
waiting for a permit or for its turn at a shared rebuild (`RebuildFlights.turn`) releases a
replaced index within `STALE_CHECK_SECONDS` of the reload. Before it waits, a request can
hold a replaced index for as long as its head checks, its graph reads, member resolution, the
estimate and the budget decision take, beside the frames of the requests that hold permits.

A permit covers a frame for as long as its work runs. An error's traceback can keep the frame
until the door has answered the error, after the permit is released.

`RebuildFlights` lets concurrent callers share one rebuild of a subject. `sync_estimate`
models dispatch from one coverage snapshot: it groups by request-batch key and window years,
then applies the same lane and StatVar chunk rules as materialisation.

Uncovered StatVars contribute zero priced rows or requests. The estimate counts them separately
and refuses more than 700 with a 422 member-cardinality error. Sync budgets assess covered costs,
so an uncovered request always runs inline by design. A subject-only request that binds no entity
is uncovered and is priced as the same request naming each country would be: zero, with the
materialiser's row ceiling as its bound. Over-budget requests face separate async row and request
ceilings. `ContainedIn` bindings are unbound to avoid a graph read. A range window clips the row
estimate only: requests are priced across all years, as the fetch sends them. `latest` uses all
years for row estimates and preserves `c[TIME_PERIOD]=LATEST` for SDMX request-size estimates. A
member that binds no role prices `CoverageIndex.exhaustive_single_axis_filter`, the entity list
its fetch sends.

Excel estimates count data rows against one sheet's room, or two sheets' when the render
can keep totals on a second sheet (`xlsx_splits_totals`). The writer enforces each sheet's
rows, columns and cell lengths after fetch.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Collection, Iterator, Mapping, Sequence
from contextlib import contextmanager

from code_index.types import PartyRow
from datasets_api.doors.mint_errors import MintBackpressure
from datasets_api.mint_models import MintOptions
from datasets_api.output_options import movable_totals_roles
from datasets_api.timings import count, phase
from dc_sdmx import ContainedIn
from materialise.errors import OverCeilingError
from materialise.formats import EXCEL_MAX_ROWS, ExcelLimitError, check_excel_rows
from materialise.generation import (
    _time_period_for_window,
    dispatch_lanes,
    request_batch_key,
    stat_var_chunk_positions,
)
from materialise.pipeline import MaterialisationRequest
from series_coverage.index import BoundRoles, CoverageIndex, Estimate, Role
from series_coverage.shared import CoverageSnapshot

_STAT_VAR_PREFIX = "dcid:"

# Coverage cannot price uncovered members, so this separate cap bounds their count.
UNCOVERED_MEMBER_CAP = 700

# How many times a request starts again on the served coverage snapshot before it refuses.
COVERAGE_RESTARTS = 2

# How long a request waits for a permit or a rebuild turn between looks at whether its
# coverage snapshot is still the served one.
STALE_CHECK_SECONDS = 0.1


class StaleCoverageSnapshot(Exception):
    """A coverage reload replaced the snapshot a request holds before its permit was taken."""


def _sliced_wait(
    attempt: Callable[[float], bool], *, ends: float, keep_waiting: Callable[[], bool] | None
) -> bool:
    """Wait until `attempt` succeeds or the monotonic time `ends`, and return whether it did.

    `attempt` waits up to the seconds it is given and says whether it got what it waits for.
    With `keep_waiting` it is called in slices of `STALE_CHECK_SECONDS`, `keep_waiting` is
    asked before each, and a false answer ends the wait. The slices add up to the time left
    until `ends` and no more. A past `ends` makes one zero-wait attempt.
    """
    if keep_waiting is None:
        return attempt(max(0.0, ends - time.monotonic()))
    while keep_waiting():
        remaining = ends - time.monotonic()
        if attempt(max(0.0, min(remaining, STALE_CHECK_SECONDS))):
            return True
        if remaining <= STALE_CHECK_SECONDS:
            return False
    return False


class MaterialisePermits:
    """The process's materialise permits: `permits` concurrent frames among the paths that
    take one (`bounded_materialise_slot`). The permits are counted, with no owner, so a thread
    that takes a second one while it holds one waits on itself.

    `acquire` passes a gate before it waits for a permit. `every_permit` closes the gate, so
    callers that arrive while it drains wait at the gate, each for no longer than its own
    timeout, and the permits that frames release go to the drain alone.
    """

    def __init__(self, permits: int) -> None:
        self.permits = permits
        self._semaphore = threading.BoundedSemaphore(permits)
        self._gate = threading.Condition()
        self._draining = 0

    def acquire(self, *, timeout: float, keep_waiting: Callable[[], bool] | None = None) -> bool:
        """Take one permit within `timeout` seconds, counting the wait at the gate. A zero
        timeout makes one attempt. Returns whether a permit was taken.

        `keep_waiting` is asked before the wait starts, when the gate opens, and then every
        `STALE_CHECK_SECONDS` of the wait for a permit (`_sliced_wait`). A false answer ends
        the wait with no permit taken. The slices add up to `timeout` and no more, so one
        call waits `timeout` at most. A request that starts again on a replaced coverage
        snapshot calls this again with its full wait (`under_current_coverage`).
        """
        ends = time.monotonic() + timeout
        if keep_waiting is not None and not keep_waiting():
            return False
        with self._gate:
            if not self._gate.wait_for(lambda: self._draining == 0, timeout=timeout):
                return False
        return _sliced_wait(
            lambda seconds: self._semaphore.acquire(timeout=seconds),
            ends=ends,
            keep_waiting=keep_waiting,
        )

    def release(self) -> None:
        self._semaphore.release()

    @contextmanager
    def every_permit(self) -> Iterator[None]:
        """Hold every permit, waiting as long as the frames alive now take to finish, so no
        materialisation overlaps the block. A coverage reload runs under it, since the old and
        new index are both resident while it loads."""
        with self._gate:
            self._draining += 1
        held = 0
        try:
            for _ in range(self.permits):
                self._semaphore.acquire()
                held += 1
            yield
        finally:
            for _ in range(held):
                self._semaphore.release()
            with self._gate:
                self._draining -= 1
                self._gate.notify_all()


@contextmanager
def bounded_materialise_slot(
    materialise_semaphore: MaterialisePermits | None,
    *,
    wait_seconds: float,
    deadline: float | None = None,
    coverage_current: Callable[[], bool] | None = None,
) -> Iterator[None]:
    """Acquire a slot, waiting up to the smaller of `wait_seconds` and the remaining deadline.

    `deadline` is an absolute monotonic timestamp. A past deadline makes one
    zero-wait acquire attempt. A missing semaphore skips the limit. Failure
    to acquire raises retryable `MintBackpressure` (503). The wait records as
    the `materialise_wait` phase and adds its whole milliseconds to the `permit_wait_ms` count.

    `coverage_current` says whether the coverage snapshot the caller took before the slot is
    still the served one (`CoverageService.serves`). It is asked before the wait, when a
    reload ends, and every `STALE_CHECK_SECONDS` of the wait (`MaterialisePermits.acquire`),
    and a false answer raises `StaleCoverageSnapshot` (`under_current_coverage`) with no
    permit taken, so a waiting caller drops a replaced index within one such slice. It is
    asked again once the permit is held, when no reload can run, and a false answer there
    releases the permit and raises the same. Every check runs inside the `materialise_wait`
    phase. A caller that takes its snapshot inside the slot passes none.

    The permit covers a frame for as long as the work inside the slot runs. The function that
    enters the slot binds no frame in its own locals, so a slot that ends without an error
    releases its permit with the frame freed. An error's traceback can keep the frame until
    the door has answered the error, after the permit is released.
    """
    if materialise_semaphore is None:
        yield
        return
    _wait_seconds = float(wait_seconds)
    if deadline is not None:
        _wait_seconds = max(0.0, min(_wait_seconds, deadline - time.monotonic()))
    started = time.perf_counter()
    with phase("materialise_wait"):
        acquired = materialise_semaphore.acquire(
            timeout=_wait_seconds, keep_waiting=coverage_current
        )
        # A wait that ended with no permit ended on a replaced snapshot or on its bound.
        stale = not acquired and coverage_current is not None and not coverage_current()
    count("permit_wait_ms", round((time.perf_counter() - started) * 1000))
    if stale:
        raise StaleCoverageSnapshot
    if not acquired:
        raise MintBackpressure()
    try:
        if coverage_current is not None:
            with phase("materialise_wait"):
                current = coverage_current()
            if not current:
                raise StaleCoverageSnapshot
        yield
    finally:
        materialise_semaphore.release()


def under_current_coverage[T](
    take: Callable[[], CoverageSnapshot], work: Callable[[CoverageSnapshot], T]
) -> T:
    """Run `work` on a snapshot from `take` and return its result.

    `work` covers everything the request derives from the snapshot before it reaches a
    materialise slot: head checks, graph reads, member resolution, the estimate and the
    budget decision. When a slot or a rebuild turn inside it raises `StaleCoverageSnapshot`,
    `work` runs again from a new `take()`, at most `COVERAGE_RESTARTS` times, and one more
    stale snapshot raises retryable `MintBackpressure` (503). Each run waits for its permit
    up to the configured wait again, so a request can wait `1 + COVERAGE_RESTARTS` times
    that wait in all, inside its request deadline. Each stale run adds one to the
    `coverage_restarts` count of the timings collection open around this call. The worker
    opens its collection inside `work`, one per run, so a restart there shows as a second
    timings line and no count. This function holds no snapshot between runs.
    """
    for _ in range(1 + COVERAGE_RESTARTS):
        try:
            return work(take())
        except StaleCoverageSnapshot:
            # Leaving this handler frees the exception, and with its traceback the frames
            # that hold the replaced snapshot, before the next run takes the served one.
            count("coverage_restarts", 1)
    raise MintBackpressure()


class RebuildFlights:
    """Lets the callers that need one rebuild of a subject share it, within this process.

    A rebuild that advances a subject's head serves every caller that needs it, so the
    definition and frame refresh doors take a turn here before re-materialising. Two
    instances can still rebuild one subject together: version inserts are
    `ON CONFLICT DO NOTHING` and `HeadWriter.advance_head` is a compare-and-swap on the
    observed run, so the second records its run and moves no head.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._running: dict[str, threading.Event] = {}

    def _claim(self, subject_id: str, finished: threading.Event) -> threading.Event | None:
        """Register `finished` as the running rebuild of `subject_id` and return `None`, or
        return the event of the rebuild already running."""
        with self._lock:
            running = self._running.get(subject_id)
            if running is None:
                self._running[subject_id] = finished
            return running

    @contextmanager
    def turn(
        self,
        subject_id: str,
        *,
        deadline: float | None,
        wait_seconds: float,
        coverage_current: Callable[[], bool] | None = None,
    ) -> Iterator[bool]:
        """Enter once no rebuild of `subject_id` the caller could share is running, yielding
        whether the caller waited for one.

        A caller arriving while a rebuild runs waits for it to end, whether it advanced the
        head, raised, or was abandoned by its request. The caller then checks the head and
        rebuilds only if it is still the one it read. A caller that did not wait rebuilds from
        the head it read. After a rebuild that failed, each waiter
        runs its own, as it would have alone. The first of them is the rebuild later arrivals
        wait on, so no caller waits on more than one.

        The wait lasts until `deadline`, the caller's own absolute monotonic deadline, since it
        stands in for work the caller would otherwise do itself. `wait_seconds` bounds it only
        for a caller with no deadline. A rebuild still running at the bound raises retryable
        `MintBackpressure` (503). The wait records as the `rebuild_wait` phase.

        `coverage_current` says whether the coverage snapshot the caller holds is still the
        served one (`CoverageService.serves`). It is asked every `STALE_CHECK_SECONDS` of the
        wait (`_sliced_wait`), and a wait that ended on a false answer raises
        `StaleCoverageSnapshot` (`under_current_coverage`), so a waiting caller drops a
        replaced index within one such slice while the rebuild it waits on still runs.
        """
        finished = threading.Event()
        running = self._claim(subject_id, finished)
        waited = running is not None
        if running is not None:
            ends = time.monotonic() + (
                float(wait_seconds) if deadline is None else max(0.0, deadline - time.monotonic())
            )
            with phase("rebuild_wait"):
                ended = _sliced_wait(running.wait, ends=ends, keep_waiting=coverage_current)
                # A wait that ended with the rebuild running ended on a replaced snapshot or
                # on its bound.
                stale = not ended and coverage_current is not None and not coverage_current()
            if stale:
                raise StaleCoverageSnapshot
            if not ended:
                raise MintBackpressure()
            running = self._claim(subject_id, finished)
        try:
            yield waited
        finally:
            if running is None:
                with self._lock:
                    del self._running[subject_id]
                finished.set()


def _bare_dcid(stat_var: str) -> str:
    return stat_var.removeprefix(_STAT_VAR_PREFIX)


def _bound_roles(request: MaterialisationRequest, *, index: CoverageIndex) -> BoundRoles:
    """Map declared roles to coverage's donor/recipient labels.

    Omit unbound roles and `ContainedIn` filters.
    """
    names = index.role_names.get(_bare_dcid(request.stat_var))
    if names is None:
        return {}
    bound: dict[Role, frozenset[str]] = {}
    for role, declared_role in (("donor", names.donor), ("recipient", names.recipient)):
        refs = request.role_bindings.get(declared_role)
        if refs is None or isinstance(refs, ContainedIn):
            continue
        bound[role] = frozenset(refs)
    return bound


def _window_years(request: MaterialisationRequest) -> tuple[int | None, int | None]:
    window = request.window
    if window.kind == "range":
        return window.start_year, window.end_year
    return None, None


def sync_estimate(
    requests: Sequence[MaterialisationRequest],
    *,
    index: CoverageIndex,
    member_concurrency: int,
) -> Estimate:
    """Estimate covered rows and SDMX requests, plus uncovered members, before network access.

    `uncovered_requests` counts uncovered materialisation requests separately from SDMX calls.
    More than 700 raises `OverCeilingError` (`member_cardinality`, 422)
    before covered costs are computed. `index` must come from the caller's
    coverage snapshot so its estimate and any coverage stamp share one artifact.

    Group by dispatch key and window years, then use the same lanes and
    StatVar chunks as materialisation. Estimate each chunk with `lanes=1`,
    using the first member's roles and window. The `latest` window keeps
    `LATEST` in request-size estimates while coverage prices all years.
    """
    lanes = dispatch_lanes(member_concurrency, len(requests))
    groups: dict[
        tuple[tuple[object, ...], int | None, int | None], list[MaterialisationRequest]
    ] = {}
    uncovered_requests = 0
    for request in requests:
        if index.role_names.get(_bare_dcid(request.stat_var)) is None:
            uncovered_requests += 1
        first_year, last_year = _window_years(request)
        key = (request_batch_key(request), first_year, last_year)
        groups.setdefault(key, []).append(request)
    if uncovered_requests > UNCOVERED_MEMBER_CAP:
        raise OverCeilingError(
            f"{uncovered_requests} member(s) carry no coverage row at all, exceeding "
            f"member_cap={UNCOVERED_MEMBER_CAP}. Coverage has no price to put on an "
            "uncovered member, so this ceiling counts how many of them one request "
            "names, independent of the covered part's own rows/requests estimate. "
            "Narrow the identity (another slot binding) to "
            "bring the uncovered member count under the cap.",
            ceiling_kind="member_cardinality",
            count=uncovered_requests,
            cap=UNCOVERED_MEMBER_CAP,
        )
    total_rows = 0
    total_requests = 0
    for (_batch_key, first_year, last_year), members in groups.items():
        bound = _bound_roles(members[0], index=index)
        variables = [_bare_dcid(member.stat_var) for member in members]
        time_period = _time_period_for_window(members[0].window)
        for position_group in stat_var_chunk_positions(variables, lanes=lanes):
            chunk_variables = [variables[position] for position in position_group]
            chunk_estimate = index.estimate(
                variables=chunk_variables,
                bound=bound,
                first_year=first_year,
                last_year=last_year,
                lanes=1,
                time_period=time_period,
            )
            total_rows += chunk_estimate.rows
            total_requests += chunk_estimate.requests
    return Estimate(rows=total_rows, requests=total_requests, uncovered_requests=uncovered_requests)


def within_sync_budget(
    estimate: Estimate, *, sync_budget_rows: int, sync_budget_requests: int
) -> bool:
    """Return whether covered rows and requests both fit their limits.

    Only covered costs enter this test. `sync_estimate` caps uncovered
    members separately. Callers submit over-budget work asynchronously and
    run in-budget work inline.
    """
    return estimate.rows <= sync_budget_rows and estimate.requests <= sync_budget_requests


def xlsx_splits_totals(
    options: MintOptions,
    *,
    identity: Mapping[str, object],
    role_columns: Sequence[str],
    parties: Mapping[str, PartyRow],
    datasets: Collection[str],
) -> bool:
    """Whether an `xlsx` render of `identity` can keep total rows on a second sheet.

    The renderer splits totals (`render._TotalRows.for_render`) for a `csv` or `xlsx` render
    without a pivot, on the roles of `movable_totals_roles`, and this reads that rule with the
    same arguments: the frame's `role_columns`, the code index's `parties` and the provenances
    the read selects, `datasets`. Before fetch a caller passes the identity's role keys
    (`identity_role_keys`), which name the same axes as the frame's role columns, and the
    elected provenances (`elected_provenances`), empty when the read can select any, which
    reads a total on an open axis in any dataset.
    """
    return (
        options.format == "xlsx"
        and options.pivot is None
        and bool(movable_totals_roles(identity, role_columns, parties=parties, datasets=datasets))
    )


def identity_role_keys(identity: Mapping[str, object]) -> tuple[str, ...]:
    """The role keys `identity` binds (`entities`) or leaves open (`open_roles`), in the
    caller's spelling, as `xlsx_splits_totals` takes them before a frame names its columns."""
    entities = identity.get("entities")
    open_roles = identity.get("open_roles")
    keys = [str(role) for role in entities] if isinstance(entities, Mapping) else []
    if isinstance(open_roles, Sequence) and not isinstance(open_roles, str):
        keys.extend(str(role) for role in open_roles)
    return tuple(dict.fromkeys(keys))


def elected_provenances(requests: Sequence[MaterialisationRequest]) -> tuple[str, ...]:
    """The provenances a read of `requests` can select before fetch: each request's elected
    dataset (`MaterialisationRequest.dataset`), whose provenance alone its facet selection
    reads. Empty when a request elects none, since its read can then select any provenance."""
    if not requests or any(request.dataset is None for request in requests):
        return ()
    return tuple(
        dict.fromkeys(request.dataset.provenance for request in requests if request.dataset)
    )


def _excel_data_row_limit(*, splits_totals: bool) -> int:
    """The data rows an `xlsx` file holds beside its headers: one sheet's, or two when
    totals can take a second sheet. How the rows divide between the sheets is known only to
    the writer, so this is the most any division could hold."""
    return (2 if splits_totals else 1) * (EXCEL_MAX_ROWS - 1)


def _two_sheet_detail(rows: int, *, estimated: bool) -> str:
    return (
        f"{'estimated ' if estimated else ''}{rows} data row(s) exceeds the "
        f"{_excel_data_row_limit(splits_totals=True)} that Excel's {EXCEL_MAX_ROWS}-row limit "
        "allows on the data sheet and the totals sheet together; request csv or parquet instead"
    )


def refuse_excel_over_estimate(
    estimate: Estimate, *, format: str, splits_totals: bool = False
) -> None:
    """For `xlsx`, raise `ExcelLimitError` (`excel_limit`, 422) before fetch when the
    estimated rows plus a header exceed `EXCEL_MAX_ROWS`, or, with `splits_totals`
    (`xlsx_splits_totals`), when they exceed what two sheets hold.

    The estimate counts total rows with the rest, so a split render can fit where one sheet
    would not. The writer judges each sheet exactly after fetch, along with columns and cell
    lengths.
    """
    if format != "xlsx" or estimate.rows <= _excel_data_row_limit(splits_totals=splits_totals):
        return
    if splits_totals:
        raise ExcelLimitError(detail=_two_sheet_detail(estimate.rows, estimated=True))
    raise ExcelLimitError(
        detail=(
            f"estimated {estimate.rows + 1} row(s) (including the header) exceeds Excel's "
            f"{EXCEL_MAX_ROWS}-row limit; request csv or parquet instead"
        )
    )


def refuse_excel_over_stored_rows(rows: int, *, splits_totals: bool) -> None:
    """Raise `ExcelLimitError` when `rows`, a stored version's data rows total rows
    included, cannot fit an `xlsx` file: the writer's own check for one sheet, or the
    two-sheet bound with `splits_totals` (`xlsx_splits_totals`)."""
    if not splits_totals:
        check_excel_rows(min(rows + 1, EXCEL_MAX_ROWS + 1))
    elif rows > _excel_data_row_limit(splits_totals=True):
        raise ExcelLimitError(detail=_two_sheet_detail(rows, estimated=False))


def refuse_inline_over_ceiling(estimate: Estimate, *, ceiling: int) -> None:
    """Raise `OverCeilingError` (`rows`, 422) before fetch when the estimate exceeds `ceiling`,
    the row ceiling the inline materialisation runs under.

    A door that runs inline and cannot submit an operation calls this, so a request the
    mid-fetch guard would stop refuses without an SDMX call.
    """
    if estimate.rows <= ceiling:
        return
    raise OverCeilingError(
        f"estimated {estimate.rows} row(s) exceeds the inline materialisation ceiling "
        f"cap={ceiling}. Narrow the identity (another slot binding, a shorter window) to "
        "bring the estimate under the cap.",
        ceiling_kind="rows",
        count=estimate.rows,
        cap=ceiling,
    )


def refuse_async_over_ceiling(
    estimate: Estimate, *, async_max_rows: int, async_max_requests: int
) -> None:
    """Raise `OverCeilingError` (422) when async row or request limits are exceeded.

    Rows are checked first, so a request over both reports `operation_rows`.
    """
    if estimate.rows > async_max_rows:
        raise OverCeilingError(
            f"estimated {estimate.rows} row(s) exceeds the async admission ceiling "
            f"cap={async_max_rows}. Narrow the identity (another slot "
            "binding, a shorter window) to bring the estimate under the cap.",
            ceiling_kind="operation_rows",
            count=estimate.rows,
            cap=async_max_rows,
        )
    if estimate.requests > async_max_requests:
        raise OverCeilingError(
            f"estimated {estimate.requests} SDMX request(s) exceeds the async admission "
            f"ceiling cap={async_max_requests}. Narrow the identity (another "
            "slot binding, a shorter window) to bring the estimate under the cap.",
            ceiling_kind="operation_requests",
            count=estimate.requests,
            cap=async_max_requests,
        )


__all__ = [
    "COVERAGE_RESTARTS",
    "STALE_CHECK_SECONDS",
    "UNCOVERED_MEMBER_CAP",
    "MaterialisePermits",
    "RebuildFlights",
    "StaleCoverageSnapshot",
    "bounded_materialise_slot",
    "refuse_async_over_ceiling",
    "refuse_excel_over_estimate",
    "refuse_inline_over_ceiling",
    "sync_estimate",
    "under_current_coverage",
    "within_sync_budget",
]
