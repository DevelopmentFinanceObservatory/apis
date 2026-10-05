"""The default place scope: the places a subject-only StatVar covers in a request that names no
place.

A StatVar's scope is every place where it holds observations, at every level: countries, regions
and aggregates, Earth, administrative areas, cities, organisations and any other place SDMX
`/availability` lists for its `observationAbout`. Search's default scope and the datasets mint
both read it here (`read_default_presence`), so the two agree by construction. The scope is a
function of the StatVar, the graph generation and the backends read: a StatVar gets the same
scope minted alone, in a group, in a collection, and in Search, which probes, counts and
fetches each member it answers with at that member's own scope. Places are in `scope_places`
order, by dcid with Earth last.

The rule, per StatVar:

1. A StatVar whose scope is settled under the graph client's generation stamp
   (`GenerationStamp.key`) and the backends its reads take (each reader's `source_key`) takes
   it from the process-wide memo with no read.
2. Otherwise its unfiltered availability read runs on the module's settle pool
   (`AVAILABILITY_WORKERS` threads), single-flight per stamp key, backends and StatVar, for at
   most `SETTLE_SECONDS` from when it holds its GET slot, capped at `MAX_EVERY_PLACE` values
   and `MAX_EVERY_PLACE_BYTES`, and for at most `SETTLE_DEADLINE_SECONDS` from its submission,
   slot waits included. The request waits for it for `EVERY_PLACE_SECONDS`. The read
   settles whether or not the request still waits, and its outcome is memoised when it is
   definite: a list within the caps, a value or byte cap, or an HTTP 5xx other than 502, 503
   and 504. A gateway status or a read past either time bound is not memoised.
3. A list is checked against the graph: one batched `typeOf` read (`node_arcs_many`) over the
   places of every list the request holds, for at most `PLACE_CHECK_SECONDS`. A place with no
   node leaves the scope, and the checked scope is memoised. Availability lists data keys the
   graph holds no node for: on dev `LifeExpectancy_Person_Female` lists `country/SouthAfrica`
   beside `country/ZAF`.
4. Every other StatVar takes its countries with data, from one `entities_with_data_by_variable`
   read over Earth's countries and Earth for all of them, else Earth when only Earth holds data,
   else nothing. That scope is memoised when availability's answer was definite. A StatVar
   whose read is still running after `EVERY_PLACE_SECONDS`, whose read ended in a gateway
   status or at either time bound, whose node check ran past its window or failed on the
   server, that arrives while every settle thread is busy, or that passes
   `MAX_COLD_READS_PER_REQUEST`, takes its countries for that request only and is listed in
   `DefaultPresence.unsettled`. A read not yet started is queued to settle while fewer than
   `MAX_QUEUED_SETTLES` wait, and is dropped past that.
5. The request's deadline bounds every wait and graph read. A wait or read it ends raises
   `PlaceScopeDeadlineError`, which both doors answer 504 `timeout`.
6. Any other failed availability read raises `PlaceScopeReadError`, a `GraphReadError` carrying
   the response's `upstream_status`, whichever door's reader made the read. Reads are
   single-flight across doors whose readers name the same backends, so a request can wait on a
   read another door's reader started, and both doors refuse the error as a retryable upstream
   failure.

Each countries fallback an availability read or node check takes, and each read that ends in
an error, logs one warning naming its StatVars and the cause.

Search answers fast and marks unsettled scopes. The mint settles. A settling read
(`read_default_presence(settle=True)`) takes no fallback for time, a busy pool or the per-request
cap: it starts or joins every StatVar's read and waits for its outcome, up to the request's
deadline and `SETTLE_DEADLINE_SECONDS`, so a cold mint and a warm re-mint state the same scopes.
It takes countries on a definite fallback, and on a transient one, which it lists in
`DefaultPresence.unsettled`.

On dev, GDP per capita lists 215 places in 0.6 to 1.0 s and life expectancy 1,846 in 0.7 to
0.8 s, so both answer at every level, while `Count_Person` fails HTTP 500 after about 9 s and
`Count_Person_Female` lists 701,813 places, so both take their countries.

DFO and base Data Commons variables read our mixer's `/availability`. UN System Data Commons
variables are absent from our mixer's SDMX service and read the UN deployment's, which the
origin graph client holds (`PresenceReader.un_availability`).

The two sources count provenance differently. Availability counts observations under every
provenance, and origin admission applies later, at the facet probe, so an every-place scope can
hold a place whose only data a withheld provenance holds, which serves no row there. The
countries fallback counts admitted facets only.
"""

from __future__ import annotations

import contextlib
import logging
import sys
import threading
import time
from collections.abc import Iterable, Mapping, Sequence
from concurrent.futures import Future, InvalidStateError, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from enum import Enum
from typing import Final, NamedTuple, Protocol

from cachetools import TLRUCache

from dc_graph import GraphDeadlineExceededError, GraphReadError
from dc_sdmx import EntityFilter, SdmxCapExceededError, SdmxClientError
from dfo_platform.executor import submit_with_context
from graph_generation import GenerationStamp
from graph_generation.stamp import ttu_from_stamp_key
from materialise.errors import MaterialisationUsageError
from origin_catalogue.graph import is_un_dcid

logger = logging.getLogger(__name__)

DEFAULT_SCOPE_PARENT: Final = "Earth"
"""The worldwide place. A scope holding it alone is a worldwide total."""

DEFAULT_SCOPE_CHILD_TYPE: Final = "Country"
"""The place type of the countries fallback's members under `DEFAULT_SCOPE_PARENT`."""

MAX_EVERY_PLACE: Final = 5_000
"""The most places a StatVar's unfiltered read returns before its scope falls back to countries.
It admits GDP (1,996 places on dev) and life expectancy (1,846). A collection's facet probe
counts each member's own places against `facets._MAX_COLLECTION_PROBE_PAIRS` (200,000), so 40
members at a full 5,000-place scope fit."""

MAX_EVERY_PLACE_BYTES: Final = MAX_EVERY_PLACE * 64
"""The most body bytes the unfiltered read downloads before its scope falls back to countries,
as it does past `MAX_EVERY_PLACE`. It allows 64 bytes a place, and a listed place costs its dcid
plus 3 bytes of quoting and separator: on dev GDP's 1,996 places arrive in 46,215 bytes (23 a
place) and `Count_Person_Female`'s 701,813 in 23.8 MB (34 a place). The cap, 320,000 bytes,
stops that read after its first 1.3 percent."""

EVERY_PLACE_SECONDS: Final = 2.0
"""How long a request waits for a StatVar's unfiltered read before it takes the StatVar's
countries for that request. The read runs on to `SETTLE_SECONDS`. On dev a light read takes 0.5
to 1.0 s cold, and 1.0 s at the median beside dense reads 16 at once. Every read past 2 s
listed hundreds of thousands of places or failed."""

SETTLE_SECONDS: Final = 15.0
"""How long an unfiltered read runs, from when it holds its GET slot, before it ends unsettled.
On dev the slowest full answer, `Count_Person_Female`'s 701,813 places, took 4.5 to 13.8 s, and
`Count_Person`'s HTTP 500 came after 8.9 to 10.9 s."""

SETTLE_DEADLINE_SECONDS: Final = 30.0
"""How long an unfiltered read may take from when it is submitted to the settle pool: its wait
for a settle thread and a GET slot, plus its own `SETTLE_SECONDS`. A read past it ends unsettled
and is not memoised, and a queued read whose bound has passed ends without a request, so no read
holds a settle thread, or delays process exit, longer than this after its submission."""

PLACE_CHECK_SECONDS: Final = 4.0
"""How long the batched node read that confirms listed places may run before every StatVar it
covers takes its countries for the request. On dev it reads GDP's 1,996 places in 0.52 to
0.66 s and 2,648 places in 0.99 to 1.2 s; about one read in 20 runs past 1.9 s."""

AVAILABILITY_WORKERS: Final = 16
"""The settle pool's threads, so the most availability reads running at once in the process.
It matches Search's SDMX GET cap (`SDMX_TOTAL_SEMAPHORE`). On dev 100 light reads take 1.7 s
16 at once and 2.9 to 3.4 s 8 at once."""

MAX_QUEUED_SETTLES: Final = 1_000
"""The most reads waiting for a settle thread. At the dev rate of 100 light reads in 1.7 s a
full queue drains in about 17 s. A read arriving past it is dropped and starts when a later
request needs it."""

MAX_COLD_READS_PER_REQUEST: Final = 100
"""The most availability reads one request starts and waits for. A request reading more
StatVars takes their countries for the rest and queues their reads to settle. On dev 100 light
reads take 1.7 s 16 at once, inside `EVERY_PLACE_SECONDS`."""

MEMO_MAX_PLACES: Final = 2_000_000
"""The memo's bound in memoised places, a fallback marker counting one. Listed places are
interned, so an entry costs a pointer a place: about 16 MB at the bound."""

_SUBJECT_ROLE: Final = "observationAbout"
_NODE_PROBE: Final = ("typeOf",)
"""The arc the place check reads. Every place node carries one."""
_SERVER_ERROR: Final = 500
_GATEWAY_STATUSES: Final = frozenset({502, 503, 504})
"""Server statuses a later read can answer differently, so their fallback is not memoised."""
_READ_FAILURES: Final = (SdmxClientError, GraphReadError, ValueError)
"""Errors a failed availability read raises besides any carrying `upstream_status`: the SDMX
client's, a graph client's, and a malformed body's `ValueError`."""


class DefaultScopeUnknownError(GraphReadError):
    """Earth's containment read named no country. The graph places every country under
    Earth, so the read failed; taken as "no country", it would serve Earth's worldwide total
    in place of every country. The doors refuse it as a retryable upstream failure."""


class PlaceScopeReadError(GraphReadError):
    """A StatVar's availability read failed in a way the countries fallback does not cover: a
    transport failure, a 4xx status or a malformed response. `upstream_status` is the
    response's status, or `None` when no response arrived, and the reader's own error is the
    `__cause__`. The doors refuse it as a retryable upstream failure."""


class PlaceScopeDeadlineError(TimeoutError):
    """The request's deadline ran out before the default place scope of `stat_vars` was read.
    Those StatVars get no scope and the request refuses as a timeout, which both doors answer
    504 `timeout`."""

    def __init__(self, stat_vars: Sequence[str]) -> None:
        super().__init__(
            f"the request's deadline ran out before the default place scope of "
            f"{list(stat_vars)!r} was read"
        )
        self.stat_vars = tuple(stat_vars)


class AvailabilityReader(Protocol):
    """The SDMX `/availability` read: the values of `component` across the observations of
    `variables` that pass `filters`, raising `SdmxCapExceededError` past `max_values`.
    `max_bytes` bounds the body it downloads, raising the same error past it. `deadline` bounds
    the read and `read_seconds` bounds it from when it holds its GET slot. Either expiry raises
    `TimeoutError`. A failed read raises an error carrying the response's `upstream_status`,
    which `read_default_presence` raises as `PlaceScopeReadError`. `dc_sdmx.SdmxClient` and
    Search's SDMX clients satisfy it."""

    def source_key(self) -> str:
        """The backend the reader reads, equal for two readers exactly when they read the same
        backend. The memo keys on it."""
        ...

    def availability_values(
        self,
        *,
        variables: Sequence[str],
        component: str,
        filters: Mapping[str, EntityFilter] | None = None,
        max_values: int | None = None,
        max_bytes: int | None = None,
        deadline: float | None = None,
        read_seconds: float | None = None,
    ) -> list[str]: ...


class PresenceReader(Protocol):
    """The graph reads the default scope takes. `dc_graph.GraphClient`,
    `origin_catalogue.graph.OriginGraphClient` and Search's graph clients satisfy it."""

    def generation_stamp(self) -> GenerationStamp:
        """The stamp the client's caches key on, without I/O."""
        ...

    def source_key(self) -> str:
        """The backends the client reads, equal for two clients exactly when they read the
        same backends. The memo keys on it."""
        ...

    def un_availability(self) -> AvailabilityReader | None:
        """The UN deployment's `/availability`, or `None` when the client reaches no UN
        deployment."""
        ...

    def node_arcs_many(
        self, dcids: Sequence[str], *, properties: Sequence[str], deadline: float | None = None
    ) -> Mapping[str, Mapping[str, tuple[str, ...]] | None]:
        """Each of `dcids` mapped to its arcs, or to `None` when the graph holds no node for it.
        A failed read raises `dc_graph.GraphReadError`, and a read `deadline` ends raises
        `dc_graph.GraphDeadlineExceededError`."""
        ...

    def child_places_batch(
        self, parent_dcids: list[str], child_type: str, *, deadline: float | None = None
    ) -> dict[str, list[str]]: ...

    def entities_with_data_by_variable(
        self,
        variable_dcids: list[str],
        entity_dcids: list[str],
        *,
        deadline: float | None = None,
    ) -> dict[str, list[str]]: ...


@dataclass(frozen=True)
class DefaultPlaceScope:
    """A default scope, of one StatVar or the union of several, in `scope_places` order.

    `every_place` is true when the places are every place with data, and false when they are
    countries with data, or Earth, because a full list was too large or unavailable. A union
    is every place only when each of its scopes is.
    """

    places: tuple[str, ...]
    every_place: bool


NO_DEFAULT_SCOPE: Final = DefaultPlaceScope(places=(), every_place=True)
"""The scope of StatVars with data nowhere, and of a request that binds a place."""


@dataclass(frozen=True)
class DefaultPresence:
    """The default scopes of a set of StatVars.

    `countries` lists Earth's countries as the graph names them, or nothing when no read needed
    them. `scopes` holds each StatVar's own scope. `unsettled` names the StatVars whose scope is
    a countries fallback for this request only (rule 4 of the module docstring), which a later
    request can answer differently. `cold_read` names the StatVars whose availability read this
    presence's request started and waited for, which `cold_reads` counts against
    `MAX_COLD_READS_PER_REQUEST`.
    """

    countries: tuple[str, ...]
    scopes: Mapping[str, DefaultPlaceScope]
    unsettled: frozenset[str] = field(default_factory=frozenset)
    cold_read: frozenset[str] = field(default_factory=frozenset)

    @property
    def cold_reads(self) -> int:
        """The availability reads this presence's request started and waited for."""
        return len(self.cold_read)

    def scope(self, stat_var: str) -> DefaultPlaceScope:
        """The default scope of `stat_var`. A StatVar the read did not cover raises
        `AssertionError`."""
        scope = self.scopes.get(stat_var)
        if scope is None:
            raise AssertionError(f"no default-scope presence was read for {stat_var!r}")
        return scope

    def facet_entities(self, stat_vars: Iterable[str]) -> list[str]:
        """The union of the scopes of `stat_vars`, in `scope_places` order. One facet read at
        these places covers every member's own scope."""
        return list(self.union_scope(stat_vars).places)

    def union_scope(self, stat_vars: Iterable[str]) -> DefaultPlaceScope:
        """The places a set of StatVars covers when each is served at its own scope, every
        place only when each scope is."""
        scopes = [self.scope(stat_var) for stat_var in dict.fromkeys(stat_vars)]
        return scope_of_places(
            (place for scope in scopes for place in scope.places),
            every_place=all(scope.every_place for scope in scopes),
        )

    def merged(self, other: DefaultPresence) -> DefaultPresence:
        """This presence with the StatVars `other` covers and this one does not. Both lists
        name Earth's countries, so this one's stands unless it is empty. `cold_read` is the
        union of both sets of StatVars, so a cold read that `other` inherited from this
        presence through `known` counts once."""
        return DefaultPresence(
            countries=self.countries or other.countries,
            scopes={**other.scopes, **self.scopes},
            unsettled=self.unsettled | other.unsettled.difference(self.scopes),
            cold_read=self.cold_read | other.cold_read,
        )


@dataclass(frozen=True, slots=True)
class _Listed:
    """Availability's every-place list, in `scope_places` order, not yet checked against the
    graph's nodes."""

    places: tuple[str, ...]


class _Fallback(Enum):
    """An availability read that answers with countries. `DEFINITE` is memoised. `TRANSIENT`
    holds for one request."""

    DEFINITE = "definite"
    TRANSIENT = "transient"


type _Outcome = _Listed | _Fallback


class _MemoKey(NamedTuple):
    """A memo and flight key. `stamp` leads, as `ttu_from_stamp_key` reads it first.
    `availability` is the `source_key` of the StatVar's availability reader and `graph` the
    graph client's, so readers of different backends never share an entry."""

    stamp: str
    availability: str
    graph: str
    stat_var: str


type _Memoised = DefaultPlaceScope | _Listed | _Fallback


def _memo_size(value: _Memoised) -> int:
    return 1 if isinstance(value, _Fallback) else max(1, len(value.places))


_lock = threading.Lock()
_memo: TLRUCache[_MemoKey, _Memoised] = TLRUCache(
    maxsize=MEMO_MAX_PLACES, ttu=ttu_from_stamp_key, getsizeof=_memo_size
)
"""Settled scopes, and definite availability outcomes a request has not yet turned into one: a
list awaiting its node check, or `_Fallback.DEFINITE` awaiting its countries. Guarded by
`_lock`."""
_flights: dict[_MemoKey, Future[_Outcome]] = {}
"""Availability reads registered and not finished, one per key. Guarded by `_lock`."""
_pool = ThreadPoolExecutor(max_workers=AVAILABILITY_WORKERS, thread_name_prefix="place-scope")


def clear_memo() -> None:
    """Drop every memoised scope. A generation provider calls it on a token or catalogue change
    (`GenerationProvider.register`). Reads still running memoise under their own, older key."""
    with _lock:
        _memo.clear()


def memo_stats() -> Mapping[str, int]:
    """The memo's entries and places, and the availability reads in flight."""
    with _lock:
        return {"entries": len(_memo), "places": int(_memo.currsize), "in_flight": len(_flights)}


def read_default_presence(
    *,
    graph: PresenceReader,
    availability: AvailabilityReader,
    stat_vars: Iterable[str],
    known: DefaultPresence | None = None,
    deadline: float | None = None,
    settle: bool = False,
) -> DefaultPresence:
    """The default scopes of `stat_vars` by the module's rule, added to `known`.

    `availability` reads our mixer's `/availability`. A UN variable reads the one
    `graph.un_availability()` returns, and a graph that reaches none raises
    `MaterialisationUsageError` before any read. StatVars `known` covers are not read again,
    and `known.cold_reads` counts toward `MAX_COLD_READS_PER_REQUEST`. With nothing to read it
    makes no request, and an empty `stat_vars` without `known` returns an empty presence.
    Earth's countries come from `known.countries`, else from one `earth_countries` read when a
    StatVar falls back. A listing naming none raises `DefaultScopeUnknownError`. `deadline`,
    the request's, bounds every wait and graph read, and one it ends raises
    `PlaceScopeDeadlineError`. A failed availability read the countries fallback does not cover
    raises `PlaceScopeReadError`. Any other error propagates. With `settle`, it waits for every
    StatVar's read to end, past `EVERY_PLACE_SECONDS`, a busy pool and the per-request cap.
    """
    missing = [sv for sv in dict.fromkeys(stat_vars) if known is None or sv not in known.scopes]
    if not missing:
        return known if known is not None else DefaultPresence(countries=(), scopes={})
    readers = {sv: _un_reader(sv, graph) if is_un_dcid(sv) else availability for sv in missing}
    if _past(deadline):
        raise PlaceScopeDeadlineError(missing)
    stamp_key = graph.generation_stamp().key
    graph_source = graph.source_key()
    keys = {
        sv: _MemoKey(stamp_key, reader.source_key(), graph_source, sv)
        for sv, reader in readers.items()
    }
    cold_read = set(known.cold_read) if known is not None else set()

    settled: dict[str, DefaultPlaceScope] = {}
    outcomes: dict[str, _Outcome] = {}
    waiting: dict[str, Future[_Outcome]] = {}
    started: list[tuple[_MemoKey, AvailabilityReader, Future[_Outcome]]] = []
    with _lock:
        for sv in missing:
            key = keys[sv]
            memoised = _memo.get(key)
            if isinstance(memoised, DefaultPlaceScope):
                settled[sv] = memoised
            elif memoised is not None:
                outcomes[sv] = memoised
            elif (flight := _flights.get(key)) is not None:
                waiting[sv] = flight
            elif settle or (
                len(cold_read) < MAX_COLD_READS_PER_REQUEST and len(_flights) < AVAILABILITY_WORKERS
            ):
                waiting[sv] = _register(key)
                started.append((key, readers[sv], waiting[sv]))
                cold_read.add(sv)
            else:
                if len(_flights) < AVAILABILITY_WORKERS + MAX_QUEUED_SETTLES:
                    started.append((key, readers[sv], _register(key)))
                outcomes[sv] = _Fallback.TRANSIENT
    for key, reader, flight in started:
        _submit(key, reader, flight)

    window = SETTLE_DEADLINE_SECONDS if settle else EVERY_PLACE_SECONDS
    outcomes |= _await(waiting, window=window, deadline=deadline)
    listed = {sv: out.places for sv, out in outcomes.items() if isinstance(out, _Listed)}
    if listed:
        known_places = _graph_places(listed, graph=graph, deadline=deadline)
        for sv, places in listed.items():
            if known_places is None:
                outcomes[sv] = _Fallback.TRANSIENT
                continue
            settled[sv] = DefaultPlaceScope(
                places=tuple(place for place in places if place in known_places),
                every_place=True,
            )
            _memoise(keys[sv], settled[sv])

    fallback = [sv for sv in missing if isinstance(outcomes.get(sv), _Fallback)]
    countries = known.countries if known is not None else ()
    if fallback:
        countries = countries or _countries(graph, stat_vars=fallback, deadline=deadline)
        for sv, scope in _countries_scopes(fallback, countries, graph, deadline=deadline).items():
            settled[sv] = scope
            if outcomes[sv] is _Fallback.DEFINITE:
                _memoise(keys[sv], scope)

    unsettled = frozenset(sv for sv in fallback if outcomes[sv] is _Fallback.TRANSIENT)
    return DefaultPresence(
        countries=countries,
        scopes={**(known.scopes if known is not None else {}), **settled},
        unsettled=(known.unsettled if known is not None else frozenset()) | unsettled,
        cold_read=frozenset(cold_read),
    )


def earth_countries(graph: PresenceReader, *, deadline: float | None = None) -> tuple[str, ...]:
    """Earth's countries as the graph names them, from one read. None raises
    `DefaultScopeUnknownError`."""
    countries = tuple(
        graph.child_places_batch(
            [DEFAULT_SCOPE_PARENT], DEFAULT_SCOPE_CHILD_TYPE, deadline=deadline
        ).get(DEFAULT_SCOPE_PARENT, [])
    )
    if not countries:
        raise DefaultScopeUnknownError(
            f"the containment read of {DEFAULT_SCOPE_PARENT!r} named no "
            f"{DEFAULT_SCOPE_CHILD_TYPE!r}, so the default place scope is unknown"
        )
    return countries


def default_place_scope(
    stat_var: str,
    *,
    graph: PresenceReader,
    availability: AvailabilityReader,
    deadline: float | None = None,
    settle: bool = False,
) -> DefaultPlaceScope:
    """The default scope of `stat_var` (`read_default_presence`)."""
    return read_default_presence(
        graph=graph,
        availability=availability,
        stat_vars=[stat_var],
        deadline=deadline,
        settle=settle,
    ).scope(stat_var)


def scope_places(places: Iterable[str]) -> tuple[str, ...]:
    """`places` in the one order a default scope states them in, on every door: every place
    but Earth by dcid, then Earth. Search's scope, its facet reads and the mint's group scope
    all order their places here."""
    distinct = set(places)
    earth = (DEFAULT_SCOPE_PARENT,) if DEFAULT_SCOPE_PARENT in distinct else ()
    return (*sorted(distinct - {DEFAULT_SCOPE_PARENT}), *earth)


def scope_of_places(places: Iterable[str], *, every_place: bool) -> DefaultPlaceScope:
    """The default scope whose places are `places`, in `scope_places` order. An empty scope keeps
    `every_place`, so an empty countries fallback reads as a fallback."""
    return DefaultPlaceScope(places=scope_places(places), every_place=every_place)


def is_worldwide(places: Sequence[str]) -> bool:
    """Whether a scope is Earth's worldwide total alone."""
    return tuple(places) == (DEFAULT_SCOPE_PARENT,)


def _register(key: _MemoKey) -> Future[_Outcome]:
    """A new single flight for `key`, which `_submit` starts. The caller holds `_lock`. A
    flight leaves `_flights` on every end: resolved by `_settle`, failed to submit, its pool
    task cancelled, or itself cancelled."""
    flight: Future[_Outcome] = Future()
    flight.add_done_callback(lambda done: _on_cancel(key, done) if done.cancelled() else None)
    _flights[key] = flight
    return flight


def _on_cancel(key: _MemoKey, flight: Future[_Outcome]) -> None:
    """End a cancelled flight. `Future.cancel` alone leaves `concurrent.futures.wait` blind to
    the cancellation until its timeout, so notify its waiters first."""
    flight.set_running_or_notify_cancel()
    _end_flight(key, flight)


def _submit(key: _MemoKey, reader: AvailabilityReader, flight: Future[_Outcome]) -> None:
    """Submit `flight`'s read to `_pool`, under a deadline `SETTLE_DEADLINE_SECONDS` from now.
    The caller does not hold `_lock`, so an executor that runs the read inline cannot deadlock
    on it. A failed submission ends the flight with its error, and a pool task cancelled before
    it runs, as a pool shutdown cancels queued tasks, cancels the flight."""
    deadline = time.monotonic() + SETTLE_DEADLINE_SECONDS
    try:
        task = submit_with_context(_pool, _settle, key, reader, flight, deadline=deadline)
    except BaseException as exc:
        _end_flight(key, flight)
        with contextlib.suppress(InvalidStateError):
            flight.set_exception(exc)
        return
    task.add_done_callback(lambda done: flight.cancel() if done.cancelled() else None)


def _settle(
    key: _MemoKey, reader: AvailabilityReader, flight: Future[_Outcome], *, deadline: float
) -> None:
    """One StatVar's unfiltered availability read under `deadline`, memoised under `key` when
    definite, resolving `flight`. The flight ends before it resolves, so a later request finds
    the memo or no flight. `flight` resolves outside `_lock`, so its waiters' callbacks never
    run under it. A read that ends in an error logs it once, whether or not a request still
    waits for it."""
    try:
        outcome = _every_place(key.stat_var, reader, deadline=deadline)
    except BaseException as exc:
        if isinstance(exc, Exception):
            logger.warning(
                "The availability read of %r under stamp %r ended in an error: %s",
                key.stat_var,
                key.stamp,
                exc,
            )
        _end_flight(key, flight)
        with contextlib.suppress(InvalidStateError):
            flight.set_exception(exc)
        return
    _end_flight(key, flight, outcome=outcome)
    # A flight cancelled while its read ran stays cancelled. The outcome is memoised above.
    with contextlib.suppress(InvalidStateError):
        flight.set_result(outcome)


def _end_flight(
    key: _MemoKey, flight: Future[_Outcome], *, outcome: _Outcome | None = None
) -> None:
    """Remove `flight` from `_flights` and memoise its definite `outcome`, under `_lock`."""
    with _lock:
        if _flights.get(key) is flight:
            del _flights[key]
        if outcome is not None and outcome is not _Fallback.TRANSIENT and key not in _memo:
            _memo[key] = outcome


def _every_place(stat_var: str, reader: AvailabilityReader, *, deadline: float) -> _Outcome:
    """The unfiltered read under the caps, `SETTLE_SECONDS` and `deadline`. A cap or a server
    error answers `DEFINITE`. A gateway status, either time bound, or a `deadline` already past
    before the read answers `TRANSIENT`. Each fallback logs a warning. Any other read failure,
    an error carrying `upstream_status` or one of `_READ_FAILURES`, raises
    `PlaceScopeReadError`. Any other error is a defect and propagates as itself."""
    if _past(deadline):
        return _fallback(
            stat_var, _Fallback.TRANSIENT, "its settle deadline passed before the read"
        )
    try:
        places = reader.availability_values(
            variables=[stat_var],
            component=_SUBJECT_ROLE,
            max_values=MAX_EVERY_PLACE,
            max_bytes=MAX_EVERY_PLACE_BYTES,
            deadline=deadline,
            read_seconds=SETTLE_SECONDS,
        )
    except SdmxCapExceededError as exc:
        return _fallback(stat_var, _Fallback.DEFINITE, f"the read passed a cap: {exc}")
    except TimeoutError as exc:
        return _fallback(stat_var, _Fallback.TRANSIENT, f"the read ran past a time bound: {exc}")
    except Exception as exc:
        if not hasattr(exc, "upstream_status") and not isinstance(exc, _READ_FAILURES):
            raise
        status = getattr(exc, "upstream_status", None)
        if not isinstance(status, int) or status < _SERVER_ERROR:
            raise PlaceScopeReadError(
                f"the availability read of {stat_var!r} failed: {exc}",
                upstream_status=status if isinstance(status, int) else None,
            ) from exc
        outcome = _Fallback.TRANSIENT if status in _GATEWAY_STATUSES else _Fallback.DEFINITE
        return _fallback(stat_var, outcome, f"the read failed with HTTP {status}: {exc}")
    return _Listed(tuple(sys.intern(place) for place in scope_places(places)))


def _fallback(stat_var: str, outcome: _Fallback, cause: str) -> _Fallback:
    """`outcome`, logged with `stat_var` and its `cause`."""
    logger.warning(
        "The default place scope of %r falls back to its countries (%s): %s",
        stat_var,
        outcome.value,
        cause,
    )
    return outcome


def _await(
    waiting: Mapping[str, Future[_Outcome]], *, window: float, deadline: float | None
) -> dict[str, _Outcome]:
    """The outcome of each read in `waiting` that ends within `window` seconds, else
    `TRANSIENT`. A wait the request's `deadline` ends first raises `PlaceScopeDeadlineError`
    for the reads still running. A read's own error propagates."""
    if not waiting:
        return {}
    cut_by_deadline = False
    if deadline is not None:
        remaining = max(0.0, deadline - time.monotonic())
        cut_by_deadline = remaining <= window
        window = min(window, remaining)
    done, running = wait(waiting.values(), timeout=window)
    if running and cut_by_deadline:
        raise PlaceScopeDeadlineError([sv for sv, flight in waiting.items() if flight in running])
    return {
        sv: flight.result() if flight in done and not flight.cancelled() else _Fallback.TRANSIENT
        for sv, flight in waiting.items()
    }


def _graph_places(
    listed: Mapping[str, Sequence[str]], *, graph: PresenceReader, deadline: float | None
) -> frozenset[str] | None:
    """The places of `listed` the graph holds a node for, from one batched `typeOf` read that
    runs for at most `PLACE_CHECK_SECONDS` under the request's `deadline`. `None` when that
    read runs past its own window or fails with an HTTP status of 500 or more, which logs a
    warning naming the StatVars of `listed`. A read the request's deadline ends raises
    `PlaceScopeDeadlineError`. Any other error propagates."""
    distinct = list(dict.fromkeys(place for places in listed.values() for place in places))
    if not distinct:
        return frozenset()
    if _past(deadline):
        raise PlaceScopeDeadlineError(list(listed))
    window = time.monotonic() + PLACE_CHECK_SECONDS
    try:
        arcs = graph.node_arcs_many(
            distinct,
            properties=_NODE_PROBE,
            deadline=window if deadline is None else min(window, deadline),
        )
    except GraphDeadlineExceededError as exc:
        if _past(deadline):
            raise PlaceScopeDeadlineError(list(listed)) from exc
        _check_fallback(listed, places=len(distinct), cause=exc)
        return None
    except GraphReadError as exc:
        if not isinstance(exc.upstream_status, int) or exc.upstream_status < _SERVER_ERROR:
            raise
        _check_fallback(listed, places=len(distinct), cause=exc)
        return None
    return frozenset(place for place in distinct if arcs.get(place) is not None)


def _check_fallback(listed: Mapping[str, Sequence[str]], *, places: int, cause: Exception) -> None:
    """Log that the node check of `places` places ended in `cause`, so every StatVar of
    `listed` takes its countries for this request."""
    logger.warning(
        "The node check of %d places failed, so %r fall back to their countries (transient): %s",
        places,
        list(listed),
        cause,
    )


def _countries(
    graph: PresenceReader, *, stat_vars: Sequence[str], deadline: float | None
) -> tuple[str, ...]:
    """`earth_countries`, with a read the request's deadline ends raising
    `PlaceScopeDeadlineError` for `stat_vars`."""
    try:
        return earth_countries(graph, deadline=deadline)
    except GraphDeadlineExceededError as exc:
        if _past(deadline):
            raise PlaceScopeDeadlineError(stat_vars) from exc
        raise


def _countries_scopes(
    stat_vars: Sequence[str],
    countries: Sequence[str],
    graph: PresenceReader,
    *,
    deadline: float | None,
) -> dict[str, DefaultPlaceScope]:
    """Each of `stat_vars`' countries with admitted data, else Earth when only Earth holds
    data, else nothing, from one presence read over `countries` and Earth."""
    try:
        read = graph.entities_with_data_by_variable(
            list(stat_vars), [*countries, DEFAULT_SCOPE_PARENT], deadline=deadline
        )
    except GraphDeadlineExceededError as exc:
        if _past(deadline):
            raise PlaceScopeDeadlineError(stat_vars) from exc
        raise
    scopes: dict[str, DefaultPlaceScope] = {}
    for sv in stat_vars:
        present = frozenset(read.get(sv, ()))
        places = scope_places(country for country in countries if country in present)
        if not places and DEFAULT_SCOPE_PARENT in present:
            places = (DEFAULT_SCOPE_PARENT,)
        scopes[sv] = DefaultPlaceScope(places=places, every_place=False)
    return scopes


def _memoise(key: _MemoKey, scope: DefaultPlaceScope) -> None:
    with _lock:
        _memo[key] = scope


def _past(deadline: float | None) -> bool:
    return deadline is not None and time.monotonic() >= deadline


def _un_reader(stat_var: str, graph: PresenceReader) -> AvailabilityReader:
    reader = graph.un_availability()
    if reader is None:
        raise MaterialisationUsageError(
            f"the UN System Data Commons variable {stat_var!r} needs a graph client that "
            f"reaches the UN deployment for its default place scope; {type(graph).__name__} "
            "reaches none"
        )
    return reader
