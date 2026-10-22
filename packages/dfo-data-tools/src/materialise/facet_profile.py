"""Each variable's facets at its default place scope, and the default a request naming nothing
receives.

A variable's profile is what a mint that binds no place, no years and no facet probes and
elects, read the same way, so a card that reports it cannot disagree with that mint:

- A role-bearing variable, one with coverage role names, probes coverage under unbound roles
  (`facets.coverage_probe`).
- Subject-only variables take their default place scopes from one presence read for the batch
  (`place_scope.read_default_presence`, settling as a mint does), then
  `observation_facets_batch_many` over the union of those places, in reads of at most
  `PAIRS_PER_READ` variable and place pairs, each bounded by a byte cap, side by side on the
  executor given for them, merged by place. Each variable is cut to its own scope
  (`facets.subject_probe`), so its profile does not depend on its companions.

The default is `facets.served_facets` with no window and the probe's breadth across every
place or role tuple. `served_facets` reads the origin catalogue bound to the context, so a
read needs one bound.

`FacetProfileCache` holds profiles in a `SingleFlightCache` keyed on variable, catalogue build,
coverage build and source scope, under the graph generation. Each missing variable reads in a
flight of its own, so a batch's profiles store as each finishes and a retry reads only the
variables still missing. Its flights run on the executor it is given, and their facet reads
on a second one, so that executor bounds the facet reads of every flight together. A flight
runs in an empty context binding only the catalogue it is keyed on and its deadline, so it
writes into no request's records. A variable without a profile is `ProfileUnavailable`, which
is never stored, whether its scope is unsettled, its own probe raised, one of its reads failed
or its flight passed its deadline.
"""

from __future__ import annotations

import contextvars
import logging
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from concurrent.futures import BrokenExecutor, CancelledError, Executor
from dataclasses import dataclass
from enum import StrEnum
from functools import partial
from typing import TYPE_CHECKING, NamedTuple, Protocol

from dc_graph import GraphCapExceededError, GraphDeadlineExceededError, GraphReadError
from dc_sdmx import SdmxClientError
from dfo_platform.executor import submit_with_context
from dfo_platform.request_deadline import request_deadline
from dfo_platform.single_flight_cache import SingleFlightCache
from graph_generation import GenerationProvider
from graph_release.context import bound_release, current_release
from materialise.facets import (
    BreadthUnread,
    MemberProbe,
    PlaceAdmission,
    SeriesCounts,
    coverage_probe,
    served_facets,
    subject_probe,
)
from materialise.graph import Facet
from materialise.place_scope import (
    MAX_EVERY_PLACE,
    AvailabilityReader,
    PresenceReader,
    read_default_presence,
)
from origin_catalogue import Catalogue, bound_catalogue, current_catalogue

if TYPE_CHECKING:
    # Keep coverage annotation-only. Importing its modules adds NumPy to the pure install.
    from series_coverage.index import CoverageIndex
    from series_coverage.shared import CoverageSnapshot

logger = logging.getLogger(__name__)

_STAT_VAR_REF_PREFIX = "dcid:"

FLIGHT_TIMEOUT_S = 5.0
"""How long one variable's profile flight may run, queue time included, before the variable
answers `ProfileGap.TIMEOUT`. The flight holds one variable's reads: its default scope's
presence read, then at most `CHUNK_READS` facet reads side by side."""

MAX_BYTES = 2 * 1024 * 1024
"""The byte cap on each facet read a flight makes (`read_facet_profiles`)."""

PAIRS_PER_READ = 1_000
"""The most variable and place pairs one facet read of a flight names. The widest
default-scope reads measured on the dev graph cost at most about 670 bytes a pair: IMF WEO
nominal GDP's 279 facets read 134 KB at its 210 places, base nominal GDP 580 KB at 1,996
places and female life expectancy 246 KB at 1,884. A read of 1,000 pairs therefore stays near
0.65 MiB, under a third of `MAX_BYTES`, where one read of a 5,000-place scope
(`place_scope.MAX_EVERY_PLACE`) at that rate would pass it."""

CHUNK_READS = -(-MAX_EVERY_PLACE // PAIRS_PER_READ)
"""The most facet reads one variable's flight makes: its widest default scope,
`place_scope.MAX_EVERY_PLACE` places, in `PAIRS_PER_READ` chunks. They run side by side while
the cache's read executor has free threads, so a flight's `FLIGHT_TIMEOUT_S` covers about one
read's latency rather than one per chunk. That executor's size, not the number of flights,
bounds the facet reads a cache has in flight (`shared_profiles.FACET_READ_WORKERS`)."""

PROFILE_CACHE_MAXSIZE = 64
"""The variables a process's `FacetProfileCache` keeps, one size for every surface. An entry's
size follows its facets times its places, held in `places_by_facet`, measured with
`tracemalloc` on profiles built from undated facet reads (what `observation_facets_batch_many`
returns) at 200 places: about 120 KiB for a variable with 5 facets, which most variables have
at most, and about 2.8 MiB for WEO GDP's 279. The cache therefore holds about 7.5 MiB when its
entries are typical and at most about 180 MiB if every entry were as wide as WEO GDP."""


_TIMEOUTS = (TimeoutError, GraphDeadlineExceededError)
_FAILURES = (
    GraphReadError,
    GraphCapExceededError,
    SdmxClientError,
    BrokenExecutor,
    CancelledError,
)


class ProfileGraphReader(PresenceReader, Protocol):
    """The graph reads a profile takes: the default place scope's (`PresenceReader`) and
    batched facet reads, each bounded by `max_bytes`. `origin_catalogue.graph.OriginGraphClient`
    satisfies it."""

    def observation_facets_batch_many(
        self,
        *,
        stat_vars: Sequence[str],
        entities: Sequence[str],
        deadline: float | None = None,
        max_bytes: int | None = None,
    ) -> dict[str, dict[str, list[Facet]]]: ...


@dataclass(frozen=True, slots=True)
class FacetProfile:
    """One variable's facets where a request naming nothing reads it.

    `candidates` are the probe's facets, merged across its places or role tuples. `breadth`
    maps each facet id to the places, or the donor and recipient pairs for a role-bearing
    variable, at which it holds a point in any year. It holds no union of several facets'
    series: a ranking that joins facets counts theirs when it reads (`profile_breadth`).
    `default` is the facets such a request is served (`served_facets`), empty when no candidate dispatches. `places_by_facet` maps each
    facet id to the places it was admitted at, and is empty for a role-bearing variable.
    `places` is a subject-only variable's default place scope. `countries_fallback` is true
    when that scope is its countries with data because availability failed or listed too many
    places (`place_scope.DefaultPlaceScope.every_place`), as `Count_Person`'s HTTP 500 does on
    dev, so a card can say its places are countries only."""

    stat_var: str
    candidates: tuple[Facet, ...]
    breadth: Mapping[str, int]
    default: tuple[Facet, ...]
    places_by_facet: Mapping[str, frozenset[str]]
    places: tuple[str, ...] = ()
    countries_fallback: bool = False


class ProfileGap(StrEnum):
    """Why a variable has no profile. None is stored, so a response carrying one is not
    cached, and the next read tries again."""

    TIMEOUT = "timeout"
    """The flight, or the caller's wait for it, passed its deadline."""
    FAILED = "failed"
    """A read failed (a graph read error, a byte cap, an SDMX error), or the executor could
    not run the flight. A later read can answer."""
    DEFECT = "defect"
    """The variable's own probe or election raised for any other reason, logged as an error
    with the variable. A later read meets it again."""
    UNSETTLED_SCOPE = "unsettled_scope"
    """A variable's default place scope fell back to countries for that read only."""


@dataclass(frozen=True, slots=True)
class ProfileUnavailable:
    stat_var: str
    gap: ProfileGap


type ProfileOutcome = FacetProfile | ProfileUnavailable


def read_facet_profiles(
    stat_vars: Iterable[str],
    *,
    graph: ProfileGraphReader,
    availability: AvailabilityReader,
    coverage: CoverageIndex,
    max_bytes: int,
    deadline: float | None,
    reads: Executor | None = None,
) -> dict[str, ProfileOutcome]:
    """Each of `stat_vars`' outcome, keyed by bare dcid, read under the bound catalogue.
    `reads` runs the facet reads side by side; without it they run in turn on the calling
    thread.

    A variable fails alone: an unsettled default place scope (`DefaultPresence.unsettled`,
    which a later read can answer differently) is `ProfileGap.UNSETTLED_SCOPE`, and an error in
    its own probe or election is `ProfileGap.DEFECT` (`_isolated`). An unsettled variable is
    left out of the facet read. The reads the subject-only variables share, the presence read
    and the facet reads (`_read_facets`), receive `deadline`, and their errors propagate
    (`place_scope.PlaceScopeDeadlineError`, `dc_graph.GraphReadError`,
    `dc_graph.GraphCapExceededError`)."""
    bare = list(dict.fromkeys(sv.removeprefix(_STAT_VAR_REF_PREFIX) for sv in stat_vars))
    outcomes: dict[str, ProfileOutcome] = {
        sv: _isolated(sv, lambda sv=sv: coverage_probe(stat_var=sv, bound={}, coverage=coverage))
        for sv in bare
        if sv in coverage.role_names
    }
    subject_only = [sv for sv in bare if sv not in outcomes]
    if not subject_only:
        return outcomes

    presence = read_default_presence(
        graph=graph,
        availability=availability,
        stat_vars=subject_only,
        deadline=deadline,
        settle=True,
    )
    settled = [sv for sv in subject_only if sv not in presence.unsettled]
    outcomes.update(
        (sv, ProfileUnavailable(stat_var=sv, gap=ProfileGap.UNSETTLED_SCOPE))
        for sv in subject_only
        if sv in presence.unsettled
    )
    by_stat_var = _read_facets(
        settled,
        presence.facet_entities(settled),
        graph=graph,
        max_bytes=max_bytes,
        deadline=deadline,
        reads=reads,
    )
    for sv in settled:
        scope = presence.scope(sv)
        outcomes[sv] = _isolated(
            sv,
            lambda sv=sv, scope=scope: subject_probe(by_stat_var.get(sv, {}), scope.places),
            places=scope.places,
            countries_fallback=not scope.every_place,
        )
    return outcomes


def _read_facets(
    stat_vars: Sequence[str],
    places: Sequence[str],
    *,
    graph: ProfileGraphReader,
    max_bytes: int,
    deadline: float | None,
    reads: Executor | None,
) -> dict[str, dict[str, list[Facet]]]:
    """`stat_vars`' facets at `places`, read in place chunks of at most `PAIRS_PER_READ`
    pairs, each bounded by `max_bytes`, on `reads` under the caller's context, else in turn,
    waited on until `deadline` at most, and merged: each chunk holds other places, so a
    variable's facets at a place come from one read. A chunk past the cap is read again in
    halves, places first, so only one variable at one place past `max_bytes` raises
    `GraphCapExceededError`. The first failed chunk's error propagates, and the chunks not yet
    started are cancelled."""
    if not stat_vars or not places:
        return {}
    size = max(1, PAIRS_PER_READ // len(stat_vars))
    chunks = [places[start : start + size] for start in range(0, len(places), size)]
    read = partial(_read_chunk, stat_vars, graph=graph, max_bytes=max_bytes, deadline=deadline)
    by_stat_var: dict[str, dict[str, list[Facet]]] = {}
    if reads is None or len(chunks) == 1:
        for chunk in chunks:
            _merge_facets(by_stat_var, read(chunk))
        return by_stat_var
    futures = [submit_with_context(reads, read, chunk) for chunk in chunks]
    try:
        for future in futures:
            wait = None if deadline is None else max(0.0, deadline - time.monotonic())
            _merge_facets(by_stat_var, future.result(timeout=wait))
    except BaseException:
        for future in futures:
            future.cancel()
        raise
    return by_stat_var


def _read_chunk(
    stat_vars: Sequence[str],
    places: Sequence[str],
    *,
    graph: ProfileGraphReader,
    max_bytes: int,
    deadline: float | None,
) -> dict[str, dict[str, list[Facet]]]:
    try:
        return graph.observation_facets_batch_many(
            stat_vars=stat_vars, entities=places, deadline=deadline, max_bytes=max_bytes
        )
    except GraphCapExceededError:
        if len(places) == 1 and len(stat_vars) == 1:
            raise
    logger.warning(
        "facet profile read of %d variables at %d places passed the byte cap; reading halves",
        len(stat_vars),
        len(places),
    )
    found: dict[str, dict[str, list[Facet]]] = {}
    read = partial(_read_chunk, graph=graph, max_bytes=max_bytes, deadline=deadline)
    if len(places) > 1:
        half = len(places) // 2
        _merge_facets(found, read(stat_vars, places[:half]))
        _merge_facets(found, read(stat_vars, places[half:]))
    else:
        half = len(stat_vars) // 2
        _merge_facets(found, read(stat_vars[:half], places))
        _merge_facets(found, read(stat_vars[half:], places))
    return found


def _merge_facets(
    into: dict[str, dict[str, list[Facet]]], read: Mapping[str, Mapping[str, list[Facet]]]
) -> None:
    for stat_var, by_place in read.items():
        into.setdefault(stat_var, {}).update(by_place)


def _isolated(
    stat_var: str,
    probe: Callable[[], MemberProbe],
    *,
    places: tuple[str, ...] = (),
    countries_fallback: bool = False,
) -> ProfileOutcome:
    """`stat_var`'s profile from `probe`, so one variable's fault leaves its siblings'
    profiles standing. A timeout is `ProfileGap.TIMEOUT` and a failed read
    `ProfileGap.FAILED`; any other error is a defect, `ProfileGap.DEFECT`, logged as an error
    with the variable."""
    try:
        return _profile(stat_var, probe(), places=places, countries_fallback=countries_fallback)
    except _TIMEOUTS:
        logger.warning("facet profile of %s timed out", stat_var, exc_info=True)
        return ProfileUnavailable(stat_var=stat_var, gap=ProfileGap.TIMEOUT)
    except _FAILURES:
        logger.warning("facet profile of %s failed to read", stat_var, exc_info=True)
        return ProfileUnavailable(stat_var=stat_var, gap=ProfileGap.FAILED)
    except Exception:
        logger.exception("facet profile of %s raised a defect", stat_var)
        return ProfileUnavailable(stat_var=stat_var, gap=ProfileGap.DEFECT)


def _profile(
    stat_var: str,
    probe: MemberProbe,
    *,
    places: tuple[str, ...] = (),
    countries_fallback: bool = False,
) -> FacetProfile:
    breadth = probe.breadth_in(None) or {}
    admission = probe.admission
    return FacetProfile(
        stat_var=stat_var,
        candidates=probe.candidates,
        breadth=dict(breadth),
        default=served_facets(probe.candidates, stat_vars=[stat_var], window=None, breadth=breadth),
        places_by_facet=admission.places_by_facet if isinstance(admission, PlaceAdmission) else {},
        places=places,
        countries_fallback=countries_fallback,
    )


class ProfileKey(NamedTuple):
    """A cached profile's key. The cache adds the graph generation."""

    stat_var: str
    catalogue_build: str
    coverage_build: str
    source_scope: frozenset[str] | None


class FacetProfileCache:
    """Profiles behind a single-flight cache of at most `maxsize` variables.

    Each missing variable reads in a flight of its own on `executor`, which no caller may run
    on, and each flight ends at `flight_timeout_s` from its start, queue time included. Its
    facet reads run on `reads`, which only flights submit to, so `reads` bounds the facet reads
    of every flight at once. `generation` stamps every entry, so a new graph generation reads
    afresh. `graph`, `availability` and `max_bytes` go to `read_facet_profiles`."""

    def __init__(
        self,
        *,
        graph: ProfileGraphReader,
        availability: AvailabilityReader,
        executor: Executor,
        reads: Executor,
        generation: GenerationProvider,
        flight_timeout_s: float,
        maxsize: int,
        max_bytes: int,
    ) -> None:
        self._graph = graph
        self._availability = availability
        self._max_bytes = max_bytes
        self._reads = reads
        self._cache: SingleFlightCache[ProfileKey, ProfileOutcome] = SingleFlightCache(
            maxsize=maxsize,
            flight_timeout_s=flight_timeout_s,
            executor=executor,
            generation=generation,
        )

    def profiles(
        self,
        stat_vars: Iterable[str],
        *,
        catalogue: Catalogue,
        coverage: CoverageSnapshot,
        deadline: float | None,
    ) -> dict[str, ProfileOutcome]:
        """Each of `stat_vars`' outcome, keyed by bare dcid: its cached profile, else a read in
        the variable's own flight, shared with every caller missing the same key. A flight binds
        `catalogue` and the caller's bound graph release, and reads `coverage.index`. Each
        wait ends at the earlier of its flight's deadline and `deadline`. A variable without a
        profile is `ProfileUnavailable` with its `ProfileGap` and is not stored, while the
        profiles of the variables that finished are: its own probe raised, its scope was
        unsettled, its flight failed, or its flight or the wait for it ended. Any other error of
        a flight propagates. Raises `graph_release.context.ReleaseNotBound` with no release
        bound."""
        keys, fetch = self._flight(stat_vars, catalogue=catalogue, coverage=coverage)
        found = self._cache.get_or_fetch_many(
            keys,
            fetch,
            deadline=deadline,
            is_cacheable=_is_profile,
            flight_per_key=True,
            on_error=_failed,
        )
        return {key.stat_var: found[key] for key in keys}

    async def aprofiles(
        self,
        stat_vars: Iterable[str],
        *,
        catalogue: Catalogue,
        coverage: CoverageSnapshot,
        deadline: float | None,
    ) -> dict[str, ProfileOutcome]:
        """`profiles` awaited on the event loop: it shares the same flights, and its wait holds
        no thread."""
        keys, fetch = self._flight(stat_vars, catalogue=catalogue, coverage=coverage)
        found = await self._cache.aget_or_fetch_many(
            keys,
            fetch,
            deadline=deadline,
            is_cacheable=_is_profile,
            flight_per_key=True,
            on_error=_failed,
        )
        return {key.stat_var: found[key] for key in keys}

    def _flight(
        self, stat_vars: Iterable[str], *, catalogue: Catalogue, coverage: CoverageSnapshot
    ) -> tuple[
        list[ProfileKey], Callable[[list[ProfileKey], float], dict[ProfileKey, ProfileOutcome]]
    ]:
        """The keys `stat_vars` read under, and the fetch a flight over the missing ones runs."""
        keys = [
            ProfileKey(
                stat_var=sv,
                catalogue_build=catalogue.build_id,
                coverage_build=coverage.manifest.build_id,
                source_scope=catalogue.scope,
            )
            for sv in dict.fromkeys(sv.removeprefix(_STAT_VAR_REF_PREFIX) for sv in stat_vars)
        ]
        release = current_release()

        def fetch(
            missing: list[ProfileKey], flight_deadline: float
        ) -> dict[ProfileKey, ProfileOutcome]:
            # The flight reads for every caller alike, so it runs in an empty context rather
            # than a copy of the leading caller's: it writes into no request's records (UN
            # places, degraded reads, timings), and a failed UN part fails the flight rather
            # than reading as empty. It binds only what the read needs.
            def read() -> dict[str, ProfileOutcome]:
                with (
                    bound_catalogue(catalogue),
                    bound_release(release),
                    request_deadline(flight_deadline),
                ):
                    return read_facet_profiles(
                        [key.stat_var for key in missing],
                        graph=self._graph,
                        availability=self._availability,
                        coverage=coverage.index,
                        max_bytes=self._max_bytes,
                        deadline=flight_deadline,
                        reads=self._reads,
                    )

            outcomes = contextvars.Context().run(read)
            return {key: outcomes[key.stat_var] for key in missing}

        return keys, fetch


def _failed(key: ProfileKey, exc: Exception) -> ProfileOutcome:
    """`key` unavailable after its flight or the wait for it ended in `exc`, a timeout or a
    failed read. Any other error is raised."""
    if isinstance(exc, _TIMEOUTS):
        return ProfileUnavailable(stat_var=key.stat_var, gap=ProfileGap.TIMEOUT)
    if not isinstance(exc, _FAILURES):
        raise exc
    logger.warning("facet profile read failed for %s", key.stat_var, exc_info=exc)
    return ProfileUnavailable(stat_var=key.stat_var, gap=ProfileGap.FAILED)


def _is_profile(outcome: ProfileOutcome) -> bool:
    return isinstance(outcome, FacetProfile)


@dataclass(frozen=True, slots=True)
class ProfileBreadth:
    """Each variable's places across its whole default scope (`FacetProfile.breadth`), read
    from `profiles` under the bound catalogue and `coverage`'s current snapshot, as a
    `facets.VariableBreadthReader`. A door binds one per process (`facets.bound_variable_breadth`)
    so a single-place tie breaks on it (`profile_breadth`). A variable whose own profile raised a defect
    (`ProfileGap.DEFECT`) is `facets.BreadthUnread.DEFECT`; any other variable without a
    profile is left out and logged as a warning with its gap."""

    profiles: FacetProfileCache
    coverage: Callable[[], CoverageSnapshot]

    def __call__(
        self, stat_vars: Sequence[str], *, deadline: float | None
    ) -> dict[str, Mapping[str, int] | BreadthUnread]:
        coverage = self.coverage()
        outcomes = self.profiles.profiles(
            stat_vars, catalogue=current_catalogue(), coverage=coverage, deadline=deadline
        )
        found: dict[str, Mapping[str, int] | BreadthUnread] = {}
        unavailable: dict[str, str] = {}
        for stat_var, outcome in outcomes.items():
            if isinstance(outcome, FacetProfile):
                found[stat_var] = profile_breadth(outcome, coverage=coverage.index)
            elif outcome.gap is ProfileGap.DEFECT:
                found[stat_var] = BreadthUnread.DEFECT
            else:
                unavailable[stat_var] = outcome.gap.value
        if unavailable:
            logger.warning("no whole-variable breadth for %s", unavailable)
        return found


def profile_breadth(profile: FacetProfile, *, coverage: CoverageIndex) -> SeriesCounts:
    """`profile.breadth` for one read: its counts, and the union of any set of facet ids a
    ranking joins, counted when the join forms and once per set. A role-bearing variable's
    union comes from `coverage`'s donor and recipient pairs
    (`CoverageIndex.union_series_count`), a subject-only variable's from the places each
    facet was read at (`FacetProfile.places_by_facet`)."""

    def union_of(facet_ids: frozenset[str]) -> int:
        if profile.stat_var in coverage.role_names:
            return coverage.union_series_count(variable=profile.stat_var, facet_ids=facet_ids)
        places = profile.places_by_facet
        return len(frozenset().union(*(places.get(f, frozenset()) for f in facet_ids)))

    return SeriesCounts(counts=profile.breadth, union_of=union_of)
