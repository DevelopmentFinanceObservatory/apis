"""The reads a variable's availability takes, shared by the node door's card and the members
door's `facets` relation.

A read starts from the variable's facet profile (`FacetProfileCache.aprofiles`), awaited on the
event loop under the request's deadline, with the request's origin catalogue and one snapshot
of the bound release's coverage, and reads each profiled variable's admitted facets from it once
(`availability.held_facets`). A surface with no profile cache reads no profile, and the card's
availability is `availability.NOT_SERVED`.

Two caches on `GraphDeps.card_reads` supply the rest, each keyed by graph generation, with one
read in flight per key, and each running its reads on the DCP-read executor:

- each place's national currency (`materialise.national_currency.read_national_currencies`),
  read once per generation;
- each unit's, method's and place's `name` and `typeOf`, cached per dcid and shared by every
  variable, and per bound release too, which answers a DFO variable's
  (`graph_release.context.served_origin_key`). A place counts under the kind its own
  node reports (`graph_api.kinds.classify`).

Both reads outlive the request that starts them. The request waits until its deadline; the
reads run on to a detached bound and fill the caches when they land, so a later request finds
them. The national currencies are one read. The node facts the cache holds are served in the
caller's thread. The rest are read in chunks, the unit and method names first, by a driver on a
pool of its own that reads the chunks one after another, each chunk only for the dcids the cache
misses and each filling the cache as it lands, so a scope of thousands of places converges
across requests. A read that fails stores nothing.

The pool holds `drivers` threads and admits as many drivers, so no driver waits in its queue. A
caller waits on the event loop for a free driver slot until its own deadline and starts none
after it: a burst of cold cards then holds no queued dcid lists, and no driver starts DCP reads
for a caller that has gone. A driver also starts no chunk past its own stop. Every wait a card
takes, for a slot, a driver, the national currencies or a profile, is awaited on the event
loop, so no thread is held by a caller waiting.

The `facets` relation learns whether its node is a variable from one node's facts
(`CardReadCaches.node_facts_of`), read on the DCP-read executor without a driver.

A card waits for neither past its deadline. Without its national currencies, whether the read
failed or the DCP-read executor refused it, it folds none; without its names it leaves those
labels null; without every place's kind it leaves `summary.places` and each `sources[].places`
null. Each is a transient gap.
"""

from __future__ import annotations

import asyncio
import itertools
import logging
import threading
import time
from collections.abc import Collection, Mapping, Sequence
from concurrent.futures import Executor, ThreadPoolExecutor
from dataclasses import dataclass
from typing import TYPE_CHECKING

from code_index.query import CodeIndex
from dfo_platform.executor import submit_with_context
from dfo_platform.single_flight_cache import SingleFlightCache
from entity_resolution.membership import MembershipPolicy
from graph_api.availability import (
    COUNTRIES_ONLY_GAP,
    UNAVAILABLE,
    AvailabilityRead,
    FacetFilter,
    FacetItem,
    HeldFacets,
    ReadTargets,
    build_availability,
    facet_items,
    held_facets,
)
from graph_api.deps import DcpExecutorSaturated, DcpExecutorShutDown
from graph_api.kinds import Kind, classify
from graph_api.reader import GraphNodeReader
from graph_api.refs import first_name
from graph_generation import GenerationProvider
from graph_release.context import served_origin_key
from materialise.errors import UpstreamResponseError
from materialise.facet_profile import FacetProfile, ProfileOutcome
from materialise.graph import GraphReadError
from materialise.national_currency import read_national_currencies
from origin_catalogue import Catalogue, current_catalogue

if TYPE_CHECKING:
    from graph_api.deps import GraphDeps
    from series_coverage.shared import CoverageSnapshot

logger = logging.getLogger(__name__)

LABELS_GAP = "stat_var.availability.labels"
"""The transient gap of a card whose unit and method names could not be read."""

NATIONAL_CURRENCY_GAP = "stat_var.availability.national_currency"
"""The transient gap of a card whose places' national currencies could not be read."""

PLACE_KINDS_GAPS = ("stat_var.availability.summary.places", "stat_var.availability.sources.places")
"""The transient gaps of a card whose places' kinds are not all read yet."""

_FACTS: tuple[str, ...] = ("name", "typeOf")
_NATIONAL_CURRENCIES_KEY = "national_currencies"
# A failed national-currency read, a refused one, or a wait past the deadline degrades the card.
_READ_FAILURES = (
    GraphReadError,
    UpstreamResponseError,
    TimeoutError,
    DcpExecutorSaturated,
    DcpExecutorShutDown,
)


@dataclass(frozen=True, slots=True)
class NodeFacts:
    """A node's first name unequal to its dcid, and its types. A node the graph does not hold
    has neither."""

    name: str | None
    types: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class FactsRead:
    """The facts of the requested dcids that were read by the caller's deadline. `complete` is
    true when every one was."""

    facts: dict[str, NodeFacts]
    complete: bool


class CardReadCaches:
    """The national-currency map and the node facts behind single-flight caches whose flights
    run on `executor`, the DCP-read executor. Callers await on the event loop, and the facts
    drivers run on a pool of `drivers` threads this object owns and `close` shuts. The driver
    slots belong to the event loop that first waits on them, so one object serves one loop.

    The national-currency flight ends at `detached_timeout_s` from its start. A facts driver
    reads `chunk_size` dcids per flight, each flight ending at `chunk_timeout_s`, and starts no
    chunk past `detached_timeout_s` from its submission. A caller's wait ends at its own
    deadline, and the reads run on."""

    def __init__(
        self,
        *,
        graph: GraphNodeReader,
        executor: Executor,
        drivers: int,
        generation: GenerationProvider,
        maxsize: int,
        chunk_size: int,
        chunk_timeout_s: float,
        detached_timeout_s: float,
    ) -> None:
        self._graph = graph
        self._drivers = ThreadPoolExecutor(
            max_workers=drivers, thread_name_prefix="graph-card-reads"
        )
        # One slot per driver thread, so a submitted driver never waits in the pool's queue.
        self._driver_slots = asyncio.BoundedSemaphore(drivers)
        self._chunk_size = chunk_size
        self._detached_timeout_s = detached_timeout_s
        self._currencies: SingleFlightCache[str, dict[str, str]] = SingleFlightCache(
            maxsize=1,
            flight_timeout_s=detached_timeout_s,
            executor=executor,
            generation=generation,
        )
        self._facts: SingleFlightCache[_FactsKey, NodeFacts] = SingleFlightCache(
            maxsize=maxsize,
            flight_timeout_s=chunk_timeout_s,
            executor=executor,
            generation=generation,
        )

    def close(self) -> None:
        """Wait for the running drivers. Each ends at its next chunk once the DCP-read executor
        is shut, so close that first."""
        self._drivers.shutdown(wait=True, cancel_futures=True)

    async def national_currencies(self, *, deadline: float) -> dict[str, str]:
        """Each place's one national currency, by ISO code. Raises a read's failure, the
        executor's refusal, or `TimeoutError` at `deadline` while the read runs on."""
        return await self._currencies.aget_or_fetch(
            _NATIONAL_CURRENCIES_KEY,
            lambda flight_deadline: read_national_currencies(self._graph, deadline=flight_deadline),
            deadline=deadline,
        )

    async def node_facts_of(self, dcid: str, *, deadline: float) -> NodeFacts | None:
        """`dcid`'s facts, cached or read in one flight on the DCP-read executor, without a
        driver; `None` when the read fails, the executor refuses it, or `deadline` passes while
        it runs on."""
        try:
            key = (dcid, served_origin_key())
            return await self._facts.aget_or_fetch(
                key,
                lambda flight_deadline: self._read_facts([key], flight_deadline)[key],
                deadline=deadline,
            )
        except _READ_FAILURES:
            logger.warning("the facts of %s could not be read", dcid, exc_info=True)
            return None

    async def node_facts(self, dcids: Sequence[str], *, deadline: float) -> FactsRead:
        """The facts of `dcids` read by `deadline`. The cache's hits return at once; the misses,
        chunked in their order, go to a driver once a driver slot frees before `deadline`; a
        read past `deadline` starts none. The driver runs on after `deadline` and fills the
        cache."""
        wanted = list(dict.fromkeys(dcids))
        release = served_origin_key()
        found = {
            dcid: facts
            for (dcid, _release), facts in self._facts.cached_many(
                [(dcid, release) for dcid in wanted]
            ).items()
        }
        missing = [(dcid, release) for dcid in wanted if dcid not in found]
        if not missing:
            return FactsRead(facts=found, complete=True)
        try:
            async with asyncio.timeout(max(deadline - time.monotonic(), 0.0)):
                await self._driver_slots.acquire()
        except TimeoutError:
            return FactsRead(facts=found, complete=False)
        # A free slot is taken without yielding, so the timeout above never fires for it.
        if time.monotonic() >= deadline:
            self._driver_slots.release()
            return FactsRead(facts=found, complete=False)
        lock = threading.Lock()
        stop = time.monotonic() + self._detached_timeout_s

        def drive() -> None:
            # A detached job: it logs its failure, since no caller may be left to see it.
            try:
                for chunk in itertools.batched(missing, self._chunk_size):
                    if time.monotonic() >= stop:
                        return
                    read = self._facts.get_or_fetch_many(
                        list(chunk), self._read_facts, deadline=stop
                    )
                    with lock:
                        found.update((dcid, facts) for (dcid, _release), facts in read.items())
            except Exception:
                logger.warning("Variable card's name and type read stopped", exc_info=True)

        try:
            # The driver's flights read the graph under the caller's bound origin catalogue.
            driver = asyncio.wrap_future(submit_with_context(self._drivers, drive))
        except RuntimeError:
            # `ThreadPoolExecutor.submit` raises a bare `RuntimeError` after `close`.
            self._driver_slots.release()
            return FactsRead(facts=found, complete=False)
        # The slot frees on the loop when the driver ends, whether or not this caller waits.
        driver.add_done_callback(lambda _driver: self._driver_slots.release())
        await asyncio.wait([driver], timeout=max(deadline - time.monotonic(), 0.0))
        with lock:
            facts = dict(found)
        return FactsRead(facts=facts, complete=len(facts) == len(wanted))

    def _read_facts(self, keys: list[_FactsKey], deadline: float) -> dict[_FactsKey, NodeFacts]:
        dcids = [dcid for dcid, _release in keys]
        arcs = self._graph.node_arcs_many(dcids, properties=_FACTS, deadline=deadline)
        return {key: _facts(key[0], arcs.get(key[0])) for key in keys}


type _FactsKey = tuple[str, str]
"""A node's facts key: its dcid and the bound release (`served_origin_key`)."""


def _facts(dcid: str, arcs: Mapping[str, tuple[str, ...]] | None) -> NodeFacts:
    if arcs is None:
        return NodeFacts(name=None, types=())
    return NodeFacts(name=first_name(arcs.get("name", ()), dcid), types=arcs.get("typeOf", ()))


@dataclass(frozen=True, slots=True)
class ProfileRead:
    """The profiles one request read, each profiled variable's admitted facets (`held`), and
    the catalogue and coverage snapshot they were keyed on. A variable whose profile missed has
    no `held` entry."""

    profiles: Mapping[str, ProfileOutcome]
    held: Mapping[str, HeldFacets]
    catalogue: Catalogue
    coverage: CoverageSnapshot
    caches: CardReadCaches

    @property
    def coverage_build(self) -> str:
        return self.coverage.manifest.build_id


async def read_profiles(
    deps: GraphDeps, stat_vars: Collection[str], *, deadline: float
) -> ProfileRead | None:
    """Every one of `stat_vars`' profile in one call, awaited on the event loop, and each
    profiled variable's admitted facets; `None` when the surface has no profile cache."""
    if deps.profile_cache is None or deps.card_reads is None:
        return None
    catalogue = current_catalogue()
    coverage = deps.coverage.snapshot()
    profiles = await deps.profile_cache.aprofiles(
        stat_vars, catalogue=catalogue, coverage=coverage, deadline=deadline
    )
    held = {
        stat_var: held_facets(
            stat_var=stat_var, profile=profile, coverage=coverage.index, catalogue=catalogue
        )
        for stat_var, profile in profiles.items()
        if isinstance(profile, FacetProfile)
    }
    return ProfileRead(
        profiles=profiles,
        held=held,
        catalogue=catalogue,
        coverage=coverage,
        caches=deps.card_reads,
    )


@dataclass(frozen=True, slots=True)
class _Inputs:
    """The national currencies a response's cards read, `None` where the read failed or is
    still running, and the node facts read by the deadline."""

    currencies: dict[str, str] | None
    facts: FactsRead


async def _read_inputs(
    caches: CardReadCaches, *, named: set[str], places: set[str], deadline: float
) -> _Inputs:
    """The national currencies, read only when a card counts places, and the node facts, the
    unit and method names first, the two reads awaited side by side."""

    async def currencies() -> dict[str, str] | None:
        if not places:
            return {}
        try:
            return await caches.national_currencies(deadline=deadline)
        except _READ_FAILURES:
            logger.warning("Variable card's national-currency read failed", exc_info=True)
            return None

    ordered = [*sorted(named), *sorted(places - named)]
    read_currencies, facts = await asyncio.gather(
        currencies(), caches.node_facts(ordered, deadline=deadline)
    )
    return _Inputs(currencies=read_currencies, facts=facts)


async def read_availability(
    read: ProfileRead,
    *,
    code_index: CodeIndex,
    membership: MembershipPolicy | None,
    deadline: float,
) -> dict[str, AvailabilityRead]:
    """Each variable's availability, `UNAVAILABLE` where its profile missed. The names and
    place types of every card in the response come from one chunked read, and the national
    currencies from one more."""
    targets = {sv: held.targets for sv, held in read.held.items()}
    named = {dcid for found in targets.values() for dcid in found.named}
    places = {dcid for found in targets.values() for dcid in found.places}
    inputs = await _read_inputs(read.caches, named=named, places=places, deadline=deadline)
    facts = inputs.facts.facts
    labels = {dcid: fact.name for dcid, fact in facts.items() if fact.name is not None}
    kinds: dict[str, Kind] = {
        place: classify(
            dcid=place, types=facts[place].types, code_index=code_index, membership=membership
        )
        for place in places
        if place in facts
    }
    built = dict.fromkeys(read.profiles, UNAVAILABLE)
    for stat_var, found in targets.items():
        kinds_read = found.places <= kinds.keys()
        availability = build_availability(
            held=read.held[stat_var],
            catalogue=read.catalogue,
            parties=code_index.parties,
            national_currencies=inputs.currencies or {},
            place_kind=kinds.__getitem__ if kinds_read else None,
            labels=labels,
        )
        built[stat_var] = _degraded(availability, found, inputs)
    return built


def _degraded(read: AvailabilityRead, found: ReadTargets, inputs: _Inputs) -> AvailabilityRead:
    """`read` with a transient gap for each input its fold took that is not read."""
    facts = inputs.facts.facts
    gaps = list(read.gaps)
    if not found.named <= facts.keys():
        gaps.append(LABELS_GAP)
    if not found.places <= facts.keys():
        gaps.extend(PLACE_KINDS_GAPS)
    if found.places and inputs.currencies is None:
        gaps.append(NATIONAL_CURRENCY_GAP)
    if len(gaps) == len(read.gaps):
        return read
    return AvailabilityRead(availability=read.availability, gaps=tuple(gaps), transient=True)


@dataclass(frozen=True, slots=True)
class FacetsRead:
    """A variable's facets that a filter keeps, `None` when its profile could not be read,
    whether their labels failed to read, and the profile's permanent gaps."""

    items: list[FacetItem] | None
    labels_failed: bool
    gaps: tuple[str, ...] = ()


async def read_facets(
    read: ProfileRead, stat_var: str, *, where: FacetFilter, deadline: float
) -> FacetsRead:
    """`stat_var`'s facets that `where` keeps, ordered by facet id, labelled from the node
    facts read by `deadline`."""
    held = read.held.get(stat_var)
    if held is None:
        return FacetsRead(items=None, labels_failed=False)
    facts = await read.caches.node_facts(sorted(held.targets.named), deadline=deadline)
    labels = {dcid: fact.name for dcid, fact in facts.facts.items() if fact.name is not None}
    items = facet_items(held=held, catalogue=read.catalogue, labels=labels, where=where)
    return FacetsRead(
        items=items,
        labels_failed=not facts.complete,
        gaps=(COUNTRIES_ONLY_GAP,) if held.profile.countries_fallback else (),
    )
