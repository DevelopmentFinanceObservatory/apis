"""Memoise `options_population` by identity shape, slots, role keys, and named dataset
across one surface.

The key holds the role keys the identity binds entities under, which elect the role
class, and excludes the entities, because anchors do not change a StatVar population.
Each entry carries the graph-generation stamp and expires at its `max_age_s`;
a generation change clears the memo. This bounds possible staleness while
avoiding repeated graph enumeration, which takes 2.6 to 5.3 seconds and about
110 round trips for some populations.

Single-flight derivation is per key. A miss derives on `_POPULATION_FLIGHTS`
under its own deadline, `request_timeout_s` from when it starts, so the request
that starts it bounds only its own wait.
Callers share the result or exception; failures are never stored. Every caller
runs in `asyncio.to_thread`, so a waiting caller occupies a worker thread unless
`deadline` bounds its wait or `join_inflight=False` raises `PopulationInFlight`.
The derivation always completes and publishes, even after every caller times out.

One memo is built per `search_surface` lifespan because populations use the
caller's `materialise.GraphReader`. With no memo, derivation runs directly,
without lookup or single-flight bookkeeping.
"""

from __future__ import annotations

import hashlib
import time
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor
from typing import Final

import rfc8785

import search.engine.config as config  # module-attribute access for runtime-resolved values
from code_index.members import family_query_digest
from dfo_platform.single_flight_cache import AlreadyInFlight, SingleFlightCache
from graph_generation import UNKNOWN_GENERATION, GenerationProvider
from graph_release.context import served_origin_key
from materialise import GraphReader
from search.engine.options.population import OptionsPopulation, options_population
from search.engine.retrieval.bound_values import identity_query
from search.models import BaseIdentity

_FOLLOWER_DEADLINE_EXCEEDED_DETAIL = (
    "population memo: caller deadline elapsed while the derivation was still running"
)

_POPULATION_FLIGHT_WORKERS: Final[int] = 8
"""Threads that run shared population derivations, process-wide: the place-options prefetch
pool's four plus as many /options requests."""

_POPULATION_FLIGHTS: Final = ThreadPoolExecutor(
    max_workers=_POPULATION_FLIGHT_WORKERS, thread_name_prefix="options-population"
)
"""Runs every memo's derivations. No caller of the memo runs on it, so a caller waiting for a
derivation never holds the thread that derivation needs."""


def _population_key(identity: BaseIdentity) -> str:
    """sha256 of the RFC 8785 (JCS) canonical JSON of `identity`'s shape, slots, and the
    digest of its `identity_query` (`family_query_digest`), which carries the role keys,
    the dataset and the source scope the query elects under. These are the fields
    `options_population` reads. It deliberately excludes the entities bound under each role
    key (see the module docstring). With `GRAPH_RELEASE_ORIGIN` on it holds the bound release
    (`served_origin_key`), whose DFO variables a member walk reads."""
    document: dict[str, object] = {
        "shape": identity.shape.model_dump(mode="json", exclude_none=True),
        "slots": [slot.model_dump(mode="json") for slot in identity.slots],
        "query": family_query_digest(identity_query(identity)).hex(),
    }
    if (release := served_origin_key()) is not None:
        document["release"] = release
    canonical = rfc8785.dumps(document)
    return hashlib.sha256(canonical).hexdigest()


class PopulationInFlight(Exception):
    """Raised by `get_or_derive(join_inflight=False)` when the key is
    already claimed by another caller's derivation -- no wait, no slot
    touched. See `get_or_derive`'s own docstring for who raises this and
    why."""


class OptionsPopulationMemo:
    """Bounded, generation-keyed, single-flight cache of `OptionsPopulation`.

    Defaults come from `SEARCH_OPTIONS_POPULATION_CACHE_*` and
    `time.monotonic`. `max_entries` must be at least 1. Disabled mode derives
    directly without lookup or storage. Each call captures one generation
    stamp; its key partitions in-flight work and its `max_age_s` sets entry
    lifetime. `stats()` and `clear()` support generation-provider hooks.
    """

    def __init__(
        self,
        *,
        enabled: bool | None = None,
        max_entries: int | None = None,
        clock: Callable[[], float] = time.monotonic,
        generation: GenerationProvider = UNKNOWN_GENERATION,
    ) -> None:
        self._enabled = (
            config.settings().options_population_cache_enabled if enabled is None else enabled
        )
        max_entries = (
            config.settings().options_population_cache_max_entries
            if max_entries is None
            else max_entries
        )
        if max_entries < 1:
            raise ValueError(f"max_entries must be at least 1, got {max_entries!r}")
        self._cache: SingleFlightCache[str, OptionsPopulation] = SingleFlightCache(
            maxsize=max_entries,
            flight_timeout_s=config.settings().request_timeout_s,
            executor=_POPULATION_FLIGHTS,
            generation=generation,
            clock=clock,
            timeout_message=_FOLLOWER_DEADLINE_EXCEEDED_DETAIL,
        )

    def get_or_derive(
        self,
        identity: BaseIdentity,
        graph: GraphReader,
        *,
        deadline: float | None = None,
        join_inflight: bool = True,
    ) -> OptionsPopulation:
        """Return the cached population or derive it on a miss.

        Derivation refusals and infrastructure errors propagate and are never
        stored. `deadline` bounds this caller's wait, whether it started the
        derivation or joined it; its expiry raises `TimeoutError` while the
        derivation completes under its own deadline. Disabled, `deadline` bounds
        the graph walk, whose expiry raises `MemberReadDeadline`. With
        `join_inflight=False`, a cache hit returns normally, a free key can be
        claimed, and an occupied key raises `PopulationInFlight` without
        waiting. Place-options prefetch uses this to preserve worker capacity.
        """
        if not self._enabled:
            return options_population(identity, graph, deadline=deadline)

        try:
            return self._cache.get_or_fetch(
                _population_key(identity),
                lambda flight_deadline: options_population(
                    identity, graph, deadline=flight_deadline
                ),
                deadline=deadline,
                join_inflight=join_inflight,
            )
        except AlreadyInFlight as exc:
            raise PopulationInFlight(*exc.args) from exc

    def stats(self) -> Mapping[str, int]:
        """Hit/miss/eviction counters plus the current entry count, for the
        `graph_generation` provider to read on registration (elsewhere)."""
        return self._cache.stats()

    def clear(self) -> None:
        """Drops every stored entry and every in-flight slot's own
        bookkeeping (never the owner's derivation itself, which always runs
        to completion). Called by the `graph_generation` provider on a token
        change once this memo is registered (elsewhere)."""
        self._cache.clear()
