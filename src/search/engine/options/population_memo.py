"""Memoise `options_population` by identity shape, slots, role keys, and named dataset
across one surface.

The key holds the role keys the identity binds entities under, which elect the role
class, and excludes the entities, because anchors do not change a StatVar population.
Each entry carries the graph-generation stamp and expires at its `max_age_s`;
a generation change clears the memo. This bounds possible staleness while
avoiding repeated graph enumeration, which takes 2.6 to 5.3 seconds and about
110 round trips for some populations.

Single-flight derivation is per key. Callers share the result or exception;
failures are never stored. Every caller runs in `asyncio.to_thread`, so a
follower occupies a worker thread unless `deadline` bounds its wait or
`join_inflight=False` raises `PopulationInFlight`. The owner always completes
and publishes, even after followers time out.

One memo is built per `search_surface` lifespan because populations use the
caller's `materialise.GraphReader`. With no memo, derivation runs directly,
without lookup or single-flight bookkeeping.
"""

from __future__ import annotations

import hashlib
import time
from collections.abc import Callable, Mapping

import rfc8785

import search.engine.config as config  # module-attribute access for runtime-resolved values
from dfo_platform.single_flight_cache import AlreadyInFlight, SingleFlightCache
from graph_generation import UNKNOWN_GENERATION, GenerationProvider
from search.engine.options.population import OptionsPopulation, options_population
from search.engine.retrieval.bound_values import identity_query
from search.models import BaseIdentity
from statvar_members import MemberArcReader

_FOLLOWER_DEADLINE_EXCEEDED_DETAIL = (
    "population memo: follower deadline elapsed while the owner was still deriving"
)


def _population_key(identity: BaseIdentity) -> str:
    """sha256 of the RFC 8785 (JCS) canonical JSON of `identity`'s shape, slots, and the
    role keys and dataset its `identity_query` names -- the fields `options_population`
    reads. Deliberately excludes the entities bound under each role key: see the module
    docstring. An identity that binds no entity keys as it does without role keys."""
    document: dict[str, object] = {
        "shape": identity.shape.model_dump(mode="json", exclude_none=True),
        "slots": [slot.model_dump(mode="json") for slot in identity.slots],
    }
    query = identity_query(identity)
    if query.source is not None:
        document["source"] = query.source
    if query.role_keys:
        document["role_keys"] = sorted(query.role_keys)
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
            generation=generation,
            clock=clock,
            timeout_message=_FOLLOWER_DEADLINE_EXCEEDED_DETAIL,
        )

    def get_or_derive(
        self,
        identity: BaseIdentity,
        graph: MemberArcReader,
        *,
        deadline: float | None = None,
        join_inflight: bool = True,
    ) -> OptionsPopulation:
        """Return the cached population or derive it on a miss.

        Derivation refusals and infrastructure errors propagate and are never
        stored. `deadline` bounds only a follower's wait; expiry raises
        `asyncio.TimeoutError`, while the owner completes. With
        `join_inflight=False`, a cache hit returns normally, a free key can be
        claimed, and an occupied key raises `PopulationInFlight` without
        waiting. Place-options prefetch uses this to preserve worker capacity.
        """
        if not self._enabled:
            return options_population(identity, graph)

        try:
            return self._cache.get_or_fetch(
                _population_key(identity),
                lambda: options_population(identity, graph),
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
