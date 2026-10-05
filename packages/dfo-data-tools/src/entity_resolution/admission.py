"""Check role admission against `rangeIncludes` and its classes' `subClassOf` closure.

Reads use generation-keyed caches. Missing graph nodes contribute no parent or range arcs."""

from __future__ import annotations

import functools
import threading
from collections.abc import Mapping, Sequence
from concurrent.futures import Executor

from cachetools import TLRUCache

from dfo_platform.executor import run_in_executor_with_context
from entity_resolution.types import GraphLookup
from graph_generation import UNKNOWN_GENERATION, GenerationProvider, GenerationStamp
from graph_generation.stamp import ttu_from_stamp_key

# Cache entries expire with their generation stamp.
_RANGE_CACHE_MAXSIZE = 4096
_PARENTS_CACHE_MAXSIZE = 4096

_CacheKey = tuple[str, str]

# Derive expiry from the stamp key shared with `EntityResolver`.
_ttu = ttu_from_stamp_key


class Admission:
    """`graph_executor=None` uses the running loop's default executor for
    every graph call, matching `EntityResolver`'s own discipline.
    `generation` defaults to `UNKNOWN_GENERATION`, so an uninjected
    instance keys and expires every entry under the "unknown" stamp (1h).
    """

    def __init__(
        self,
        graph: GraphLookup,
        *,
        graph_executor: Executor | None = None,
        generation: GenerationProvider = UNKNOWN_GENERATION,
    ) -> None:
        self._graph = graph
        self._graph_executor = graph_executor
        self._generation = generation
        # Generation refresh clears caches on a background thread. Guard every cache access.
        self._cache_lock = threading.Lock()
        self._range_cache: TLRUCache[_CacheKey, frozenset[str]] = TLRUCache(
            maxsize=_RANGE_CACHE_MAXSIZE, ttu=_ttu
        )
        self._parents_cache: TLRUCache[_CacheKey, tuple[str, ...]] = TLRUCache(
            maxsize=_PARENTS_CACHE_MAXSIZE, ttu=_ttu
        )

    def stats(self) -> Mapping[str, int]:
        with self._cache_lock:
            return {
                "range_entries": len(self._range_cache),
                "parents_entries": len(self._parents_cache),
            }

    def clear(self) -> None:
        with self._cache_lock:
            self._range_cache.clear()
            self._parents_cache.clear()

    async def admits(self, role_property: str, graph_type: str) -> bool:
        """`role_property` is a graph property dcid such as `observationAbout`.

        `graph_type` is the entity's `typeOf` value. Return `False` when it is empty."""
        if not graph_type:
            return False
        # Capture before lookups and outside the cache lock so this call uses one stamp.
        stamp = self._generation.stamp()
        range_classes = await self._range_includes(role_property, stamp)
        if not range_classes:
            return False
        if graph_type in range_classes:
            return True
        visited: set[str] = {graph_type}
        frontier: list[str] = [graph_type]
        while frontier:
            parents_by_node = await self._subclass_parents_many(frontier, stamp)
            next_frontier: list[str] = []
            for node in frontier:
                for parent in parents_by_node[node]:
                    if parent in range_classes:
                        return True
                    if parent not in visited:
                        visited.add(parent)
                        next_frontier.append(parent)
            frontier = next_frontier
        return False

    async def _range_includes(self, role_property: str, stamp: GenerationStamp) -> frozenset[str]:
        key = (stamp.key, role_property)
        with self._cache_lock:
            cached = self._range_cache.get(key)
        if cached is not None:
            return cached
        fetched = await self._node_arcs_many((role_property,), properties=("rangeIncludes",))
        values = frozenset((fetched.get(role_property) or {}).get("rangeIncludes", ()))
        with self._cache_lock:
            self._range_cache[key] = values
        return values

    async def _subclass_parents_many(
        self, dcids: Sequence[str], stamp: GenerationStamp
    ) -> dict[str, tuple[str, ...]]:
        """Return cached and fetched parents without re-reading after fetch.

        A concurrent `clear()` can evict a just-written cache entry between lock acquisitions."""
        result: dict[str, tuple[str, ...]] = {}
        uncached: list[str] = []
        with self._cache_lock:
            for dcid in dcids:
                cached = self._parents_cache.get((stamp.key, dcid))
                if cached is not None:
                    result[dcid] = cached
                else:
                    uncached.append(dcid)
        if uncached:
            fetched = await self._node_arcs_many(uncached, properties=("subClassOf",))
            with self._cache_lock:
                for dcid in uncached:
                    arcs = fetched.get(dcid) or {}
                    parents = tuple(arcs.get("subClassOf", ()))
                    self._parents_cache[(stamp.key, dcid)] = parents
                    result[dcid] = parents
        return result

    async def _node_arcs_many(
        self, dcids: Sequence[str], *, properties: Sequence[str]
    ) -> dict[str, dict[str, tuple[str, ...]] | None]:
        fetch = functools.partial(self._graph.node_arcs_many, properties=properties)
        return await run_in_executor_with_context(self._graph_executor, fetch, dcids)
