"""Cache option sets produced by `options.surface.compute_options`.

Keyed by the resolved target after identity and role validation.
Resolution checks run on every request, including identity validation and role/anchor refusals (`invalid_role_
ref`, `role_incompatible`, `role_signature_incompatible`, `option_target_
invalid`), and none of those checks may be skipped just because an identical
target was already answered once. Only `options.surface.compute_options`'s
own compute -- the graph/SDMX work that turns a resolved target
into candidates -- is what a hit skips.

The key includes the caller-supplied
`build_commit` and `req.schema_version` (so rebuilds and schema changes produce distinct keys),
`options.population_memo._population_key(target.identity)` (anchors do not change a
population, and the identity carries the requested flow axis as an open role),
`req.for_.role` and
`req.include_coverage`, `target.facets` and `target.requested_window`
(carried through from the identity, unchanged), and finally one piece that
depends on which half of `compute_options` will run: for `for.role ==
"observationAbout"`, the bound anchors' own entity_type(s) for a `peers` ask
(so a Kenya-anchored ask and a Ghana-anchored ask -- both Country -- share
one entry) or the single bound anchor's own dcid for a `children` ask (which
has no anchor-invariant answer, unlike peers); for a directional role
(`origin`/`destination`), the opposite axis's own bound-entity set
(`options.surface._directional_opposite_bound`'s own return value) -- reused
rather than re-derived, since it is already the exact fact
`_directional_options_set`'s own coverage lookup filters by.

The caller passes one `CoverageSnapshot` to the cache key and response
builder. Its stamp ensures a newer release's coverage view cannot reuse
results built against earlier donor or recipient membership. The key also holds the
build id and UN System setting of the origin catalogue the request bound, which decide
what provenances the options admit and which places the UN front door refuses. A
catalogue reload changes them and not the graph generation. The caller reads the bound
catalogue once and passes it to the key; `compute_options` reads the same bound snapshot.

The cache stores results whose reads all answered (`is_cacheable`), with their
warnings. A coverage probe failure, a failed UN System Data Commons read and a UN
front door refusal of a place outside the origin catalogue build's refused set make a
result transient and leave it uncached. Places in that set are a fact of the build,
which the key holds, so a result naming only them is stored and a hit
returns their warning. A compute that noted a degraded read
(`degraded_reads.note_degraded_read`), such as a place's facet tie left to the unit's
name because its whole-variable breadth was not read, remains uncached; refusals and
infrastructure failures raise and remain uncached. A later request retries every
uncached result.

Each entry's generation stamp is part of its key and sets its `max_age_s`.
A coverage republish changes the caller-supplied key. Single-flight starts
one `asyncio.Task` per key and runs `compute` in `asyncio.to_thread`. The
flight's deadline, `request_timeout_s` from when it starts, is passed to
`compute` and bound as the task's request deadline, so no caller's deadline
bounds the reads `compute` makes. Callers await that task through
`asyncio.shield`, each with its own deadline.
Timeout or cancellation detaches that caller's wait; the shared task continues.
Concurrent callers receive the computed result, including a transient one that
is never stored. Followers wait on the
event loop and do not occupy worker threads.

Both option caches are injected into `search_surface` and created in its
lifespan. They remain inactive when the lifespan does not run, so tests that
use different fake graphs cannot share entries accidentally.
"""

from __future__ import annotations

import hashlib
import time
from collections.abc import Callable, Mapping
from dataclasses import replace
from typing import cast

import rfc8785

import search.engine.config as config  # module-attribute access for runtime-resolved values
from dfo_platform.single_flight_cache import SingleFlightCache
from graph_generation import UNKNOWN_GENERATION, GenerationProvider
from materialise.roles import RoleAxis
from origin_catalogue import Catalogue
from search.engine.degraded_reads import recording_degraded_reads
from search.engine.options.population_memo import _population_key
from search.engine.options.resolve import ResolvedOptionsTarget
from search.engine.options.surface import (
    _DEFAULT_PEER_TYPE,
    ComputedOptions,
    _directional_opposite_bound,
    _entities_with_role,
)
from search.models import OptionsRequest
from series_coverage.shared import CoverageSnapshot

# ---------------------------------------------------------------------------
# The response cache's own key
# ---------------------------------------------------------------------------


def _observation_about_key_component(req: OptionsRequest, target: ResolvedOptionsTarget) -> dict:
    """The `for.role == "observationAbout"` slice of the cache key: `peers`
    keys on the bound anchors' own entity_type(s) (no anchor at all reads as
    the identical `_DEFAULT_PEER_TYPE` `options.surface._observation_about_
    options_set` itself falls back to, so an unbound ask and a
    `_DEFAULT_PEER_TYPE`-anchored ask share one entry); `children` keys on
    the bound anchors' own dcid(s). Either branch can carry more than one
    distinct value only on a request `compute_options` itself refuses
    (`mixed_peers_anchor_types`, `children_anchor_ambiguous`) -- never
    stored either way (see the module docstring), so this need not special-
    case that shape beyond staying a plain, order-invariant, JSON-safe
    value.
    """
    anchors = _entities_with_role(list(target.anchors), "observationAbout")
    scope = req.for_.scope or "peers"
    if scope == "peers":
        anchor_types = {a.entity_type for a in anchors if a.entity_type} or {_DEFAULT_PEER_TYPE}
        return {"scope": "peers", "anchor_entity_types": sorted(anchor_types)}
    anchor_dcids = {a.ref for a in anchors}
    return {"scope": "children", "anchor_dcids": sorted(anchor_dcids)}


def _directional_key_component(
    target: ResolvedOptionsTarget, *, axis: RoleAxis
) -> dict[str, list[str]] | None:
    """The `origin`/`destination` slice of the cache key: the opposite
    axis's own bound-entity set, exactly as `options.surface.
    _directional_opposite_bound` computes it for `_directional_options_set`'s
    own `series_coverage.CoverageIndex.open_axis` lookup -- reused rather
    than re-derived so this can never drift from what the surface half
    actually filters by. Keyed by coverage's own "donor"/"recipient" label
    rather than the axis's declared role property name: an internal cache
    key, not a wire value, so either shape works as long as it stays
    deterministic and collision-free, and reusing coverage's own label
    keeps this key in agreement with the fact the surface half actually
    reads. Every dcid set is sorted here: two requests naming the same
    opposite-axis dcids in a different order denote the identical filter
    and must share one key, mirroring the identity core's own
    order-invariant entity buckets. None (no component) when the opposite
    axis carries no bound entity, matching `_directional_opposite_bound`'s
    own empty-mapping case.
    """
    bound = _directional_opposite_bound(target, axis=axis)
    if not bound:
        return None
    return {coverage_role: sorted(dcids) for coverage_role, dcids in bound.items()}


def options_response_cache_key(
    req: OptionsRequest,
    target: ResolvedOptionsTarget,
    *,
    build_commit: str,
    coverage: CoverageSnapshot,
    catalogue: Catalogue,
) -> str:
    """The `OptionsResponseCache` key for `req`/`target`. See the module
    docstring for every ingredient and why each is there. `build_commit`,
    `coverage` and `catalogue` are passed in by the caller (`app.py`'s
    `list_options`, which takes one `get_coverage_service().snapshot()` read and
    one `current_catalogue()` read per request and threads them here and into
    `compute_options`, rather than this function reading either again on its own)
    rather than read here, keeping this function pure -- no I/O, no config import --
    the same discipline `response_cache_keys.py`'s own key builders follow.
    """
    if req.for_.role == "observationAbout":
        axis_component: object = _observation_about_key_component(req, target)
    else:
        axis_component = _directional_key_component(target, axis=cast(RoleAxis, req.for_.role))

    window = target.requested_window
    if window is None or isinstance(window, str):
        window_doc: object = window
    else:
        window_doc = window.model_dump(mode="json", exclude_none=True)

    facets_doc = (
        {
            axis_name: binding.model_dump(mode="json", exclude_none=True)
            for axis_name, binding in target.facets.items()
        }
        if target.facets
        else None
    )

    document = {
        "build_commit": build_commit,
        "schema_version": req.schema_version,
        "population_key": _population_key(target.identity),
        "role": req.for_.role,
        "include_coverage": bool(req.include_coverage),
        "axis": axis_component,
        "facets": facets_doc,
        "requested_window": window_doc,
        "coverage_stamp": coverage.stamp,
        "origin_catalogue": catalogue.build_id,
        "un_system_enabled": catalogue.un_system_enabled,
    }
    canonical = rfc8785.dumps(document)
    return f"opt_{hashlib.sha256(canonical).hexdigest()}"


# ---------------------------------------------------------------------------
# OptionsResponseCache: TTL/LRU, single-flight, keyed by
# options_response_cache_key
# ---------------------------------------------------------------------------


def is_cacheable(computed: ComputedOptions) -> bool:
    """Whether `computed` may be stored: no read failed during its compute
    (`ComputedOptions.transient`) and it noted no degraded read. A stored result keeps its
    warnings, so a hit returns the UN front door's refusals as the compute did."""
    return not computed.transient and not computed.degraded


def _recording_degraded(
    compute: Callable[[float], ComputedOptions],
) -> Callable[[float], ComputedOptions]:
    """`compute` under its own degraded-read record, marking the result `degraded` when the
    record noted any read."""

    def recorded(deadline: float) -> ComputedOptions:
        with recording_degraded_reads() as lost:
            computed = compute(deadline)
        return replace(computed, degraded=True) if lost.degraded else computed

    return recorded


class OptionsResponseCache:
    """Bounded, generation-keyed, single-flight cache of option results.

    Configuration supplies entry count and enabled state by default.
    `max_entries` must be at least 1. Disabled mode runs `compute` without
    caching or single-flight. Each call captures a generation stamp for its
    key and entry lifetime; `stats()` and `clear()` support provider hooks.
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
            config.settings().options_response_cache_enabled if enabled is None else enabled
        )
        max_entries = (
            config.settings().options_response_cache_max_entries
            if max_entries is None
            else max_entries
        )
        if max_entries < 1:
            raise ValueError(f"max_entries must be at least 1, got {max_entries!r}")
        self._cache: SingleFlightCache[str, ComputedOptions] = SingleFlightCache(
            maxsize=max_entries if self._enabled else 0,
            flight_timeout_s=config.settings().request_timeout_s,
            generation=generation,
            clock=clock,
        )

    async def get_or_compute(
        self, key: str, compute: Callable[[float], ComputedOptions], *, deadline: float
    ) -> ComputedOptions:
        """Return the cached result or compute it on a miss.

        Exceptions propagate and are not stored. Transient or degraded results are
        shared with concurrent callers but not cached. `deadline` bounds only
        this caller's wait; timeout raises `asyncio.TimeoutError` without
        cancelling the shared task. `compute` receives the absolute deadline
        its reads must honour: the flight's, or this caller's when the cache
        is disabled.
        """
        return await self._cache.aget_or_fetch(
            key, _recording_degraded(compute), deadline=deadline, is_cacheable=is_cacheable
        )

    def cached(self, key: str) -> ComputedOptions | None:
        """The stored result under `key`, without computing or joining a flight. A result
        found counts a hit. A key not found counts nothing, since the caller computes it
        next and `get_or_compute` counts the miss."""
        return self._cache.cached_many([key]).get(key)

    def stamp_key(self) -> str:
        """The current generation stamp's key, read before a compute whose result is then
        passed to `offer`."""
        return self._cache.stamp_key()

    def offer(self, key: str, value: ComputedOptions, *, stamp_key: str) -> bool:
        """Store `value`, computed outside the cache, without joining or waiting on a flight.

        Stores only when the generation stamp still has `stamp_key`, `value` is cacheable and
        no live entry holds `key`, so an options request computing the same key can neither
        delay nor fail the caller. Return whether it stored."""
        return is_cacheable(value) and self._cache.offer(key, value, stamp_key=stamp_key)

    def stats(self) -> Mapping[str, int]:
        """Hit/miss/eviction counters plus the current entry count, for the
        `graph_generation` provider to read on registration (elsewhere)."""
        return self._cache.stats()

    def clear(self) -> None:
        """Drops every stored entry and every in-flight task's own
        bookkeeping (never the task itself). Called by the `graph_generation`
        provider on a token change once this cache is registered (elsewhere).
        A task already running keeps running to completion regardless. It
        still shields every caller still awaiting it, and its own eventual
        publish is a harmless no-op against an entry this call already
        dropped."""
        self._cache.clear()
