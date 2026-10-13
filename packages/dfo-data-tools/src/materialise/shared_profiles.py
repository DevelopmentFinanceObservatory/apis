"""The process's facet profile caches, one per source, shared by every surface and worker
that reads facet profiles and reference-counted like the shared graph generation
(`graph_generation.shared`).

A source is the graph, SDMX and UN roots a cache reads and the generation it is keyed under.
Surfaces naming the same source share one cache, so their profile reads share one flight per
cold variable and one memory bound. That is the deployed case, where every surface reads one
mixer. A process whose surfaces name different roots holds one cache per source. The first
acquire of a source builds its cache over clients and an executor of its own: an origin graph
client at `ProfileSource.graph_base_url`, which routes `undata/` dcids to
`ProfileSource.un_base_url` and reads UN availability there, the mixer's SDMX availability at
`ProfileSource.sdmx_base_url`, `PROFILE_WORKERS` flight threads no caller runs on, and
`FACET_READ_WORKERS` threads for the flights' facet reads. Within a
cache a profile's key is the variable, the catalogue build, the coverage build and the source
scope, under the generation, so callers binding different catalogues or coverage builds read
apart. The token is part of the source, so surfaces holding different credentials never share
a cache and each flight reads under its own surface's token.

The last release of a source shuts its executors, waits up to `FLIGHT_TIMEOUT_S` for running
flights and their reads to end, then closes its clients, so no flight reads through a closed client unless it
outlives that wait. Release blocks for that long at most, so an async caller runs it off the
event loop. Building and closing a cache log, at info, the sources the process holds.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from dc_sdmx import SdmxClient
from graph_generation import GenerationProvider
from materialise.facet_profile import (
    FLIGHT_TIMEOUT_S,
    MAX_BYTES,
    PROFILE_CACHE_MAXSIZE,
    FacetProfileCache,
    ProfileBreadth,
)
from origin_catalogue.graph import (
    ARCS_CACHE_SIZE,
    LABEL_TYPE_CACHE_SIZE,
    OBS_CACHE_SIZE,
    OBS_PROPERTIES_CACHE_SIZE,
    OriginGraphClient,
    build_graph_client,
)

if TYPE_CHECKING:
    from series_coverage.shared import CoverageSnapshot

logger = logging.getLogger(__name__)

PROFILE_WORKERS = 16
"""The threads a source's profile flights run on, for every surface together: the variables
whose profiles read at once. A flight spends its time waiting on its presence read, which runs
on `place_scope`'s settle pool of as many threads, and on its facet reads."""

FACET_READ_WORKERS = 16
"""The threads a source's profile flights run their facet reads on, so the most facet reads
its flights have in flight at once."""


@dataclass(frozen=True, slots=True)
class ProfileSource:
    """Where a process's profile flights read, and as whom. Each address is an API root
    including `/core/api`, compared without a trailing slash. `token` is sent as
    `x-graph-token` to the mixer; it is compared, so one cache serves one credential, and left
    out of the repr the cache logs."""

    graph_base_url: str
    sdmx_base_url: str
    un_base_url: str
    token: str | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        for name in ("graph_base_url", "sdmx_base_url", "un_base_url"):
            object.__setattr__(self, name, getattr(self, name).rstrip("/"))


@dataclass(frozen=True, slots=True, eq=False)
class SharedProfiles:
    """One source's profile cache and the clients and executor it owns. Use
    `acquire_shared_profiles` and `release_shared_profiles`."""

    cache: FacetProfileCache
    source: ProfileSource
    generation: GenerationProvider
    executor: ThreadPoolExecutor
    reads: ThreadPoolExecutor
    graph: OriginGraphClient
    availability: SdmxClient

    def breadth(self, coverage: Callable[[], CoverageSnapshot]) -> ProfileBreadth:
        """A whole-variable breadth reader over this cache, keyed on the coverage build
        `coverage` returns (`facets.bound_variable_breadth`)."""
        return ProfileBreadth(profiles=self.cache, coverage=coverage)


def _build(*, source: ProfileSource, generation: GenerationProvider) -> SharedProfiles:
    graph = build_graph_client(
        base_url=source.graph_base_url,
        token=source.token,
        generation=generation,
        un_base_url=source.un_base_url,
        un_availability=SdmxClient(base_url=source.un_base_url),
        label_cache_size=LABEL_TYPE_CACHE_SIZE,
        type_cache_size=LABEL_TYPE_CACHE_SIZE,
        arcs_cache_size=ARCS_CACHE_SIZE,
        obs_properties_cache_size=OBS_PROPERTIES_CACHE_SIZE,
        obs_cache_size=OBS_CACHE_SIZE,
    )
    availability = SdmxClient(base_url=source.sdmx_base_url, token=source.token)
    executor = ThreadPoolExecutor(max_workers=PROFILE_WORKERS, thread_name_prefix="profile")
    reads = ThreadPoolExecutor(max_workers=FACET_READ_WORKERS, thread_name_prefix="profile-facets")
    cache = FacetProfileCache(
        graph=graph,
        availability=availability,
        executor=executor,
        reads=reads,
        generation=generation,
        flight_timeout_s=FLIGHT_TIMEOUT_S,
        maxsize=PROFILE_CACHE_MAXSIZE,
        max_bytes=MAX_BYTES,
    )
    return SharedProfiles(
        cache=cache,
        source=source,
        generation=generation,
        executor=executor,
        reads=reads,
        graph=graph,
        availability=availability,
    )


def _close(shared: SharedProfiles) -> None:
    # A running flight and its reads end by its deadline, `FLIGHT_TIMEOUT_S` from its start, so
    # the wait is bounded by the same figure; queued flights and reads are cancelled.
    def drain() -> None:
        shared.executor.shutdown(wait=True, cancel_futures=True)
        shared.reads.shutdown(wait=True, cancel_futures=True)

    draining = threading.Thread(target=drain, name="profile-shutdown", daemon=True)
    draining.start()
    draining.join(FLIGHT_TIMEOUT_S)
    if draining.is_alive():
        logger.warning(
            "a facet profile flight still runs %.0f s after shutdown; closing its clients",
            FLIGHT_TIMEOUT_S,
        )
    shared.availability.close()
    shared.graph.close()


type _SourceKey = tuple[ProfileSource, int]
"""A source and its generation's identity. A held cache keeps its generation alive, so the
identity cannot be reused while the key is held."""

_lock = threading.Lock()
_held: dict[_SourceKey, tuple[SharedProfiles, int]] = {}


def _held_sources() -> list[ProfileSource]:
    return [shared.source for shared, _count in _held.values()]


def acquire_shared_profiles(
    *, source: ProfileSource, generation: GenerationProvider
) -> SharedProfiles:
    """Acquire the profile cache for `source` under `generation`, building it on the source's
    first acquire. Callers naming the same roots and generation share one cache."""
    key = (source, id(generation))
    with _lock:
        held = _held.get(key)
        if held is None:
            shared, count = _build(source=source, generation=generation), 0
        else:
            shared, count = held
        _held[key] = (shared, count + 1)
        if held is None:
            logger.info("facet profile caches held: %s", _held_sources())
    return shared


def release_shared_profiles(shared: SharedProfiles) -> None:
    """Release one holder of `shared`. The source's last release shuts its executors, waits up
    to `FLIGHT_TIMEOUT_S` for running flights and reads, then closes its clients. Raises `RuntimeError`
    for a stale or extra release."""
    key = (shared.source, id(shared.generation))
    with _lock:
        held = _held.get(key)
        if held is None or held[0] is not shared:
            raise RuntimeError(
                "release_shared_profiles() called on a facet profile cache that "
                "acquire_shared_profiles() never returned, or more times than it was acquired"
            )
        count = held[1] - 1
        if count:
            _held[key] = (shared, count)
            return
        del _held[key]
        logger.info("facet profile caches held: %s", _held_sources())
    _close(shared)
