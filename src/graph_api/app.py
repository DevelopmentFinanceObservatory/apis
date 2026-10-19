"""Compose the Graph API's routes, dependencies, and error classifiers into one surface.

The lifespan acquires the graph generation first, then builds one shared graph
reader keyed on it, which routes `undata/` dcids to the UN System Data Commons
and reads UN availability from that deployment's SDMX root. It also builds the
bounded DCP-read executor that refuses a read when every worker is busy. It
acquires the process's shared facet profile cache (`materialise.shared_profiles`)
over the same API root for graph reads and the mixer's SDMX availability, which
Search and Datasets share when mounted beside it and naming the same roots.
Beside it sit the caches of the national currencies and node facts a variable's
availability reads, whose flights run on the DCP-read executor, and a small
executor for the drivers that read node facts in chunks after a request answers.
It publishes the shared resolver's membership policy, which the node, resolve
and members doors classify groups and regions by. Surface
construction loads the origin catalogue, so a missing or unloadable
`ORIGIN_CATALOGUE_URI` fails startup, and each request binds its loaded
snapshot. Resolve confirmation and region reads, node reads, and
source-document reads share them. Resolvekit matching uses its
own single-worker executor. A live surface also acquires the process-wide
coverage service when `COVERAGE_ARTIFACT_URI` is set, and the sources door
reads currency and granularity from it. A missing URI, a missing `coverage`
extra, or an unreadable artifact leaves coverage unset and startup continues; only the
unreadable artifact is marked as a failure (`GraphDeps.coverage_load_failed`).

Routes: `GET /healthz`, `GET /capabilities`, `GET /schema`, `POST /resolve`,
`GET /node/{dcid}`, `POST /nodes`, `GET /node/{dcid}/members`, `GET /roots`,
`GET /documentation/{dcid}`, and `GET /sources`, all under `/api/graph`.

Door classifiers return `ProblemSpec`. The door whose route raised classifies
first (`dfo_platform.doors.classify_by_door`), so a graph read failure is logged
by its own door, and the platform renders RFC 9457 responses. The fronting Data access proxy handles request
authentication and enforces `401`, `413`, `415`, and `429`.
"""

from __future__ import annotations

import asyncio
import functools
import logging
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

from fastapi import FastAPI

from code_index.shared import get_index
from dc_sdmx import SdmxClient
from dfo_platform import ProblemSpec
from dfo_platform.app import Surface
from dfo_platform.doors import DoorContribution, classify_by_door, flatten
from dfo_platform.problem import BACKPRESSURE_HEADERS
from entity_resolution import EntityResolver, SharedResolver, acquire_shared, release_shared
from graph_api.card_reads import CardReadCaches
from graph_api.code_lookup import CodeIndexCodes
from graph_api.codes import GRAPH_CODE_CATALOGUE
from graph_api.config import (
    GRAPH_NODE_POOL_SIZE,
    GRAPH_NODE_TIMEOUT_SECONDS,
    GRAPH_SDMX_TIMEOUT_SECONDS,
    GraphSettings,
)
from graph_api.deps import (
    GRAPH_DEPS_SLOT,
    DcpExecutorSaturated,
    DcpExecutorShutDown,
    DcpReadExecutor,
    GraphDeps,
    GraphDepsNotSet,
)
from graph_api.doors.health import build_health_door
from graph_api.doors.members import build_members_door
from graph_api.doors.node import NODE_MEMBERSHIP_SLOT, build_node_door
from graph_api.doors.resolve import build_resolve_door
from graph_api.doors.schema import build_schema_door
from graph_api.doors.sources import build_sources_door, new_provenance_doc_cache
from graph_api.node_cache import NodeEntryCache
from graph_api.reader import GraphNodeReader
from graph_api.version import GRAPH_CAPABILITY_VERSION
from graph_generation import UNKNOWN_GENERATION, GenerationProvider, RegistrationHandle
from graph_generation import acquire_shared as acquire_shared_generation
from graph_generation import release_shared as release_shared_generation
from materialise.facet_profile import (
    FLIGHT_TIMEOUT_S,
    MAX_BYTES,
    PROFILE_CACHE_MAXSIZE,
    FacetProfileCache,
    ProfileGraphReader,
)
from materialise.place_scope import AvailabilityReader
from materialise.shared_profiles import (
    FACET_READ_WORKERS,
    PROFILE_WORKERS,
    ProfileSource,
    SharedProfiles,
    acquire_shared_profiles,
    release_shared_profiles,
)
from origin_catalogue import bound_catalogue
from origin_catalogue.graph import (
    ARCS_CACHE_SIZE,
    LABEL_TYPE_CACHE_SIZE,
    OBS_CACHE_SIZE,
    OBS_PROPERTIES_CACHE_SIZE,
    OriginGraphClient,
    build_graph_client,
)
from origin_catalogue.service import get_catalogue_service

if TYPE_CHECKING:
    from series_coverage.shared import CoverageService

logger = logging.getLogger(__name__)

# The names and types of the units, methods and places variable cards read: a name and a
# types tuple each, about 6 MiB when full.
_CARD_FACTS_CACHE_SIZE = 20_000
# On the dev graph a 500-dcid chunk of names and types reads in 0.2 to 0.5 s, the 3,536
# places of the countries and US counties in 1.9 s, and the national currencies in about
# 3.5 s (14 s once, cold). The graph stalls 2 to 4 s about once a minute. A detached read
# stops at 30 s, twice the slowest read seen, which covers a 5,000-place scope.
_CARD_FACTS_CHUNK_SIZE = 500
_CARD_DETACHED_TIMEOUT_SECONDS = 30.0
# Each facts driver holds one DCP-read worker at a time, so two leave the rest for requests.
_CARD_DRIVERS = 2


def _classify_deps_exception(exc: BaseException) -> ProblemSpec | None:
    """Classify requests that arrive before the lifespan publishes `GRAPH_DEPS_SLOT`, reads
    refused because every DCP-read worker is busy, and reads submitted after teardown shut
    the DCP-read executor down."""
    if isinstance(exc, GraphDepsNotSet):
        return ProblemSpec(
            status=503,
            code="service_unavailable",
            detail="the Graph surface has not finished starting; retry",
            retryable=True,
        )
    if isinstance(exc, DcpExecutorSaturated):
        return ProblemSpec(
            status=503,
            code="service_unavailable",
            detail="the Graph surface is busy; retry",
            retryable=True,
            headers=BACKPRESSURE_HEADERS,
        )
    if isinstance(exc, DcpExecutorShutDown):
        return ProblemSpec(
            status=503,
            code="service_unavailable",
            detail="the Graph surface is shutting down; retry",
            retryable=True,
        )
    return None


async def _acquire_coverage() -> tuple[CoverageService | None, bool]:
    """Return the process-wide coverage service, or `None` when it is unavailable, and whether
    the unavailability is a failure. An unset `COVERAGE_ARTIFACT_URI` and a missing `coverage`
    extra are configuration, so they are not failures; an artifact that failed to load is.

    The first acquisition loads the artifact, so it runs off the event loop.
    The service is a process singleton that no surface releases.
    """
    try:
        from series_coverage.shared import (
            CoverageArtifactUriMissingError,
            get_coverage_service,
        )
    except ImportError as exc:
        logger.info(
            "Graph surface serves no currency or granularity: the coverage extra is missing (%s)",
            exc,
        )
        return None, False
    try:
        return await asyncio.to_thread(get_coverage_service), False
    except CoverageArtifactUriMissingError:
        logger.info(
            "Graph surface serves no currency or granularity: COVERAGE_ARTIFACT_URI is not set"
        )
        return None, False
    except Exception:
        logger.warning(
            "Graph surface serves no currency or granularity: the coverage artifact failed to load",
            exc_info=True,
        )
        return None, True


def graph_surface(
    *,
    graph: GraphNodeReader | None = None,
    resolver: EntityResolver | None = None,
    node_pool_size: int | None = None,
    coverage: CoverageService | None = None,
    profile_graph: ProfileGraphReader | None = None,
    availability: AvailabilityReader | None = None,
) -> Surface:
    """Build the Graph surface.

    Inject `graph` or `resolver` to bypass their live startup resources.
    `node_pool_size` sets the bounded DCP executor's worker count; `None` uses
    `GRAPH_NODE_POOL_SIZE`. `coverage`
    feeds the sources door's currency and granularity. When it is omitted, a
    live surface (no injected `graph`) acquires the process-wide service and an
    injected graph runs without coverage. `profile_graph` and `availability`
    feed a facet profile cache of the surface's own, on an executor of its own, for an
    injected graph. A live surface ignores them and acquires the process's shared cache, and
    an injected graph runs without a profile cache unless both are given. An injected
    resolver leaves the membership policy for its caller to publish.
    """
    catalogue = get_catalogue_service()
    # Distinct literal prefixes keep documentation and node routes disjoint. The members
    # route goes ahead of the node route, whose dcid convertor accepts slashes and would
    # take `<dcid>/members` as one dcid.
    doors: tuple[DoorContribution, ...] = (
        build_health_door(),
        build_schema_door(),
        build_resolve_door(),
        build_sources_door(),
        build_members_door(),
        build_node_door(),
    )

    router = flatten(doors)

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        # Read live settings only for a live reader. Injected graphs need no environment.
        # Fresh locals make repeated lifespan entries independent.
        live_reader: OriginGraphClient | None = None
        shared_profiles: SharedProfiles | None = None
        shared_resolver: SharedResolver | None = None
        shared_generation: GenerationProvider | None = None
        generation_registration: RegistrationHandle | None = None
        doc_cache_registration: RegistrationHandle | None = None
        node_entries_registration: RegistrationHandle | None = None
        dcp_executor: DcpReadExecutor | None = None
        profile_executor: ThreadPoolExecutor | None = None
        profile_reads: ThreadPoolExecutor | None = None
        card_reads: CardReadCaches | None = None
        graph_dcp_base_url: str | None = None
        resolved_coverage = coverage
        coverage_load_failed = False

        async def _release_owned() -> None:
            """Drain queued and running DCP reads off-loop, then wait for running card-read
            drivers, whose next read the shut DCP executor refuses. Release the shared profile
            cache, or cancel the injected surface's queued profile flights and facet reads and
            wait for running ones. Close registrations, release the resolver, then generation, then
            close the graph reader last. Release only resources this lifespan owns. Injected
            graph, resolver and availability objects remain caller-owned.
            """
            if dcp_executor is not None:
                await asyncio.to_thread(dcp_executor.shutdown, wait=True)
            if card_reads is not None:
                await asyncio.to_thread(card_reads.close)
            if shared_profiles is not None:
                await asyncio.to_thread(release_shared_profiles, shared_profiles)
            if profile_executor is not None:
                await asyncio.to_thread(profile_executor.shutdown, wait=True, cancel_futures=True)
            if profile_reads is not None:
                await asyncio.to_thread(profile_reads.shutdown, wait=True, cancel_futures=True)
            if doc_cache_registration is not None:
                doc_cache_registration.close()
            if node_entries_registration is not None:
                node_entries_registration.close()
            if generation_registration is not None:
                generation_registration.close()
            if shared_resolver is not None:
                release_shared(shared_resolver)
            if shared_generation is not None:
                release_shared_generation(shared_generation)
            if live_reader is not None:
                live_reader.close()

        try:
            settings = GraphSettings.from_env() if graph is None else None
            if settings is not None:
                graph_dcp_base_url = settings.dcp_base_url

            resolved_generation: GenerationProvider = UNKNOWN_GENERATION
            # A live reader keys its caches on the generation, and a live resolver its
            # cache, so either acquires it. Only injected graph and resolver skip it.
            if graph is None or resolver is None:
                # The first generation read is synchronous, so acquire it off the event loop.
                # An injected graph has no DCP base URL to cross-check.
                shared_generation = await asyncio.to_thread(
                    functools.partial(
                        acquire_shared_generation,
                        catalogue=catalogue,
                        expected_graph_base=graph_dcp_base_url,
                    )
                )
                resolved_generation = shared_generation

            if settings is not None:
                live_reader = build_graph_client(
                    base_url=settings.dcp_base_url,
                    token=settings.dcp_token,
                    generation=resolved_generation,
                    un_base_url=settings.un_base_url,
                    un_availability=SdmxClient(
                        base_url=settings.un_base_url, timeout=GRAPH_SDMX_TIMEOUT_SECONDS
                    ),
                    label_cache_size=LABEL_TYPE_CACHE_SIZE,
                    type_cache_size=LABEL_TYPE_CACHE_SIZE,
                    arcs_cache_size=ARCS_CACHE_SIZE,
                    obs_properties_cache_size=OBS_PROPERTIES_CACHE_SIZE,
                    obs_cache_size=OBS_CACHE_SIZE,
                )
                # The mixer serves SDMX under the same API root as its graph reads.
                shared_profiles = acquire_shared_profiles(
                    source=ProfileSource(
                        graph_base_url=settings.dcp_base_url,
                        sdmx_base_url=settings.dcp_base_url,
                        un_base_url=settings.un_base_url,
                        token=settings.dcp_token or None,
                    ),
                    generation=resolved_generation,
                )
            resolved_graph = live_reader if live_reader is not None else graph
            assert resolved_graph is not None

            effective_pool_size = (
                node_pool_size if node_pool_size is not None else GRAPH_NODE_POOL_SIZE
            )
            dcp_executor = DcpReadExecutor(
                max_workers=effective_pool_size, thread_name_prefix="graph-dcp"
            )

            resolved_resolver = resolver
            # Injected resolvers skip shared acquisition. The live resolver's graph
            # confirmation reads use this surface's bounded DCP executor.
            if resolver is None:
                assert shared_generation is not None
                shared_resolver = acquire_shared()
                resolver_instance = EntityResolver(
                    graph=resolved_graph,
                    shared=shared_resolver,
                    graph_executor=dcp_executor,
                    generation=shared_generation,
                    codes=CodeIndexCodes(get_index()),
                    code_index=get_index(),
                )
                generation_registration = shared_generation.register(
                    "graph_api.entity_resolver",
                    stats=resolver_instance.stats,
                    clear=resolver_instance.clear,
                )
                resolved_resolver = resolver_instance

            provenance_docs = new_provenance_doc_cache(
                executor=dcp_executor, generation=resolved_generation
            )
            doc_cache_registration = resolved_generation.register(
                "graph_api.provenance_docs",
                stats=provenance_docs.stats,
                clear=provenance_docs.clear,
            )
            node_entries = NodeEntryCache(generation=resolved_generation)
            node_entries_registration = resolved_generation.register(
                "graph_api.node_entries", stats=node_entries.stats, clear=node_entries.clear
            )

            if resolved_coverage is None and graph is None:
                resolved_coverage, coverage_load_failed = await _acquire_coverage()

            profile_cache: FacetProfileCache | None = None
            if shared_profiles is not None:
                profile_cache = shared_profiles.cache
            elif profile_graph is not None and availability is not None:
                # A single-flight cache needs a pool no caller runs on.
                profile_executor = ThreadPoolExecutor(
                    max_workers=PROFILE_WORKERS, thread_name_prefix="graph-profile"
                )
                profile_reads = ThreadPoolExecutor(
                    max_workers=FACET_READ_WORKERS, thread_name_prefix="graph-profile-facets"
                )
                profile_cache = FacetProfileCache(
                    graph=profile_graph,
                    availability=availability,
                    executor=profile_executor,
                    reads=profile_reads,
                    generation=resolved_generation,
                    flight_timeout_s=FLIGHT_TIMEOUT_S,
                    maxsize=PROFILE_CACHE_MAXSIZE,
                    max_bytes=MAX_BYTES,
                )
            if profile_cache is not None:
                card_reads = CardReadCaches(
                    graph=resolved_graph,
                    executor=dcp_executor,
                    drivers=_CARD_DRIVERS,
                    generation=resolved_generation,
                    maxsize=_CARD_FACTS_CACHE_SIZE,
                    chunk_size=_CARD_FACTS_CHUNK_SIZE,
                    chunk_timeout_s=GRAPH_NODE_TIMEOUT_SECONDS,
                    detached_timeout_s=_CARD_DETACHED_TIMEOUT_SECONDS,
                )
        except BaseException:
            await _release_owned()
            raise

        assert resolved_resolver is not None
        deps = GraphDeps(
            graph=resolved_graph,
            resolver=resolved_resolver,
            dcp_executor=dcp_executor,
            generation=resolved_generation,
            provenance_docs=provenance_docs,
            node_entries=node_entries,
            coverage=resolved_coverage,
            coverage_load_failed=coverage_load_failed,
            profile_cache=profile_cache,
            card_reads=card_reads,
        )
        GRAPH_DEPS_SLOT.set(_app, deps)
        if shared_resolver is not None:
            NODE_MEMBERSHIP_SLOT.set(_app, shared_resolver.policy)
        try:
            yield
        finally:
            NODE_MEMBERSHIP_SLOT.clear(_app)
            GRAPH_DEPS_SLOT.clear(_app)
            await _release_owned()

    return Surface(
        name="graph",
        prefix="/api/graph",
        router=router,
        capability_version=GRAPH_CAPABILITY_VERSION,
        codes=GRAPH_CODE_CATALOGUE,
        classify=classify_by_door(doors, _classify_deps_exception),
        exception_types=(
            GraphDepsNotSet,
            DcpExecutorSaturated,
            DcpExecutorShutDown,
            *(exc_type for door in doors for exc_type in door.exception_types),
        ),
        exception_exemplars=(
            GraphDepsNotSet(),
            DcpExecutorSaturated(),
            DcpExecutorShutDown(),
            *(exemplar for door in doors for exemplar in door.exception_exemplars),
        ),
        lifespan=lifespan,
        registry_endpoints={
            path: endpoint for door in doors for path, endpoint in door.registry_endpoints.items()
        },
        request_scope=lambda: bound_catalogue(catalogue.snapshot()),
    )
