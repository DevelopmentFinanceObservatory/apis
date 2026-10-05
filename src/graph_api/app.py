"""Compose the Graph API's routes, dependencies, and error classifiers into one surface.

The lifespan builds one shared graph reader, which routes `undata/` dcids to
the UN System Data Commons, and a bounded DCP-read executor with an admission
semaphore. Surface construction loads the origin catalogue, so a missing or
unloadable `ORIGIN_CATALOGUE_URI` fails startup, and each request binds its
loaded snapshot. Resolve confirmation and region reads, node
reads, and source-document reads share them. Resolvekit matching uses its
own single-worker executor. A live surface also acquires the process-wide
coverage service when `COVERAGE_ARTIFACT_URI` is set, and the sources door
reads currency and granularity from it. A missing URI, a missing `coverage`
extra, or an unreadable artifact leaves coverage unset and startup continues.

Routes: `GET /healthz`, `GET /capabilities`, `POST /resolve`,
`GET /node/{dcid}`, `GET /documentation/{dcid}`, and `GET /sources`, all
under `/api/graph`.

Door classifiers return `ProblemSpec`. The platform combines them and
renders RFC 9457 responses. The fronting Data access proxy handles request
authentication and enforces `401`, `413`, `415`, and `429`.
"""

from __future__ import annotations

import asyncio
import functools
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

from fastapi import FastAPI

from dfo_platform import ProblemSpec
from dfo_platform.app import Surface
from dfo_platform.doors import DoorContribution, combine_classifiers, flatten
from entity_resolution import EntityResolver, SharedResolver, acquire_shared, release_shared
from graph_api.codes import GRAPH_CODE_CATALOGUE
from graph_api.config import GRAPH_NODE_POOL_SIZE, GraphSettings
from graph_api.deps import GRAPH_DEPS_SLOT, GraphDeps, GraphDepsNotSet
from graph_api.doors.health import build_health_door
from graph_api.doors.node import build_node_door
from graph_api.doors.resolve import build_resolve_door
from graph_api.doors.sources import build_sources_door
from graph_api.reader import GraphNodeReader
from graph_api.version import GRAPH_CAPABILITY_VERSION
from graph_generation import UNKNOWN_GENERATION, GenerationProvider, RegistrationHandle
from graph_generation import acquire_shared as acquire_shared_generation
from graph_generation import release_shared as release_shared_generation
from origin_catalogue import bound_catalogue
from origin_catalogue.graph import OriginGraphClient, build_graph_client
from origin_catalogue.service import get_catalogue_service

if TYPE_CHECKING:
    from series_coverage.shared import CoverageService

logger = logging.getLogger(__name__)


def _classify_deps_exception(exc: BaseException) -> ProblemSpec | None:
    """Classify requests that arrive before the lifespan publishes `GRAPH_DEPS_SLOT`."""
    if isinstance(exc, GraphDepsNotSet):
        return ProblemSpec(
            status=503,
            code="service_unavailable",
            detail="the Graph surface has not finished starting; retry",
            retryable=True,
        )
    return None


async def _acquire_coverage() -> CoverageService | None:
    """Return the process-wide coverage service, or `None` when it is unavailable.

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
        return None
    try:
        return await asyncio.to_thread(get_coverage_service)
    except CoverageArtifactUriMissingError:
        logger.info(
            "Graph surface serves no currency or granularity: COVERAGE_ARTIFACT_URI is not set"
        )
    except Exception:
        logger.warning(
            "Graph surface serves no currency or granularity: the coverage artifact failed to load",
            exc_info=True,
        )
    return None


def graph_surface(
    *,
    graph: GraphNodeReader | None = None,
    resolver: EntityResolver | None = None,
    node_pool_size: int | None = None,
    coverage: CoverageService | None = None,
) -> Surface:
    """Build the Graph surface.

    Inject `graph` or `resolver` to bypass their live startup resources.
    `node_pool_size` sets both the bounded DCP executor size and its
    admission semaphore; `None` uses `GRAPH_NODE_POOL_SIZE`. `coverage`
    feeds the sources door's currency and granularity. When it is omitted, a
    live surface (no injected `graph`) acquires the process-wide service and an
    injected graph runs without coverage.
    """
    catalogue = get_catalogue_service()
    # Distinct literal prefixes keep documentation and node routes disjoint.
    doors: tuple[DoorContribution, ...] = (
        build_health_door(),
        build_resolve_door(),
        build_sources_door(),
        build_node_door(),
    )

    router = flatten(doors)

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        # Read live settings only for a live reader. Injected graphs need no environment.
        # Fresh locals make repeated lifespan entries independent.
        live_reader: OriginGraphClient | None = None
        shared_resolver: SharedResolver | None = None
        shared_generation: GenerationProvider | None = None
        generation_registration: RegistrationHandle | None = None
        dcp_executor: ThreadPoolExecutor | None = None
        dcp_semaphore: threading.BoundedSemaphore | None = None
        graph_dcp_base_url: str | None = None
        resolved_coverage = coverage

        async def _release_owned() -> None:
            """Drain queued and running work off-loop. Close registration, release the
            resolver, then generation, and close the reader last. Release only resources
            this lifespan owns. Injected graph and resolver objects remain caller-owned.
            """
            if dcp_executor is not None:
                await asyncio.to_thread(dcp_executor.shutdown, wait=True)
            if generation_registration is not None:
                generation_registration.close()
            if shared_resolver is not None:
                release_shared(shared_resolver)
            if shared_generation is not None:
                release_shared_generation(shared_generation)
            if live_reader is not None:
                live_reader.close()

        try:
            if graph is None:
                settings = GraphSettings.from_env()
                live_reader = build_graph_client(
                    base_url=settings.dcp_base_url,
                    token=settings.dcp_token,
                    un_base_url=settings.un_base_url,
                )
                graph_dcp_base_url = settings.dcp_base_url
            resolved_graph = live_reader if live_reader is not None else graph
            assert resolved_graph is not None

            effective_pool_size = (
                node_pool_size if node_pool_size is not None else GRAPH_NODE_POOL_SIZE
            )
            dcp_executor = ThreadPoolExecutor(
                max_workers=effective_pool_size, thread_name_prefix="graph-dcp"
            )
            dcp_semaphore = threading.BoundedSemaphore(effective_pool_size)

            resolved_resolver = resolver
            resolved_generation: GenerationProvider = UNKNOWN_GENERATION
            # Injected resolvers skip shared acquisition. The live resolver's graph
            # confirmation reads use this surface's bounded DCP executor.
            if resolver is None:
                # The first generation read is synchronous, so acquire it off the event loop.
                # An injected graph has no DCP base URL to cross-check.
                shared_generation = await asyncio.to_thread(
                    functools.partial(
                        acquire_shared_generation,
                        catalogue=catalogue,
                        expected_graph_base=graph_dcp_base_url,
                    )
                )
                shared_resolver = acquire_shared()
                resolver_instance = EntityResolver(
                    graph=resolved_graph,
                    shared=shared_resolver,
                    graph_executor=dcp_executor,
                    generation=shared_generation,
                )
                generation_registration = shared_generation.register(
                    "graph_api.entity_resolver",
                    stats=resolver_instance.stats,
                    clear=resolver_instance.clear,
                )
                resolved_resolver = resolver_instance
                resolved_generation = shared_generation

            if resolved_coverage is None and graph is None:
                resolved_coverage = await _acquire_coverage()
        except BaseException:
            await _release_owned()
            raise

        assert resolved_resolver is not None
        deps = GraphDeps(
            graph=resolved_graph,
            resolver=resolved_resolver,
            dcp_executor=dcp_executor,
            dcp_semaphore=dcp_semaphore,
            generation=resolved_generation,
            coverage=resolved_coverage,
        )
        GRAPH_DEPS_SLOT.set(_app, deps)
        try:
            yield
        finally:
            GRAPH_DEPS_SLOT.clear(_app)
            await _release_owned()

    return Surface(
        name="graph",
        prefix="/api/graph",
        router=router,
        capability_version=GRAPH_CAPABILITY_VERSION,
        codes=GRAPH_CODE_CATALOGUE,
        classify=combine_classifiers(_classify_deps_exception, *(door.classify for door in doors)),
        exception_types=(
            GraphDepsNotSet,
            *(exc_type for door in doors for exc_type in door.exception_types),
        ),
        exception_exemplars=(
            GraphDepsNotSet(),
            *(exemplar for door in doors for exemplar in door.exception_exemplars),
        ),
        lifespan=lifespan,
        request_scope=lambda: bound_catalogue(catalogue.snapshot()),
    )
