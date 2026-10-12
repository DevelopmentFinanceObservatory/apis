"""Dependencies shared by the Graph doors and published during the surface lifespan.

The node, sources, and resolve doors share the graph reader and the bounded
DCP-read executor. Resolve also uses the resolver and generation provider.
Resolvekit owns a separate single-worker executor. The sources door reads
currency from the coverage service when one is configured, and keeps the
source documents it reads in `provenance_docs`.

This module holds the bundle and the doors' one way onto the executor,
`run_dcp_read`, to keep door imports separate from the composition root,
which imports every door.
"""

from __future__ import annotations

import asyncio
import functools
import logging
from collections.abc import Callable
from concurrent.futures import BrokenExecutor, ThreadPoolExecutor
from dataclasses import dataclass
from time import perf_counter
from typing import TYPE_CHECKING, Annotated, Any, override

from fastapi import Depends

from dfo_platform.executor import SaturationTrackingExecutor, submit_with_context
from dfo_platform.single_flight_cache import SingleFlightCache
from dfo_platform.state import StateSlot
from entity_resolution import EntityResolver
from graph_api.config import DCP_QUEUE_WAIT_WARN_SECONDS
from graph_api.reader import GraphNodeReader
from graph_generation import GenerationProvider

if TYPE_CHECKING:
    from materialise.provenance_doc import ProvenanceDoc

    # `series_coverage.shared` needs numpy, which the `graph` extra does not install.
    from series_coverage.shared import CoverageService


logger = logging.getLogger(__name__)


class GraphDepsNotSet(Exception):
    """A request arrived before lifespan setup. The surface classifies this as retryable 503."""

    def __init__(self) -> None:
        super().__init__("Graph dependencies are not set: the surface's lifespan has not run")


@dataclass(frozen=True, slots=True)
class GraphDeps:
    graph: GraphNodeReader
    resolver: EntityResolver
    dcp_executor: ThreadPoolExecutor
    generation: GenerationProvider
    # Source documents keyed by `(dcid, origin catalogue build id)` and `generation`'s stamp,
    # read on `dcp_executor`.
    provenance_docs: SingleFlightCache[tuple[str, str], ProvenanceDoc]
    # `None` when `COVERAGE_ARTIFACT_URI` is unset or the artifact failed to load.
    coverage: CoverageService | None = None


class DcpExecutorShutDown(Exception):
    """The DCP-read executor refused new work because it is shut down, as during lifespan
    teardown. The surface classifies this as retryable 503."""

    def __init__(self) -> None:
        super().__init__("the Graph surface's DCP-read executor is shut down")


class DcpReadExecutor(SaturationTrackingExecutor):
    """The Graph surface's DCP-read executor. It refuses work after shutdown with
    `DcpExecutorShutDown`, so every graph read answers retryable 503, whether it comes
    from a door through `run_dcp_read` or from the resolver's confirmation."""

    @override
    def submit(self, fn, /, *args: Any, **kwargs: Any):  # type: ignore[override]
        try:
            return super().submit(fn, *args, **kwargs)
        except BrokenExecutor:
            raise
        except RuntimeError:
            # `ThreadPoolExecutor.submit` raises a bare `RuntimeError` after shutdown.
            raise DcpExecutorShutDown() from None


async def run_dcp_read[**P, R](
    executor: ThreadPoolExecutor, fn: Callable[P, R], /, *args: P.args, **kwargs: P.kwargs
) -> R:
    """Run `fn` on the DCP-read executor under the caller's context and await its result.

    A busy executor queues the call. Cancelling the await while the call is still queued
    cancels it, so a door's `wait_for` timeout stops a queued read from ever running. A
    running call keeps its worker until it returns. Logs and re-raises `BrokenExecutor`.
    The call logs its queue wait (`log_queue_wait`).
    """
    timed = log_queue_wait(fn)
    try:
        future = submit_with_context(executor, timed, *args, **kwargs)
    except BrokenExecutor:
        logger.error(
            "the Graph surface's DCP-read executor is permanently broken; "
            "every read submitted to it will fail the same way",
            exc_info=True,
        )
        raise
    return await asyncio.wrap_future(future)


def log_queue_wait[**P, R](fn: Callable[P, R], /) -> Callable[P, R]:
    """`fn` wrapped to log, as it starts, the time since the wrap. Wrap `fn` as it is
    submitted to the DCP-read executor, so the time is its queue wait. The line is DEBUG, or
    WARNING with the threshold when the wait passed `DCP_QUEUE_WAIT_WARN_SECONDS`."""
    name = getattr(fn, "__qualname__", type(fn).__qualname__)
    submitted = perf_counter()

    @functools.wraps(fn)
    def run(*args: P.args, **kwargs: P.kwargs) -> R:
        waited_ms = (perf_counter() - submitted) * 1000
        threshold_ms = DCP_QUEUE_WAIT_WARN_SECONDS * 1000
        if waited_ms > threshold_ms:
            logger.warning(
                "DCP read %s waited %.0f ms for a worker, past the %.0f ms queue-wait threshold",
                name,
                waited_ms,
                threshold_ms,
            )
        else:
            logger.debug("DCP read %s waited %.0f ms for a worker", name, waited_ms)
        return fn(*args, **kwargs)

    return run


# A distinct slot name keeps composed surfaces separate. Missing dependencies map to 503.
GRAPH_DEPS_SLOT: StateSlot[GraphDeps] = StateSlot("graph_deps", unset_exception=GraphDepsNotSet)

# `Annotated` avoids a `Depends(...)` call in a default argument (ruff B008).
GraphDepsDep = Annotated[GraphDeps, Depends(GRAPH_DEPS_SLOT)]
