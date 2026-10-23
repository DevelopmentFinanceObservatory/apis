"""Dependencies shared by the Graph doors and published during the surface lifespan.

The node, sources, and resolve doors share the graph reader and the bounded
DCP-read executor, which refuses a read when every worker is busy instead of
queueing it. Resolve also uses the resolver and generation provider.
Resolvekit owns a separate single-worker executor. The sources door reads
currency from the bound release's coverage, and keeps the
source documents it reads in `provenance_docs`. A variable's facet
profile comes from the process's shared `FacetProfileCache` for its source
(`materialise.shared_profiles`), whose flights run on an executor no request
runs on. The national currencies and node facts a
variable's availability reads beside its profile (`CardReadCaches`) run their
flights on the DCP-read executor. The node routes keep the envelopes they build in
`node_entries`, which guards every operation with its own lock.

This module holds the bundle and the doors' one way onto the executor,
`run_dcp_read`, to keep door imports separate from the composition root,
which imports every door.
"""

from __future__ import annotations

import asyncio
import functools
import logging
import threading
from collections.abc import Callable
from concurrent.futures import BrokenExecutor, Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from time import perf_counter
from typing import TYPE_CHECKING, Annotated, Any, override

from fastapi import Depends

from dfo_platform.executor import SaturationTrackingExecutor, submit_with_context
from dfo_platform.single_flight_cache import SingleFlightCache
from dfo_platform.state import StateSlot
from entity_resolution import EntityResolver
from graph_api.config import DCP_QUEUE_WAIT_WARN_SECONDS
from graph_api.node_cache import NodeEntryCache
from graph_api.reader import GraphNodeReader
from graph_generation import GenerationProvider
from materialise.facet_profile import FacetProfileCache

if TYPE_CHECKING:
    # `card_reads` imports this module's executor refusals.
    from graph_api.card_reads import CardReadCaches
    from materialise.provenance_doc import ProvenanceDoc
    from series_coverage.shared import ServedCoverage


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
    coverage: ServedCoverage
    # The process's shared facet profiles on a live surface. An injected graph has its own
    # only when the surface was given a profile graph and availability reader, else `None`.
    profile_cache: FacetProfileCache | None = None
    # The national currencies and node facts a variable's availability reads beside its
    # profile, on the DCP-read executor. Set whenever `profile_cache` is.
    card_reads: CardReadCaches | None = None
    # The envelopes the node routes built, keyed by generation stamp, table builds and dcid.
    node_entries: NodeEntryCache = field(default_factory=NodeEntryCache)


class DcpExecutorShutDown(Exception):
    """The DCP-read executor refused new work because it is shut down, as during lifespan
    teardown. The surface classifies this as retryable 503."""

    def __init__(self) -> None:
        super().__init__("the Graph surface's DCP-read executor is shut down")


class DcpExecutorSaturated(Exception):
    """Every worker of the DCP-read executor is busy, so the read is refused instead of
    queued. The surface classifies this as retryable 503."""

    def __init__(self) -> None:
        super().__init__("every worker of the Graph surface's DCP-read executor is busy")


class DcpReadExecutor(SaturationTrackingExecutor):
    """The Graph surface's DCP-read executor. It admits at most one read per worker and
    refuses the next with `DcpExecutorSaturated`, and refuses work after shutdown with
    `DcpExecutorShutDown`. Every graph read therefore answers retryable 503 at once,
    whether it comes from a door through `run_dcp_read` or from one of the resolver's
    sequential reads, so a resolve admitted at its first read can be refused at a later one.

    A slot is taken at submit and released with the parent's in-flight count, when the read
    returns or the parent refuses the submit. A running read therefore holds its worker's
    slot until it finishes, even after its request has timed out.
    """

    saturation_consequence = "new reads are refused"

    def __init__(self, *, max_workers: int, thread_name_prefix: str, **kwargs: Any) -> None:
        super().__init__(max_workers=max_workers, thread_name_prefix=thread_name_prefix, **kwargs)
        self._admission = threading.BoundedSemaphore(max_workers)

    @override
    def submit(self, fn, /, *args: Any, **kwargs: Any):  # type: ignore[override]
        if not self._admission.acquire(blocking=False):
            raise DcpExecutorSaturated()
        try:
            return super().submit(fn, *args, **kwargs)
        except BrokenExecutor:
            raise
        except RuntimeError:
            # `ThreadPoolExecutor.submit` raises a bare `RuntimeError` after shutdown.
            raise DcpExecutorShutDown() from None

    @override
    def _release_slot(self, _future: Future | None) -> None:
        # Free admission first, so a caller that sees the in-flight count drop can submit.
        self._admission.release()
        super()._release_slot(_future)


async def run_dcp_read[**P, R](
    executor: ThreadPoolExecutor, fn: Callable[P, R], /, *args: P.args, **kwargs: P.kwargs
) -> R:
    """Run `fn` on the DCP-read executor under the caller's context and await its result.

    A `DcpReadExecutor` with every worker busy raises `DcpExecutorSaturated` here. A
    running call keeps its worker until it returns, even after a door's `wait_for`
    timeout has given up on it. Logs and re-raises `BrokenExecutor`. The call logs its
    wait for a worker (`log_queue_wait`).
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
