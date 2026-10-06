"""Adapters between Search clients and the shared ``materialise`` core.

Search's ``LiveGraphClient`` also implements ``materialise.GraphReader`` and
serves both paths. Materialise uses a separate SDMX client, which caches its
observation reads, and both SDMX clients share one GET semaphore.

``observation_facets_via_materialise`` reads facet metadata in bounded,
concurrent graph batches for option coverage.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor

import httpx

import search.engine.config as config  # module-attribute access for runtime-resolved values
from dc_graph import (
    Facet,
    GraphDeadlineExceededError,
    GraphReadError,
)
from dfo_platform.executor import submit_with_context
from graph_generation import UNKNOWN_GENERATION, GenerationProvider
from materialise.graph import GraphReader
from search.engine.clients.materialise_sdmx_cache import CachingMaterialiseSdmxClient
from search.engine.clients.sdmx import SDMX_TOTAL_SEMAPHORE
from search.engine.errors import GraphInfraError

logger = logging.getLogger(__name__)


def build_materialise_sdmx_client(
    *,
    base: str | None = None,
    timeout: float | None = None,
    transport: httpx.BaseTransport | None = None,
    generation: GenerationProvider = UNKNOWN_GENERATION,
) -> CachingMaterialiseSdmxClient:
    """Build a ``CachingMaterialiseSdmxClient`` pointed at ``{base}/core/api``.

    It caches observation reads under ``generation``, holding at most
    ``SEARCH_MATERIALISE_SDMX_CACHE_MAX_BYTES`` in total and
    ``SEARCH_MATERIALISE_SDMX_CACHE_MAX_ENTRY_BYTES`` per read. It shares ``SDMX_TOTAL_SEMAPHORE`` with ``LiveSdmxClient``. The
    ``transport`` argument is for tests.
    """
    _base = (base or config.settings().graph_base).rstrip("/")
    _timeout = timeout if timeout is not None else config.settings().graph_timeout_s
    return CachingMaterialiseSdmxClient(
        base_url=f"{_base}/core/api",
        token=config.settings().graph_token or None,
        timeout=_timeout,
        transport=transport,
        chunk_concurrency=config.settings().sdmx_chunk_concurrency,
        total_concurrency=SDMX_TOTAL_SEMAPHORE,
        max_bytes=config.settings().materialise_sdmx_cache_max_bytes,
        max_entry_bytes=config.settings().materialise_sdmx_cache_max_entry_bytes,
        generation=generation,
    )


# Maximum (StatVar, entity) pairs per facet probe. Live dev measurements gave
# 362,310 bytes for 25,200 pairs and 633,643 bytes for 50,400 pairs. The
# The marginal payload rate is 147.6 bytes per non-empty pair. Roughly 90% empty cells
# dilute all-pair rates. An 8-StatVar by 15-entity control
# returned real facets, with 14.2% of pairs non-empty. Responses were uncompressed, so
# measured bytes match the downloaded payload.
#
# At 50,000 pairs, the 32 MiB response cap permits 671.1 bytes per pair, while
# a quarter-cap target permits 167.8 bytes, only 1.1 times the measured rate.
# At 25,000 pairs, breaching 32 MiB requires 1,342.2 bytes per pair, 9.1
# times the measured rate. Dense indicators with four or five provenances
# could reach 500-1,000 bytes per pair, so this lower bound leaves headroom.
# The reachable maximum is 700 StatVars by SEARCH_MAX_OPTIONS (500) entities,
# or 350,000 pairs. That takes 14 POSTs in two waves at concurrency 8.
#
# options/surface.py probes subject-only populations. CRS variables use provider/recipient
# roles, so observationAbout probes return empty results. standard_option_coverage uses
# at most 40 StatVars on recall and 200 on resubmit.
# The latter fits 200 StatVars by 200-250 places in one request.
# Measurements cover dev only.
_FACET_PROBE_MAX_PAIRS_PER_POST = 25_000

_FACET_PROBE_SEMAPHORE = threading.BoundedSemaphore(config.settings().facet_probe_total_concurrency)
"""Process-wide limit for concurrent ``/v2/observation`` facet probes.

Each chunk holds a permit only during its POST. Acquisition waits up to
``graph_timeout_s`` or, with a deadline, the smaller of that timeout and the
remaining budget. Deadline expiry raises ``GraphDeadlineExceededError``.
Other wait timeouts raise ``GraphReadError``. Tests can replace this
attribute because ``BoundedSemaphore`` has no resize method.
"""


def observation_facets_via_materialise(
    *,
    stat_vars: Sequence[str],
    entities: Sequence[str],
    materialise_graph: GraphReader,
    deadline: float | None = None,
) -> dict[str, dict[str, list[Facet]]]:
    """Read facet summaries as ``{stat_var: {entity: [Facet]}}``.

    Empty axes return ``{}``. The reader has no pair cap, so StatVars are
    chunked by ``max(1, _FACET_PROBE_MAX_PAIRS_PER_POST // len(entities))``.
    Chunks submit concurrently. Results are consumed in submission order, and
    the first failure in that order wins. Each POST holds the shared semaphore
    for its duration.
    The absolute monotonic ``deadline`` reaches every read and permit wait.

    A ``GraphReadError`` becomes ``GraphInfraError`` with its upstream status.
    Results return only after all chunks succeed. Failure cancels queued chunks
    and waits for running ones. Empty pairs map to ``[]``.
    """
    stat_vars_list = list(stat_vars)
    entities_list = list(entities)
    if not stat_vars_list or not entities_list:
        return {}

    chunk_size = max(1, _FACET_PROBE_MAX_PAIRS_PER_POST // len(entities_list))
    chunks = [stat_vars_list[i : i + chunk_size] for i in range(0, len(stat_vars_list), chunk_size)]

    def _fetch_chunk(chunk: list[str]) -> dict[str, dict[str, list[Facet]]]:
        if deadline is None:
            wait = config.settings().graph_timeout_s
        else:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise GraphDeadlineExceededError(
                    "facet-probe deadline already passed before acquiring a permit"
                )
            wait = min(config.settings().graph_timeout_s, remaining)
        if not _FACET_PROBE_SEMAPHORE.acquire(timeout=wait):
            if deadline is not None and deadline - time.monotonic() <= 0:
                raise GraphDeadlineExceededError(
                    f"facet-probe permit not acquired within {wait}s "
                    "(process-wide semaphore held by other in-flight facet probes) "
                    "and the request deadline has passed"
                )
            raise GraphReadError(
                f"facet-probe permit not acquired within {wait}s "
                "(process-wide semaphore held by other in-flight facet probes)"
            )
        try:
            return materialise_graph.observation_facets_batch_many(
                stat_vars=chunk, entities=entities_list, deadline=deadline
            )
        finally:
            _FACET_PROBE_SEMAPHORE.release()

    try:
        if len(chunks) == 1:
            by_stat_var = _fetch_chunk(chunks[0])
        else:
            by_stat_var = {}
            max_workers = min(len(chunks), config.settings().facet_probe_total_concurrency)
            pool = ThreadPoolExecutor(max_workers=max_workers)
            try:
                futures = [submit_with_context(pool, _fetch_chunk, chunk) for chunk in chunks]
                for future in futures:
                    by_stat_var.update(future.result())
            finally:
                # Cancel queued POSTs after failure. Wait for running POSTs to finish.
                pool.shutdown(wait=True, cancel_futures=True)
    except GraphReadError as exc:
        raise GraphInfraError(str(exc), upstream_status=exc.upstream_status) from exc

    return by_stat_var
