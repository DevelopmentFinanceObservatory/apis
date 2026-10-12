"""Build trailing peers and children events for a Spec's observationAbout place.

The resolver runs once on a dedicated `ThreadPoolExecutor`. `join_inflight=False`
avoids waiting on another caller's population derivation. Scope reads make
independent graph calls (450 to 690 ms each). Two permits run them in parallel.
One runs both in sequence. A full pool skips the stage. Refusals, graph errors,
saturation, and population collisions drop affected events.

Permits belong to submitted jobs and release from completion callbacks, including
after caller cancellation. Cancellation between stages prevents later submissions,
while an already-running job finishes and may fill the population memo. Shutdown
races release the permit and drop the stage. An unbound observationAbout returns
`[]` without deriving a target. The coroutine never raises. The caller wraps it in
`concurrency.create_owned_task` and does not use `asyncio.to_thread`.
"""

from __future__ import annotations

import asyncio
import functools
import logging
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Literal

from dc_sdmx import SdmxClient as MaterialiseSdmxClient
from dfo_platform.executor import submit_with_context
from materialise import GraphReader
from search.engine.clients.graph import EngineGraphClient
from search.engine.clients.sdmx import EngineSdmxClient
from search.engine.config import settings
from search.engine.errors import EngineInfraError, EngineInputError
from search.engine.events import PlaceOptionsEvent
from search.engine.options.population_memo import OptionsPopulationMemo, PopulationInFlight
from search.engine.options.resolve import ResolvedOptionsTarget, resolve_options_target
from search.engine.options.surface import build_options_response
from search.models import BaseIdentity, OptionsRequest
from series_coverage.shared import CoverageSnapshot

logger = logging.getLogger(__name__)

_SCOPES: tuple[Literal["peers"], Literal["children"]] = ("peers", "children")


@dataclass(slots=True, frozen=True)
class PlaceOptionsPool:
    """Dedicated executor and non-blocking admission semaphore for place options.

    Both are sized by `SEARCH_PLACE_OPTIONS_WORKERS`. The ASGI lifespan builds
    and shuts down the pool. Callers pass it as one object. Direct callers
    without a pool skip place-options scheduling.
    """

    executor: ThreadPoolExecutor
    admission: threading.BoundedSemaphore

    @classmethod
    def build(cls, *, max_workers: int) -> PlaceOptionsPool:
        """Build an executor and semaphore, each with `max_workers` slots."""
        return cls(
            executor=ThreadPoolExecutor(
                max_workers=max_workers, thread_name_prefix="search-place-options"
            ),
            admission=threading.BoundedSemaphore(max_workers),
        )

    def shutdown(self) -> None:
        """Cancel queued jobs and stop submissions without waiting for running jobs."""
        self.executor.shutdown(wait=False, cancel_futures=True)


async def _submit_admitted[T](pool: PlaceOptionsPool, fn: Callable[[], T | None]) -> T | None:
    """Submit a job whose admission permit is already held.

    The job's completion callback releases the permit, including on failure or
    cancellation before start. A shutdown race releases the permit and returns
    `None`.
    """
    try:
        cf = submit_with_context(pool.executor, fn)
    except RuntimeError:
        pool.admission.release()
        return None
    cf.add_done_callback(lambda _cf: pool.admission.release())
    return await asyncio.wrap_future(cf)


async def _run_admitted[T](pool: PlaceOptionsPool, fn: Callable[[], T | None]) -> T | None:
    """Submit `fn` with a non-blocking permit, or return `None` when full."""
    if not pool.admission.acquire(blocking=False):
        logger.info("place_options: stage skipped, no admission permit free")
        return None
    return await _submit_admitted(pool, fn)


async def _run_scope_stage(
    pool: PlaceOptionsPool,
    event_for_scope: Callable[[Literal["peers", "children"]], PlaceOptionsEvent | None],
) -> list[PlaceOptionsEvent]:
    """Run both scopes, using two permits in parallel or one in sequence.

    Acquire both permits before submitting anything: two independent acquires would race,
    and at `SEARCH_PLACE_OPTIONS_WORKERS=1` drop the second scope. One permit serves both scopes.
    Skip the stage when all permits are held.
    """
    if not pool.admission.acquire(blocking=False):
        logger.info("place_options: scope stage skipped, no admission permit free")
        return []
    if pool.admission.acquire(blocking=False):
        results = await asyncio.gather(
            *(
                _submit_admitted(pool, functools.partial(event_for_scope, scope))
                for scope in _SCOPES
            )
        )
        return [event for event in results if event is not None]

    logger.info(
        "place_options: only one admission permit free for the scope stage; "
        "running peers then children sequentially on it"
    )

    def _both_scopes() -> list[PlaceOptionsEvent]:
        events = []
        for scope in _SCOPES:
            event = event_for_scope(scope)
            if event is not None:
                events.append(event)
        return events

    return await _submit_admitted(pool, _both_scopes) or []


async def build_place_options_events(
    spec_id: str,
    identity: BaseIdentity,
    *,
    graph: EngineGraphClient,
    sdmx: EngineSdmxClient,
    materialise_graph: GraphReader,
    pool: PlaceOptionsPool,
    coverage: CoverageSnapshot,
    materialise_sdmx: MaterialiseSdmxClient | None = None,
    population_memo: OptionsPopulationMemo | None = None,
    start_ms: int,
) -> list[PlaceOptionsEvent]:
    """Build trailing peers and children events for `spec_id`.

    Resolve the target once, then run each scope on the dedicated pool. Emit
    events for successful scopes and never raise. An unbound observationAbout returns `[]`.
    Submitted jobs finish after caller cancellation and release permits on completion.

    `identity` is the posted Spec identity. `coverage` is the request snapshot,
    and `start_ms` is the request timestamp the budget runs from.
    `materialise_sdmx` supports unit checks. Callers require `sdmx`, which is
    unused here.
    """
    anchor_dcids = identity.entities.get("observationAbout")
    if not anchor_dcids:
        return []
    anchor_ref = anchor_dcids[0]
    deadline = start_ms / 1000.0 + settings().request_timeout_s

    def _request(scope: Literal["peers", "children"] | None) -> OptionsRequest:
        # `for` is an aliased reserved word, so build the model from wire keys.
        # `model_validate` also preserves the validated identity instance.
        return OptionsRequest.model_validate(
            {
                "identity": identity,
                "for": {"role": "observationAbout", "scope": scope},
                "include_coverage": True,
            }
        )

    def _resolve_target() -> ResolvedOptionsTarget | None:
        try:
            # The target is scope-independent. Avoid joining another caller's derivation.
            return resolve_options_target(
                _request(None),
                graph=graph,
                materialise_graph=materialise_graph,
                population_memo=population_memo,
                deadline=deadline,
                join_inflight=False,
            )
        except PopulationInFlight:
            logger.info("place_options: prefetch dropped, population key already in flight")
            return None
        except (EngineInputError, EngineInfraError, TimeoutError):
            return None

    target = await _run_admitted(pool, _resolve_target)
    if target is None:
        return []

    def _event_for_scope(scope: Literal["peers", "children"]) -> PlaceOptionsEvent | None:
        try:
            response = build_options_response(
                _request(scope),
                target,
                graph=graph,
                deadline=deadline,
                materialise_graph=materialise_graph,
                materialise_sdmx=materialise_sdmx,
                coverage=coverage,
            )
        except (EngineInputError, EngineInfraError):
            return None
        return PlaceOptionsEvent(
            spec_id=spec_id,
            role="observationAbout",
            scope=scope,
            anchor_ref=anchor_ref,
            options=response.options.items,
            truncated=response.options.truncated,
        )

    return await _run_scope_stage(pool, _event_for_scope)
