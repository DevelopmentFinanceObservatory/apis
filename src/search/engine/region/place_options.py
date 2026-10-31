"""Build trailing peers and children events for a Spec's observationAbout place.

The resolver runs once on a dedicated `ThreadPoolExecutor`. `join_inflight=False`
avoids waiting on another caller's population derivation. Scope reads make
independent graph calls (450 to 690 ms each). Two permits run them in parallel.
One runs both in sequence. A full pool skips the stage. Refusals, graph errors,
saturation, and population collisions drop affected events.

A pool built with the app's `OptionsResponseCache` reads each scope through it, under
the key `POST /api/search/options` uses (`options_response_cache_key`), so either path
fills the entry the other reads. Target resolution takes a permit first, so a full pool
skips even scopes that are stored. A stored scope then takes no further permit. The
scopes not stored run on the pool under the same admission as without the cache, and the
job that computes each one offers it to the cache (`OptionsResponseCache.offer`) under the
generation stamp read before submission. A compute that outlives the stream's budget
therefore still fills the cache for the next request, and the offer never joins or waits
on an options request computing the same key.

Place options are not part of the response cache's bundle. A cache hit runs this builder
again from the replayed response (`replayed_place_options`), and the options cache
serves it when warm.

Permits belong to submitted jobs and release from completion callbacks, including
after caller cancellation. Cancellation between stages prevents later submissions,
while an already-running job finishes, may fill the population memo, and offers its
scope to the options cache. Shutdown races release the permit and drop the stage. An
unbound observationAbout returns `[]` without deriving a target. The coroutine never
raises. The caller wraps it in `concurrency.create_owned_task` and does not use
`asyncio.to_thread`.
"""

from __future__ import annotations

import asyncio
import functools
import logging
import threading
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from typing import Literal

from dc_sdmx import SdmxClient as MaterialiseSdmxClient
from dfo_platform.build import build_commit
from dfo_platform.executor import submit_with_context
from materialise import GraphReader
from origin_catalogue import current_catalogue
from search.engine.clients.graph import EngineGraphClient
from search.engine.clients.sdmx import EngineSdmxClient
from search.engine.concurrency import create_owned_task
from search.engine.config import settings
from search.engine.degraded_reads import DegradedReads, recording_degraded_reads
from search.engine.envelope import now_ms
from search.engine.errors import EngineInfraError, EngineInputError
from search.engine.events import PlaceOptionsEvent
from search.engine.options.options_response_cache import (
    OptionsResponseCache,
    options_response_cache_key,
)
from search.engine.options.population_memo import OptionsPopulationMemo, PopulationInFlight
from search.engine.options.resolve import ResolvedOptionsTarget, resolve_options_target
from search.engine.options.surface import ComputedOptions, compute_options
from search.models import BaseIdentity, DefinitePart, OptionsRequest, ResolveResponse, Spec
from series_coverage.shared import CoverageSnapshot

logger = logging.getLogger(__name__)

type Scope = Literal["peers", "children"]

_SCOPES: tuple[Scope, Scope] = ("peers", "children")


class _Refused:
    """A scope whose request the options surface refused. It streams no event."""


_REFUSED = _Refused()


@dataclass(slots=True, frozen=True)
class PlaceOptionsPool:
    """Dedicated executor, admission semaphore and option-set cache for place options.

    The executor and semaphore are sized by `SEARCH_PLACE_OPTIONS_WORKERS`. The ASGI
    lifespan builds and shuts down the pool and gives it the options route's
    `OptionsResponseCache`. A pool without one computes every scope. Callers pass it as
    one object. Direct callers without a pool skip place-options scheduling.
    """

    executor: ThreadPoolExecutor
    admission: threading.BoundedSemaphore
    options_cache: OptionsResponseCache | None = None

    @classmethod
    def build(
        cls, *, max_workers: int, options_cache: OptionsResponseCache | None = None
    ) -> PlaceOptionsPool:
        """Build an executor and semaphore, each with `max_workers` slots."""
        return cls(
            executor=ThreadPoolExecutor(
                max_workers=max_workers, thread_name_prefix="search-place-options"
            ),
            admission=threading.BoundedSemaphore(max_workers),
            options_cache=options_cache,
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


async def _run_scope_stage[T](
    pool: PlaceOptionsPool,
    result_for_scope: Callable[[Scope], T | None],
    scopes: Sequence[Scope] = _SCOPES,
) -> dict[Scope, T]:
    """Run `scopes`, using a permit each in parallel or one permit for all in sequence.

    Acquire every permit before submitting anything: two independent acquires would race,
    and at `SEARCH_PLACE_OPTIONS_WORKERS=1` drop the second scope. Skip the stage when all
    permits are held. Scopes whose result is `None` are left out.
    """
    if not pool.admission.acquire(blocking=False):
        logger.info("place_options: scope stage skipped, no admission permit free")
        return {}
    if len(scopes) > 1 and pool.admission.acquire(blocking=False):
        results = await asyncio.gather(
            *(
                _submit_admitted(pool, functools.partial(result_for_scope, scope))
                for scope in scopes
            )
        )
        return {
            scope: result
            for scope, result in zip(scopes, results, strict=True)
            if result is not None
        }

    if len(scopes) > 1:
        logger.info(
            "place_options: only one admission permit free for the scope stage; "
            "running peers then children sequentially on it"
        )

    def _in_sequence() -> dict[Scope, T]:
        results = {}
        for scope in scopes:
            result = result_for_scope(scope)
            if result is not None:
                results[scope] = result
        return results

    return await _submit_admitted(pool, _in_sequence) or {}


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
    # The request's bound origin catalogue, which every compute below reads, keys the entries.
    catalogue = current_catalogue()

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

    # The target's reads mark every scope's options and leave the region unmarked (`_computed_for_scope`).
    target_lost = DegradedReads()

    def _resolve_target() -> ResolvedOptionsTarget | _Refused | None:
        try:
            with recording_degraded_reads(detached=True) as lost:
                # The target is scope-independent. Avoid joining another caller's derivation.
                resolved = resolve_options_target(
                    _request(None),
                    graph=graph,
                    materialise_graph=materialise_graph,
                    population_memo=population_memo,
                    deadline=deadline,
                    join_inflight=False,
                )
            lost.merge_into(target_lost)
            return resolved
        except PopulationInFlight:
            logger.info("place_options: prefetch dropped, population key already in flight")
            return None
        except EngineInputError:
            return _REFUSED
        except (EngineInfraError, TimeoutError):
            return None

    target = await _run_admitted(pool, _resolve_target)
    if target is None or isinstance(target, _Refused):
        return []

    def _computed_for_scope(scope: Scope) -> ComputedOptions | _Refused | None:
        try:
            # Detached: the task copies the context its region's resolution bound, and place
            # options are not in the response bundle, so their reads must not mark the part.
            with recording_degraded_reads(detached=True) as lost:
                computed = compute_options(
                    _request(scope),
                    target,
                    graph=graph,
                    deadline=deadline,
                    materialise_graph=materialise_graph,
                    materialise_sdmx=materialise_sdmx,
                    coverage=coverage,
                )
        except EngineInputError:
            return _REFUSED
        except EngineInfraError:
            return None
        return replace(computed, degraded=lost.degraded or target_lost.degraded)

    cache = pool.options_cache
    if cache is None:
        computed = await _run_scope_stage(pool, _computed_for_scope)
    else:
        computed = await _through_cache(
            pool,
            cache,
            _computed_for_scope,
            key_for_scope=lambda scope: options_response_cache_key(
                _request(scope),
                target,
                build_commit=build_commit(),
                coverage=coverage,
                catalogue=catalogue,
            ),
        )
    return [
        PlaceOptionsEvent(
            spec_id=spec_id,
            role="observationAbout",
            scope=scope,
            anchor_ref=anchor_ref,
            options=result.options.items,
            truncated=result.options.truncated,
        )
        for scope in _SCOPES
        if isinstance(result := computed.get(scope), ComputedOptions)
    ]


async def _through_cache(
    pool: PlaceOptionsPool,
    cache: OptionsResponseCache,
    computed_for_scope: Callable[[Scope], ComputedOptions | _Refused | None],
    *,
    key_for_scope: Callable[[Scope], str],
) -> dict[Scope, ComputedOptions | _Refused]:
    """Return each scope's result. Stored ones come from `cache` with no permit, the rest
    from the scope stage as the uncached path runs it.

    Each job offers its own result to `cache` under the generation stamp read before
    submission, so a result computed while the generation moves is not stored under the
    newer stamp, and a job still running when the caller is cancelled stores what it
    computed. The offer neither joins nor waits on an options request computing the same
    key, and a result the cache refuses (`is_cacheable`) is still returned.
    """
    keys = {scope: key_for_scope(scope) for scope in _SCOPES}
    found = {scope: cache.cached(keys[scope]) for scope in _SCOPES}
    results: dict[Scope, ComputedOptions | _Refused] = {
        scope: options for scope, options in found.items() if options is not None
    }
    missing = [scope for scope in _SCOPES if scope not in results]
    if not missing:
        return results
    stamp_key = cache.stamp_key()

    def computed_and_offered(scope: Scope) -> ComputedOptions | _Refused | None:
        result = computed_for_scope(scope)
        if isinstance(result, ComputedOptions):
            cache.offer(keys[scope], result, stamp_key=stamp_key)
        return result

    results.update(await _run_scope_stage(pool, computed_and_offered, missing))
    return results


def place_options_task(
    spec: Spec,
    *,
    graph: EngineGraphClient,
    sdmx: EngineSdmxClient,
    materialise_graph: GraphReader,
    materialise_sdmx: MaterialiseSdmxClient | None,
    pool: PlaceOptionsPool,
    coverage: CoverageSnapshot,
    population_memo: OptionsPopulationMemo | None,
) -> asyncio.Task[list[PlaceOptionsEvent]]:
    """The place-options task for `spec`, on the event loop: its events carry the spec id
    and read the identity `spec` states (`Spec.definition.identity`). A fresh run and a
    response cache hit both launch through here, so a Spec built by a same-shape merge reads
    the merged identity on either path."""
    return create_owned_task(
        build_place_options_events(
            spec.spec_id,
            spec.definition.identity,
            graph=graph,
            sdmx=sdmx,
            materialise_graph=materialise_graph,
            materialise_sdmx=materialise_sdmx,
            pool=pool,
            coverage=coverage,
            population_memo=population_memo,
            start_ms=now_ms(),
        )
    )


def replayed_place_options(
    *,
    stream: bool,
    graph: EngineGraphClient,
    sdmx: EngineSdmxClient,
    materialise_graph: GraphReader | None,
    materialise_sdmx: MaterialiseSdmxClient | None,
    pool: PlaceOptionsPool | None,
    coverage: CoverageSnapshot,
    population_memo: OptionsPopulationMemo | None,
) -> Callable[[ResolveResponse], list[asyncio.Task[list[PlaceOptionsEvent]]]] | None:
    """What launches, for a response cache hit to stream after its replayed events, the
    place-options tasks a fresh run of the hit's response launches. `None` when a fresh run
    launches none: `stream` (trailing option coverage) is off, or the request has no
    materialise graph or pool.

    A fresh run launches one task per definite Spec (`place_options_task`), which the
    response's part carries. Parts folded into one Spec share its task.
    The tasks read `coverage`, the hit's own snapshot, which the bundle's key holds."""
    if not stream or materialise_graph is None or pool is None:
        return None
    # Bound once narrowed, so the closure reads them as present.
    reader: GraphReader = materialise_graph
    admitted: PlaceOptionsPool = pool

    def launch(response: ResolveResponse) -> list[asyncio.Task[list[PlaceOptionsEvent]]]:
        specs = {
            part.spec.spec_id: part.spec
            for part in (response.parts if response.kind == "resolution" else [])
            if isinstance(part, DefinitePart)
        }
        return [
            place_options_task(
                spec,
                graph=graph,
                sdmx=sdmx,
                materialise_graph=reader,
                materialise_sdmx=materialise_sdmx,
                pool=admitted,
                coverage=coverage,
                population_memo=population_memo,
            )
            for spec in specs.values()
        ]

    return launch
