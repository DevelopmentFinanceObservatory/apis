"""The mint door: `POST /api/datasets`.

Mints run on the surface's work executor (`datasets_api.deps`), whose non-blocking semaphore
is sized by `DATASETS_WORK_MAX_CONCURRENCY`. A full executor returns retryable 503
`MintBackpressure`. A mint's deadline is `DATASETS_MINT_TIMEOUT_SECONDS` from its arrival, less
an unwind margin, and is bound as the request deadline (`dfo_platform.request_deadline`), so every
SDMX, graph and metadata read of the work takes it. `asyncio.wait_for` stops waiting at the
timeout itself and maps it to retryable 504. Cancelling a queued future releases its permit. A
running thread continues after request cancellation and holds its permit until it returns, and
each of its reads stops at the deadline.

The door reads its clients and stores from the surface's `Deps` and builds none. Exception
mapping lives in `datasets_api.doors.mint_errors` and its inventory in
`datasets_api.doors.mint_inventory`.
"""

from __future__ import annotations

import asyncio
import functools
import logging
import time

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from datasets_api.deps import MATERIALISE_RAW_FETCH_GUARD_ROWS, Deps, DepsCell
from datasets_api.doors.definition import OPERATION_SUBMITTED_RESPONSES, ROUTE_EXCEPTION_TYPES
from datasets_api.doors.mint_errors import MintBackpressure, classify_mint_exception
from datasets_api.doors.mint_inventory import MINT_EXCEPTION_EXEMPLARS, MINT_EXCEPTION_TYPES
from datasets_api.headers import no_store_headers
from datasets_api.kinds.spec_id import mint_spec_id
from datasets_api.kinds.statvar_set import mint_statvar_set
from datasets_api.manifest_models import Manifest
from datasets_api.mint_deps import MintDeps
from datasets_api.mint_models import MintRequest, StatVarSetInput
from datasets_api.operations.submit import (
    OperationSubmitted,
    operation_submitted_resource,
)
from datasets_api.render import check_options_supported
from datasets_api.schema import OperationResource
from datasets_api.timings import door_call, run_with_timings
from datasets_api.version import DATASETS_CAPABILITY_VERSION
from dfo_platform.doors import DoorContribution, route_registry_endpoints
from dfo_platform.request_deadline import request_deadline
from dfo_platform.trace import get_trace_id
from statvar_members import shared_enumerator

logger = logging.getLogger(__name__)


def _sdmx_deadline(timeout: float) -> float:
    """The work deadline of a mint answered within `timeout` from now: up to one second, or 10%
    of the timeout, is left for `wait_for` to unwind."""
    grace = min(1.0, timeout * 0.1)
    return time.monotonic() + max(0.0, timeout - grace)


def mint_deps(deps: Deps, *, deadline: float | None) -> MintDeps:
    """The mint's view of the surface's `Deps`, with this request's SDMX deadline."""
    return MintDeps(
        registry=deps.registry,
        spec_reader=deps.spec_reader,
        sdmx=deps.sdmx,
        graph=deps.graph,
        artifact_store=deps.artifact_store,
        materialiser_build=deps.materialiser_build,
        capability_version=DATASETS_CAPABILITY_VERSION,
        ceiling=MATERIALISE_RAW_FETCH_GUARD_ROWS,
        member_concurrency=deps.settings.mint_member_max_concurrency,
        enumerator=shared_enumerator(),
        enrichment=deps.enrichment,
        code_index=deps.code_index,
        rate_cache=deps.rate_cache,
        coverage=deps.coverage,
        version_writer=deps.version_writer,
        snapshot_reader=deps.snapshot_reader,
        head_store=deps.head_store,
        materialise_semaphore=deps.materialise_semaphore,
        deadline=deadline,
        operation_store=deps.operation_store,
        task_client=deps.task_client,
        settings=deps.settings,
        now=deps.now,
        rebuilds=deps.rebuilds,
        variable_breadth=deps.variable_breadth,
    )


async def _mint_response(body: MintRequest, *, deps: Deps) -> JSONResponse:
    """Run a mint on the work executor and return its manifest, or 202 when work is deferred.

    Reject unsupported render options before resolution, writes, or SDMX reads.
    """
    trace_id = get_trace_id()
    settings = deps.settings
    deadline = _sdmx_deadline(settings.mint_timeout_seconds)
    request_deps = mint_deps(deps, deadline=deadline)
    check_options_supported(body.options, code_index=request_deps.code_index)
    input_ = body.input
    if isinstance(input_, StatVarSetInput):
        mint = functools.partial(
            mint_statvar_set,
            statvars=input_.statvars,
            scope=input_.scope,
            options=body.options,
            deps=request_deps,
            trace_id=trace_id,
        )
    else:
        mint = functools.partial(
            mint_spec_id,
            spec_id=input_.spec_id,
            definition=input_.definition,
            scope=input_.scope,
            options=body.options,
            deps=request_deps,
            trace_id=trace_id,
        )
    submitted_resource: OperationResource | None = None
    with door_call("mint", logger=logger) as call, request_deadline(deadline):
        try:
            manifest = await asyncio.wait_for(
                deps.work_executor.run(
                    functools.partial(run_with_timings, mint, call.timings),
                    refuse=MintBackpressure,
                    timings=call.timings,
                ),
                timeout=settings.mint_timeout_seconds,
            )
        except OperationSubmitted as submitted:
            # A deferred 202 is a successful outcome, not a refusal.
            submitted_resource = operation_submitted_resource(submitted)
    if submitted_resource is not None:
        assert submitted_resource.operation.poll is not None
        return JSONResponse(
            submitted_resource.model_dump(mode="json", exclude_none=True),
            status_code=202,
            headers={
                **no_store_headers(server_timing=call.server_timing),
                "Location": submitted_resource.operation.poll,
            },
        )
    return JSONResponse(
        manifest.model_dump(mode="json", exclude_none=True),
        headers=no_store_headers(server_timing=call.server_timing),
    )


def build_mint_door(*, deps: DepsCell) -> DoorContribution:
    """Build the mint door over the surface's dependency cell."""
    router = APIRouter()

    @router.post("/api/datasets", response_model=Manifest, responses=OPERATION_SUBMITTED_RESPONSES)
    async def mint(body: MintRequest) -> JSONResponse:
        return await _mint_response(body, deps=deps.get())

    return DoorContribution(
        router=router,
        registry_endpoints=route_registry_endpoints(router, "datasets.mint"),
        classify=classify_mint_exception,
        # The worker runs the operations a mint queues on the definition route's serving path and
        # classifies their failures under this endpoint (`datasets_api.operations.worker`).
        exception_types=tuple(
            dict.fromkeys((*MINT_EXCEPTION_TYPES, *ROUTE_EXCEPTION_TYPES["datasets.definition"]))
        ),
        exception_exemplars=MINT_EXCEPTION_EXEMPLARS,
    )
