"""The frames door serves a structural frame view, one dimension's values, or values
filtered by query. It checks `spec_id` and `content_version` formats before calling
`assembler.pipeline.get_frame_view`. Frames are shared, so the door takes no caller or session.

`GET /api/presentation/frames/{spec_id}` names a dataset id alone and reads the version the
analysis runner reads for a Spec root of that id
(`presentation_api.newest_frames.resolve_bare_spec`), answering the pair route's body, read from
that version's canonical frame, with the `spec_id` and `content_version` it resolved. Only a `spec_` id of 32 hex digits resolves. A
derived `ops_` id names its version.

Its executor and non-blocking semaphore are independent of the components door. Saturation
returns retryable 503. Each permit covers the pipeline call and JSON encoding. The bare route
can rebuild a dataset, so it runs on a smaller executor and semaphore of its own, shared with a
component create's bare resolve, and bare reads of stale datasets refuse among themselves while
pair reads keep their permits.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import cast

from fastapi import APIRouter, FastAPI
from fastapi.responses import JSONResponse

from assembler.errors import BindingInvalid, FrameUnavailable
from assembler.pipeline import BARE_SPEC_ID, get_frame_view
from assembler.ports import FrameRef, FrameStore
from dataset_store import close_shared_pool as close_dataset_pool
from datasets_api.door_executor import DoorExecutor, build_door_executor
from datasets_api.shutdown import ShutdownBudget
from datasets_api.store_errors import DatasetsStoreUnavailable
from dfo_platform import ProblemSpec, refused
from dfo_platform.doors import DoorContribution
from dfo_platform.store_errors import store_failure_is_permanent
from presentation_api.codes import retryable_for
from presentation_api.config import (
    PRESENTATION_FRAMES_MAX_CONCURRENCY,
    PRESENTATION_FRAMES_TIMEOUT_SECONDS,
    PresentationSettings,
)
from presentation_api.live import build_live_frame_store
from presentation_api.models import FrameViewResponse, NewestFrameViewResponse
from presentation_api.newest_frames import (
    NewestFrames,
    NewestNotReady,
    newest_not_ready_spec,
    resolve_bare_spec,
)
from presentation_api.path_params import (
    PathParamInvalid,
    require_content_version,
    require_spec_id,
)
from presentation_api.refusals import (
    FRAME_UNAVAILABLE_EXEMPLARS,
    binding_invalid_spec,
    frame_unavailable_spec,
)
from presentation_api.routes import (
    BARE_FRAME_PATH,
    FRAME_PATH,
    FRAMES_ENDPOINT,
    door_registry_endpoints,
)
from presentation_store import frame_reads_bounded_by
from spec_registry.errors import SpecRegistryStoreError, SpecRegistryUnavailable

logger = logging.getLogger(__name__)


class FramesBackpressure(Exception):
    """Raised when the frames semaphore is full."""


def classify_frames_exception(exc: BaseException) -> ProblemSpec | None:
    if isinstance(exc, PathParamInvalid):
        return refused(
            "request_invalid",
            detail=str(exc),
            retryable=False,
        )
    if isinstance(exc, BindingInvalid):
        return binding_invalid_spec(exc)
    if isinstance(exc, FrameUnavailable):
        return frame_unavailable_spec(exc)
    if isinstance(exc, NewestNotReady):
        return newest_not_ready_spec(exc)
    if isinstance(exc, FramesBackpressure):
        return refused(
            "service_unavailable",
            detail="the frames door is at its concurrent-read capacity right now; retry",
            retryable=retryable_for("service_unavailable"),
        )
    if isinstance(exc, (SpecRegistryUnavailable, SpecRegistryStoreError)):
        # A frame load reads the registry to scope its revocation check, and fails closed.
        logger.error("Spec registry read scoping a frame load's revocation check failed: %s", exc)
        return refused(
            "service_unavailable",
            detail="the spec registry could not scope a frame's revocation check; retry",
            retryable=retryable_for("service_unavailable"),
        )
    if isinstance(exc, DatasetsStoreUnavailable):
        if store_failure_is_permanent(exc.sqlstate):
            logger.error(
                "Datasets store call underlying a frame load failed with a permanent cause: %s",
                exc,
            )
            return refused(
                "internal_error",
                detail="a frame store call failed with a non-retryable cause",
                retryable=retryable_for("internal_error"),
            )
        logger.error("Datasets store call underlying a frame load failed: %s", exc)
        return refused(
            "service_unavailable",
            detail="a frame store call failed; retry",
            retryable=retryable_for("service_unavailable"),
        )
    return None


@dataclass
class FramesDeps:
    frame_store: FrameStore


_VIEW_DESCRIPTION = (
    "Neither `dimension` nor `query` given returns the frame's structural view. `dimension` "
    "alone returns that one dimension's full value list. `query` alone returns every dimension "
    "filtered to values matching the query. `dimension` and `query` together return that one "
    "dimension's values filtered to the query."
)


def build_frames_door(
    *,
    frame_store: FrameStore | None = None,
    newest: NewestFrames | None = None,
    bare_reads: DoorExecutor,
    pool_size: int | None = None,
    budget: ShutdownBudget | None = None,
) -> DoorContribution:
    """Build a live frame store in this door's lifespan when none is injected. `newest`
    resolves a bare `spec_id`; with none, the bare route refuses `frame_unavailable`.
    `bare_reads` runs the bare route and the components door's bare resolve, and this door's
    lifespan shuts it down."""
    _budget = budget if budget is not None else ShutdownBudget()
    effective_pool_size = (
        pool_size if pool_size is not None else PRESENTATION_FRAMES_MAX_CONCURRENCY
    )
    door_executor = build_door_executor(effective_pool_size, "frames", surface="presentation")

    deps = FramesDeps(frame_store=cast(FrameStore, frame_store))

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        opened_dataset_pool = False
        owned_graph = None

        def _close_live_clients_built_here() -> None:
            if owned_graph is not None:
                owned_graph.close()
            if opened_dataset_pool:
                close_dataset_pool()

        try:
            if frame_store is None:
                settings = PresentationSettings.from_env()
                # No frames route pins.
                live = build_live_frame_store(settings=settings, allowances=None)
                opened_dataset_pool = True
                owned_graph = live.graph
                deps.frame_store = live.frame_store
        except BaseException:
            _close_live_clients_built_here()
            raise

        try:
            yield
        finally:
            door_executor.executor.shutdown(wait=False, cancel_futures=True)
            bare_reads.executor.shutdown(wait=False, cancel_futures=True)
            await asyncio.to_thread(door_executor.drain, _budget)
            await asyncio.to_thread(bare_reads.drain, _budget)
            _close_live_clients_built_here()

    async def _run(
        job: Callable[[], JSONResponse], /, *, on: DoorExecutor = door_executor
    ) -> JSONResponse:
        with frame_reads_bounded_by(timeout_seconds=PRESENTATION_FRAMES_TIMEOUT_SECONDS):
            return await asyncio.wait_for(
                on.run(job, refuse=FramesBackpressure),
                timeout=PRESENTATION_FRAMES_TIMEOUT_SECONDS,
            )

    router = APIRouter()

    @router.get(
        FRAME_PATH,
        responses={200: {"model": FrameViewResponse, "description": _VIEW_DESCRIPTION}},
    )
    async def get_frame(
        spec_id: str,
        content_version: str,
        dimension: str | None = None,
        query: str | None = None,
    ) -> JSONResponse:
        require_spec_id(spec_id)
        require_content_version(content_version)

        def job() -> JSONResponse:
            document = get_frame_view(
                frame_ref=FrameRef(spec_id=spec_id, content_version=content_version),
                dimension=dimension,
                query=query,
                frame_store=deps.frame_store,
            )
            return JSONResponse(document, status_code=200)

        return await _run(job)

    @router.get(
        BARE_FRAME_PATH,
        responses={
            200: {
                "model": NewestFrameViewResponse,
                "description": f"{_VIEW_DESCRIPTION} The frame is the newest version of the "
                "dataset `spec_id` names, the version the analysis runner reads for a Spec "
                "root; `spec_id` and `content_version` name it. The view reads that version's "
                "canonical frame, as the pair route does, so it holds no default enrichment "
                "column, and a `dimension` naming one refuses `binding_invalid`.",
            }
        },
    )
    async def get_newest_frame(
        spec_id: str,
        dimension: str | None = None,
        query: str | None = None,
    ) -> JSONResponse:
        if not BARE_SPEC_ID.fullmatch(spec_id):
            raise PathParamInvalid(name="spec_id", value=spec_id)

        def job() -> JSONResponse:
            frame_ref = resolve_bare_spec(spec_id, newest=newest)
            document = get_frame_view(
                frame_ref=frame_ref,
                dimension=dimension,
                query=query,
                frame_store=deps.frame_store,
            )
            return JSONResponse(
                {
                    **document,
                    "spec_id": frame_ref.spec_id,
                    "content_version": frame_ref.content_version,
                },
                status_code=200,
            )

        return await _run(job, on=bare_reads)

    return DoorContribution(
        router=router,
        registry_endpoints=door_registry_endpoints(router),
        classify=classify_frames_exception,
        exception_types=(
            PathParamInvalid,
            BindingInvalid,
            FrameUnavailable,
            NewestNotReady,
            FramesBackpressure,
            DatasetsStoreUnavailable,
            SpecRegistryUnavailable,
            SpecRegistryStoreError,
        ),
        endpoint_exemplars={
            FRAMES_ENDPOINT: (
                PathParamInvalid(name="spec_id", value="exemplar"),
                BindingInvalid(
                    pointer="/dimension",
                    dimension="exemplar",
                    legal_values=("recipient",),
                    truncated=False,
                    detail="exemplar",
                ),
                *FRAME_UNAVAILABLE_EXEMPLARS,
                NewestNotReady("exemplar", retry_after_s=1.0),
                FramesBackpressure(),
                DatasetsStoreUnavailable("exemplar"),
                DatasetsStoreUnavailable("exemplar", sqlstate="23503"),
                SpecRegistryUnavailable(),
                SpecRegistryStoreError("exemplar"),
            )
        },
        lifespan=lifespan,
    )
