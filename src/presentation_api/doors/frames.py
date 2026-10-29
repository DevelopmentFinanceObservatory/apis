"""The frames door serves a structural frame view, one dimension's values, or values
filtered by query. It checks `spec_id` and `content_version` formats before calling
`assembler.pipeline.get_frame_view`. Frames are shared, so the door takes no caller or session.

Its executor and non-blocking semaphore are independent of the components door. Saturation
returns retryable 503. Each permit covers the pipeline call and JSON encoding.
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
from assembler.pipeline import get_frame_view
from assembler.ports import FrameRef, FrameStore
from dataset_store import close_shared_pool as close_dataset_pool
from datasets_api.door_executor import build_door_executor
from datasets_api.shutdown import ShutdownBudget
from datasets_api.store_errors import DatasetsStoreUnavailable
from dfo_platform import ProblemSpec
from dfo_platform.doors import DoorContribution
from dfo_platform.store_errors import store_failure_is_permanent
from presentation_api.codes import retryable_for
from presentation_api.config import (
    PRESENTATION_FRAMES_MAX_CONCURRENCY,
    PRESENTATION_FRAMES_TIMEOUT_SECONDS,
    PresentationSettings,
)
from presentation_api.live import build_live_frame_store
from presentation_api.models import FrameViewResponse
from presentation_api.path_params import (
    PathParamInvalid,
    require_content_version,
    require_spec_id,
)
from presentation_api.refusals import binding_invalid_spec, frame_unavailable_spec
from presentation_store import frame_reads_bounded_by
from spec_registry.errors import SpecRegistryStoreError, SpecRegistryUnavailable

logger = logging.getLogger(__name__)


class FramesBackpressure(Exception):
    """Raised when the frames semaphore is full."""


def classify_frames_exception(exc: BaseException) -> ProblemSpec | None:
    if isinstance(exc, PathParamInvalid):
        return ProblemSpec(
            status=400,
            code="request_invalid",
            detail=str(exc),
            retryable=False,
        )
    if isinstance(exc, BindingInvalid):
        return binding_invalid_spec(exc)
    if isinstance(exc, FrameUnavailable):
        return frame_unavailable_spec(exc)
    if isinstance(exc, FramesBackpressure):
        return ProblemSpec(
            status=503,
            code="service_unavailable",
            detail="the frames door is at its concurrent-read capacity right now; retry",
            retryable=retryable_for("service_unavailable"),
        )
    if isinstance(exc, (SpecRegistryUnavailable, SpecRegistryStoreError)):
        # A frame load reads the registry to scope its revocation check, and fails closed.
        logger.error("Spec registry read scoping a frame load's revocation check failed: %s", exc)
        return ProblemSpec(
            status=503,
            code="service_unavailable",
            detail="the spec registry could not scope a frame's revocation check; retry",
            retryable=retryable_for("service_unavailable"),
        )
    if isinstance(exc, DatasetsStoreUnavailable):
        if store_failure_is_permanent(exc.sqlstate):
            logger.error(
                "Datasets store call underlying a frame load failed with a permanent cause: %s",
                exc,
            )
            return ProblemSpec(
                status=500,
                code="internal_error",
                detail="a frame store call failed with a non-retryable cause",
                retryable=retryable_for("internal_error"),
            )
        logger.error("Datasets store call underlying a frame load failed: %s", exc)
        return ProblemSpec(
            status=503,
            code="service_unavailable",
            detail="a frame store call failed; retry",
            retryable=retryable_for("service_unavailable"),
        )
    return None


@dataclass
class FramesDeps:
    frame_store: FrameStore


def build_frames_door(
    *,
    frame_store: FrameStore | None = None,
    pool_size: int | None = None,
    budget: ShutdownBudget | None = None,
) -> DoorContribution:
    """Build a live frame store in this door's lifespan when none is injected."""
    _budget = budget if budget is not None else ShutdownBudget()
    effective_pool_size = (
        pool_size if pool_size is not None else PRESENTATION_FRAMES_MAX_CONCURRENCY
    )
    door_executor = build_door_executor(effective_pool_size, "frames", surface="presentation")
    executor = door_executor.executor

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
            executor.shutdown(wait=False, cancel_futures=True)
            await asyncio.to_thread(door_executor.drain, _budget)
            _close_live_clients_built_here()

    async def _run(job: Callable[[], JSONResponse], /) -> JSONResponse:
        with frame_reads_bounded_by(timeout_seconds=PRESENTATION_FRAMES_TIMEOUT_SECONDS):
            return await asyncio.wait_for(
                door_executor.run(job, refuse=FramesBackpressure),
                timeout=PRESENTATION_FRAMES_TIMEOUT_SECONDS,
            )

    router = APIRouter()

    @router.get(
        "/api/presentation/frames/{spec_id}/{content_version}",
        responses={
            200: {
                "model": FrameViewResponse,
                "description": "Neither `dimension` nor `query` given returns the frame's "
                "structural view. `dimension` alone returns that one dimension's full value "
                "list. `query` alone returns every dimension filtered to values matching the "
                "query. `dimension` and `query` together return that one dimension's values "
                "filtered to the query.",
            }
        },
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

    return DoorContribution(
        router=router,
        classify=classify_frames_exception,
        exception_types=(
            PathParamInvalid,
            BindingInvalid,
            FrameUnavailable,
            FramesBackpressure,
            DatasetsStoreUnavailable,
            SpecRegistryUnavailable,
            SpecRegistryStoreError,
        ),
        exception_exemplars=(
            PathParamInvalid(name="spec_id", value="exemplar"),
            BindingInvalid(
                pointer="/dimension",
                dimension="exemplar",
                legal_values=("recipient",),
                truncated=False,
                detail="exemplar",
            ),
            FrameUnavailable(reason="io", detail="exemplar"),
            FramesBackpressure(),
            DatasetsStoreUnavailable("exemplar"),
            DatasetsStoreUnavailable("exemplar", sqlstate="23503"),
            SpecRegistryUnavailable(),
            SpecRegistryStoreError("exemplar"),
        ),
        lifespan=lifespan,
    )
