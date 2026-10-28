"""The analysis door: `POST /api/datasets/analysis` and `POST /api/datasets/analysis/anchor`.

`POST /api/datasets/analysis` takes a recipe tree (`analysis.AnalysisRequest`) and answers:

- 200 `AnalysisReady`: the format 2 ref of the stored frame, its frame `cv_` and output digest,
  the top step's `summary` (`dfo_analysis_engine.calls.VerbCall.summary`) and the shape facts
  it holds, the execution identity, and each period the recipe states relative to the latest;
- 202 `AnalysisPendingBody` with `Retry-After`, when a root's dataset is still building;
- a problem document otherwise. An engine refusal, `several_members`, `over_ceiling` and a root
  the runner cannot read (`snapshot_not_found`, `artifact_expired`, `recipe_not_found`,
  `rendering_not_recorded`) carry `retry_mode` and `fix` beside the code, as the engine's refusal body
  (`dfo_analysis_engine.refusal.Refusal.as_json`) does.

`POST /api/datasets/analysis/anchor` takes roots and the populations a rolling step counts over
(`analysis.AnchorRequest`) and answers the latest period, its rule and the series it leaves out,
storing nothing.

Both routes admit every caller and spend no allowance. The caller is resolved
(`dfo_platform.caller.resolve_caller`) for the call's log line alone, and a caller this
service cannot name logs as unresolved. A website request whose caller header is malformed
still refuses, as on every route that resolves callers. Until the code registry maps the reused Datasets codes on `datasets.analysis`, such a refusal goes
out at the one status the registry gives it on every other endpoint. A run takes a thread of the analysis
executor (`DATASETS_ANALYSIS_MAX_CONCURRENCY`), refusing retryable `service_unavailable`
(503) when every thread is busy, and one materialise permit for its frames; it answers timeout
(504) after `DATASETS_ANALYSIS_TIMEOUT_SECONDS`.
"""

from __future__ import annotations

import functools
import logging
import time
from typing import Any, Literal

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, JsonValue

from datasets_api.admission import bounded_materialise_slot
from datasets_api.analysis import (
    AnalysisBounds,
    AnalysisPending,
    AnalysisRequest,
    AnalysisStores,
    AnalysisTooLarge,
    AnchorRequest,
    FrameUnavailable,
    RootExpired,
    RootNotStored,
    RootRecipeMissing,
    RootRenderingUnreadable,
    SeveralMembers,
    run_analysis,
    run_anchor,
)
from datasets_api.analysis_roots import LiveRootSource
from datasets_api.backpressure import Backpressure
from datasets_api.codes import CODE_CATALOG
from datasets_api.codes_analysis import ANALYSIS_STATUSES
from datasets_api.current import DefinitionDeps
from datasets_api.deps import Deps, DepsCell
from datasets_api.doors.definition import SERVING_EXCEPTION_TYPES, _run_retrieval, definition_deps
from datasets_api.doors.refresh import RefreshRebuildFailed, RenderingNotRecorded
from datasets_api.headers import NO_STORE_HEADERS
from datasets_api.refusals import classify_datasets_refusal
from datasets_api.timings import door_call
from datasets_api.version import DATASETS_CAPABILITY_VERSION
from dfo_analysis_engine.refusal import AnalysisRefused, RetryMode
from dfo_analysis_engine.refusal import Refusal as EngineRefusal
from dfo_platform import ProblemSpec
from dfo_platform.caller import CallerHeaderInvalid, CallerUnresolved, resolve_caller
from dfo_platform.cloud_logging import JSON_FIELDS_ATTR
from dfo_platform.codes import bound_registry_endpoint
from dfo_platform.doors import DoorContribution, route_registry_endpoints
from dfo_platform.problem import Envelope, Refusal
from dfo_platform.trace import get_trace_id

logger = logging.getLogger(__name__)

_ANALYSIS_PATH = "/api/datasets/analysis"
_ANCHOR_PATH = "/api/datasets/analysis/anchor"
_ENVELOPE = Envelope(catalogue=CODE_CATALOG, capability_version=DATASETS_CAPABILITY_VERSION)
_CORRECT = RetryMode.CORRECT_AND_RETRY.value


class AnalysisBackpressure(Backpressure):
    """Every analysis thread is busy. Maps to retryable `service_unavailable` (503)."""

    code = "service_unavailable"

    def __init__(self) -> None:
        super().__init__("the analysis door is at its concurrent capacity right now; retry")


class AnalysisReady(BaseModel):
    """The stored frame of a computed tree."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["ready"] = "ready"
    ref: dict[str, JsonValue] = Field(description="the format 2 DatasetRef naming the frame")
    frame_content_version: str = Field(pattern=r"^cv_[0-9a-f]{32}$")
    output_digest: str = Field(pattern=r"^out_[0-9a-f]{32}$")
    summary: dict[str, JsonValue] = Field(
        description="the facts of the tree's top step: the verb, its canonical parameters, the "
        "result's shape and exclusions, and the engine result's own fields"
    )
    rows: int = Field(ge=0)
    role_keys: list[str]
    stat_var_count: int = Field(ge=0)
    exclusions: list[dict[str, JsonValue]]
    time: dict[str, JsonValue]
    execution_identity: dict[str, str]
    resolved_periods: list[dict[str, JsonValue]] = Field(
        description="each period the recipe states relative to the latest, with its period"
    )


class AnalysisPendingBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal["pending"] = "pending"
    retry_after_s: float
    operations: list[str] = Field(description="the queued rebuilds the run waits on")


class LeftOutSeriesBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    root: int = Field(ge=0, description="the population the series belongs to, by position")
    coordinates: dict[str, str]


class AnchorAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid")

    period: str
    rule: str
    left_out: list[LeftOutSeriesBody]
    roots: list[dict[str, str]] = Field(
        description="the dataset versions the period was taken over"
    )


def _problem(*, status: int, code: str, detail: str, retry_mode: str, fix: object) -> JSONResponse:
    problem = _ENVELOPE.details(status=status, code=code, detail=detail, retryable=False)
    return JSONResponse(
        status_code=status,
        content={**problem.to_json(), "retry_mode": retry_mode, "fix": fix},
        media_type="application/problem+json",
        headers={"Cache-Control": "no-store"},
    )


def _refused(exc: Exception) -> JSONResponse:
    """The problem document of an analysis refusal, with its retry mode and fix."""
    if isinstance(exc, AnalysisRefused):
        refusal = exc.refusal
        return _problem(
            status=ANALYSIS_STATUSES[refusal.code.value],
            code=refusal.code.value,
            detail=refusal.message,
            retry_mode=refusal.retry_mode.value,
            fix=refusal.fix,
        )
    assert isinstance(exc, SeveralMembers | FrameUnavailable | AnalysisTooLarge)
    spec = classify_analysis_exception(exc)
    assert spec is not None
    fix: object = (
        {"ceiling_kind": exc.ceiling_kind, "count": exc.count, "cap": exc.cap}
        if isinstance(exc, AnalysisTooLarge)
        else exc.fix
    )
    return _problem(
        status=spec.status, code=spec.code, detail=str(exc), retry_mode=_CORRECT, fix=fix
    )


def _pending(pending: AnalysisPending) -> JSONResponse:
    body = AnalysisPendingBody(
        retry_after_s=pending.retry_after_s, operations=list(pending.operations)
    )
    return JSONResponse(
        body.model_dump(mode="json"),
        status_code=202,
        headers={**NO_STORE_HEADERS, "Retry-After": str(max(1, round(pending.retry_after_s)))},
    )


def _log_caller(request: Request, deps: Deps, *, route: str) -> None:
    """Log the caller of `request`. A caller the service cannot name logs as unresolved."""
    try:
        key: str | None = resolve_caller(request, deps.settings.caller).key
    except CallerUnresolved:
        key = None
    logger.info("analysis call", extra={JSON_FIELDS_ATTR: {"route": route, "caller": key}})


def classify_analysis_exception(exc: BaseException) -> ProblemSpec | None:
    """An analysis refusal raised outside the routes' own handling, at its status, else the
    Datasets refusals."""
    if isinstance(exc, AnalysisRefused):
        return ProblemSpec(
            status=ANALYSIS_STATUSES[exc.refusal.code.value],
            code=exc.refusal.code.value,
            retryable=False,
            detail=exc.refusal.message,
        )
    try:
        return classify_datasets_refusal(exc)
    except LookupError:
        # The registry maps two codes on `datasets.analysis` so far. A Datasets refusal a run
        # raises (a spec not found, a rebuild's refusal, a full pool) goes out at the one
        # status the registry gives it everywhere else, and the gap is logged.
        logger.warning(
            "code %s has no status on datasets.analysis; answering its status elsewhere",
            getattr(exc, "code", type(exc).__name__),
        )
        with bound_registry_endpoint(None):
            return classify_datasets_refusal(exc)


def _stores(current: Deps) -> AnalysisStores:
    return AnalysisStores(
        snapshot_reader=current.snapshot_reader,
        version_writer=current.version_writer,
        artifact_store=current.artifact_store,
        recipe_store=current.recipe_store,
        now=current.now,
    )


def _bounds(current: Deps) -> AnalysisBounds:
    settings = current.settings
    return AnalysisBounds(
        max_nodes=settings.analysis_max_nodes,
        max_root_rows=settings.analysis_max_root_rows,
        max_join_rows=settings.analysis_max_join_rows,
    )


def _permit(current: Deps, deps: DefinitionDeps) -> functools.partial[Any]:
    return functools.partial(
        bounded_materialise_slot,
        current.materialise_semaphore,
        wait_seconds=current.settings.materialise_wait_seconds,
        deadline=deps.deadline,
    )


def _analysis_job(
    *, deps: DefinitionDeps, current: Deps, body: AnalysisRequest, trace_id: str | None
) -> dict[str, Any]:
    result = run_analysis(
        body,
        source=LiveRootSource(deps=deps, recipes=current.recipe_store),
        stores=_stores(current),
        bounds=_bounds(current),
        permit=_permit(current, deps),
        trace_id=trace_id,
    )
    return AnalysisReady.model_validate(result.body()).model_dump(mode="json")


def _anchor_job(
    *, deps: DefinitionDeps, current: Deps, body: AnchorRequest, trace_id: str | None
) -> dict[str, Any]:
    anchor, roots = run_anchor(
        body,
        source=LiveRootSource(deps=deps, recipes=current.recipe_store),
        permit=_permit(current, deps),
        trace_id=trace_id,
    )
    return AnchorAnswer(
        period=anchor.period,
        rule=anchor.rule,
        left_out=[
            LeftOutSeriesBody(root=series.root, coordinates=dict(series.coordinates))
            for series in anchor.left_out
        ],
        roots=[root.dataset.model_dump() for root in roots],
    ).model_dump(mode="json")


_REFUSED = (AnalysisRefused, SeveralMembers, FrameUnavailable, AnalysisTooLarge)


class _AnalysisRoutes:
    """The door's two handlers over the surface's dependency cell."""

    def __init__(self, cell: DepsCell) -> None:
        self._cell = cell

    async def _run(self, job: Any, *, name: str, request: Request, body: BaseModel) -> JSONResponse:
        current = self._cell.get()
        _log_caller(request, current, route=name)
        deps = definition_deps(
            current, deadline=time.monotonic() + current.settings.analysis_timeout_seconds
        )
        with door_call(name, logger=logger) as call:
            try:
                answer = await _run_retrieval(
                    current.analysis_executor,
                    job,
                    deps=deps,
                    timings=call.timings,
                    refuse=AnalysisBackpressure,
                    current=current,
                    body=body,
                    trace_id=get_trace_id(),
                )
            except AnalysisPending as pending:
                outcome: JSONResponse = _pending(pending)
            except _REFUSED as refused:
                outcome = _refused(refused)
            except Refusal as refused:
                spec = classify_analysis_exception(refused)
                assert spec is not None
                outcome = _ENVELOPE.render(spec)
            else:
                outcome = JSONResponse(answer, headers=dict(NO_STORE_HEADERS))
        outcome.headers["Server-Timing"] = call.server_timing
        return outcome

    async def post_analysis(self, body: AnalysisRequest, request: Request) -> JSONResponse:
        """Compute a recipe tree and store its frame. Answers the stored frame's format 2 ref,
        `pending` while a root builds, or the refusal with its fix."""
        return await self._run(_analysis_job, name="analysis", request=request, body=body)

    async def post_anchor(self, body: AnchorRequest, request: Request) -> JSONResponse:
        """The latest period of the roots over the populations, and the series it leaves
        out. Stores nothing."""
        return await self._run(_anchor_job, name="analysis_anchor", request=request, body=body)


def build_analysis_door(*, deps: DepsCell) -> DoorContribution:
    """Build the analysis door over the surface's dependency cell."""
    routes = _AnalysisRoutes(deps)
    router = APIRouter()
    router.add_api_route(
        _ANALYSIS_PATH,
        routes.post_analysis,
        methods=["POST"],
        name="post_analysis",
        response_model=AnalysisReady,
        responses={202: {"model": AnalysisPendingBody}},
    )
    router.add_api_route(
        _ANCHOR_PATH,
        routes.post_anchor,
        methods=["POST"],
        name="post_analysis_anchor",
        response_model=AnchorAnswer,
    )
    return DoorContribution(
        router=router,
        registry_endpoints=route_registry_endpoints(router, "datasets.analysis"),
        classify=classify_analysis_exception,
        # A Spec root is served as the definition door serves it, and a root under `latest`
        # moves as the frame refresh moves it, so their refusals reach this route too.
        exception_types=(
            *SERVING_EXCEPTION_TYPES,
            RenderingNotRecorded,
            RefreshRebuildFailed,
            AnalysisRefused,
            SeveralMembers,
            RootNotStored,
            RootExpired,
            RootRecipeMissing,
            RootRenderingUnreadable,
            AnalysisTooLarge,
            AnalysisBackpressure,
            CallerHeaderInvalid,
        ),
        exception_exemplars=(
            AnalysisRefused(EngineRefusal.op_spec_invalid(detail="exemplar")),
            SeveralMembers(verb="quotient_of", side="data", spec_ids=["spec_a", "spec_b"]),
            RootNotStored(spec_id="spec_exemplar", content_version="cv_exemplar", reason="absent"),
            RootExpired(spec_id="spec_exemplar", content_version="cv_exemplar", reason="expired"),
            RootRecipeMissing(
                spec_id="ops_exemplar", content_version="ocv_exemplar", reason="none"
            ),
            RootRenderingUnreadable(
                spec_id="spec_exemplar", content_version="cv_exemplar", reason="a groupby"
            ),
            AnalysisTooLarge(ceiling_kind="analysis_nodes", count=65, cap=64),
            AnalysisBackpressure(),
            CallerHeaderInvalid(),
        ),
    )


__all__ = [
    "AnalysisBackpressure",
    "AnalysisPendingBody",
    "AnalysisReady",
    "AnchorAnswer",
    "build_analysis_door",
    "classify_analysis_exception",
]
