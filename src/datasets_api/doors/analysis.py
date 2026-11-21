"""The analysis door: `POST /api/datasets/analysis` and `POST /api/datasets/analysis/anchor`.

`POST /api/datasets/analysis` takes a recipe tree (`analysis.AnalysisRequest`) and answers:

- 200 `AnalysisReady`: the format 2 ref of the stored frame, its frame `cv_` and output digest,
  the top step's `summary` (`dfo_analysis_engine.calls.VerbCall.summary`) and the shape facts
  it holds, the execution identity, and each period the recipe states relative to the latest;
- 202 `AnalysisPendingBody` with `Retry-After`, when a root's dataset is still building;
- 500 `internal_error` when the pair the run computed already holds a different recipe or frame
  (`analysis.AnalysisStoreDiverges`), a fault this service logs with both identities;
- a problem document otherwise. An engine refusal, `several_members`, `over_ceiling` and a root
  the runner cannot read (`snapshot_not_found`, `artifact_expired`, `recipe_not_found`,
  `rendering_not_recorded`) carry `retry_mode` and `fix` beside the code, as the engine's refusal body
  (`dfo_analysis_engine.refusal.Refusal.as_json`) does.

`POST /api/datasets/analysis/anchor` takes roots and the populations a rolling step counts over
(`analysis.AnchorRequest`) and answers the latest period, its rule and the series it leaves out,
storing nothing.

A request with `verify` set computes the tree again at the versions its roots name and
answers 200 `AnalysisVerified` (`analysis.verify_analysis`): `outcome` (`same`, `different`,
`not_stored` or `roots_unavailable`), the pair the computation names with its frame `cv_` and
output digest, the pair compared with and the `cv_` of the frame stored there, the stored
frame's output digest when the two `cv_`s differ, the execution identity that computed it and
the one the stored recipe records, the refusal the computation met (`refusal_code`,
`refusal_reason`), and `unavailable_roots`. An answer reproduces when the `cv_`s or the output
digests match. An engine refusal or a cap the computation meets answers `different` when a
frame is stored under the pair compared with. `unavailable_roots` names every root that cannot
be read at its version, up to the root-row bound (`over_ceiling` then), with its code:
`artifact_expired` once its stored frame has aged out, `snapshot_revoked` once its version is
taken down. A root whose rendering's file has aged out is rendered in memory from its stored
frame. A verification reads no stored answer and writes nothing: no frame, recipe, answer,
version record or rendering, and it pins nothing. It admits the caller holding the `verify`
grant (`datasets_api.principals`, `VERIFY_PRINCIPAL`) alone and refuses any other
`verify_not_permitted` (403). It takes an analysis thread and a materialise permit only while
another of each stays free for runs, anchors and mints, else refuses retryable
`service_unavailable` (503), so verifications never hold the last of either; settings that
grant `verify` need two of each (`DatasetsBuildSettings`). It logs no `datasets.delivery`
line, since it serves no frame. A `verify` request that also sets `latest`, or names a Spec
root, refuses `request_invalid` (400), since neither names a version.

Both routes admit every caller and spend no allowance. The caller is resolved
(`dfo_platform.caller.resolve_caller`) for the call's log line alone, and a caller this
service cannot name logs as unresolved, a malformed `x-dfo-caller` of a vouched request
included. Every Datasets code the routes answer
goes out at the status the code registry maps on `datasets.analysis`. A run takes a thread of
the analysis executor (`DATASETS_ANALYSIS_MAX_CONCURRENCY`), refusing retryable
`service_unavailable` (503) when every thread is busy, and one materialise permit for its frames.
It answers timeout (504) after `DATASETS_ANALYSIS_TIMEOUT_SECONDS`, and its thread runs on to
store the frame.

A run is keyed (`analysis.prepare_analysis`) before it takes its materialise permit. A key that
holds a stored answer whose frame file is still served answers that body
(`analysis.stored_answer`). A key another request of this process is computing answers the
pending body, 202 with `Retry-After` 5 and no operations, so a retry after a timeout joins the
one run and later finds its answer stored. The key is released when the run ends, whether it
succeeds or fails.
"""

from __future__ import annotations

import functools
import logging
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any, Literal

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, JsonValue

from datasets_api.admission import bounded_materialise_slot
from datasets_api.analysis import (
    AnalysisBounds,
    AnalysisPending,
    AnalysisRequest,
    AnalysisStoreDiverges,
    AnalysisStores,
    AnalysisTooLarge,
    AnchorRequest,
    FrameUnavailable,
    RootExpired,
    RootNotStored,
    RootRecipeMissing,
    RootRenderingUnreadable,
    SeveralMembers,
    prepare_analysis,
    run_anchor,
    run_prepared,
    stored_answer,
    verify_analysis,
)
from datasets_api.analysis_roots import LiveRootSource
from datasets_api.background_admission import leaving_headroom
from datasets_api.backpressure import Backpressure
from datasets_api.codes import CODE_CATALOG
from datasets_api.codes_analysis import ANALYSIS_STATUSES
from datasets_api.config import DatasetsBuildSettings
from datasets_api.current import DefinitionDeps
from datasets_api.delivery import record_analysis
from datasets_api.deps import Deps, DepsCell
from datasets_api.doors.definition import (
    SERVING_EXCEPTION_EXEMPLARS,
    SERVING_EXCEPTION_TYPES,
    _run_retrieval,
    definition_deps,
)
from datasets_api.doors.mint_errors import MintBackgroundDeclined
from datasets_api.doors.refresh import RefreshRebuildFailed
from datasets_api.headers import NO_STORE_HEADERS
from datasets_api.principals import Grant, Principals, VerifyNotPermitted
from datasets_api.refusals import classify_datasets_refusal
from datasets_api.timings import door_call
from datasets_api.version import DATASETS_CAPABILITY_VERSION
from dfo_analysis_engine.refusal import AnalysisRefused, RefusalCode, RetryMode
from dfo_analysis_engine.refusal import Refusal as EngineRefusal
from dfo_platform import ProblemSpec
from dfo_platform.caller import CallerHeaderInvalid, CallerUnresolved, resolve_caller
from dfo_platform.cloud_logging import JSON_FIELDS_ATTR
from dfo_platform.doors import DoorContribution, route_registry_endpoints
from dfo_platform.problem import Envelope, Refusal, log_problem
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


class UnavailableRootBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    spec_id: str
    content_version: str
    code: str = Field(description="the Datasets code a run refuses the root with")
    reason: str


class AnalysisVerified(BaseModel):
    """A tree computed again at the versions it names, beside the frame stored under its pair
    (`analysis.Verification`)."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["verified"] = "verified"
    outcome: Literal["same", "different", "not_stored", "roots_unavailable"]
    frame: dict[str, str] | None = Field(description="the pair the computation names")
    frame_content_version: str | None = Field(pattern=r"^cv_[0-9a-f]{32}$")
    output_digest: str | None = Field(pattern=r"^out_[0-9a-f]{32}$")
    stored: dict[str, str] | None = Field(description="the pair compared with")
    stored_frame_content_version: str | None = Field(pattern=r"^cv_[0-9a-f]{32}$")
    stored_output_digest: str | None = Field(
        pattern=r"^out_[0-9a-f]{32}$",
        description="the stored frame's output digest, read when the two cv_s differ",
    )
    stored_execution_identity: str | None = Field(
        description="the execution identity the stored frame's recipe records"
    )
    execution_identity: dict[str, str]
    refusal_code: str | None = Field(
        description="the engine's refusal code, or over_ceiling, the computation met"
    )
    refusal_reason: str | None
    unavailable_roots: list[UnavailableRootBody]


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


def _refused(exc: Exception) -> JSONResponse:
    """The problem document of an analysis refusal (`classify_analysis_exception`), with its
    retry mode and fix when the route names them (`_REFUSED`). The route renders it outside
    the platform's exception handler, so it logs the line the problem names."""
    spec = classify_analysis_exception(exc)
    assert spec is not None
    log_problem(spec, exc, logger=logger, context="datasets")
    if not isinstance(exc, _REFUSED):
        return _ENVELOPE.render(spec)
    fix: object
    if isinstance(exc, AnalysisRefused):
        retry_mode, fix = exc.refusal.retry_mode.value, exc.refusal.fix
    elif isinstance(exc, AnalysisTooLarge):
        retry_mode = _CORRECT
        fix = {"ceiling_kind": exc.ceiling_kind, "count": exc.count, "cap": exc.cap}
    else:
        assert isinstance(exc, SeveralMembers | FrameUnavailable)
        retry_mode, fix = _CORRECT, exc.fix
    problem = _ENVELOPE.details(
        status=spec.status, code=spec.code, detail=spec.detail, retryable=False
    )
    return JSONResponse(
        status_code=spec.status,
        content={**problem.to_json(), "retry_mode": retry_mode, "fix": fix},
        media_type="application/problem+json",
        headers={"Cache-Control": "no-store"},
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
    except (CallerUnresolved, CallerHeaderInvalid):
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
    return classify_datasets_refusal(exc)


def analysis_stores(current: Deps) -> AnalysisStores:
    """What a run over the surface's `Deps` stores its frame and recipe through."""
    return AnalysisStores(
        snapshot_reader=current.snapshot_reader,
        version_writer=current.version_writer,
        artifact_store=current.artifact_store,
        recipe_store=current.recipe_store,
        enrichment=current.enrichment,
        site=current.settings.site,
        now=current.now,
    )


def analysis_bounds(current: Deps) -> AnalysisBounds:
    """The tree-size, root-row and join-row bounds the settings give a run."""
    settings = current.settings
    return AnalysisBounds(
        max_nodes=settings.analysis_max_nodes,
        max_root_rows=settings.analysis_max_root_rows,
        max_join_rows=settings.analysis_max_join_rows,
    )


def analysis_permit(current: Deps, deps: DefinitionDeps) -> functools.partial[Any]:
    """The materialise permit a run holds while it reads its root frames."""
    return functools.partial(
        bounded_materialise_slot,
        current.materialise_semaphore,
        wait_seconds=current.settings.materialise_wait_seconds,
        deadline=deps.deadline,
    )


# The `Retry-After` of a request that joins a run already computing its key.
_JOIN_RETRY_AFTER_S = 5.0


class RunningAnalyses:
    """The request keys this process is computing.

    One instance serves every caller that runs a keyed analysis in the process: the analysis
    door, and the live share slot check (`presentation_api.newest_frames`) when the app mounts
    both surfaces. So a key computes once, and every other caller answers pending meanwhile.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._keys: set[str] = set()

    @contextmanager
    def claim(self, key: str) -> Iterator[None]:
        """Hold `key` while the block runs, and release it however the block ends.

        Raises:
            AnalysisPending: Another request holds `key`, with no operations and
                `_JOIN_RETRY_AFTER_S`.
        """
        with self._lock:
            if key in self._keys:
                raise AnalysisPending(operations=(), retry_after_s=_JOIN_RETRY_AFTER_S)
            self._keys.add(key)
        try:
            yield
        finally:
            with self._lock:
                self._keys.discard(key)


# The analysis threads and materialise permits a `verify` request leaves free for runs,
# anchors and mints.
_VERIFY_HEADROOM = 1


def _verify_job(
    *, deps: DefinitionDeps, current: Deps, body: AnalysisRequest, trace_id: str | None
) -> dict[str, Any]:
    """Verify `body`, taking each materialise permit only while `_VERIFY_HEADROOM` stay free
    after it (`background_admission.leaving_headroom`).

    Raises:
        AnalysisBackpressure: A permit could be taken only by leaving fewer free.
    """
    bounds = analysis_bounds(current)
    try:
        with leaving_headroom(_VERIFY_HEADROOM):
            verification = verify_analysis(
                body,
                source=LiveRootSource(
                    deps=deps,
                    recipes=current.recipe_store,
                    render=False,
                    max_root_rows=bounds.max_root_rows,
                ),
                stores=analysis_stores(current),
                bounds=bounds,
                permit=analysis_permit(current, deps),
                trace_id=trace_id,
            )
    except MintBackgroundDeclined:
        raise AnalysisBackpressure() from None
    return AnalysisVerified.model_validate(verification.body()).model_dump(mode="json")


def _admit_verify(request: Request, settings: DatasetsBuildSettings) -> None:
    """Admit a `verify` request from the caller holding the `verify` grant alone.

    Raises:
        VerifyNotPermitted: The caller lacks the grant.
    """
    Principals.resolve(settings=settings).require(Grant.verify)(request)


def _analysis_job(
    *,
    deps: DefinitionDeps,
    current: Deps,
    body: AnalysisRequest,
    trace_id: str | None,
    running: RunningAnalyses,
) -> dict[str, Any]:
    source = LiveRootSource(deps=deps, recipes=current.recipe_store)
    stores = analysis_stores(current)
    bounds = analysis_bounds(current)
    prepared = prepare_analysis(body, source=source, bounds=bounds, trace_id=trace_id)
    answer = stored_answer(prepared, stores=stores)
    if answer is None:
        with running.claim(prepared.key):
            answer = run_prepared(
                prepared,
                source=source,
                stores=stores,
                bounds=bounds,
                permit=analysis_permit(current, deps),
                trace_id=trace_id,
            ).body()
    return AnalysisReady.model_validate(answer).model_dump(mode="json")


def _anchor_job(
    *, deps: DefinitionDeps, current: Deps, body: AnchorRequest, trace_id: str | None
) -> dict[str, Any]:
    anchor, roots = run_anchor(
        body,
        source=LiveRootSource(deps=deps, recipes=current.recipe_store),
        bounds=analysis_bounds(current),
        permit=analysis_permit(current, deps),
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

    def __init__(self, cell: DepsCell, *, running: RunningAnalyses) -> None:
        self._cell = cell
        self._analysis_job = functools.partial(_analysis_job, running=running)

    async def _run(
        self,
        job: Any,
        *,
        name: str,
        request: Request,
        body: BaseModel,
        record: Callable[[dict[str, Any]], None] | None = None,
        headroom: int = 0,
    ) -> JSONResponse:
        """Run `job` as door `name`. A door given `record` delivers: `record` records its
        answer for the call's `datasets.delivery` line. `headroom` is the analysis threads
        that must stay free after `job` takes one."""
        current = self._cell.get()
        _log_caller(request, current, route=name)
        deps = definition_deps(
            current, deadline=time.monotonic() + current.settings.analysis_timeout_seconds
        )
        with door_call(name, logger=logger, delivers=record is not None) as call:
            try:
                answer = await _run_retrieval(
                    current.analysis_executor,
                    job,
                    deps=deps,
                    timings=call.timings,
                    refuse=AnalysisBackpressure,
                    headroom=headroom,
                    current=current,
                    body=body,
                    trace_id=get_trace_id(),
                )
            except AnalysisPending as pending:
                outcome: JSONResponse = _pending(pending)
            except (AnalysisRefused, Refusal) as refused:
                outcome = _refused(refused)
            else:
                if record is not None:
                    record(answer)
                outcome = JSONResponse(answer, headers=dict(NO_STORE_HEADERS))
        outcome.headers["Server-Timing"] = call.server_timing
        return call.delivered(outcome)

    async def post_analysis(self, body: AnalysisRequest, request: Request) -> JSONResponse:
        """Compute a recipe tree and store its frame. Answers the stored frame's format 2 ref,
        `pending` while a root builds, or the refusal with its fix. With `verify`, compute it
        again at its pinned versions, storing nothing, for the `verify` grant's holder."""
        if body.verify:
            _admit_verify(request, self._cell.get().settings)
            return await self._run(
                _verify_job,
                name="analysis_verify",
                request=request,
                body=body,
                headroom=_VERIFY_HEADROOM,
            )
        return await self._run(
            self._analysis_job,
            name="analysis",
            request=request,
            body=body,
            record=record_analysis,
        )

    async def post_anchor(self, body: AnchorRequest, request: Request) -> JSONResponse:
        """The latest period of the roots over the populations, and the series it leaves
        out. Stores nothing."""
        return await self._run(_anchor_job, name="analysis_anchor", request=request, body=body)


def build_analysis_door(*, deps: DepsCell, running: RunningAnalyses) -> DoorContribution:
    """Build the analysis door over the surface's dependency cell. `running` holds the keys
    the process computes, shared with every other caller that runs a keyed analysis."""
    routes = _AnalysisRoutes(deps, running=running)
    router = APIRouter()
    router.add_api_route(
        _ANALYSIS_PATH,
        routes.post_analysis,
        methods=["POST"],
        name="post_analysis",
        # `verify` answers `AnalysisVerified`, every other request `AnalysisReady`.
        response_model=AnalysisReady | AnalysisVerified,
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
        # A Spec root, and a dataset root under `latest`, is served as the definition door
        # serves it, so its refusals reach this route too.
        exception_types=(
            *SERVING_EXCEPTION_TYPES,
            AnalysisRefused,
            SeveralMembers,
            RootNotStored,
            RootExpired,
            RootRecipeMissing,
            RootRenderingUnreadable,
            AnalysisTooLarge,
            AnalysisStoreDiverges,
            AnalysisBackpressure,
            # A dataset root whose queued rebuild failed (`datasets_api.analysis_roots`).
            RefreshRebuildFailed,
            # A `verify` request from a caller without the `verify` grant.
            VerifyNotPermitted,
        ),
        endpoint_exemplars={
            "datasets.analysis": (
                *SERVING_EXCEPTION_EXEMPLARS,
                # A run refuses with any engine code.
                *(
                    AnalysisRefused(EngineRefusal(code=code, message="exemplar", fix={}))
                    for code in RefusalCode
                ),
                SeveralMembers(verb="quotient_of", side="data", spec_ids=["spec_a", "spec_b"]),
                RootNotStored(
                    spec_id="spec_exemplar", content_version="cv_exemplar", reason="absent"
                ),
                RootExpired(
                    spec_id="spec_exemplar", content_version="cv_exemplar", reason="expired"
                ),
                RootRecipeMissing(
                    spec_id="ops_exemplar", content_version="ocv_exemplar", reason="none"
                ),
                RootRenderingUnreadable(
                    spec_id="spec_exemplar", content_version="cv_exemplar", reason="a groupby"
                ),
                AnalysisTooLarge(ceiling_kind="analysis_nodes", count=65, cap=64),
                AnalysisStoreDiverges(spec_id="ops_exemplar", content_version="ocv_exemplar"),
                AnalysisBackpressure(),
                RefreshRebuildFailed(operation_id="op_exemplar", refusal_code="over_ceiling"),
                VerifyNotPermitted(presented=False),
            )
        },
    )


__all__ = [
    "AnalysisBackpressure",
    "AnalysisPendingBody",
    "AnalysisReady",
    "AnalysisVerified",
    "AnchorAnswer",
    "RunningAnalyses",
    "analysis_bounds",
    "analysis_permit",
    "analysis_stores",
    "build_analysis_door",
    "classify_analysis_exception",
]
