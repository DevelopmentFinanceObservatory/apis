"""The analysis door: `POST /api/datasets/analysis` and `POST /api/datasets/analysis/anchor`.

`POST /api/datasets/analysis` takes a recipe tree (`analysis.AnalysisRequest`) and answers:

- 200 `AnalysisReady`: the format 2 ref of the stored frame, its frame `cv_` and output digest,
  the top step's `summary` (`dfo_analysis_engine.calls.VerbCall.summary`) and the shape facts
  it holds, the execution identity, and each period the recipe states relative to the latest.
  A `latest: "stored"` request adds `resolved_roots`, each dataset's stored head version and
  freshness from the roots it resolved (`analysis.stored_heads`), composed here and never
  stored. A request stating `include` receives `summary` only when it names it, reading a
  stored answer without it, and `manifest`, the derived manifest of the answer's pair read now
  (`doors.derived.derived_manifest_model`), when it names that. A request naming neither gets the
  body it always has;
- 202 `AnalysisPendingBody` with `Retry-After`, when a root's dataset is still building, or
  under `latest: "stored"` has no servable head and its rebuild is queued;
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
`verify_not_permitted` (403). It takes an analysis thread only while another stays free for
runs and anchors, waiting for one as a run waits, and a materialise permit only while another
stays free, else refuses retryable `service_unavailable` (503), so verifications never hold
the last of either; settings that
grant `verify` need two of each (`DatasetsBuildSettings`). It takes no thread of the analysis
key executor. It logs no `datasets.delivery` line, since it serves no frame. A `verify` request
that also sets `latest`, `true` or `"stored"`, or names a Spec root, refuses `request_invalid`
(400), since neither names a version, and so does one that sets `include`.

Both routes admit every caller and spend no allowance. The caller is resolved
(`dfo_platform.caller.resolve_caller`) for the call's log line alone, and a caller this
service cannot name logs as unresolved, a malformed `x-dfo-caller` of a vouched request
included. Every Datasets code the routes answer
goes out at the status the code registry maps on `datasets.analysis`.

A run is keyed first (`analysis.prepare_analysis`), on a thread of the analysis key executor
(`DATASETS_ANALYSIS_KEY_MAX_CONCURRENCY`). A request waits for one up to `_KEY_WAIT_SECONDS` or
its deadline, whichever comes first, and refuses retryable `service_unavailable` (503) at that
bound, when `_KEY_QUEUE_PER_THREAD` requests per thread already wait, or as soon as a swap
replaces the graph release it bound (`admission.bound_release_served`), as a request waiting for
a materialise permit does. Keying resolves the roots at the versions they are read at, so every
refusal of a root, and the pending body of a root still building, answers from it. A
`latest: "stored"` request keys from records alone, rendering nothing
(`LiveRootSource(defer_renders=True)`); one whose roots need a rendering no record names to be
keyed is keyed on its analysis thread instead. A key that holds a stored answer whose frame
file is still served answers that body there (`analysis.stored_answer`), taking no analysis
thread and no materialise permit, so stored answers keep answering while every analysis thread
computes. Any other run then waits for a thread of the analysis executor
(`DATASETS_ANALYSIS_MAX_CONCURRENCY`) as keying waits for its own, up to `_RUN_WAIT_SECONDS`
or its deadline with `_RUN_QUEUE_PER_THREAD` runs waiting per thread, refusing retryable
`service_unavailable` (503) at either bound or on a release swap, and computes the key it was
given, keying nothing again. An anchor and a verification wait for a thread under the same
bounds and queue, and while one waits every waiting run leaves free the threads it needs
(`_Reservations`), so runs queued for every thread never starve it. A key another request of this process is computing answers the pending body, 202
with `Retry-After` 5 and no operations, so a retry after a timeout joins the one run and later
finds its answer stored. A run that holds its key reads the stored answer again before it
computes, so an answer another request stored after this one keyed answers as stored. The key is
released when the run ends, whether it succeeds or fails. The run holds one materialise permit
for its frames. The request answers timeout (504) after `DATASETS_ANALYSIS_TIMEOUT_SECONDS`,
counted from admission over keying and computing together. A timeout while the run computes
leaves its thread running on to store the frame and answer. A timeout while it keys stores
nothing, since the run never reaches an analysis thread, and a retry keys it again.

A run's `Server-Timing` header and `datasets.timings` line carry phases whose sum covers
`total`: `queue`, from the request's wait for an executor thread to the job's start, and
`handoff`, from the job's end back to the event loop, summed over the keying and analysis hops;
`resolve_roots`, planning the tree and resolving its roots (`analysis.prepare_analysis`);
`request_key`; `stored_answer`, the answer read, its frame file check and the renewal of a file
near the end of its retention; `derived_manifest`, under `include: ["manifest"]`; `body`,
validating and dumping `AnalysisReady`; and `serialise`, encoding the response. A run that
computes adds `locate_roots`, reading its root frames; `evaluate`, the engine's evaluation of
the tree over them (`analysis._evaluated_frames`); `store_frame`, writing the computed frame; and
`write_answer`, recording the answer under its request key. Phases recorded inside these say where
their time went, such as `snapshot_state`, `head_read`, `coverage` and `head_check` inside
`resolve_roots`, and the connection pool's `db_pool_wait`, `db_connect` and `db_ping` inside
whichever phase borrowed the connection.

A `latest: "stored"` request keys through `dataset_store.snapshot.SharedVersionReads`, so the
roots' resolution, the stored answer's file check and a hit's derived manifest read each
version once for each set of revoking ids.
"""

from __future__ import annotations

import functools
import logging
import threading
import time
from collections.abc import Awaitable, Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, replace
from typing import Any, Literal

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, JsonValue

from dataset_store.snapshot import SharedVersionReads
from datasets_api.admission import bound_release_served, bounded_materialise_slot
from datasets_api.analysis import (
    STORED,
    AnalysisBounds,
    AnalysisPending,
    AnalysisRequest,
    AnalysisStoreDiverges,
    AnalysisStores,
    AnalysisTooLarge,
    AnchorRequest,
    FrameUnavailable,
    PreparedAnalysis,
    RootExpired,
    RootNotStored,
    RootRecipeMissing,
    RootRenderingUnreadable,
    RootSource,
    SeveralMembers,
    prepare_analysis,
    run_anchor,
    run_prepared,
    stored_answer,
    stored_heads,
    verify_analysis,
)
from datasets_api.analysis_roots import LiveRootSource, RenderNeeded
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
from datasets_api.doors.derived import DerivedManifest, derived_manifest_model
from datasets_api.doors.mint_errors import MintBackgroundDeclined
from datasets_api.doors.refresh import RefreshRebuildFailed
from datasets_api.headers import NO_STORE_HEADERS
from datasets_api.operations.renewal import renew_on_hit
from datasets_api.operations.task_create_pool import joining_creates
from datasets_api.principals import Grant, Principals, VerifyNotPermitted
from datasets_api.refusals import classify_datasets_refusal
from datasets_api.timings import PhaseTimings, door_call, phase
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


class StatVarsSummed(BaseModel):
    """A root whose groupby summed its StatVars into one series, the version it summed into
    and how many StatVars with data it summed (its `sum_variables` step's `summed.count`)."""

    model_config = ConfigDict(extra="forbid")

    spec_id: str
    content_version: str
    count: int = Field(ge=2)


class RootVersion(BaseModel):
    model_config = ConfigDict(extra="forbid")

    spec_id: str
    content_version: str = Field(description="the version the root reads")
    base_content_version: str | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
        description="the dataset version `content_version` renders, for a groupby sum or a "
        "party-totals cut; absent when the root reads the dataset version itself",
    )


class ResolvedRootBody(BaseModel):
    """A dataset a `latest: "stored"` request's roots read: the version its stored head
    names and the freshness the head GET states for it."""

    model_config = ConfigDict(extra="forbid")

    spec_id: str
    content_version: str
    freshness: Literal["current", "pending", "stale"]


class AnalysisReady(BaseModel):
    """The stored frame of a computed tree. A request without `include` and `latest:
    "stored"` receives the body it always has, `summary` included."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["ready"] = "ready"
    ref: dict[str, JsonValue] = Field(description="the format 2 DatasetRef naming the frame")
    frame_content_version: str = Field(pattern=r"^cv_[0-9a-f]{32}$")
    output_digest: str = Field(pattern=r"^out_[0-9a-f]{32}$")
    summary: dict[str, JsonValue] | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
        description="the facts of the tree's top step: the verb, its canonical parameters, the "
        "result's shape and exclusions, and the engine result's own fields. Absent when the "
        "request states `include` without `summary`",
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
    derived_roots: list[RootVersion] = Field(
        default_factory=list,
        description="each root the tree read whose rendering is a version of its own, a groupby "
        "sum or a party-totals cut, with the dataset version it derives from, so a caller "
        "matches it to the dataset version an anchor records",
    )
    stat_vars_summed: list[StatVarsSummed] = Field(
        default_factory=list,
        description="each root read summed over its StatVars, with how many it summed; a root "
        "that states no groupby, holds several StatVars and is read under steps that name no "
        "stat_var is summed by the service itself, keeping the summable axes the steps name",
    )
    resolved_roots: list[ResolvedRootBody] | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
        description='for a `latest: "stored"` request alone: each dataset its roots read, '
        "once, in the order the tree first reaches it, with the version its stored head names "
        "and that head's freshness, stated for this request",
    )
    manifest: DerivedManifest | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
        description="for a request whose `include` names `manifest` alone: the derived manifest "
        "of the ref's member as `GET /api/datasets/derived/{spec_id}/{content_version}` "
        "answers it now",
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
    roots: list[RootVersion] = Field(
        description="the dataset versions the period was taken over, each with the dataset "
        "version its rendering derives from"
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
    """The tree-size, root-row and join-row bounds the settings give a run, and whether its
    inferred sums read the codes their filters keep."""
    settings = current.settings
    return AnalysisBounds(
        max_nodes=settings.analysis_max_nodes,
        max_root_rows=settings.analysis_max_root_rows,
        max_join_rows=settings.analysis_max_join_rows,
        sum_selected_codes=settings.analysis_sum_selected_codes,
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


@dataclass(frozen=True, kw_only=True)
class _Keyed:
    """A run keyed on the analysis key executor: the source that resolved its roots, which
    the run reads them through, and the prepared run, or the stored answer of its key. A
    `latest: "stored"` run whose roots cannot be keyed from records (`RenderNeeded`) holds
    neither, and is prepared on its analysis thread."""

    body: AnalysisRequest
    source: RootSource | None
    prepared: PreparedAnalysis | None
    answer: dict[str, Any] | None


def _wants_summary(body: AnalysisRequest) -> bool:
    """Whether `body`'s 200 carries `summary`: always without `include`, else when named."""
    return body.include is None or "summary" in body.include


def _ready(
    answer: dict[str, Any], *, body: AnalysisRequest, prepared: PreparedAnalysis, current: Deps
) -> dict[str, Any]:
    """The 200 of `body` from the stored or computed `answer`: `summary` only when the request
    asks for it, `resolved_roots` for a `latest: "stored"` request, composed from the heads its
    roots resolved to, and the derived manifest of the answer's pair read now when `include`
    names it. A request naming neither member gets `answer` as stored.

    Raises:
        SnapshotNotFound, SnapshotRevoked, ArtifactExpired: As the derived manifest route.
    """
    ready = dict(answer)
    if not _wants_summary(body):
        ready.pop("summary", None)
    if body.latest == STORED:
        ready["resolved_roots"] = [
            {
                "spec_id": head.spec_id,
                "content_version": head.content_version,
                "freshness": head.freshness,
            }
            for head in stored_heads(prepared)
        ]
    if body.include is not None and "manifest" in body.include:
        member = ready["ref"]["members"][0]
        with phase("derived_manifest"):
            # The validated model, which `AnalysisReady` takes as it stands and dumps once.
            ready["manifest"] = derived_manifest_model(
                id=member["spec"]["spec_id"],
                content_version=member["content_version"],
                current=current,
            )
    with phase("body"):
        return AnalysisReady.model_validate(ready).model_dump(mode="json")


def _key_job(
    *, deps: DefinitionDeps, current: Deps, body: AnalysisRequest, trace_id: str | None
) -> _Keyed:
    """Key `body` (`analysis.prepare_analysis`) and read the answer stored under its key. A
    `latest: "stored"` request resolves its roots from records alone, rendering nothing
    (`LiveRootSource` `defer_renders`), so keying writes nothing a version state holds (it
    writes a head read mark and may queue renewals and head checks), and its reads share each
    version's record, state and locations (`SharedVersionReads`): the roots' resolution, the
    stored answer's file check and the derived manifest of a hit read each once for each set
    of revoking ids. The sharing ends with this job. The source it
    hands a run that misses, and that run's own reads on its analysis thread, read the stores
    as they stand then, so a version revoked or renewed since keying is judged anew."""
    if body.latest != STORED:
        return _keyed(deps=deps, current=current, body=body, trace_id=trace_id)
    reads = SharedVersionReads(current.snapshot_reader)
    try:
        return _keyed(
            deps=replace(deps, snapshot_reader=reads),
            current=replace(current, snapshot_reader=reads),
            body=body,
            trace_id=trace_id,
        )
    finally:
        reads.close()


def _keyed(
    *, deps: DefinitionDeps, current: Deps, body: AnalysisRequest, trace_id: str | None
) -> _Keyed:
    """`_key_job` over the stores `deps` and `current` read."""
    source = LiveRootSource(
        deps=deps, recipes=current.recipe_store, defer_renders=body.latest == STORED
    )
    try:
        prepared = prepare_analysis(
            body, source=source, bounds=analysis_bounds(current), trace_id=trace_id
        )
    except RenderNeeded:
        return _Keyed(body=body, source=None, prepared=None, answer=None)
    answer = stored_answer(
        prepared,
        stores=analysis_stores(current),
        summary=_wants_summary(body),
        on_hit=renew_on_hit(current),
    )
    return _Keyed(
        body=body,
        source=source,
        prepared=prepared,
        answer=None
        if answer is None
        else _ready(answer, body=body, prepared=prepared, current=current),
    )


def _analysis_job(
    *,
    deps: DefinitionDeps,
    current: Deps,
    keyed: _Keyed,
    trace_id: str | None,
    running: RunningAnalyses,
) -> dict[str, Any]:
    """Compute and store the keyed run, joining a run of its key this process is computing.
    The answer stored under the key since it was keyed answers in its place. A run keying
    could not prepare from records is prepared here, rendering what its roots need."""
    stores = analysis_stores(current)
    body, source, prepared = keyed.body, keyed.source, keyed.prepared
    if prepared is None or source is None:
        source = LiveRootSource(deps=deps, recipes=current.recipe_store)
        prepared = prepare_analysis(
            body, source=source, bounds=analysis_bounds(current), trace_id=trace_id
        )
    with running.claim(prepared.key):
        answer = stored_answer(
            prepared,
            stores=stores,
            summary=_wants_summary(body),
            on_hit=renew_on_hit(current),
        )
        if answer is None:
            answer = run_prepared(
                prepared,
                source=source,
                stores=stores,
                bounds=analysis_bounds(current),
                permit=analysis_permit(current, deps),
                trace_id=trace_id,
            ).body()
    return _ready(answer, body=body, prepared=prepared, current=current)


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
        roots=[RootVersion.model_validate(root.version()) for root in roots],
    ).model_dump(mode="json")


_REFUSED = (AnalysisRefused, SeveralMembers, FrameUnavailable, AnalysisTooLarge)

# The longest a request waits for a thread of the analysis key executor, as the shares door waits
# for its own. Warm keys take well under a second, so a burst of stored answers queues within it.
_KEY_WAIT_SECONDS = 5.0
# The requests that may wait for each thread of the analysis key executor. One more refuses 503
# at once, so the waits polling the executor's semaphore stay few.
_KEY_QUEUE_PER_THREAD = 4
# The longest a run with no stored answer, an anchor or a verification waits for an analysis
# thread, and the requests that may wait for each one, so two readers that miss at once both
# compute in turn within the bound.
_RUN_WAIT_SECONDS = 5.0
_RUN_QUEUE_PER_THREAD = 4


class _Reservations:
    """The analysis threads each waiting anchor or verification needs free, so a run waiting
    beside it leaves them free (`_AnalysisRoutes._on_analysis_thread`). Written from the event
    loop and from the job as it starts on its thread."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._held: dict[int, int] = {}
        self._next = 0

    def add(self, threads: int) -> int:
        with self._lock:
            self._next += 1
            self._held[self._next] = threads
            return self._next

    def drop(self, token: int) -> None:
        with self._lock:
            self._held.pop(token, None)

    def headroom(self) -> int:
        """The threads a run must leave free after it takes one: the most any waiting anchor
        or verification needs."""
        with self._lock:
            return max(self._held.values(), default=0)


# What a door answers with: the request's `Deps`, its retrieval view and the call's timings in,
# the answer body out.
type _Answering = Callable[[Deps, DefinitionDeps, PhaseTimings], Awaitable[dict[str, Any]]]


class _AnalysisRoutes:
    """The door's two handlers over the surface's dependency cell."""

    def __init__(self, cell: DepsCell, *, running: RunningAnalyses) -> None:
        self._cell = cell
        self._analysis_job = functools.partial(_analysis_job, running=running)
        # The requests waiting for or holding a thread of the analysis key executor, and of the
        # analysis executor for a run, an anchor or a verification. Read and written on the
        # event loop alone.
        self._keying = 0
        self._running = 0
        self._reservations = _Reservations()

    async def _run(
        self,
        answering: _Answering,
        *,
        name: str,
        request: Request,
        record: Callable[[dict[str, Any]], None] | None = None,
    ) -> JSONResponse:
        """Answer door `name` with `answering`. A door given `record` delivers: `record`
        records its answer for the call's `datasets.delivery` line."""
        current = self._cell.get()
        _log_caller(request, current, route=name)
        deps = definition_deps(
            current, deadline=time.monotonic() + current.settings.analysis_timeout_seconds
        )
        with door_call(name, logger=logger, delivers=record is not None) as call:
            # A `latest: "stored"` root's head check task create runs beside the rest of
            # the call.
            async with joining_creates(call.timings):
                try:
                    answer = await answering(current, deps, call.timings)
                except AnalysisPending as pending:
                    outcome: JSONResponse = _pending(pending)
                except (AnalysisRefused, Refusal) as refused:
                    outcome = _refused(refused)
                else:
                    if record is not None:
                        record(answer)
                    started = time.perf_counter()
                    outcome = JSONResponse(answer, headers=dict(NO_STORE_HEADERS))
                    call.timings.record("serialise", (time.perf_counter() - started) * 1000)
        outcome.headers["Server-Timing"] = call.server_timing
        return call.delivered(outcome)

    async def _on_analysis_thread(
        self,
        job: Callable[..., Any],
        *,
        current: Deps,
        deps: DefinitionDeps,
        timings: PhaseTimings,
        headroom: int = 0,
        reserve: bool = False,
        **kwargs: object,
    ) -> Any:
        """Run `job` on a thread of the analysis executor, waiting for one up to
        `_RUN_WAIT_SECONDS` or the deadline, with `_RUN_QUEUE_PER_THREAD` requests waiting per
        thread, and refusing 503 past either bound or once a swap replaces the release the
        request bound. `headroom` is the threads that must stay free after `job` takes one. A
        `reserve` job, an anchor or a verification, holds the threads it needs while it waits,
        and a run waiting beside it leaves them free (`_Reservations`), so runs queued for
        every thread never starve it.

        Raises:
            AnalysisBackpressure: As above.
        """
        assert deps.deadline is not None
        executor = current.analysis_executor
        if self._running >= executor.pool_size * (1 + _RUN_QUEUE_PER_THREAD):
            raise AnalysisBackpressure()
        reservations = self._reservations
        token = reservations.add(1 + headroom) if reserve else None

        def started(**arguments: object) -> Any:
            if token is not None:
                reservations.drop(token)
            return job(**arguments)

        self._running += 1
        try:
            return await _run_retrieval(
                executor,
                started,
                deps=deps,
                timings=timings,
                refuse=AnalysisBackpressure,
                wait_seconds=min(_RUN_WAIT_SECONDS, max(0.0, deps.deadline - time.monotonic())),
                headroom=headroom if reserve else reservations.headroom,
                keep_waiting=bound_release_served(),
                timed_hop=True,
                current=current,
                **kwargs,
            )
        finally:
            self._running -= 1
            if token is not None:
                reservations.drop(token)

    def _on_analysis_executor(self, job: Any, *, body: BaseModel, headroom: int = 0) -> _Answering:
        """Answer with `job`, an anchor or a verification, on a thread of the analysis
        executor (`_on_analysis_thread`). `headroom` is the threads that must stay free after
        `job` takes one."""

        async def answering(
            current: Deps, deps: DefinitionDeps, timings: PhaseTimings
        ) -> dict[str, Any]:
            return await self._on_analysis_thread(
                job,
                current=current,
                deps=deps,
                timings=timings,
                headroom=headroom,
                reserve=True,
                body=body,
                trace_id=get_trace_id(),
            )

        return answering

    def _keyed_run(self, body: AnalysisRequest) -> _Answering:
        """Key `body` and read its stored answer on the analysis key executor, then compute any
        other run on the analysis executor. The wait for a keying thread is bounded as the
        module docstring says."""

        async def answering(
            current: Deps, deps: DefinitionDeps, timings: PhaseTimings
        ) -> dict[str, Any]:
            assert deps.deadline is not None
            trace_id = get_trace_id()
            executor = current.analysis_key_executor
            if self._keying >= executor.pool_size * (1 + _KEY_QUEUE_PER_THREAD):
                raise AnalysisBackpressure()
            self._keying += 1
            try:
                keyed = await _run_retrieval(
                    executor,
                    _key_job,
                    deps=deps,
                    timings=timings,
                    refuse=AnalysisBackpressure,
                    wait_seconds=min(_KEY_WAIT_SECONDS, max(0.0, deps.deadline - time.monotonic())),
                    keep_waiting=bound_release_served(),
                    timed_hop=True,
                    current=current,
                    body=body,
                    trace_id=trace_id,
                )
            finally:
                self._keying -= 1
            if keyed.answer is not None:
                return keyed.answer
            return await self._on_analysis_thread(
                self._analysis_job,
                current=current,
                deps=deps,
                timings=timings,
                keyed=keyed,
                trace_id=trace_id,
            )

        return answering

    async def post_analysis(self, body: AnalysisRequest, request: Request) -> JSONResponse:
        """Compute a recipe tree and store its frame. Answers the stored frame's format 2 ref,
        `pending` while a root builds, or the refusal with its fix. With `verify`, compute it
        again at its pinned versions, storing nothing, for the `verify` grant's holder."""
        if body.verify:
            _admit_verify(request, self._cell.get().settings)
            return await self._run(
                self._on_analysis_executor(_verify_job, body=body, headroom=_VERIFY_HEADROOM),
                name="analysis_verify",
                request=request,
            )
        return await self._run(
            self._keyed_run(body), name="analysis", request=request, record=record_analysis
        )

    async def post_anchor(self, body: AnchorRequest, request: Request) -> JSONResponse:
        """The latest period of the roots over the populations, and the series it leaves
        out. Stores nothing."""
        return await self._run(
            self._on_analysis_executor(_anchor_job, body=body),
            name="analysis_anchor",
            request=request,
        )


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
    "ResolvedRootBody",
    "RunningAnalyses",
    "analysis_bounds",
    "analysis_permit",
    "analysis_stores",
    "build_analysis_door",
    "classify_analysis_exception",
]
