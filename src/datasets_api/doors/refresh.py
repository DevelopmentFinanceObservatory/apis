"""The frame refresh door serves `POST /api/datasets/frames/refresh`.

The body names a dataset frame by `spec_id` and `content_version`. The door answers whether
that frame is still the current rendering of its dataset, rebuilding the dataset when its data
moved or its head is too old:

- `current`: the named frame is the current frame of its rendering.
- `moved`: the current frame of the same rendering has another `content_version`, whether this
  call rebuilt the dataset or an earlier one did.
- `pending`: the rebuild is over the inline budget and runs as a queued operation. The caller
  calls again after `retry_after_s`. A call made while the operation is queued or running names
  the same operation and enqueues nothing.

The dataset and rendering come from stored records. A version record
with no `base_content_version` is the dataset's default rendering. A derived version, a groupby
collapse or a party-totals cut (`without_party_totals`) or both, is stored under its own
`content_version` and names the raw version in `base_content_version`. Its minting options
(`recorded_rendering.MintingOptions`) are read from the requested-format file linked under the
derived version, since the stored frame's rendering is the same for every frame
(`datasets_api.recorded_rendering`). A derived version with no such file refuses
`rendering_not_recorded` (422). An unregistered `spec_id`
refuses `spec_not_found` (404), an unknown version of a registered one `snapshot_not_found`
(404), and a derived (`ops_`) frame `frame_not_refreshable` (422). Resolution does not check
the named version's revocation, so a frame taken down still refreshes to a newer current frame.

A groupby sum that leaves the frame unchanged, as over a dataset with one contributing StatVar,
and a party-totals cut that leaves nothing out have no version of their own: their file is
linked under the raw version. Renderings that produce the same rows share one version, so a
version can be recorded under several renderings: a raw version under the plain rendering and
such files, a derived version under a groupby and a groupby with a cut that left nothing out.
The records do not say which of them the caller holds, and they need not say: the refresh
renders each of them from the head with the current code index (`_rendering_frame`) and answers
with the one frame they make, linking a file of each under it. Once they differ it refuses
`rendering_not_recorded` (422). A head whose stored frame has expired, so that a rendering
cannot be made from it, rebuilds or answers `pending` as for any other frame. The frame
refreshes, and rebuilds, as the plain rendering when that is recorded, and otherwise as the
first of its minting options by `recorded_rendering.ordered_mintings`, so the order the files
were linked in moves nothing.

Freshness is the definition door's (`current.serve_current` through `serve_spec`): a head
confirmed within `head_recheck_seconds` answers with no read, a later call compares the head's
dependency fingerprint in one graph read, and a changed fingerprint, a newly imported release,
or a head older than its maximum age (`head_max_age_seconds`, or
`head_imported_data_max_age_seconds` for a head keyed on an import) rebuilds. A rebuild mints the options `ai` mints a frame with (bare
parquet in the frame layout) plus the frame's minting options, so it writes the canonical frame
Presentation reads, and a frame minted without party totals is minted without them again.
Concurrent refreshes and definition reads of one dataset share one rebuild
(`admission.RebuildFlights`).

For a derived frame the answer is the derived version under the current head with the same
minting options. The door looks it up among the dataset's stored versions and, when none is
recorded yet, mints it from the head's stored frame as a head-hit groupby read does.

A stored frame answers only while its stored frame file (`render.servable_frame_location`) is
servable, for the plain rendering and a derived one alike, since Presentation loads the frame
from that file. When a derived version's frame file has expired the door renders it again from
the head's stored frame, replaying the party-totals cut its trace records, and records it. When
that cannot be done, or the head's own frame file has expired, the door rebuilds the dataset.

A rebuild over the inline budget submits the request's operation
(`operations.submit.submit_over_budget`). The submission resets a done operation, so the door
answers `pending` for it, and the worker's run serves through `serve_current`, which advances
the head, so a later refresh answers from that head. An operation that failed and will not be
retried refuses non-retryable `internal_error` (500).

The door reads the surface's `Deps`, so it shares the definition door's clients, materialise
semaphore, and `RebuildFlights`. Refreshes run on the work executor beside mints and rebuilds, so
they take no thread from the read executor's head hits. A call arriving when every work thread is
busy refuses retryable `service_unavailable` (503). Only `LIVE_REFRESH_PRINCIPAL` may call it
(`datasets_api.live_refresh`). The route class checks the caller before FastAPI reads the body.
The door records no reference and pins nothing.
"""

from __future__ import annotations

import logging
from typing import Annotated, Literal

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from dataset_store.snapshot import SnapshotNotFound, SnapshotReader
from datasets_api.backpressure import Backpressure
from datasets_api.config import DatasetsBuildSettings
from datasets_api.current import DefinitionDeps, head_rendering, load_stored
from datasets_api.deps import DepsCell
from datasets_api.doors.definition import (
    SERVING_EXCEPTION_EXEMPLARS,
    SERVING_EXCEPTION_TYPES,
    _run_retrieval,
    request_definition_deps,
    serve_spec,
    served_spec,
)
from datasets_api.headers import no_store_headers
from datasets_api.manifest_models import Manifest
from datasets_api.operations.submit import OPERATION_SUBMIT_RETRY_AFTER_MS, OperationSubmitted
from datasets_api.principals import Grant, Principals, RefreshNotPermitted, granted_route_class
from datasets_api.recorded_rendering import (
    MintingOptions,
    RenderingNotRecorded,
    StoredRendering,
    frame_options,
    ordered_mintings,
    recorded_mintings,
    recorded_party_cut,
    records_plain_file,
    rendering_of,
)
from datasets_api.refusals import classify_datasets_refusal
from datasets_api.render import servable_frame_location
from datasets_api.timings import door_call, phase
from dfo_platform import ProblemSpec
from dfo_platform.doors import DoorContribution, route_registry_endpoints
from dfo_platform.problem import Refusal
from dfo_platform.trace import get_trace_id
from materialise.formats import ParquetSizeBoundError
from materialise.reshape import RollupNotSummableError
from series_coverage.shared import CoverageSnapshot

logger = logging.getLogger(__name__)

_DERIVED_SPEC_ID_PREFIX = "ops_"


class FrameNotRefreshable(Refusal):
    """The named frame is a derived frame. Maps to `frame_not_refreshable` (422)."""

    code = "frame_not_refreshable"

    def __init__(self, *, spec_id: str) -> None:
        super().__init__(
            f"{spec_id!r} is a derived frame; only a dataset frame (`spec_`) is refreshed here"
        )


class RefreshRebuildFailed(Refusal):
    """The queued rebuild of the frame's dataset failed and will not be retried. Maps to
    non-retryable `internal_error` (500)."""

    code = "internal_error"

    def __init__(self, *, operation_id: str, refusal_code: str | None) -> None:
        super().__init__(
            f"the queued rebuild {operation_id!r} failed with {refusal_code or 'no recorded code'}"
        )


class RefreshBackpressure(Backpressure):
    """The refresh door's executor is full. Maps to retryable `service_unavailable` (503)."""

    code = "service_unavailable"

    def __init__(self) -> None:
        super().__init__("the frame refresh door is at its concurrent capacity right now; retry")


def classify_refresh_exception(exc: BaseException) -> ProblemSpec | None:
    return classify_datasets_refusal(exc)


class FramePair(BaseModel):
    model_config = ConfigDict(extra="forbid")

    spec_id: str
    content_version: str


class RefreshFrameRequest(BaseModel):
    """The frame to refresh. An `ops_` id is accepted by the grammar and refused
    `frame_not_refreshable`."""

    model_config = ConfigDict(extra="forbid")

    spec_id: str = Field(pattern=r"^(?:spec|ops)_[^/\s]+$", max_length=200)
    content_version: str = Field(pattern=r"^(?:cv|ocv)_[^/\s]+$", max_length=200)


class FrameCurrent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal["current"] = "current"
    frame: FramePair = Field(description="the named frame, which is its rendering's current one")


class FrameMoved(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal["moved"] = "moved"
    frame: FramePair = Field(description="the current frame of the named frame's rendering")


class FramePending(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal["pending"] = "pending"
    operation_id: str = Field(description="the queued rebuild, pollable on the operations route")
    retry_after_s: float = Field(description="seconds to wait before calling again")


FrameRefreshResponse = Annotated[
    FrameCurrent | FrameMoved | FramePending, Field(discriminator="status")
]


def stored_rendering(
    *, spec_id: str, content_version: str, snapshot_reader: SnapshotReader
) -> StoredRendering:
    """Resolve a stored dataset frame's rendering from its version record and linked files
    (`recorded_rendering.rendering_of`).

    A version is recorded under each rendering a file linked under it records, and a raw
    version under the plain rendering too when it links a plain file or no file of minting
    options. A version recorded under several is `ambiguous` and refreshes as the plain
    rendering when that is recorded, else as the first of its `mintings`; the refresh answers it
    only while all of them render the current data as one frame. The registration of `spec_id`
    is not checked.

    Raises:
        SnapshotNotFound: The dataset holds no version at `content_version`.
        RenderingNotRecorded: The version is derived and its linked files record no minting
            options.
    """
    version = snapshot_reader.read_version(spec_id, content_version)
    if version is None:
        raise SnapshotNotFound(id=spec_id, content_version=content_version)
    return rendering_of(
        spec_id=spec_id,
        version=version,
        renderings=[
            location.rendering
            for location in snapshot_reader.read_artifact_locations(spec_id, content_version)
        ],
    )


def refresh_can_answer(*, stored: StoredRendering, target: StoredRendering) -> bool:
    """Whether a frame resolved as `target` is one the refresh of a frame resolved as `stored`
    answers with, for some state of the same dataset's data. The render routes accept a dataset
    target by this rule, so they draw a card on every frame its refresh returns and on no
    other.

    The refresh answers with the frame every rendering `stored` is recorded under makes from the
    head, and links a file of each of its minting options under that frame. A plain rendering
    makes the raw head, so a `stored` recording it answers with a raw version alone. Each of
    `stored.mintings` makes a derived version recording it, or the raw head when it leaves the
    frame unchanged and its file is linked there. So the target is a frame recording every one
    of `stored.mintings`, and a raw one when `stored` records the plain rendering.
    """
    if stored.plain and target.collapsed:
        return False
    return all(minting in target.mintings for minting in stored.mintings)


def _recorded_derived(
    *, spec_id: str, base_content_version: str, minting: MintingOptions, deps: DefinitionDeps
) -> str | None:
    """The stored version derived from `base_content_version` with `minting`, or `None` when
    none is on record."""
    if deps.version_lister is None:
        return None
    for version in deps.version_lister.list_versions(spec_id):
        if version.base_content_version == base_content_version and minting in recorded_mintings(
            version.renderings
        ):
            return version.content_version
    return None


def _servable_frame(*, spec_id: str, content_version: str, deps: DefinitionDeps) -> Manifest | None:
    """The stored frame at `content_version`, read without its artifacts, or `None` when no
    servable stored frame file is linked under it. Presentation loads a frame from that file, so an
    expired one makes the frame unloadable there (`missing_rendering`)."""
    locations = deps.snapshot_reader.read_artifact_locations(spec_id, content_version)
    if servable_frame_location(locations, now=deps.now()) is None:
        return None
    return load_stored(
        subject_id=spec_id, content_version=content_version, deps=deps, include_artifacts=False
    )


def _rendering_frame(
    *,
    spec_id: str,
    head_content_version: str,
    minting: MintingOptions | None,
    deps: DefinitionDeps,
    trace_id: str | None,
    coverage_snapshot: CoverageSnapshot,
) -> Manifest | None:
    """The stored frame of the `minting` rendering under the head, or of the plain rendering
    when `minting` is `None`, answered only while its stored frame file is servable
    (`_servable_frame`). The plain frame is the head. A derived rendering is the derived version
    on record under the head; with none on record, or with that version's frame file expired,
    the rendering is made from the head's stored frame, as a head-hit groupby read makes it. A
    version on record replays the party-totals cut its trace records
    (`recorded_rendering.recorded_party_cut`), so it renders to its own version whatever the code
    index and the cut's rule now decide; with none on record the cut is derived afresh. A
    rendering with a version of its own is recorded under it with its canonical frame, and one
    that leaves the frame unchanged is linked under the head. `None` sends the caller to a
    rebuild: the head's frame file has expired or cannot render it."""
    if minting is None:
        return _servable_frame(spec_id=spec_id, content_version=head_content_version, deps=deps)
    recorded = _recorded_derived(
        spec_id=spec_id, base_content_version=head_content_version, minting=minting, deps=deps
    )
    recorded_cut = None
    if recorded is not None:
        stored = _servable_frame(spec_id=spec_id, content_version=recorded, deps=deps)
        if stored is not None:
            return stored
        version = deps.snapshot_reader.read_version(spec_id, recorded)
        if version is not None:
            recorded_cut = recorded_party_cut(version)
    options = frame_options(minting)
    rendered = head_rendering(
        subject_id=spec_id,
        content_version=head_content_version,
        deps=deps,
        options=options,
        trace_id=trace_id,
        coverage_snapshot=coverage_snapshot,
        recorded_cut=recorded_cut,
    )
    if rendered is None:
        return None
    return _servable_frame(spec_id=spec_id, content_version=rendered.content_version, deps=deps)


def _records_every(
    *, spec_id: str, content_version: str, rendering: StoredRendering, deps: DefinitionDeps
) -> bool:
    """Whether the version at `content_version` links a file of each of `rendering.mintings`,
    and a plain file when `rendering` records the plain rendering, so every rendering
    `rendering` is recorded under makes that version."""
    renderings = [
        location.rendering
        for location in deps.snapshot_reader.read_artifact_locations(spec_id, content_version)
    ]
    recorded = ordered_mintings(renderings)
    return all(minting in recorded for minting in rendering.mintings) and (
        not rendering.plain or records_plain_file(renderings)
    )


def _agreed_frame(
    *,
    spec_id: str,
    content_version: str,
    head_content_version: str,
    rendering: StoredRendering,
    deps: DefinitionDeps,
    trace_id: str | None,
    coverage_snapshot: CoverageSnapshot,
) -> Manifest | None:
    """The stored frame every rendering `rendering` is recorded under makes from the head, with
    a file of each linked under it, or `None` when one of them cannot be made from the head's
    frame (`_rendering_frame`), which sends the caller to a rebuild. `content_version` is the
    named frame's.

    The first rendering is the one the frame refreshes as, and a frame recorded under one
    rendering is that rendering's frame. When the first rendering's frame already links a file
    of every rendering, they are one frame there and nothing more renders. Otherwise each other
    rendering is made from the head with the current code index, a plain file is linked under
    the head, and the frame answers once every rendering made the same version and that version
    links a file of each.

    Raises:
        RenderingNotRecorded: The renderings make different frames, or their files did not
            all link under the one they make.
    """
    renderings = ((None,) if rendering.plain else ()) + rendering.mintings

    def frame_of(minting: MintingOptions | None) -> Manifest | None:
        return _rendering_frame(
            spec_id=spec_id,
            head_content_version=head_content_version,
            minting=minting,
            deps=deps,
            trace_id=trace_id,
            coverage_snapshot=coverage_snapshot,
        )

    def records_every(made: str) -> bool:
        return _records_every(spec_id=spec_id, content_version=made, rendering=rendering, deps=deps)

    differ = RenderingNotRecorded(
        spec_id=spec_id, content_version=content_version, renderings=len(renderings)
    )
    frame = frame_of(renderings[0])
    if frame is None or len(renderings) == 1 or records_every(frame.content_version):
        return frame
    for other in renderings[1:]:
        made = frame_of(other)
        if made is None:
            return None
        if made.content_version != frame.content_version:
            raise differ
    if rendering.plain:
        head_rendering(
            subject_id=spec_id,
            content_version=head_content_version,
            deps=deps,
            options=frame_options(None),
            trace_id=trace_id,
            coverage_snapshot=coverage_snapshot,
        )
    if not records_every(frame.content_version):
        raise differ
    return frame


def answer_frame_refresh(
    *,
    spec_id: str,
    content_version: str,
    deps: DefinitionDeps,
    trace_id: str | None,
    recorded: StoredRendering | None = None,
) -> FrameCurrent | FrameMoved | FramePending:
    """Answer the refresh of one dataset frame. Runs on the refresh door's executor.

    The named frame's records are read under the id that names it, so a frame recorded under a
    posted id that is not its served id keeps resolving. The head, any rebuild, and the answer
    use the served id (`kinds.spec_id.resolve_served_definition`), so such a frame answers
    `moved` to its served dataset's current frame.

    `recorded` is the frame's rendering as a card revision recorded it when the card was made
    (`presentation.component_revision.frame_rendering`, read by `StoredRendering.from_record`).
    With it the refresh answers under that rendering and reads no rendering from the frame's
    records, so a file of another rendering linked later under the named version changes
    nothing, as on the render routes (`refresh_can_answer`). Without it the rendering is the
    one `stored_rendering` reads today."""
    with phase("resolve"):
        served = served_spec(spec_id=spec_id, deps=deps, trace_id=trace_id)
        rendering = (
            recorded
            if recorded is not None
            else stored_rendering(
                spec_id=spec_id,
                content_version=content_version,
                snapshot_reader=deps.snapshot_reader,
            )
        )
    named_spec_id, spec_id = spec_id, served.spec_id
    minting = rendering.refreshes_as
    options = rendering.options
    # The versions `_agreed_frame` found every rendering makes, so `_answer` checks only a
    # frame a rebuild served.
    agreed: set[str] = set()

    def _agreed(head_content_version: str, coverage_snapshot: CoverageSnapshot) -> Manifest | None:
        frame = _agreed_frame(
            spec_id=spec_id,
            content_version=content_version,
            head_content_version=head_content_version,
            rendering=rendering,
            deps=deps,
            trace_id=trace_id,
            coverage_snapshot=coverage_snapshot,
        )
        if frame is not None:
            agreed.add(frame.content_version)
        return frame

    def _load_head_frame(
        head_content_version: str, coverage_snapshot: CoverageSnapshot
    ) -> Manifest | None:
        """The stored frame of this rendering under the head, read without its artifacts."""
        if (
            minting is not None
            and named_spec_id == spec_id
            and head_content_version == rendering.base_content_version
        ):
            # The named frame is its rendering of the version it was rendered from.
            stored = _servable_frame(spec_id=spec_id, content_version=content_version, deps=deps)
            if stored is not None:
                return stored
        return _agreed(head_content_version, coverage_snapshot)

    def _answer(current_content_version: str) -> FrameCurrent | FrameMoved:
        frame = FramePair(spec_id=spec_id, content_version=current_content_version)
        if (spec_id, current_content_version) == (named_spec_id, content_version):
            return FrameCurrent(frame=frame)
        if rendering.ambiguous and current_content_version not in agreed:
            # A rebuild served the frame, so no head read checked the other renderings.
            version = deps.snapshot_reader.read_version(spec_id, current_content_version)
            head_content_version = (
                version.base_content_version
                if version is not None and version.base_content_version is not None
                else current_content_version
            )
            made = _agreed(head_content_version, deps.coverage.snapshot())
            if made is None or made.content_version != current_content_version:
                raise RenderingNotRecorded(
                    spec_id=spec_id,
                    content_version=content_version,
                    renderings=len(rendering.mintings) + rendering.plain,
                )
        return FrameMoved(frame=frame)

    try:
        manifest = serve_spec(
            stored=served,
            options=options,
            trigger="refresh",
            deps=deps,
            trace_id=trace_id,
            load_head=_load_head_frame,
        )
    except OperationSubmitted as submitted:
        operation = submitted.operation
    else:
        return _answer(manifest.content_version)
    if operation.state == "failed":
        raise RefreshRebuildFailed(operation_id=operation.id, refusal_code=operation.refusal_code)
    return FramePending(
        operation_id=operation.id, retry_after_s=OPERATION_SUBMIT_RETRY_AFTER_MS / 1000
    )


def build_refresh_door(
    *,
    deps: DepsCell,
    settings: DatasetsBuildSettings,
    principal_header: str | None = None,
    principal: str | None = None,
) -> DoorContribution:
    """Build the frame refresh door over the surface's dependency cell. The route admits the
    `refresh` grant alone, held by `principal`, which defaults to
    `settings.live_refresh_principal`. An empty principal refuses every caller.
    """
    require_refresh = Principals.resolve(
        settings=settings,
        principal_header=principal_header,
        refresh=None if principal is None else frozenset({principal}),
    ).require(Grant.refresh)
    router = APIRouter(route_class=granted_route_class(require_refresh))

    @router.post("/api/datasets/frames/refresh", response_model=FrameRefreshResponse)
    async def refresh_frame(body: RefreshFrameRequest) -> JSONResponse:
        """Whether the named dataset frame is still its rendering's current frame. Answers
        `current`, `moved` with the current frame, or `pending` with the queued rebuild."""
        if body.spec_id.startswith(_DERIVED_SPEC_ID_PREFIX):
            raise FrameNotRefreshable(spec_id=body.spec_id)
        current = deps.get()
        with door_call("frame_refresh", logger=logger) as call:
            answer = await _run_retrieval(
                current.work_executor,
                answer_frame_refresh,
                deps=request_definition_deps(current),
                timings=call.timings,
                refuse=RefreshBackpressure,
                spec_id=body.spec_id,
                content_version=body.content_version,
                trace_id=get_trace_id(),
            )
        return JSONResponse(
            answer.model_dump(mode="json"),
            headers=no_store_headers(server_timing=call.server_timing),
        )

    return DoorContribution(
        router=router,
        registry_endpoints=route_registry_endpoints(router, "datasets.refresh"),
        classify=classify_refresh_exception,
        # `answer_frame_refresh` serves the frame's dataset as the definition door serves a
        # registered id, rendering the frame's parquet and its groupby sum.
        exception_types=(
            *SERVING_EXCEPTION_TYPES,
            ParquetSizeBoundError,
            RollupNotSummableError,
            FrameNotRefreshable,
            RenderingNotRecorded,
            RefreshRebuildFailed,
            RefreshBackpressure,
            RefreshNotPermitted,
            # `stored_rendering` reads the frame's version, which may not exist.
            SnapshotNotFound,
        ),
        endpoint_exemplars={
            "datasets.refresh": (
                *SERVING_EXCEPTION_EXEMPLARS,
                ParquetSizeBoundError("exemplar"),
                RollupNotSummableError(
                    "exemplar", axis="exemplar-axis", precondition="exemplar-precondition"
                ),
                FrameNotRefreshable(spec_id="ops_exemplar"),
                RenderingNotRecorded(spec_id="spec_exemplar", content_version="cv_exemplar"),
                RefreshRebuildFailed(operation_id="op_exemplar", refusal_code="over_ceiling"),
                RefreshBackpressure(),
                RefreshNotPermitted(presented=False),
                SnapshotNotFound(id="ops_exemplar", content_version="ocv_exemplar"),
            )
        },
    )


__all__ = [
    "FrameCurrent",
    "FrameMoved",
    "FrameNotRefreshable",
    "FramePair",
    "FramePending",
    "RefreshBackpressure",
    "RefreshFrameRequest",
    "RefreshRebuildFailed",
    "RenderingNotRecorded",
    "StoredRendering",
    "answer_frame_refresh",
    "build_refresh_door",
    "classify_refresh_exception",
    "refresh_can_answer",
    "stored_rendering",
]
