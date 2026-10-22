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
linked under the raw version. A raw version linking a file of one set of minting options and no
plain requested-format file is that rendering. One linking a plain file and files of one or more
minting options stands for several renderings, and its records do not say which the caller
holds. While those renderings of the current data are still one frame, the refresh answers with
it and links every file under it. Once they differ, because a rendering of the current data has
a version of its own, the refresh refuses `rendering_not_recorded` (422) and names the rendering
ambiguous. A raw version linking files of two or more different minting options and no plain
file refuses `rendering_not_recorded` at once, since no rendering of the current data is the
one to answer with. So does a derived version linking files of two or more different minting
options, which two renderings producing the same rows share, such as a cut and a groupby that
left the cut frame unchanged.

Freshness is the definition door's (`current.serve_current` through `_serve_spec`): a head
confirmed within `head_recheck_seconds` answers with no read, a later call compares the head's
dependency fingerprint in one graph read, and a changed fingerprint or a head older than
`head_max_age_seconds` rebuilds. A rebuild mints the options `ai` mints a frame with (bare
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
the head's stored frame and records it. When that cannot be done, or the head's own frame file has
expired, the door rebuilds the dataset.

A dataset whose rebuild is always over the inline budget holds no head, because the worker
advances none. Each refresh of it probes and then reads the operation its request submits
(`operation_store.compute_operation_id`): done answers that operation's version. A done
operation answers only while a snapshot read still lists its rendering's file
(`snapshot.rendering_servable`, the check the operations poll makes). When the file has left
storage the door fails the row (`OperationStore.fail_unservable_done`) and submits again, which
re-runs the operation and answers `pending`.

The door reads the surface's `Deps`, so it shares the definition door's clients, materialise
semaphore, and `RebuildFlights`. Refreshes run on the work executor beside mints and rebuilds, so
they take no thread from the read executor's head hits. A call arriving when every work thread is
busy refuses retryable `service_unavailable` (503). Only `LIVE_REFRESH_PRINCIPAL` may call it
(`datasets_api.live_refresh`). The route class checks the caller before FastAPI reads the body.
The door records no reference and pins nothing.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
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
    SERVING_EXCEPTION_TYPES,
    _run_retrieval,
    _serve_spec,
    _served_spec,
    request_definition_deps,
)
from datasets_api.doors.snapshot import rendering_servable
from datasets_api.headers import no_store_headers
from datasets_api.manifest_models import Manifest
from datasets_api.mint_models import MintOptions
from datasets_api.operations.submit import OPERATION_SUBMIT_RETRY_AFTER_MS, OperationSubmitted
from datasets_api.principals import Grant, Principals, RefreshNotPermitted, granted_route_class
from datasets_api.recorded_rendering import (
    MintingOptions,
    recorded_mintings,
    records_plain_file,
)
from datasets_api.refusals import classify_datasets_refusal
from datasets_api.render import artifact_rendering_key, servable_frame_location
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


class RenderingNotRecorded(Refusal):
    """The stored records do not say which rendering the frame is: a derived frame with no
    linked file recording its minting options, or with `ambiguous` a raw version linking a file
    of minting options and a plain file whose renderings of the current data differ, or with
    `sums` above one a derived version, or a raw version with no plain file, linking files of
    that many different minting options. Maps to `rendering_not_recorded` (422)."""

    code = "rendering_not_recorded"

    def __init__(
        self, *, spec_id: str, content_version: str, ambiguous: bool = False, sums: int = 1
    ) -> None:
        self.ambiguous = ambiguous
        if sums > 1:
            detail = (
                f"the rendering of {spec_id!r} at content_version {content_version!r} is "
                f"ambiguous: the version links files of {sums} different renderings that "
                "mint a version of their own, and its records do not say which the frame is"
            )
        elif ambiguous:
            detail = (
                f"the rendering of {spec_id!r} at content_version {content_version!r} is "
                "ambiguous: the version links a groupby sum or party-totals cut and a plain "
                "file, and the renderings of the current data are no longer one frame"
            )
        else:
            detail = (
                f"{spec_id!r} at content_version {content_version!r} is a derived version, "
                "and no file linked under it records the groupby or party-totals cut it was "
                "minted with"
            )
        super().__init__(detail)


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


class RefreshRebuildUnservable(Refusal):
    """The queued rebuild is done and its file has left storage, again after this call
    submitted it to run once more. Maps to retryable `service_unavailable` (503)."""

    code = "service_unavailable"
    retryable = True

    def __init__(self, *, operation_id: str) -> None:
        super().__init__(
            f"the queued rebuild {operation_id!r} is done but its file is no longer in "
            "storage; retry"
        )


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


def _frame_options(minting: MintingOptions | None) -> MintOptions:
    """The options a mint of a frame takes: the bare frame-layout parquet `ai` mints a frame
    with, plus the minting options."""
    if minting is None:
        return MintOptions(format="parquet", bundle=False, layout="frame")
    return MintOptions(
        format="parquet",
        bundle=False,
        groupby=minting.groupby,
        without_party_totals=minting.without_party_totals,
        layout="frame",
    )


@dataclass(frozen=True)
class StoredRendering:
    """The rendering of its dataset a stored frame is, read from its version record."""

    # The raw version the frame is, or was derived from.
    base_content_version: str
    # The groupby and party-totals cut the frame was minted with, `None` for the plain one.
    minting: MintingOptions | None
    # The frame's version is a raw one linking a plain file and a file of each of `mintings`,
    # so the frame is one of several renderings that were the same rows when it was minted.
    ambiguous: bool = False
    # The frame's version is a derived version of its own, not a raw version.
    collapsed: bool = False
    # Each different set of minting options a file linked under the frame's version records.
    # `minting` is the first.
    mintings: tuple[MintingOptions, ...] = ()

    @property
    def options(self) -> MintOptions:
        """The options a mint of this rendering takes (`_frame_options`)."""
        return _frame_options(self.minting)


def stored_rendering(
    *, spec_id: str, content_version: str, snapshot_reader: SnapshotReader
) -> StoredRendering:
    """Resolve a stored dataset frame's rendering from its version record and linked files.

    A version derived from another takes its minting options (a groupby, a party-totals cut,
    or both) from the files linked under it. It refuses when none records any, and when they
    record two or more different ones, since renderings that produce the same rows share one
    derived version and nothing says which the frame is. A raw version linking no file of
    minting options is the plain rendering. One linking files of one set of minting options
    and no plain file is that rendering. One linking files of two or more different ones and
    no plain file refuses, since nothing says which the frame is. One linking a plain file and
    any files of minting options is `ambiguous`: it is served as the plain rendering, which is
    determined because the refresh answers only while every one of `mintings` still renders
    the current data as the plain frame. The registration of `spec_id` is not checked.

    Raises:
        SnapshotNotFound: The dataset holds no version at `content_version`.
        RenderingNotRecorded: The version is derived and its linked files record no minting
            options or several different ones, or a raw version linking several different
            ones and no plain file.
    """
    version = snapshot_reader.read_version(spec_id, content_version)
    if version is None:
        raise SnapshotNotFound(id=spec_id, content_version=content_version)
    renderings = [
        location.rendering
        for location in snapshot_reader.read_artifact_locations(spec_id, content_version)
    ]
    mintings: tuple[MintingOptions, ...] = ()
    for recorded in recorded_mintings(renderings):
        if recorded not in mintings:
            mintings += (recorded,)
    minting = mintings[0] if mintings else None
    if version.base_content_version is not None:
        if minting is None:
            raise RenderingNotRecorded(spec_id=spec_id, content_version=content_version)
        if len(mintings) > 1:
            raise RenderingNotRecorded(
                spec_id=spec_id,
                content_version=content_version,
                ambiguous=True,
                sums=len(mintings),
            )
        return StoredRendering(
            base_content_version=version.base_content_version,
            minting=minting,
            collapsed=True,
            mintings=mintings,
        )
    plain_file = records_plain_file(renderings)
    if len(mintings) > 1 and not plain_file:
        raise RenderingNotRecorded(
            spec_id=spec_id, content_version=content_version, ambiguous=True, sums=len(mintings)
        )
    return StoredRendering(
        base_content_version=content_version,
        minting=minting,
        ambiguous=minting is not None and plain_file,
        mintings=mintings,
    )


def refresh_can_answer(*, stored: StoredRendering, target: StoredRendering) -> bool:
    """Whether a frame resolved as `target` is one the refresh of a frame resolved as `stored`
    answers with, for some state of the same dataset's data. The render routes accept a dataset
    target by this rule, so they draw a card on every frame its refresh returns and on no
    other.

    The refresh answers a plain frame with the dataset's raw head, whatever else is linked
    under it. It answers a derived frame with the same rendering of the head: a derived version
    recording those minting options, or the raw head when the rendering leaves the frame
    unchanged and its file is linked there. It answers an ambiguous frame only with a raw head
    that is every one of its renderings, so one linking a file of each of its minting options.
    """
    if stored.minting is None:
        return not target.collapsed
    if stored.ambiguous:
        return not target.collapsed and all(
            minting in target.mintings for minting in stored.mintings
        )
    return stored.minting in target.mintings


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
    rendering with a version of its own is recorded under it with its canonical frame, and one
    that leaves the frame unchanged is linked under the head. `None` sends the caller to a
    rebuild: the head's frame file has expired or cannot render it."""
    if minting is None:
        return _servable_frame(spec_id=spec_id, content_version=head_content_version, deps=deps)
    recorded = _recorded_derived(
        spec_id=spec_id, base_content_version=head_content_version, minting=minting, deps=deps
    )
    if recorded is not None:
        stored = _servable_frame(spec_id=spec_id, content_version=recorded, deps=deps)
        if stored is not None:
            return stored
    options = _frame_options(minting)
    rendered = head_rendering(
        subject_id=spec_id,
        content_version=head_content_version,
        deps=deps,
        options=options,
        trace_id=trace_id,
        coverage_snapshot=coverage_snapshot,
    )
    if rendered is None:
        return None
    return _servable_frame(spec_id=spec_id, content_version=rendered.content_version, deps=deps)


def _refuse_differing_renderings(
    *,
    spec_id: str,
    content_version: str,
    current_content_version: str,
    minting: MintingOptions,
    deps: DefinitionDeps,
    trace_id: str | None,
) -> None:
    """Refuse an ambiguous frame once the plain and the `minting` rendering of the current data
    differ. `current_content_version` is the plain one.

    The two are one frame when the current version links a file of each, as the named version
    does. When it does not yet, both are rendered from its stored frame: a rendering that keeps
    the version is linked under it (`_rendering_frame`), and so is the plain file. Anything else
    refuses: a rendering with a version of its own, a rendering that cannot be made, and a file
    that was not linked. The current frame therefore always records both renderings, and a
    later refresh of it meets the same check.

    Raises:
        RenderingNotRecorded: The two renderings are not established to be one frame.
    """

    def _records_both() -> bool:
        renderings = [
            location.rendering
            for location in deps.snapshot_reader.read_artifact_locations(
                spec_id, current_content_version
            )
        ]
        return minting in recorded_mintings(renderings) and records_plain_file(renderings)

    def _render_both(coverage_snapshot: CoverageSnapshot) -> bool:
        """Render what the current version lacks and say whether it then records both."""
        summed = _rendering_frame(
            spec_id=spec_id,
            head_content_version=current_content_version,
            minting=minting,
            deps=deps,
            trace_id=trace_id,
            coverage_snapshot=coverage_snapshot,
        )
        if summed is None or summed.content_version != current_content_version:
            return False
        head_rendering(
            subject_id=spec_id,
            content_version=current_content_version,
            deps=deps,
            options=_frame_options(None),
            trace_id=trace_id,
            coverage_snapshot=coverage_snapshot,
        )
        return _records_both()

    if _records_both():
        return
    if _render_both(deps.coverage.snapshot()):
        return
    raise RenderingNotRecorded(spec_id=spec_id, content_version=content_version, ambiguous=True)


def _refresh_frame(
    *, spec_id: str, content_version: str, deps: DefinitionDeps, trace_id: str | None
) -> FrameCurrent | FrameMoved | FramePending:
    """Answer the refresh of one dataset frame. Runs on the refresh door's executor.

    The named frame's records are read under the id that names it, so a frame recorded under a
    posted id that is not its served id keeps resolving. The head, any rebuild, and the answer
    use the served id (`kinds.spec_id.resolve_served_definition`), so such a frame answers
    `moved` to its served dataset's current frame."""
    with phase("resolve"):
        served = _served_spec(spec_id=spec_id, deps=deps, trace_id=trace_id)
        rendering = stored_rendering(
            spec_id=spec_id, content_version=content_version, snapshot_reader=deps.snapshot_reader
        )
    named_spec_id, spec_id = spec_id, served.spec_id
    # An ambiguous frame is served as the plain rendering, and `_answer` then checks the others.
    minting = None if rendering.ambiguous else rendering.minting
    options = _frame_options(minting)

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
        return _rendering_frame(
            spec_id=spec_id,
            head_content_version=head_content_version,
            minting=minting,
            deps=deps,
            trace_id=trace_id,
            coverage_snapshot=coverage_snapshot,
        )

    def _answer(current_content_version: str) -> FrameCurrent | FrameMoved:
        frame = FramePair(spec_id=spec_id, content_version=current_content_version)
        if (spec_id, current_content_version) == (named_spec_id, content_version):
            return FrameCurrent(frame=frame)
        if rendering.ambiguous:
            for recorded in rendering.mintings:
                _refuse_differing_renderings(
                    spec_id=spec_id,
                    content_version=content_version,
                    current_content_version=current_content_version,
                    minting=recorded,
                    deps=deps,
                    trace_id=trace_id,
                )
        return FrameMoved(frame=frame)

    # A second pass runs only after this call failed a done operation whose file is gone.
    for _ in range(2):
        try:
            manifest = _serve_spec(
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
        # The operation's options are the rendering's, so its version is the current frame.
        if operation.state == "failed":
            raise RefreshRebuildFailed(
                operation_id=operation.id, refusal_code=operation.refusal_code
            )
        if operation.state != "done" or operation.content_version is None:
            return FramePending(
                operation_id=operation.id, retry_after_s=OPERATION_SUBMIT_RETRY_AFTER_MS / 1000
            )
        if rendering_servable(
            subject_id=spec_id,
            content_version=operation.content_version,
            rendering_key=artifact_rendering_key(options, site=deps.settings.site),
            snapshot_reader=deps.snapshot_reader,
            now=deps.now,
            settings=deps.settings,
        ):
            return _answer(operation.content_version)
        # The file has left storage. Failing the row makes the next submit re-run it.
        deps.operation_store.fail_unservable_done(operation.id, attempt=operation.attempt_count)
    raise RefreshRebuildUnservable(operation_id=operation.id)


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
                _refresh_frame,
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
        # `_refresh_frame` serves the frame's dataset as the definition door serves a registered
        # id, rendering the frame's parquet and its groupby sum.
        exception_types=(
            *SERVING_EXCEPTION_TYPES,
            ParquetSizeBoundError,
            RollupNotSummableError,
            FrameNotRefreshable,
            RenderingNotRecorded,
            RefreshRebuildFailed,
            RefreshRebuildUnservable,
            RefreshBackpressure,
            RefreshNotPermitted,
        ),
        exception_exemplars=(
            FrameNotRefreshable(spec_id="ops_exemplar"),
            RenderingNotRecorded(spec_id="spec_exemplar", content_version="cv_exemplar"),
            RefreshRebuildFailed(operation_id="op_exemplar", refusal_code="over_ceiling"),
            RefreshRebuildUnservable(operation_id="op_exemplar"),
            RefreshBackpressure(),
            RefreshNotPermitted(presented=False),
        ),
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
    "RefreshRebuildUnservable",
    "RenderingNotRecorded",
    "StoredRendering",
    "build_refresh_door",
    "classify_refresh_exception",
    "refresh_can_answer",
    "stored_rendering",
]
