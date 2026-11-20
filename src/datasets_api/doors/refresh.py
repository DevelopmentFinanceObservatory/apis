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
The refresh makes each of them from the head with the current code index (`_rendering_frame`)
and answers with the one frame they make, linking a file of each under it, so the caller's
frame is that frame whichever rendering it was made as. Once they differ it refuses
`rendering_not_recorded` (422). A head whose stored frame has expired, so that a rendering
cannot be made from it, rebuilds or answers `pending` as for any other frame. With the plain
rendering among them the head is that rendering's frame, so a servable head that cannot make
another rendering refuses `rendering_not_recorded` at once. After a
rebuild the renderings are made from the raw version it materialised, whether or not that
became the head: one that cannot be made, and renderings that make one frame other than the
one the rebuild served, refuse `rendering_not_recorded`. The frame
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
recorded yet, mints it from the head's stored frame as a head-hit groupby read does. A version
is recorded under a minting from the head when a file of that minting linked under it names the
head among those it was made from (`datasets.version_artifact_head`): a record keeps the base
it was first recorded from, and a render from another head that makes the same version links
its file there. A link naming no head, as every link written before links named heads does,
counts for the head only when the version's record names that head as its base and every file
of minting options linked under the version records one minting, since a version's record and
its first link are written in one transaction and no code path deletes a link
(`datasets.version_artifact`), so that minting made it from its base. The named frame answers
when it is recorded under the minting from the head, and otherwise the newest version that is.
A frame of one minting with no party-totals cut that leaves a raw version unchanged answers at
that version's own head with its own stored frame while no link under the version names a head,
since files are then linked under it only from its own frame. A cut is decided by the code
index at each render, so a frame of a cut is made again.

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
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Annotated, Literal, cast

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
from datasets_api.versions import HeadVersion
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
    options. A version recorded under several is `ambiguous` and refreshes as the first of its
    `renderings`: the plain one when recorded, else the first minting options by
    `recorded_rendering.ordered_mintings`. The refresh answers it only while all of them render
    the current data as one frame. The registration of `spec_id` is not checked.

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
    if None in stored.renderings and target.collapsed:
        return False
    return all(minting in target.mintings for minting in stored.mintings)


def _head_mintings(version: HeadVersion, *, head_content_version: str) -> list[MintingOptions]:
    """The minting options the files linked under `version` record as made from
    `head_content_version`: each link naming that head, and each link naming none while
    `version`'s record names that head as its base and every file of minting options linked
    under it records one minting, the rule the module docstring states for links written before
    links named heads."""
    links = version.links
    headless_count = (
        version.base_content_version == head_content_version
        and len(ordered_mintings(recorded_mintings(link.rendering for link in links))) == 1
    )
    return recorded_mintings(
        link.rendering
        for link in links
        if head_content_version in link.heads or (headless_count and not link.heads)
    )


class _DerivedVersions:
    """The dataset's versions made from one head: those a file linked under them, or their
    record, says one of their mintings made from it (`_head_mintings`), read with a query by that
    head (`VersionLister.list_head_versions`) on the first lookup after `reread`, and the version
    this refresh's renders and rebuild last made from it with each minting. A lookup of another
    head starts over. `named` is the refreshed frame's version, which a lookup puts first among
    the versions recorded under a minting, else `None` when the frame is read under another id."""

    def __init__(self, *, spec_id: str, named: str | None, deps: DefinitionDeps) -> None:
        self._spec_id = spec_id
        self._named = named
        self._deps = deps
        self._head: str | None = None
        self._made: dict[MintingOptions, str] = {}
        self._listed: list[tuple[list[MintingOptions], str]] | None = None

    def _at(self, head_content_version: str) -> None:
        if self._head != head_content_version:
            self._head = head_content_version
            self._made = {}
            self._listed = None

    def _recorded(self, head_content_version: str) -> list[tuple[list[MintingOptions], str]]:
        if self._listed is None:
            lister = self._deps.version_lister
            self._listed = (
                [
                    (
                        _head_mintings(version, head_content_version=head_content_version),
                        version.content_version,
                    )
                    for version in lister.list_head_versions(
                        self._spec_id, head_content_version=head_content_version
                    )
                ]
                if lister is not None
                else []
            )
        return self._listed

    def recorded(self, *, base_content_version: str, minting: MintingOptions) -> str | None:
        """The stored version made from `base_content_version` with `minting`: the one this
        refresh last made from it, else the named version when the read records it under
        `minting`, else the newest the read records under it, or `None`."""
        self._at(base_content_version)
        made = self._made.get(minting)
        if made is not None:
            return made
        candidates = [
            cv for mintings, cv in self._recorded(base_content_version) if minting in mintings
        ]
        if self._named in candidates:
            return self._named
        return next(iter(candidates), None)

    def add(
        self, *, base_content_version: str, minting: MintingOptions, content_version: str
    ) -> None:
        """Note that a render or rebuild of this refresh made `content_version` from
        `base_content_version` with `minting`, replacing what it made with `minting` before."""
        self._at(base_content_version)
        self._made[minting] = content_version

    def reread(self) -> None:
        """Read the recorded versions again on the next lookup, keeping what this refresh made,
        so a lookup sees what another caller recorded meanwhile."""
        self._listed = None


@dataclass(frozen=True)
class _Frame:
    """A stored frame with a servable stored frame file, read without its artifacts, and the
    renderings of the files linked under its version when it was read."""

    manifest: Manifest
    renderings: tuple[Mapping[str, object], ...]

    @property
    def content_version(self) -> str:
        return self.manifest.content_version


def _servable_frame(*, spec_id: str, content_version: str, deps: DefinitionDeps) -> _Frame | None:
    """The stored frame at `content_version`, or `None` when no servable stored frame file is
    linked under it. Presentation loads a frame from that file, so an expired one makes the
    frame unloadable there (`missing_rendering`)."""
    locations = deps.snapshot_reader.read_artifact_locations(spec_id, content_version)
    if servable_frame_location(locations, now=deps.now()) is None:
        return None
    return _Frame(
        manifest=load_stored(
            subject_id=spec_id, content_version=content_version, deps=deps, include_artifacts=False
        ),
        renderings=tuple(location.rendering for location in locations),
    )


def _rendering_frame(
    *,
    spec_id: str,
    head_content_version: str,
    minting: MintingOptions | None,
    derived: _DerivedVersions,
    deps: DefinitionDeps,
    trace_id: str | None,
    coverage_snapshot: CoverageSnapshot,
) -> _Frame | None:
    """The stored frame of the `minting` rendering under the head, or of the plain rendering
    when `minting` is `None`, answered only while its stored frame file is servable
    (`_servable_frame`). The plain frame is the head. A derived rendering is the derived version
    on record under the head (`derived`); with none on record, or with that version's frame file
    expired, the rendering is made from the head's stored frame, as a head-hit groupby read makes
    it. A version on record replays the party-totals cut its trace records
    (`recorded_rendering.recorded_party_cut`), so it renders to its own version whatever the code
    index and the cut's rule now decide; with none on record the cut is derived afresh. A
    rendering with a version of its own is recorded under it with its canonical frame, and one
    that leaves the frame unchanged is linked under the head. `None` sends the caller to a
    rebuild: the head's frame file has expired or cannot render it."""
    if minting is None:
        return _servable_frame(spec_id=spec_id, content_version=head_content_version, deps=deps)
    recorded = derived.recorded(base_content_version=head_content_version, minting=minting)
    recorded_cut = None
    if recorded is not None:
        stored = _servable_frame(spec_id=spec_id, content_version=recorded, deps=deps)
        if stored is not None:
            return stored
        version = deps.snapshot_reader.read_version(spec_id, recorded)
        if version is not None:
            recorded_cut = recorded_party_cut(version)
    rendered = head_rendering(
        subject_id=spec_id,
        content_version=head_content_version,
        deps=deps,
        options=frame_options(minting),
        trace_id=trace_id,
        coverage_snapshot=coverage_snapshot,
        recorded_cut=recorded_cut,
    )
    if rendered is None:
        return None
    if recorded is None and rendered.content_version != head_content_version:
        derived.add(
            base_content_version=head_content_version,
            minting=minting,
            content_version=rendered.content_version,
        )
    return _servable_frame(spec_id=spec_id, content_version=rendered.content_version, deps=deps)


def _link_heads(*, spec_id: str, content_version: str, deps: DefinitionDeps) -> frozenset[str]:
    """Every head a link under the version names (`VersionLister.link_heads`), none without a
    version lister."""
    lister = deps.version_lister
    if lister is None:
        return frozenset()
    return lister.link_heads(spec_id, content_version=content_version)


def _records_every(
    renderings: Iterable[Mapping[str, object]],
    *,
    rendering: StoredRendering,
    base_content_version: str,
    collapsed: bool,
) -> bool:
    """Whether a version linking files of `renderings` is one the refresh of `rendering` answers
    with (`refresh_can_answer`), derived from `base_content_version` when `collapsed`, and links
    a plain file when the plain rendering is one of `rendering`'s, so every rendering it is
    recorded under makes that version."""
    renderings = list(renderings)
    target = StoredRendering(
        base_content_version=base_content_version,
        collapsed=collapsed,
        mintings=tuple(recorded_mintings(renderings)),
    )
    return refresh_can_answer(stored=rendering, target=target) and (
        None not in rendering.renderings or records_plain_file(renderings)
    )


def _agreed_frame(
    *,
    spec_id: str,
    content_version: str,
    head_content_version: str,
    rendering: StoredRendering,
    derived: _DerivedVersions,
    deps: DefinitionDeps,
    trace_id: str | None,
    coverage_snapshot: CoverageSnapshot,
) -> _Frame | None:
    """The stored frame every rendering `rendering` is recorded under makes from the head, with
    a file of each linked under it, or `None` when one of them cannot be made from the head's
    frame (`_rendering_frame`), which sends the caller to a rebuild. `content_version` is the
    named frame's.

    Every rendering is made from the head with the current code index, the first being the one
    the frame refreshes as; a frame recorded under one rendering is that rendering's frame. With
    several, each rendering is looked up by the head that made it (`_DerivedVersions`) even when
    its file is already linked under the first one's frame, since that file may have been made
    from another head. A plain file is linked under the head unless one is, and the frame
    answers once every rendering made the same version and that version links a file of each.
    Each check reads the files the frame's own read listed. When the plain rendering is one of
    them, the head is servable, so a rendering that cannot be made from it refuses rather than
    rebuilds.

    Raises:
        RenderingNotRecorded: The renderings make different frames, their files did not all
            link under the one they make, or, with the plain rendering among them, one cannot
            be made from the head.
    """
    renderings = rendering.renderings

    def frame_of(minting: MintingOptions | None) -> _Frame | None:
        return _rendering_frame(
            spec_id=spec_id,
            head_content_version=head_content_version,
            minting=minting,
            derived=derived,
            deps=deps,
            trace_id=trace_id,
            coverage_snapshot=coverage_snapshot,
        )

    differ = RenderingNotRecorded(
        spec_id=spec_id,
        content_version=content_version,
        renderings=len(renderings),
        plain=None in renderings,
    )
    unmade = RenderingNotRecorded(
        spec_id=spec_id,
        content_version=content_version,
        renderings=len(renderings),
        unmade=True,
        plain=None in renderings,
    )
    frame = frame_of(renderings[0])
    if frame is None or len(renderings) == 1:
        return frame
    linked = list(frame.renderings)
    for other in renderings[1:]:
        made = frame_of(other)
        if made is None:
            # With the plain rendering first, `frame` is the servable head, so the rendering
            # cannot be made from a frame a rebuild would only make again.
            if None in renderings:
                raise unmade
            return None
        if made.content_version != frame.content_version:
            raise differ
        # A later read lists every file linked under the version by the time it ran.
        linked = list(made.renderings)
    # The plain rendering, when recorded, is the first, so `frame` is the head.
    if None in renderings and not records_plain_file(frame.renderings):
        plain = head_rendering(
            subject_id=spec_id,
            content_version=head_content_version,
            deps=deps,
            options=frame_options(None),
            trace_id=trace_id,
            coverage_snapshot=coverage_snapshot,
        )
        if plain is not None:
            linked += [artifact.rendering for artifact in plain.artifacts]
    if not _records_every(
        linked,
        rendering=rendering,
        base_content_version=head_content_version,
        collapsed=frame.content_version != head_content_version,
    ):
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
    one `stored_rendering` reads today.

    Raises:
        RenderingNotRecorded: The renderings the frame is recorded under make different
            frames from the current data, or, after a rebuild, one of them cannot be made from
            the raw version it materialised, or they agree on a version other than the one it
            served.
    """
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
    derived = _DerivedVersions(
        spec_id=spec_id,
        named=content_version if named_spec_id == spec_id else None,
        deps=deps,
    )
    # The (head, version) pairs `_agreed_frame` found every rendering makes from that head, so
    # `_answer` checks only a frame no head read agreed on at the head it was served from.
    agreed: set[tuple[str, str]] = set()

    def _agreed(head_content_version: str, coverage_snapshot: CoverageSnapshot) -> _Frame | None:
        frame = _agreed_frame(
            spec_id=spec_id,
            content_version=content_version,
            head_content_version=head_content_version,
            rendering=rendering,
            derived=derived,
            deps=deps,
            trace_id=trace_id,
            coverage_snapshot=coverage_snapshot,
        )
        if frame is not None:
            agreed.add((head_content_version, frame.content_version))
        return frame

    def _load_head_frame(
        head_content_version: str, coverage_snapshot: CoverageSnapshot
    ) -> Manifest | None:
        """The stored frame of this rendering under the head, read without its artifacts. The
        named frame answers while the head made it with each rendering (`_DerivedVersions`)."""
        if (
            not rendering.ambiguous
            and minting is not None
            and not minting.without_party_totals
            and not rendering.collapsed
            and named_spec_id == spec_id
            and head_content_version == content_version
            and not _link_heads(spec_id=spec_id, content_version=content_version, deps=deps)
        ):
            # The named raw version is the head, and with no link under it naming a head its
            # files were linked only from its own frame, so the minting leaves this head
            # unchanged.
            stored = _servable_frame(spec_id=spec_id, content_version=content_version, deps=deps)
            if stored is not None:
                return stored.manifest
        # A head read after another caller's rebuild sees what that rebuild recorded.
        derived.reread()
        frame = _agreed(head_content_version, coverage_snapshot)
        return frame.manifest if frame is not None else None

    # The raw version this call's rebuild materialised, set by `serve_spec`.
    materialised: str | None = None

    def _note_materialised(content_version: str) -> None:
        nonlocal materialised
        materialised = content_version

    def _answer(current_content_version: str) -> FrameCurrent | FrameMoved:
        # A frame recorded under several renderings answers `current` only once every
        # rendering makes the named version, as it answers `moved` only once they agree.
        if rendering.ambiguous and materialised is None:
            # With no rebuild of this call, a head read answered, which agreed on the frame; a
            # frame none agreed on is not established as one rendering.
            if not any(version == current_content_version for _, version in agreed):
                raise RenderingNotRecorded(
                    spec_id=spec_id,
                    content_version=content_version,
                    renderings=len(rendering.renderings),
                    plain=None in rendering.renderings,
                )
        elif rendering.ambiguous and (materialised, current_content_version) not in agreed:
            # A rebuild of this call served the frame, so the renderings are checked against
            # the raw version it materialised, whether or not that became the head, unless a
            # head read agreed on the frame there.
            head_content_version = cast(str, materialised)
            derived.reread()
            if minting is not None and current_content_version != head_content_version:
                derived.add(
                    base_content_version=head_content_version,
                    minting=minting,
                    content_version=current_content_version,
                )
            made = _agreed(head_content_version, deps.coverage.snapshot())
            if made is None or made.content_version != current_content_version:
                raise RenderingNotRecorded(
                    spec_id=spec_id,
                    content_version=content_version,
                    renderings=len(rendering.renderings),
                    agreed_on=made.content_version if made is not None else None,
                    unmade=made is None,
                    plain=None in rendering.renderings,
                )
        frame = FramePair(spec_id=spec_id, content_version=current_content_version)
        if (spec_id, current_content_version) == (named_spec_id, content_version):
            return FrameCurrent(frame=frame)
        return FrameMoved(frame=frame)

    try:
        manifest = serve_spec(
            stored=served,
            options=rendering.options,
            trigger="refresh",
            deps=deps,
            trace_id=trace_id,
            load_head=_load_head_frame,
            on_materialised=_note_materialised,
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
