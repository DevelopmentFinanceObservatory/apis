"""Renders a stored revision's intent on another frame, appending no revision.

`render_component` reads one stored revision, rebuilds its component from that revision's
`stored_intent` on a target frame through `_build_candidate`, and stamps the envelope at the
stored revision's own number. Its `provenance.inputs` names the target frame and
`provenance.refreshed_from` names the stored revision. Its description, where the target
frame's version holds a record, is composed for the request from today's words, and carries
`rendering: "live"` and the target frame. The revision store is only read, so
the stored revision and the head stay as they were. The target frame is not pinned, so a
live render leaves it in the working tier.

The stored intent is validated against this deployment's catalogue release, and the envelope
and fact summary are minted under it, whichever release the stored revision was minted under.
The stored payload is not read, so a revision whose payload shape an older release wrote
renders too.

The stored select binds the target frame by `assembler.rolling.StoredSelect`'s rules, so a
series the target no longer holds is dropped and named where a write refuses it.

`render_component` does not decide whether the target frame may stand in for the stored one.
The caller accepts the frame first. `render_on_frame` is the shared core that
`assembler.pipeline.export_component` uses to export a rendering's rows.
"""

from __future__ import annotations

import dataclasses
import datetime
import functools
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import cast

from assembler.catalogue import CatalogueRelease
from assembler.mint import DescribedRevision, MintedComponent, stamp_envelope
from assembler.mint.provenance import _refreshed_provenance
from assembler.pipeline.candidate import _build_candidate, _check_envelope_size
from assembler.pipeline.stored_revision import read_stored_revision
from assembler.pipeline.write import (
    _require_release_not_behind,
    _series_coordinates_for_storage,
    _utc_now,
)
from assembler.ports import (
    TRANSIENT_FRAME_REASONS,
    FrameRef,
    FrameStore,
    FrameUnavailable,
    ReleaseRegistry,
    ResolvedFrame,
    RevisionRecord,
    RevisionStore,
    VocabularySource,
)
from assembler.reply import WriteReply, build_reply


class ComponentHoldsNoFrame(Exception):
    """The revision binds no frame, so it cannot be rendered on one: a note."""

    def __init__(self, *, component_id: str, revision: int, detail: str) -> None:
        self.component_id = component_id
        self.revision = revision
        self.detail = detail
        super().__init__(detail)


@dataclass(frozen=True)
class Rendering:
    """A stored revision's intent rendered on `resolved_frame` under `release`.

    `series_coordinates` holds one `{id, coordinates}` entry per series of `minted`, as a
    stored revision's `series_coordinates` does."""

    release: CatalogueRelease
    stored_intent: Mapping[str, object]
    resolved_frame: ResolvedFrame
    minted: MintedComponent
    series_coordinates: tuple[dict[str, object], ...]


def render_on_frame(
    *,
    record: RevisionRecord,
    target_frame: FrameRef,
    frame_store: FrameStore,
    release_registry: ReleaseRegistry,
    vocabulary_source: VocabularySource | None = None,
    clock: Callable[[], datetime.datetime] = _utc_now,
) -> Rendering:
    """Rebuild `record`'s component from its stored intent on `target_frame`.

    Reads `target_frame` through `frame_store.load`, and the frame `record` was written on
    when a select binding misses a value of `target_frame`. Writes nothing. The minted
    provenance names `target_frame` in `inputs` and `record.revision` in `refreshed_from`.
    The caller has already accepted `target_frame` as a stand-in for the stored frame.
    With a `vocabulary_source` the rendering carries its live description where the card
    has one; without, it carries none.

    Raises as `render_component` does, apart from the revision lookup."""
    member = cast(str, record.stored_intent["type"])
    if member == "note_narration":
        raise ComponentHoldsNoFrame(
            component_id=record.component_id,
            revision=record.revision,
            detail=f"a {member} binds no frame, so it has no rendering on another one",
        )
    release = release_registry.pinned()
    _require_release_not_behind(head=record, release=release)
    resolved_frame = frame_store.load(frame_ref=target_frame)
    stored_ref = FrameRef(**cast("Mapping[str, str]", record.stored_intent["frame_ref"]))

    @functools.cache
    def load_stored_frame() -> ResolvedFrame | None:
        if stored_ref == target_frame:
            return resolved_frame
        try:
            return frame_store.load(frame_ref=stored_ref)
        except FrameUnavailable as exc:
            if exc.reason in TRANSIENT_FRAME_REASONS:
                raise
            return None

    # An empty patch merges to the stored intent. `merge_intent` refuses a patch naming the
    # create-only `type` or `frame_ref`, which the stored intent holds.
    candidate = _build_candidate(
        patch={},
        current=record,
        is_create=False,
        release=release,
        frame_store=frame_store,
        clock=clock,
        resolved_frame=resolved_frame,
        load_stored_frame=load_stored_frame,
        described=(
            DescribedRevision(
                vocabulary_source=vocabulary_source,
                component_id=record.component_id,
                revision=record.revision,
                live=True,
            )
            if vocabulary_source is not None
            else None
        ),
    )
    minted = dataclasses.replace(
        candidate.minted,
        provenance=_refreshed_provenance(
            provenance=candidate.minted.provenance, refreshed_from=record.revision
        ),
    )
    # `_build_candidate` measured the provenance before it named the revision.
    _check_envelope_size(minted=minted)
    return Rendering(
        release=release,
        stored_intent=candidate.stored_intent,
        resolved_frame=resolved_frame,
        minted=minted,
        series_coordinates=_series_coordinates_for_storage(series=candidate.series),
    )


def render_component(
    *,
    component_id: str,
    revision: int,
    target_frame: FrameRef,
    frame_store: FrameStore,
    revision_store: RevisionStore,
    release_registry: ReleaseRegistry,
    vocabulary_source: VocabularySource | None = None,
    clock: Callable[[], datetime.datetime] = _utc_now,
) -> WriteReply:
    """Render revision `revision` of `component_id` on `target_frame`, appending no revision.

    The reply is the one a read of that revision would build, for the rendering: the
    envelope carries `revision`, `provenance.inputs` names `target_frame`, and
    `provenance.refreshed_from` is `revision`. `changed` is empty and no write token is
    returned. The envelope and fact summary are minted under this deployment's release. The
    envelope's description is worded from `vocabulary_source` (`render_on_frame`).

    `target_frame` must already be accepted. This function binds the stored intent to
    whatever frame it is given, and does not compare that frame's dataset, rendering, or
    recipe with the stored frame's. The caller checks that before calling.

    The render pins nothing, so `target_frame` keeps the retention it has.

    Raises `DraftNotFound` for an unknown component, `RevisionNotFound` for a missing
    revision (`EditConflict` when the component has no head), `ComponentHoldsNoFrame` for a
    note, `ReleaseMismatch` when the revision was minted under a newer release than this
    deployment's, and `FrameUnavailable` from the frame load. A stored intent that does not bind `target_frame` raises the
    refusal a write of that intent on that frame raises: `SchemaInvalid`, `BindingInvalid`,
    `ConstraintViolation`, or `ResourceExhausted`."""
    record = read_stored_revision(
        component_id=component_id, revision=revision, revision_store=revision_store
    )
    rendering = render_on_frame(
        record=record,
        target_frame=target_frame,
        frame_store=frame_store,
        release_registry=release_registry,
        vocabulary_source=vocabulary_source,
        clock=clock,
    )
    envelope = stamp_envelope(
        minted=rendering.minted,
        component_id=component_id,
        revision=revision,
        release=rendering.release,
    )
    membership = revision_store.composition_of(component_id=component_id)
    reply = build_reply(
        composition_id=membership.composition_id if membership is not None else None,
        composition_revision=membership.revision if membership is not None else None,
        changed=(),
        stored_intent=rendering.stored_intent,
        resolved_payload=envelope,
        series_coordinates=rendering.series_coordinates,
        formatter_profile=rendering.release.formatter_profile,
    )
    return reply
