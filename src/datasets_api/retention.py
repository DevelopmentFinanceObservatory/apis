"""The retention a Manifest reports: how long its version can still be fetched.

Every file of a version renders from one stored frame: its own, or, for a derived version
(`StoredVersion.base_content_version` set: a groupby collapse or a party-totals cut), its base
version's. The frame is the one `retained_frame_location` gives, which the pin door also
promotes. A pinned frame has no expiry. A working frame is guaranteed until
`WORKING_TIER_RETENTION` after its `stored_at`. A download renews only the file it names
(`doors.files.renew_and_resolve`), so only a download of the frame itself moves that clock.
Past that point the frame also renews when a render from it re-uploads its bytes
(`doors.files.render_version_from_frame`): the file selector rendering a file the version holds
no surviving copy of, or a link click whose own file's bytes are gone. When the frame's bytes
are gone too, both refuse `artifact_expired`. A definition GET over an aged frame and a mint build
from upstream instead, and a build of unchanged data uploads the frame again, restarting its
clock. The snapshot and observations routes refuse `artifact_expired` once no listed file of
the version is in retention, whatever the frame's bytes.

A version that links no frame renders nothing, so it reports over its own listed data artifacts
instead: pinned when one is pinned, else the latest of their 30-day guarantees.

Each Manifest door sets `Manifest.retention` on its response inside the executor call that
builds it: the mint, definition, snapshot and operation doors. No stored body holds it, since
the field defaults to `None` and `manifest.version_document` leaves it out.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from datetime import datetime
from typing import cast

from artifact_store import Tier
from dataset_store.clock import Clock
from dataset_store.snapshot import (
    WORKING_TIER_RETENTION,
    SnapshotReader,
    StoredArtifactLocation,
)
from datasets_api.artifact_delivery import LISTABLE_MEDIA_TYPES
from datasets_api.manifest_models import Manifest, RetentionModel
from datasets_api.render import (
    is_statistics_rendering,
    servable_frame_location,
    stored_frame_location,
)
from datasets_api.timings import phase


class RetentionUnreadable(Exception):
    """The version a door just served has no record, or holds neither a frame nor a listed data
    artifact, so no retention can be stated. A door maps it to `internal_error`."""

    def __init__(self, *, subject_id: str, content_version: str, reason: str) -> None:
        self.subject_id = subject_id
        self.content_version = content_version
        super().__init__(
            f"no retention for subject_id {subject_id!r} content_version {content_version!r}: "
            f"{reason}"
        )


def _retention_of(locations: Sequence[StoredArtifactLocation]) -> RetentionModel:
    if any(location.tier == Tier.pinned.value for location in locations):
        return RetentionModel(pinned=True)
    stored_at = max(location.stored_at for location in locations)
    return RetentionModel(pinned=False, held_until=stored_at + WORKING_TIER_RETENTION)


def retained_frame_location(
    locations: Sequence[StoredArtifactLocation], *, now: datetime
) -> StoredArtifactLocation | None:
    """The frame among a version's `locations` whose retention the version reports: its servable
    frame (`render.servable_frame_location`), so a pinned copy wins, else its stored frame over
    aged locations (`render.stored_frame_location`), the one a renewal would renew."""
    return servable_frame_location(locations, now=now) or stored_frame_location(locations)


def version_retention(
    reader: SnapshotReader, *, subject_id: str, content_version: str, now: datetime
) -> RetentionModel:
    """The retention of the version at `content_version` under `subject_id`, its frame judged
    servable at `now`. One transaction reads the version and its locations, and a derived
    version's base locations take a second.

    Raises `RetentionUnreadable` when the version has no record, or links neither a frame nor
    a listed data artifact."""
    state = reader.read_version_state(subject_id, content_version)
    if state is None:
        raise RetentionUnreadable(
            subject_id=subject_id, content_version=content_version, reason="no version record"
        )
    base_content_version = state.version.base_content_version
    locations = (
        state.locations
        if base_content_version is None
        else reader.read_artifact_locations(subject_id, base_content_version)
    )
    frame = retained_frame_location(locations, now=now)
    if frame is not None:
        return _retention_of([frame])

    data = [
        location
        for location in state.locations
        if location.media_type in LISTABLE_MEDIA_TYPES
        and not is_statistics_rendering(location.rendering)
    ]
    if not data:
        raise RetentionUnreadable(
            subject_id=subject_id,
            content_version=content_version,
            reason="it links no frame and no data artifact",
        )
    return _retention_of(data)


def with_retention(
    manifest: Manifest, *, subject_id: str, reader: SnapshotReader, now: datetime
) -> Manifest:
    """`manifest` with the retention of its version, recorded under `subject_id`, read under
    the `retention` timing phase."""
    with phase("retention"):
        retention = version_retention(
            reader, subject_id=subject_id, content_version=manifest.content_version, now=now
        )
    return manifest.model_copy(update={"retention": retention})


def retaining[T](fn: Callable[..., T], *, reader: SnapshotReader, now: Clock) -> Callable[..., T]:
    """`fn`, whose Manifest results carry their version's retention (`with_retention`), read
    under the manifest's `definition.spec_id`, the id every version a definition build serves
    is recorded under. Other results pass through unchanged."""

    def call(*args: object, **kwargs: object) -> T:
        result = fn(*args, **kwargs)
        if not isinstance(result, Manifest):
            return result
        retained = with_retention(
            result, subject_id=result.definition.spec_id, reader=reader, now=now()
        )
        return cast(T, retained)

    return call
