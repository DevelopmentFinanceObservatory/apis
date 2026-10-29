"""The retention a Presentation reply states for the frame it draws.

A card draws its version's own stored frame, so the retention stated is that frame's, read as
`presentation_store.LiveFrameStore.load` reads it: one `read_servable_version` transaction,
which refuses a revoked version or a revoked base version, then the version's servable stored
frame (`datasets_api.render.servable_frame_location`). A pinned frame states `pinned`. A
working frame states `held_until`, `WORKING_TIER_RETENTION` after DFO stored it. A frame that
aged out of the working tier states the `held_until` that has passed, so the reply says when it
stopped loading. A collapse or cut version states its own frame's retention and never its base
version's, since a card loads only its own frame.

Retention is advisory on a read, a create, a render and a share history: a reply states `None`
when the version has no record, is revoked, links no stored frame, or the Datasets store cannot
answer, and answers the card as stored. A revocation under a served id the frame ref does not
name is checked through the spec registry, which these doors do not read, so retention states
such a frame while its load refuses `revoked`.
"""

from __future__ import annotations

import logging

from artifact_store import Tier
from assembler.ports import FrameRef
from dataset_store.clock import utc_now
from dataset_store.snapshot import (
    WORKING_TIER_RETENTION,
    SnapshotNotFound,
    SnapshotReader,
    SnapshotRevoked,
    read_servable_version,
)
from datasets_api.manifest_models import RetentionModel
from datasets_api.render import servable_frame_location, stored_frame_location
from datasets_api.retention import RetentionUnreadable

__all__ = ["frame_retention", "stated_retention"]

logger = logging.getLogger(__name__)


def frame_retention(
    frame_ref: FrameRef, *, snapshot_reader: SnapshotReader | None
) -> dict[str, object] | None:
    """The wire retention of the stored frame `frame_ref` names, by the rule the module
    docstring states, or `None` for a door with no snapshot reader.

    Raises:
        RetentionUnreadable: The version has no record, is revoked, or links no stored frame.
    """
    if snapshot_reader is None:
        return None
    now = utc_now()
    try:
        servable = read_servable_version(
            snapshot_reader,
            subject_id=frame_ref.spec_id,
            content_version=frame_ref.content_version,
            now=now,
        )
    except (SnapshotNotFound, SnapshotRevoked) as exc:
        raise RetentionUnreadable(
            subject_id=frame_ref.spec_id,
            content_version=frame_ref.content_version,
            reason=str(exc),
        ) from exc
    frame = servable_frame_location(servable.servable, now=now) or stored_frame_location(
        servable.aged_out
    )
    if frame is None:
        raise RetentionUnreadable(
            subject_id=frame_ref.spec_id,
            content_version=frame_ref.content_version,
            reason="it links no stored frame",
        )
    if frame.tier == Tier.pinned.value:
        retention = RetentionModel(pinned=True)
    else:
        retention = RetentionModel(
            pinned=False, held_until=frame.stored_at + WORKING_TIER_RETENTION
        )
    return retention.model_dump(mode="json")


def stated_retention(
    frame_ref: FrameRef, *, snapshot_reader: SnapshotReader | None
) -> dict[str, object] | None:
    """`frame_retention`, or `None` when it cannot be read for any reason, for a reply that
    states a card's frame rather than refusing over it."""
    try:
        return frame_retention(frame_ref, snapshot_reader=snapshot_reader)
    except RetentionUnreadable:
        return None
    except Exception:
        # Retention is advisory, so a Datasets store that cannot answer leaves the card's
        # reply standing with none, as every other failure to read it does.
        logger.warning(
            "retention of %s at %s could not be read",
            frame_ref.spec_id,
            frame_ref.content_version,
            exc_info=True,
        )
        return None
