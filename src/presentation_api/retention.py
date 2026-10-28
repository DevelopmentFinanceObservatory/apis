"""The retention a Presentation reply states for the frame it draws."""

from __future__ import annotations

from assembler.ports import FrameRef
from dataset_store.clock import utc_now
from dataset_store.snapshot import SnapshotReader
from datasets_api.retention import RetentionUnreadable, version_retention

__all__ = ["frame_retention", "stated_retention"]


def frame_retention(
    frame_ref: FrameRef, *, snapshot_reader: SnapshotReader | None
) -> dict[str, object] | None:
    """The wire retention of `frame_ref`'s version, counted from when DFO stored its frame, or
    `None` for a door with no snapshot reader.

    Raises:
        RetentionUnreadable: The version has no record or links nothing to serve.
    """
    if snapshot_reader is None:
        return None
    retention = version_retention(
        snapshot_reader,
        subject_id=frame_ref.spec_id,
        content_version=frame_ref.content_version,
        now=utc_now(),
    )
    return retention.model_dump(mode="json")


def stated_retention(
    frame_ref: FrameRef, *, snapshot_reader: SnapshotReader | None
) -> dict[str, object] | None:
    """`frame_retention`, or `None` when the version has no record or links nothing to serve,
    for a reply that states a stored card's frame rather than refusing over it."""
    try:
        return frame_retention(frame_ref, snapshot_reader=snapshot_reader)
    except RetentionUnreadable:
        return None
