"""Pins the frames a card revision draws, when a share first includes the card.

A card is created and revised in the working tier, so its frame is held 30 days from when DFO
stored it. A share keeps what it shows, so the share's create calls `pin_card_frames` for each
card revision it includes. The pin goes through `FrameStore.pin` under the component's own
reference (`presentation:<component_id>`), so the reconciliation of orphaned presentation
references covers it, and a derived frame pins with its roots (`LiveFrameStore.pin`).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import cast

from assembler.pipeline.stored_revision import read_stored_revision
from assembler.ports import FrameRef, FrameStore, RevisionStore
from dfo_platform.caller import Caller

__all__ = ["pin_card_frames"]


def pin_card_frames(
    *,
    component_id: str,
    revision: int,
    revision_store: RevisionStore,
    frame_store: FrameStore,
    caller: Caller | None = None,
) -> bool | None:
    """Pin the frame revision `revision` of `component_id` draws, with a derived frame's
    roots, each new pinned copy spending `caller`'s units (`FrameStore.pin`). A note draws no
    frame and pins nothing.

    Returns:
        `frame_store.pin`'s answer: whether a reference was created or reactivated, `None`
        when the store cannot say. `False` for a note.

    Raises:
        DraftNotFound: The store holds no such component.
        RevisionNotFound: The component has a head and no revision with that number.
        EditConflict: The component has no head.
        FrameUnavailable: The frame, or a root of it, cannot be pinned.
    """
    record = read_stored_revision(
        component_id=component_id, revision=revision, revision_store=revision_store
    )
    body = cast("Mapping[str, str] | None", record.stored_intent.get("frame_ref"))
    if body is None:
        return False
    return frame_store.pin(frame_ref=FrameRef(**body), component_id=component_id, caller=caller)
