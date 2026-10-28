"""What a share's slot serves: its data on the right frame, and the history of frames served.

A snapshot slot serves the frame its share revision records for the card. A live slot serves the
newest data its card's intent can be drawn on. It asks `NewestFrames.newest` for the frame of
the card's recorded frame at most once per `FRESHNESS_WINDOW` and keeps the answer as the
slot's `SlotCheck` (`presentation_store.shares`), which the reads in between serve. A check
states the rows:

- `current`: the frame the check found, which is the card's own frame until its data moves;
- `pending`: newer data is building as a queued operation, and the slot serves the frame of its
  last check until a check after the window finds it built;
- `stale`: the check failed, the card cannot be drawn on the newer frame, or no Datasets surface
  answers in this app; the slot serves the frame of its last check, or the card's own frame
  when the newer one cannot be drawn.

A snapshot slot states `current`, since its rows are the share's for good.

The rows are the card's data export (`assembler.pipeline.export_component.export_component_data`,
unbundled), on the served frame as a target when it is not the frame the card revision is
stored on. `format=json` turns the Parquet export into a JSON document of rows, and `csv` and
`parquet` serve the export's bytes. Every read records the frame it served in `served_frame`
with the card revision's recorded rendering.

The history lists those records newest first, each with the retention of its frame counted
from when DFO stored it (`presentation_api.retention.frame_retention`) and whether that
retention still holds: pinned, or held past now. A frame with no version record states no
retention and is not fetchable.
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime
from typing import TYPE_CHECKING, Literal

import pyarrow.parquet as pq

from assembler.pipeline.export_component import (
    ComponentExportFile,
    ComponentHoldsNoData,
    export_component_data,
)
from assembler.pipeline.stored_revision import read_stored_revision
from assembler.ports import FrameRef, RevisionRecord
from datasets_api.retention import RetentionUnreadable
from presentation_api.newest_frames import FRESHNESS_WINDOW, NewestPending
from presentation_api.retention import frame_retention
from presentation_store import Freshness, ServedFrame, ShareCard, ShareRecord, SlotCheck

if TYPE_CHECKING:
    from presentation_api.doors.shares import SharesDeps

__all__ = [
    "ShareSlotEmpty",
    "SlotData",
    "SlotFormat",
    "history_document",
    "serve_slot",
    "serving_frame",
    "slot_card",
]

logger = logging.getLogger(__name__)

type SlotFormat = Literal["json", "csv", "parquet"]

_JSON_MEDIA_TYPE = "application/json"


class ShareSlotEmpty(Exception):
    """The share has no slot at this number, or the slot holds no card. Maps to 404
    `not_found`."""

    def __init__(self, *, share_id: str, slot: int) -> None:
        self.share_id = share_id
        self.slot = slot
        super().__init__(f"share {share_id!r} shows no card in slot {slot}")


@dataclass(frozen=True)
class SlotData:
    """One read of a slot: the body, its media type and file name (`None` for JSON), the frame
    served, its freshness, when a live slot's check ran (`None` for a snapshot), and the
    body's strong ETag."""

    body: bytes
    media_type: str
    file_name: str | None
    frame: FrameRef
    freshness: Freshness
    checked_at: datetime | None
    etag: str


def slot_card(share: ShareRecord, slot: int) -> ShareCard:
    """The card slot `slot` of `share` shows.

    Raises:
        ShareSlotEmpty: No such slot, or it holds no card.
    """
    if 1 <= slot <= len(share.slots):
        card = share.slots[slot - 1].card
        if card is not None:
            return card
    raise ShareSlotEmpty(share_id=share.id, slot=slot)


def serving_frame(*, share: ShareRecord, slot: int, card: ShareCard, deps: SharesDeps) -> FrameRef:
    """The frame slot `slot` serves now, with no check: a live slot's last check, else the
    frame its revision records. `card.frame` must be set."""
    assert card.frame is not None
    if share.mode == "live":
        check = deps.share_store.slot_check(share_id=share.id, slot=slot, card=card)
        if check is not None:
            return check.frame
    return card.frame


def serve_slot(*, share: ShareRecord, slot: int, format: SlotFormat, deps: SharesDeps) -> SlotData:
    """Serve slot `slot` of `share` in `format` and record the frame served.

    Raises:
        ShareSlotEmpty: No such slot, or it holds no card.
        ComponentHoldsNoData: The card is a note or shows no data values.
        Any refusal of the card's export on the frame it serves.
    """
    card = slot_card(share, slot)
    record = read_stored_revision(
        component_id=card.component_id, revision=card.revision, revision_store=deps.revision_store
    )
    if card.frame is None:
        raise ComponentHoldsNoData(
            component_id=card.component_id,
            revision=card.revision,
            detail=f"a {record.stored_intent['type']} draws no frame, so it has no rows to serve",
        )
    if share.mode == "snapshot":
        frame, freshness, checked_at = card.frame, "current", None
        exported = _export(card, frame=frame, record=record, format=format, deps=deps)
    else:
        check = _live_check(share=share, slot=slot, card=card, deps=deps)
        try:
            exported = _export(card, frame=check.frame, record=record, format=format, deps=deps)
        except Exception:
            if check.frame == card.frame:
                raise
            # The newer frame no longer fits the card's intent, or cannot be read. The slot
            # serves the card's own frame, which its share pinned, until the next check.
            logger.warning(
                "Share %s slot %s cannot draw %s on %s; serving its own frame",
                share.id,
                slot,
                card.component_id,
                check.frame,
                exc_info=True,
            )
            check = SlotCheck(frame=card.frame, freshness="stale", checked_at=check.checked_at)
            deps.share_store.record_slot_check(share_id=share.id, slot=slot, card=card, check=check)
            exported = _export(card, frame=card.frame, record=record, format=format, deps=deps)
        frame, freshness, checked_at = check.frame, check.freshness, check.checked_at
    deps.share_store.record_served(
        share_id=share.id, slot=slot, card=card, frame=frame, rendering=record.frame_rendering
    )
    if format == "json":
        body = _json_document(
            share=share,
            slot=slot,
            card=card,
            frame=frame,
            freshness=freshness,
            parquet=exported.data,
        )
        media_type, file_name = _JSON_MEDIA_TYPE, None
    else:
        body, media_type, file_name = exported.data, exported.media_type, exported.file_name
    digest = hashlib.sha256(media_type.encode() + b"\0" + body).hexdigest()
    return SlotData(
        body=body,
        media_type=media_type,
        file_name=file_name,
        frame=frame,
        freshness=freshness,
        checked_at=checked_at,
        etag=f'"{digest[:32]}"',
    )


def history_document(
    *, share_id: str, served: Sequence[ServedFrame], deps: SharesDeps
) -> dict[str, object]:
    """The history route's body: every frame `served`, in the order given, with its retention."""
    now = deps.now()
    entries: list[dict[str, object]] = []
    for entry in served:
        try:
            retention = frame_retention(entry.frame, snapshot_reader=deps.snapshot_reader)
        except RetentionUnreadable:
            retention = None
        entries.append(
            {
                "slot": entry.slot,
                "component_id": entry.component_id,
                "revision": entry.revision,
                "frame": {
                    "spec_id": entry.frame.spec_id,
                    "content_version": entry.frame.content_version,
                },
                "rendering": dict(entry.rendering) if entry.rendering is not None else None,
                "first_served_at": entry.first_served_at.isoformat(),
                "retention": retention,
                "fetchable": _fetchable(retention, now=now),
            }
        )
    return {"id": share_id, "served": entries}


def _fetchable(retention: Mapping[str, object] | None, *, now: datetime) -> bool:
    if retention is None:
        return False
    if retention.get("pinned") is True:
        return True
    held_until = retention.get("held_until")
    return isinstance(held_until, str) and datetime.fromisoformat(held_until) > now


def _live_check(*, share: ShareRecord, slot: int, card: ShareCard, deps: SharesDeps) -> SlotCheck:
    """The slot's check, run again when the last one is older than `FRESHNESS_WINDOW`. Of two
    reads finding it due, the one whose claim moves its time runs it, and the other serves the
    last answer."""
    assert card.frame is not None
    store = deps.share_store
    now = deps.now()
    due_before = now - FRESHNESS_WINDOW
    prior = store.slot_check(share_id=share.id, slot=slot, card=card)
    if prior is not None:
        if prior.checked_at > due_before:
            return prior
        if not store.claim_slot_check(
            share_id=share.id, slot=slot, card=card, now=now, due_before=due_before
        ):
            return store.slot_check(share_id=share.id, slot=slot, card=card) or prior
    last = prior.frame if prior is not None else card.frame
    check = _check(card.frame, last=last, now=now, deps=deps, label=f"{share.id} slot {slot}")
    store.record_slot_check(share_id=share.id, slot=slot, card=card, check=check)
    return check


def _check(
    frame: FrameRef, *, last: FrameRef, now: datetime, deps: SharesDeps, label: str
) -> SlotCheck:
    """Ask for the newest frame standing for `frame`, serving `last` when none answers."""
    if deps.newest is None:
        logger.error("Share %s cannot check for newer data: no Datasets surface answers", label)
        return SlotCheck(frame=last, freshness="stale", checked_at=now)
    try:
        answer = deps.newest.newest(frame)
    except Exception:
        # Every refusal and failure of the refresh or the run leaves the slot on its last frame,
        # which the reply states as stale, until the next check.
        logger.warning("Share %s newer-data check of %s failed", label, frame, exc_info=True)
        return SlotCheck(frame=last, freshness="stale", checked_at=now)
    if isinstance(answer, NewestPending):
        return SlotCheck(frame=last, freshness="pending", checked_at=now)
    return SlotCheck(frame=answer, freshness="current", checked_at=now)


def _export(
    card: ShareCard,
    *,
    frame: FrameRef,
    record: RevisionRecord,
    format: SlotFormat,
    deps: SharesDeps,
) -> ComponentExportFile:
    """The card's unbundled export on `frame`, which is a target unless the card revision is
    stored on it. JSON reads the Parquet export."""
    target = None if record.stored_intent.get("frame_ref") == _frame_body(frame) else frame
    return export_component_data(
        component_id=card.component_id,
        revision=card.revision,
        format="parquet" if format == "json" else format,
        bundle=False,
        revision_store=deps.revision_store,
        frame_store=deps.frame_store,
        site_url=deps.site.public_site_url,
        target_frame=target,
        release_registry=deps.release_registry if target is not None else None,
    )


def _frame_body(frame: FrameRef) -> dict[str, str]:
    return {"spec_id": frame.spec_id, "content_version": frame.content_version}


def _json_document(
    *,
    share: ShareRecord,
    slot: int,
    card: ShareCard,
    frame: FrameRef,
    freshness: Freshness,
    parquet: bytes,
) -> bytes:
    """The JSON body: the slot, its card and frame, the freshness, and the Parquet export's rows
    as objects, a NaN as `null` and a date as ISO 8601 text."""
    rows = pq.read_table(io.BytesIO(parquet)).to_pylist()
    document = {
        "share_id": share.id,
        "slot": slot,
        "mode": share.mode,
        "freshness": freshness,
        "component_id": card.component_id,
        "revision": card.revision,
        "frame": _frame_body(frame),
        "rows": [
            {
                column: None if isinstance(value, float) and math.isnan(value) else value
                for column, value in row.items()
            }
            for row in rows
        ],
    }
    return json.dumps(
        document, default=_json_cell, ensure_ascii=False, allow_nan=False, separators=(",", ":")
    ).encode()


def _json_cell(value: object) -> object:
    if isinstance(value, date | datetime):
        return value.isoformat()
    raise TypeError(f"a {type(value).__name__} cell has no JSON form")
