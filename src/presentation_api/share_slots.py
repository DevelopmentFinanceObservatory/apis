"""What a share's slot serves: its data on the right frame, and the history of frames served.

A snapshot slot serves the frame its share revision records for the card. A live slot serves the
newest data its card's intent can be drawn on. It asks `NewestFrames.newest` for the frame of
the card's recorded frame and keeps the answer as the slot's `SlotCheck`
(`presentation_store.shares`), which the reads before its `next_check_at` serve. One read claims
a due check and runs it while the others serve the last check's answer. A `pending`
check is due again after the retry time the refresh or the run gave, between one second and
`FRESHNESS_WINDOW`. Every other check is due after `FRESHNESS_WINDOW`. A check states the rows:

- `current`: the frame the check found, which is the card's own frame until its data moves;
- `pending`: newer data is building as a queued operation, and the slot serves the frame of its
  last check until a check after the retry time finds it built; a read during the slot's first
  check states it too, on the card's own frame with no check time;
- `stale`: the check failed, the card cannot be drawn on the newer frame, or no Datasets surface
  answers in this app; the slot serves the frame of its last check, or the card's own frame
  when the newer one cannot be drawn.

A snapshot slot states `current`, since its rows are the share's for good.

`serve_on_slot_frame` makes that choice for both of a slot's reads, its data and its card, so
the two serve one frame. It draws the read on the frame the check chose, and when
that fails on a newer frame it records the slot `stale` on the card's own frame and draws there.

The rows are the card's data export (`assembler.pipeline.export_component.export_component_data`,
the one the components door's data route serves), on the served frame as a target when it is
not the frame the card revision is stored on. `format=json` turns the unbundled Parquet export
into a JSON document of rows, and `csv`, `xlsx` and `parquet` serve the export's bytes, as a
bundle when asked. An `xlsx` refusal of a cell's characters is the request's, so it answers as
it is and leaves the slot's check alone. The card (`serve_slot_card`) is the components door's read
of the card revision, drawn on the served frame (`assembler.pipeline.render_component`) when it
is not the frame the revision is stored on, with that frame's retention and the freshness
beside it. A note draws no frame, so its card is its stored read, `current`. Every read of a
frame records the frame it served in `served_frame` with the card revision's recorded
rendering.

The history lists those records newest first, a page at a time, each with the retention of its
frame counted from when DFO stored it (`presentation_api.retention.stated_retention`, read once per distinct
frame of the document) and whether that retention still holds: pinned, or held past now. A frame with no version record states no
retention and is not fetchable. An entry for a slot a replace left empty has no card, frame,
rendering or retention, and is not fetchable. A page names the next by an opaque cursor
(`history_cursor`) over its last entry's `first_served_at` and id, the keyset the store reads by.
"""

from __future__ import annotations

import base64
import binascii
import io
import json
import logging
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from datetime import UTC, date, datetime, timedelta
from typing import TYPE_CHECKING, Literal

import pyarrow.parquet as pq

from assembler.pipeline import get_component, render_component
from assembler.pipeline.export_component import (
    ComponentExportFile,
    ComponentHoldsNoData,
    export_component_data,
)
from assembler.pipeline.stored_revision import read_stored_revision
from assembler.ports import TRANSIENT_FRAME_REASONS, FrameRef, FrameUnavailable, RevisionRecord
from dfo_platform.codes import bound_registry_endpoint
from materialise.formats import ExcelIllegalCharacterError
from presentation_api.config import PRESENTATION_COMPONENTS_TIMEOUT_SECONDS
from presentation_api.doors.components import classify_components_exception
from presentation_api.newest_frames import FRESHNESS_WINDOW, NewestPending
from presentation_api.reply import serialize_read_reply
from presentation_api.retention import stated_retention
from presentation_api.routes import SHARES_ENDPOINT
from presentation_store import (
    Freshness,
    ServedPage,
    ServedPosition,
    ShareCard,
    ShareRecord,
    SlotCheck,
)

if TYPE_CHECKING:
    from presentation_api.doors.shares import SharesDeps

__all__ = [
    "HISTORY_PAGE_DEFAULT",
    "HISTORY_PAGE_MAX",
    "ShareSlotEmpty",
    "SlotData",
    "SlotFormat",
    "SlotServed",
    "history_cursor",
    "history_document",
    "history_position",
    "serve_on_slot_frame",
    "serve_slot",
    "serve_slot_card",
    "serving_frame",
    "slot_card",
]

logger = logging.getLogger(__name__)

type SlotFormat = Literal["json", "csv", "xlsx", "parquet"]

_JSON_MEDIA_TYPE = "application/json"

HISTORY_PAGE_DEFAULT = 100
"""The entries a history page holds when the request names no `limit`."""

HISTORY_PAGE_MAX = 500
"""The most entries a history page holds."""

_EPOCH = datetime(1970, 1, 1, tzinfo=UTC)
_MICROSECOND = timedelta(microseconds=1)
_MAX_ENTRY_ID = 2**63 - 1

_MIN_RETRY_S = 1.0
"""The least a pending check waits, whatever retry time it was given."""

# How long a claimed check holds off other reads before any may claim it again. It is twice the
# bound on one request, so a claim outlives the read that runs its check, and one whose reader
# died frees the slot after that time.
_CLAIM_TTL = timedelta(seconds=2 * PRESENTATION_COMPONENTS_TIMEOUT_SECONDS)


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
    served (`None` for a note's card), its freshness, when a live slot's check ran (`None` for
    a snapshot or a note)."""

    body: bytes
    media_type: str
    file_name: str | None
    frame: FrameRef | None
    freshness: Freshness
    checked_at: datetime | None


@dataclass(frozen=True)
class SlotServed[T]:
    """What `serve_on_slot_frame` drew, the frame it drew on, that frame's freshness, and when
    a live slot's check ran (`None` for a snapshot)."""

    drawn: T
    frame: FrameRef
    freshness: Freshness
    checked_at: datetime | None


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


def serve_on_slot_frame[T](
    *,
    share: ShareRecord,
    slot: int,
    card: ShareCard,
    record: RevisionRecord,
    deps: SharesDeps,
    draw: Callable[[FrameRef], T],
) -> SlotServed[T]:
    """Draw `card`, shown in slot `slot` of `share`, with `draw` on the frame the slot serves
    now, and record the frame served. A snapshot serves its recorded frame. A live slot serves
    its check's frame (`_live_check`), stored once `draw` succeeds on it. When `draw` fails on a
    frame other than the card's own, `draw` draws on the card's own frame, served `stale`. A
    transient failure (`_transient`) stores no check, so the read after the claim lapses checks
    again; any other failure stores the slot `stale` on the card's own frame.
    `card.frame` must be set and `record` must be the card revision.

    Raises:
        Any refusal of `draw` on the card's own frame.
    """
    assert card.frame is not None
    if share.mode == "snapshot":
        frame, freshness, checked_at = card.frame, "current", None
        drawn = draw(frame)
    else:
        check, ran = _live_check(
            share=share, slot=slot, card=card, rendering=record.frame_rendering, deps=deps
        )
        try:
            drawn = draw(check.frame)
        except ExcelIllegalCharacterError:
            # The format refuses the rows, whatever frame they come from.
            raise
        except Exception as exc:
            if check.frame == card.frame:
                raise
            transient = _transient(exc)
            # The newer frame does not fit the card's intent, or cannot be read. The slot
            # serves the card's own frame, which its share pinned: until the next check when
            # the failure is permanent, and for this read alone when a later read may pass.
            logger.warning(
                "Share %s slot %s cannot draw %s on %s%s; serving its own frame",
                share.id,
                slot,
                card.component_id,
                check.frame,
                " for now" if transient else "",
                exc_info=True,
            )
            check = replace(check, frame=card.frame, freshness="stale")
            drawn = draw(card.frame)
            ran = not transient
        if ran:
            deps.share_store.record_slot_check(share_id=share.id, slot=slot, card=card, check=check)
        frame, freshness, checked_at = check.frame, check.freshness, check.checked_at
    deps.share_store.record_served(
        share_id=share.id,
        revision=share.revision,
        slot=slot,
        card=card,
        frame=frame,
        rendering=record.frame_rendering,
    )
    return SlotServed(drawn=drawn, frame=frame, freshness=freshness, checked_at=checked_at)


def serve_slot(
    *, share: ShareRecord, slot: int, format: SlotFormat, bundle: bool, deps: SharesDeps
) -> SlotData:
    """Serve slot `slot` of `share` in `format`, as the export's bundle when `bundle` is set,
    and record the frame served. `bundle` must be false for `json`.

    Raises:
        ShareSlotEmpty: No such slot, or it holds no card.
        ComponentHoldsNoData: The card is a note or shows no data values.
        Any refusal of the card's export on the frame it serves.
    """
    assert not (bundle and format == "json")
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
    served = serve_on_slot_frame(
        share=share,
        slot=slot,
        card=card,
        record=record,
        deps=deps,
        draw=lambda frame: _export(
            card, frame=frame, record=record, format=format, bundle=bundle, deps=deps
        ),
    )
    if format == "json":
        body = _json_document(
            share=share,
            slot=slot,
            card=card,
            frame=served.frame,
            freshness=served.freshness,
            parquet=served.drawn.data,
        )
        media_type, file_name = _JSON_MEDIA_TYPE, None
    else:
        exported = served.drawn
        body, media_type, file_name = exported.data, exported.media_type, exported.file_name
    return SlotData(
        body=body,
        media_type=media_type,
        file_name=file_name,
        frame=served.frame,
        freshness=served.freshness,
        checked_at=served.checked_at,
    )


def serve_slot_card(*, share: ShareRecord, slot: int, deps: SharesDeps) -> SlotData:
    """Serve the card slot `slot` of `share` shows, as a component read with `freshness`,
    drawn on the frame the slot serves (`serve_on_slot_frame`), and record the frame served. A
    note's card is its stored read, `current`.

    Raises:
        ShareSlotEmpty: No such slot, or it holds no card.
        Any refusal of drawing the card on the frame it serves.
    """
    card = slot_card(share, slot)
    record = read_stored_revision(
        component_id=card.component_id, revision=card.revision, revision_store=deps.revision_store
    )
    frame: FrameRef | None = None
    freshness: Freshness = "current"
    checked_at: datetime | None = None
    if card.frame is None:
        document = _card_document(card, frame=None, record=record, deps=deps)
    else:
        served = serve_on_slot_frame(
            share=share,
            slot=slot,
            card=card,
            record=record,
            deps=deps,
            draw=lambda on: _card_document(card, frame=on, record=record, deps=deps),
        )
        document, frame = served.drawn, served.frame
        freshness, checked_at = served.freshness, served.checked_at
    body = json.dumps(
        {**document, "freshness": freshness},
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    ).encode()
    return SlotData(
        body=body,
        media_type=_JSON_MEDIA_TYPE,
        file_name=None,
        frame=frame,
        freshness=freshness,
        checked_at=checked_at,
    )


def _card_document(
    card: ShareCard, *, frame: FrameRef | None, record: RevisionRecord, deps: SharesDeps
) -> dict[str, object]:
    """The components door's read of the card revision on `frame`: its stored read on the
    frame it is stored on, or with none, else drawn on `frame`, with `frame`'s retention."""
    if frame is None or record.stored_intent.get("frame_ref") == _frame_body(frame):
        document = dict(
            get_component(
                component_id=card.component_id,
                revision=card.revision,
                revision_store=deps.revision_store,
            )
        )
    else:
        reply = render_component(
            component_id=card.component_id,
            revision=card.revision,
            target_frame=frame,
            frame_store=deps.frame_store,
            revision_store=deps.revision_store,
            release_registry=deps.release_registry,
            vocabulary_source=deps.vocabulary_source,
        )
        document = serialize_read_reply(reply, stored_intent=record.stored_intent)
    retention = (
        stated_retention(frame, snapshot_reader=deps.snapshot_reader) if frame is not None else None
    )
    return {**document, "retention": retention}


def history_document(*, share_id: str, page: ServedPage, deps: SharesDeps) -> dict[str, object]:
    """The history route's body: the entries of `page`, in the order given, each frame with its
    retention, read once per distinct frame however many entries name it, and the cursor of the
    next page (`None` on the last)."""
    now = deps.now()
    retentions: dict[FrameRef, dict[str, object] | None] = {}
    entries: list[dict[str, object]] = []
    for entry in page.entries:
        frame = entry.frame
        retention = None
        if frame is not None:
            if frame not in retentions:
                retentions[frame] = stated_retention(frame, snapshot_reader=deps.snapshot_reader)
            retention = retentions[frame]
        entries.append(
            {
                "slot": entry.slot,
                "component_id": entry.component_id,
                "revision": entry.revision,
                "frame": (
                    {"spec_id": frame.spec_id, "content_version": frame.content_version}
                    if frame is not None
                    else None
                ),
                "rendering": dict(entry.rendering) if entry.rendering is not None else None,
                "first_served_at": entry.first_served_at.isoformat(),
                "retention": retention,
                "fetchable": _fetchable(retention, now=now),
            }
        )
    return {
        "id": share_id,
        "served": entries,
        "next": history_cursor(page.next) if page.next is not None else None,
    }


def history_cursor(position: ServedPosition) -> str:
    """The opaque cursor a history page names its next page by: unpadded URL-safe base64 of
    the last entry's `first_served_at`, in microseconds since the epoch, and its id."""
    micros = (position.first_served_at - _EPOCH) // _MICROSECOND
    return base64.urlsafe_b64encode(f"{micros}.{position.id}".encode()).decode().rstrip("=")


def history_position(cursor: str) -> ServedPosition | None:
    """The position `cursor` names, or `None` when `history_cursor` gives no such cursor."""
    try:
        decoded = base64.b64decode(cursor + "=" * (-len(cursor) % 4), altchars=b"-_", validate=True)
        micros, entry_id = (int(part) for part in decoded.decode("ascii").split("."))
        position = ServedPosition(first_served_at=_EPOCH + micros * _MICROSECOND, id=entry_id)
    except (binascii.Error, UnicodeDecodeError, ValueError, OverflowError):
        return None
    if not 0 < entry_id <= _MAX_ENTRY_ID or history_cursor(position) != cursor:
        return None
    return position


def _fetchable(retention: Mapping[str, object] | None, *, now: datetime) -> bool:
    if retention is None:
        return False
    if retention.get("pinned") is True:
        return True
    held_until = retention.get("held_until")
    return isinstance(held_until, str) and datetime.fromisoformat(held_until) > now


def _transient(exc: Exception) -> bool:
    """Whether a later read may draw where `exc` refused with nothing done: a `FrameUnavailable`
    whose reason is transient (`TRANSIENT_FRAME_REASONS`), or any other failure the components
    classifier, on the share routes' endpoint, marks retryable."""
    if isinstance(exc, FrameUnavailable):
        return exc.reason in TRANSIENT_FRAME_REASONS
    with bound_registry_endpoint(SHARES_ENDPOINT):
        spec = classify_components_exception(exc)
    return spec is not None and spec.retryable


def _live_check(
    *,
    share: ShareRecord,
    slot: int,
    card: ShareCard,
    rendering: Mapping[str, object] | None,
    deps: SharesDeps,
) -> tuple[SlotCheck, bool]:
    """The slot's check, run once none is stored and again once the last one is due
    (`SlotCheck.next_check_at`), asking for the newest frame of the card's frame under
    `rendering`, the one the card revision recorded. Of reads finding it due, the one whose claim succeeds runs it,
    and the others serve what the last check recorded: its frame, freshness and check time, or
    `pending` on the card's own frame before the first check ends. A claim whose reader never
    records its check expires after `_CLAIM_TTL`. Returns the check and whether this read ran
    it, so the caller stores it once its draw succeeds."""
    assert card.frame is not None
    store = deps.share_store
    now = deps.now()
    prior = store.slot_check(share_id=share.id, slot=slot, card=card)
    if prior is not None and prior.next_check_at > now:
        return prior, False
    if not store.claim_slot_check(
        share_id=share.id, slot=slot, card=card, now=now, claim_until=now + _CLAIM_TTL
    ):
        held = store.slot_check(share_id=share.id, slot=slot, card=card)
        # A lost claim means another read stored or claimed the check, and nothing removes one.
        assert held is not None
        return held, False
    last = prior.frame if prior is not None else card.frame
    check = _check(
        card.frame,
        rendering=rendering,
        last=last,
        now=now,
        deps=deps,
        label=f"{share.id} slot {slot}",
    )
    return check, True


def _check(
    frame: FrameRef,
    *,
    rendering: Mapping[str, object] | None,
    last: FrameRef,
    now: datetime,
    deps: SharesDeps,
    label: str,
) -> SlotCheck:
    """Ask for the newest frame standing for `frame` of `rendering`, serving `last` when none answers. The
    check is due again after `FRESHNESS_WINDOW`, or after the retry time of a pending answer."""
    due = now + FRESHNESS_WINDOW
    if deps.newest is None:
        logger.error("Share %s cannot check for newer data: no Datasets surface answers", label)
        return SlotCheck(frame=last, freshness="stale", checked_at=now, next_check_at=due)
    try:
        answer = deps.newest.newest(frame, rendering=rendering)
    except Exception:
        # Every refusal and failure of the refresh or the run leaves the slot on its last frame,
        # which the reply states as stale, until the next check.
        logger.warning("Share %s newer-data check of %s failed", label, frame, exc_info=True)
        return SlotCheck(frame=last, freshness="stale", checked_at=now, next_check_at=due)
    if isinstance(answer, NewestPending):
        retry = min(max(answer.retry_after_s, _MIN_RETRY_S), FRESHNESS_WINDOW.total_seconds())
        return SlotCheck(
            frame=last,
            freshness="pending",
            checked_at=now,
            next_check_at=now + timedelta(seconds=retry),
        )
    return SlotCheck(frame=answer, freshness="current", checked_at=now, next_check_at=due)


def _export(
    card: ShareCard,
    *,
    frame: FrameRef,
    record: RevisionRecord,
    format: SlotFormat,
    bundle: bool,
    deps: SharesDeps,
) -> ComponentExportFile:
    """The card's export on `frame`, which is a target unless the card revision is stored on
    it, bundled when `bundle` is set. JSON reads the unbundled Parquet export."""
    target = None if record.stored_intent.get("frame_ref") == _frame_body(frame) else frame
    return export_component_data(
        component_id=card.component_id,
        revision=card.revision,
        format="parquet" if format == "json" else format,
        bundle=bundle,
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
