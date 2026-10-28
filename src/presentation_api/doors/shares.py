"""The shares door publishes cards under one `/s/<id>` id, serves each slot's data, and replaces,
snapshots, deletes and reads shares.

`POST /api/presentation/shares` takes `{cards: [{component_id, revision}], latest, title?, kind?,
payload?}`. `latest` has no default: true makes a live share, which follows new data, and false
a snapshot, which keeps the versions its cards drew. `kind` defaults to `card`. A `card` share
takes 1 to 24 cards, a `canvas` 0 to 24, and a `search` or `conversation` share none, and a
`conversation` share is a snapshot. `payload` is website-owned JSON of at most
`MAX_SHARE_PAYLOAD_BYTES` as compact UTF-8, stored as given and read back on GET. The create spends one unit of the caller's `share`
allowance (`datasets_api.allowance`, caller named by `dfo_platform.caller.resolve_caller`),
refusing 429 `allowance_exhausted` once it is spent, then pins the frame each card revision
draws (`presentation_store.pin_card_frames`), live or snapshot, since a share first including a
card pins it. A unit stays spent when the pin or the write then fails. The reply carries the
share's page, embed and per-slot data URLs under the public site, and the `shw_` write token,
which appears only there; the store keeps its SHA-256 digest.

`PUT /api/presentation/shares/{share_id}` takes `{cards}` and writes the share's next revision.
Slot n shows the nth card, and slots past the last card stay empty, so every slot keeps its URLs.
It pins the card revisions the share did not show before. `DELETE` takes the share down. Both
need the share's token in `x-dfo-write-token`, checked before anything else about the share, so
a caller without it learns nothing more: no header refuses 403 `write_token_required`, another
token 403 `write_token_invalid`, and a share stored without a token (a guest's, imported from the
website) refuses every token. A replace of a snapshot then refuses 409 `share_not_replaceable`.

`POST /api/presentation/shares/{share_id}/snapshot` turns a live share into a snapshot in place,
under the same URLs, with the share's token. Each slot keeps the frame it serves now (its last
newest-data check, else its card's own frame): the door pins those frames, then writes the
share's next revision with them and the snapshot mode. A snapshot answers as it is, and a share
replaced meanwhile refuses 409 `edit_conflict`. No route turns a snapshot live.

`GET /api/presentation/shares/{share_id}` is public and allows any origin: kind, mode, title,
revision, slots, the website's payload and created_at, never the token.

`GET /api/presentation/shares/{share_id}/slots/{slot}/data?format=json|csv|parquet` is public
and allows any origin. It serves the rows behind the slot's card, JSON by default, on the frame
`presentation_api.share_slots` chooses: a snapshot's recorded frame, or a live slot's newest
data, checked at most once per `newest_frames.FRESHNESS_WINDOW`. The JSON body and the
`x-dfo-freshness` header state `current`, `stale` or `pending`; a live slot's
`x-dfo-checked-at` says when the check ran. The reply carries a strong `ETag` over its bytes and
answers `If-None-Match` with 304. A live slot's reply is `no-cache`, so a cache revalidates it;
a snapshot's is `public, max-age=300`. A slot number the share has not had, or a slot a
replacement left empty, answers 404 `not_found`, and so does a note's slot.

`GET /api/presentation/shares/{share_id}/history` is public and allows any origin: every frame
the share's slots served, newest first, each with its retention counted from when DFO stored it
and whether it is still fetchable (`share_slots.history_document`).

A share id that names no share, or one taken down, answers 404 `not_found` on every route.

Handlers are module-level functions that read `SHARES_DEPS_SLOT`, which the door's lifespan
sets. Every route runs on one executor behind a non-blocking semaphore, and saturation answers
retryable 503.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Callable, Iterable, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Annotated, Literal, Self, cast

from fastapi import APIRouter, Depends, FastAPI, Header, Path, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    StrictBool,
    StrictInt,
    model_validator,
)

from assembler.catalogue import BakedReleaseRegistry
from assembler.errors import (
    BindingInvalid,
    ConstraintViolation,
    FrameUnavailable,
    ReleaseMismatch,
    ResourceExhausted,
    RevisionNotFound,
    SchemaInvalid,
)
from assembler.pipeline.export_component import ComponentExportMismatch, ComponentHoldsNoData
from assembler.pipeline.stored_revision import read_stored_revision
from assembler.ports import (
    DraftNotFound,
    EditConflict,
    FrameRef,
    FrameStore,
    ReleaseRegistry,
    RevisionStore,
)
from assembler.write_token import (
    SHARE_WRITE_TOKEN_PREFIX,
    mint_write_token,
    write_token_digest,
    write_token_matches,
)
from dataset_store import LiveDatasetStore
from dataset_store import close_shared_pool as close_dataset_pool
from dataset_store import shared_pool as dataset_shared_pool
from dataset_store.snapshot import SnapshotReader
from datasets_api.allowance import (
    AllowanceExhausted,
    AllowanceKind,
    AllowanceLedger,
    LiveAllowanceLedger,
    allowance_limits,
)
from datasets_api.artifact_delivery import artifact_content_disposition
from datasets_api.config import DatasetsBuildSettings
from datasets_api.door_executor import DoorExecutor, build_door_executor
from datasets_api.shutdown import ShutdownBudget
from datasets_api.store_errors import DatasetsStoreUnavailable
from dfo_platform import ProblemSpec
from dfo_platform.caller import CallerHeaderInvalid, CallerUnresolved, resolve_caller
from dfo_platform.config import CallerSettings, PublicSiteSettings
from dfo_platform.doors import DoorContribution, route_registry_endpoints
from dfo_platform.state import StateSlot
from presentation_api.config import (
    PRESENTATION_COMPONENTS_MAX_CONCURRENCY,
    PRESENTATION_COMPONENTS_TIMEOUT_SECONDS,
    PresentationSettings,
)
from presentation_api.doors.component_data import classify_component_data_exception
from presentation_api.doors.components import WRITE_TOKEN_HEADER, classify_components_exception
from presentation_api.live import build_live_frame_store
from presentation_api.models import (
    SharedResponse,
    ShareHistoryResponse,
    ShareReadResponse,
    ShareSlotDataResponse,
    ShareWriteResponse,
)
from presentation_api.newest_frames import NewestFrames
from presentation_api.path_params import PathParamInvalid, require_component_id, require_share_id
from presentation_api.share_slots import (
    ShareSlotEmpty,
    SlotFormat,
    history_document,
    serve_slot,
    serving_frame,
)
from presentation_store import (
    GeneratedIdCollision,
    LivePresentationStore,
    LiveShareStore,
    PresentationStoreUnavailable,
    ShareCard,
    ShareNotFound,
    ShareRecord,
    ShareStore,
    frame_reads_bounded_by,
    pin_card_frames,
)
from presentation_store import close_shared_pool as close_presentation_pool
from presentation_store import shared_pool as presentation_shared_pool
from spec_registry.errors import SpecRegistryStoreError, SpecRegistryUnavailable

__all__ = [
    "MAX_SHARE_CARDS",
    "MAX_SHARE_PAYLOAD_BYTES",
    "SHARES_DEPS_SLOT",
    "SHARES_ENDPOINT",
    "SharesDeps",
    "build_shares_door",
    "classify_shares_exception",
]

logger = logging.getLogger(__name__)

SHARES_ENDPOINT = "presentation.shares"
"""The code registry endpoint every share route serves."""

SHARES_PATH = "/api/presentation/shares"
SHARE_PATH = "/api/presentation/shares/{share_id}"
SHARE_SNAPSHOT_PATH = f"{SHARE_PATH}/snapshot"
SHARE_HISTORY_PATH = f"{SHARE_PATH}/history"
SHARE_SLOT_DATA_PATH = f"{SHARE_PATH}/slots/{{slot}}/data"

MAX_SHARE_CARDS = 24
"""The most cards one create or replace takes, and so the most slots a write adds."""

MAX_SHARE_PAYLOAD_BYTES = 1_048_576
"""The largest `payload` a create takes, as compact UTF-8 JSON."""

_MAX_TITLE_LENGTH = 300

_FRESHNESS_HEADER = "x-dfo-freshness"
_CHECKED_AT_HEADER = "x-dfo-checked-at"
_SLOT_DATA_EXPOSED = f"ETag, {_FRESHNESS_HEADER}, {_CHECKED_AT_HEADER}"
_LIVE_CACHE_CONTROL = "no-cache"
_SNAPSHOT_CACHE_CONTROL = "public, max-age=300"

# How many cards each kind takes on create, and whether it may be live.
_KIND_CARDS: Mapping[str, tuple[int, int]] = {
    "card": (1, MAX_SHARE_CARDS),
    "canvas": (0, MAX_SHARE_CARDS),
    "search": (0, 0),
    "conversation": (0, 0),
}


class ShareCardBody(BaseModel):
    """One card revision a share shows."""

    model_config = ConfigDict(extra="forbid")

    component_id: str = Field(description="the card's cmp_ id")
    revision: StrictInt = Field(ge=1, description="the card revision the slot shows")


class ShareCardsBody(BaseModel):
    """A replace's body: the cards the share's next revision shows, slot 1 first."""

    model_config = ConfigDict(extra="forbid")

    cards: list[ShareCardBody] = Field(
        min_length=1,
        max_length=MAX_SHARE_CARDS,
        description="the cards in slot order; two entries may name the same revision",
    )


class CreateShareBody(BaseModel):
    """A create's body. `latest` has no default, because a live link and a fixed one answer
    different questions under a public URL."""

    model_config = ConfigDict(extra="forbid")

    cards: list[ShareCardBody] = Field(
        default_factory=list,
        max_length=MAX_SHARE_CARDS,
        description="the cards in slot order; two entries may name the same revision. A card "
        "share takes at least one, a canvas any number, a search or conversation share none",
    )
    latest: StrictBool = Field(
        description="true for a live share that follows new data and takes replacements; "
        "false for a snapshot that keeps the versions its cards draw now, for good"
    )
    title: str | None = Field(
        default=None, min_length=1, max_length=_MAX_TITLE_LENGTH, description="the share's title"
    )
    kind: Literal["card", "canvas", "search", "conversation"] = Field(
        default="card",
        description="what the share publishes; the website renders a canvas, search or "
        "conversation share from its payload. A conversation share is a snapshot",
    )
    payload: JsonValue = Field(
        default=None,
        description="website-owned JSON stored with the share as given and read back on GET, "
        f"at most {MAX_SHARE_PAYLOAD_BYTES} bytes as compact UTF-8",
    )

    @model_validator(mode="after")
    def _fits_its_kind(self) -> Self:
        least, most = _KIND_CARDS[self.kind]
        if not least <= len(self.cards) <= most:
            wanted = "none" if most == 0 else f"{least} to {most}"
            raise ValueError(f"a {self.kind} share takes {wanted} cards, not {len(self.cards)}")
        if self.kind == "conversation" and self.latest:
            raise ValueError("a conversation share is a snapshot: send latest=false")
        size = len(json.dumps(self.payload, ensure_ascii=False, separators=(",", ":")).encode())
        if size > MAX_SHARE_PAYLOAD_BYTES:
            raise ValueError(
                f"the payload is {size} bytes, over the {MAX_SHARE_PAYLOAD_BYTES}-byte bound"
            )
        return self


class WriteTokenRequired(Exception):
    """A replace or delete carries no write token. Maps to 403 `write_token_required`."""

    def __init__(self, share_id: str) -> None:
        self.share_id = share_id
        super().__init__(
            f"replacing or deleting share {share_id!r} needs its write token in "
            f"{WRITE_TOKEN_HEADER}"
        )


class WriteTokenInvalid(Exception):
    """The token sent is not the share's, or the share has none. Maps to 403
    `write_token_invalid`."""

    def __init__(self, share_id: str) -> None:
        self.share_id = share_id
        super().__init__(
            f"the {WRITE_TOKEN_HEADER} sent does not grant changing share {share_id!r}; send "
            "the write_token its create returned"
        )


class ShareNotReplaceable(Exception):
    """A replace names a snapshot. Maps to 409 `share_not_replaceable`."""

    def __init__(self, share_id: str) -> None:
        self.share_id = share_id
        super().__init__(
            f"share {share_id!r} is a snapshot, whose cards are fixed for good; create a new "
            "share for new cards, or create it with latest=true to replace its cards later"
        )


class SharesUnavailable(Exception):
    """The door is at capacity, or its lifespan has not set its dependencies. Maps to
    retryable 503 `service_unavailable`."""

    def __init__(self) -> None:
        super().__init__("the shares door cannot take this request right now; retry")


@dataclass
class SharesDeps:
    """What the share routes read, set on the app by the door's lifespan."""

    share_store: ShareStore
    revision_store: RevisionStore
    frame_store: FrameStore
    release_registry: ReleaseRegistry
    allowances: AllowanceLedger
    caller: CallerSettings
    site: PublicSiteSettings
    executor: DoorExecutor
    # The version records history entries take their retention from; `None` states none.
    snapshot_reader: SnapshotReader | None = None
    # Where a live slot asks for newer data; `None` leaves every live slot stale.
    newest: NewestFrames | None = None
    now: Callable[[], datetime] = field(default=lambda: datetime.now(UTC))


SHARES_DEPS_SLOT: StateSlot[SharesDeps] = StateSlot(
    "presentation_shares_deps", unset_exception=SharesUnavailable
)

SharesDepsDep = Annotated[SharesDeps, Depends(SHARES_DEPS_SLOT)]

WriteTokenHeader = Annotated[str | None, Header(alias=WRITE_TOKEN_HEADER)]


def authorise_share_write(share: ShareRecord, *, presented: str | None) -> None:
    """Refuse a replace or delete of `share` unless `presented` is its write token.

    Raises:
        WriteTokenRequired: No token was sent.
        WriteTokenInvalid: The token is not the share's, or the share has none.
    """
    if presented is None:
        raise WriteTokenRequired(share.id)
    if not write_token_matches(presented=presented, stored_digest=share.write_token_sha256):
        raise WriteTokenInvalid(share.id)


def _read_cards(
    cards: Iterable[ShareCardBody], *, revision_store: RevisionStore
) -> list[ShareCard]:
    """The stored card revisions `cards` name, each with the frame it draws.

    Raises:
        PathParamInvalid: A component id is malformed.
        DraftNotFound: No component exists at an id.
        RevisionNotFound: A component has no such revision.
        EditConflict: A component has no revision at all.
    """
    read: list[ShareCard] = []
    for card in cards:
        require_component_id(card.component_id)
        record = read_stored_revision(
            component_id=card.component_id, revision=card.revision, revision_store=revision_store
        )
        frame = cast("Mapping[str, str] | None", record.stored_intent.get("frame_ref"))
        read.append(
            ShareCard(
                component_id=card.component_id,
                revision=card.revision,
                frame=FrameRef(**frame) if frame is not None else None,
            )
        )
    return read


def _pin(cards: Iterable[ShareCard], *, deps: SharesDeps) -> None:
    """Pin the frame each distinct card revision in `cards` draws."""
    for component_id, revision in dict.fromkeys((c.component_id, c.revision) for c in cards):
        pin_card_frames(
            component_id=component_id,
            revision=revision,
            revision_store=deps.revision_store,
            frame_store=deps.frame_store,
        )


def _locations(share: ShareRecord, *, site: str) -> dict[str, object]:
    return {
        "id": share.id,
        "url": f"{site}/s/{share.id}",
        "embed_url": f"{site}/embed/{share.id}",
        "data_urls": [
            _data_url(share.id, slot, site=site) for slot in range(1, len(share.slots) + 1)
        ],
        "revision": share.revision,
        "latest": share.mode == "live",
    }


def _data_url(share_id: str, slot: int, *, site: str) -> str:
    return f"{site}/api/v1/presentation/shares/{share_id}/slots/{slot}/data"


def _read_document(share: ShareRecord, *, site: str) -> dict[str, object]:
    return {
        "id": share.id,
        "kind": share.kind,
        "mode": share.mode,
        "title": share.title,
        "revision": share.revision,
        "slots": [
            {
                "slot": slot.slot,
                "card": (
                    {
                        "component_id": slot.card.component_id,
                        "revision": slot.card.revision,
                        "frame": (
                            {
                                "spec_id": slot.card.frame.spec_id,
                                "content_version": slot.card.frame.content_version,
                            }
                            if slot.card.frame is not None
                            else None
                        ),
                    }
                    if slot.card is not None
                    else None
                ),
                "data_url": _data_url(share.id, slot.slot, site=site),
            }
            for slot in share.slots
        ],
        "payload": share.payload,
        "created_at": share.created_at.isoformat(),
    }


async def _run[R: Response](deps: SharesDeps, job: Callable[[], R], /) -> R:
    with frame_reads_bounded_by(timeout_seconds=PRESENTATION_COMPONENTS_TIMEOUT_SECONDS):
        return await asyncio.wait_for(
            deps.executor.run(job, refuse=SharesUnavailable),
            timeout=PRESENTATION_COMPONENTS_TIMEOUT_SECONDS,
        )


async def create_share(body: CreateShareBody, request: Request, deps: SharesDepsDep) -> Response:
    caller = resolve_caller(request, deps.caller)

    def job() -> Response:
        cards = _read_cards(body.cards, revision_store=deps.revision_store)
        deps.allowances.spend(caller, AllowanceKind.share, deps.now())
        _pin(cards, deps=deps)
        token = mint_write_token(prefix=SHARE_WRITE_TOKEN_PREFIX)
        share = deps.share_store.create_share(
            kind=body.kind,
            mode="live" if body.latest else "snapshot",
            title=body.title,
            write_token_sha256=write_token_digest(token),
            cards=cards,
            payload=body.payload,
        )
        document = {**_locations(share, site=deps.site.public_site_url), "write_token": token}
        return JSONResponse(document, status_code=201, headers={"Cache-Control": "no-store"})

    return await _run(deps, job)


async def replace_share(
    share_id: str,
    body: ShareCardsBody,
    deps: SharesDepsDep,
    write_token: WriteTokenHeader = None,
) -> Response:
    require_share_id(share_id)

    def job() -> Response:
        current = deps.share_store.read_share(share_id=share_id)
        authorise_share_write(current, presented=write_token)
        if current.mode == "snapshot":
            raise ShareNotReplaceable(share_id)
        cards = _read_cards(body.cards, revision_store=deps.revision_store)
        shown = {
            (slot.card.component_id, slot.card.revision)
            for slot in current.slots
            if slot.card is not None
        }
        _pin((card for card in cards if (card.component_id, card.revision) not in shown), deps=deps)
        share = deps.share_store.replace_share(share_id=share_id, cards=cards)
        return JSONResponse(_locations(share, site=deps.site.public_site_url))

    return await _run(deps, job)


async def delete_share(
    share_id: str, deps: SharesDepsDep, write_token: WriteTokenHeader = None
) -> Response:
    require_share_id(share_id)

    def job() -> Response:
        current = deps.share_store.read_share(share_id=share_id)
        authorise_share_write(current, presented=write_token)
        deps.share_store.delete_share(share_id=share_id)
        return Response(status_code=204)

    return await _run(deps, job)


async def read_share(share_id: str, deps: SharesDepsDep) -> Response:
    require_share_id(share_id)

    def job() -> Response:
        share = deps.share_store.read_share(share_id=share_id)
        return JSONResponse(
            _read_document(share, site=deps.site.public_site_url),
            # A share is public, so any page may read it across origins.
            headers={"Access-Control-Allow-Origin": "*"},
        )

    return await _run(deps, job)


async def post_share_snapshot(
    share_id: str, deps: SharesDepsDep, write_token: WriteTokenHeader = None
) -> Response:
    require_share_id(share_id)

    def job() -> Response:
        current = deps.share_store.read_share(share_id=share_id)
        authorise_share_write(current, presented=write_token)
        if current.mode == "live":
            frames: dict[int, FrameRef] = {}
            pins: dict[tuple[str, FrameRef], None] = {}
            for slot in current.slots:
                if slot.card is None or slot.card.frame is None:
                    continue
                frame = serving_frame(share=current, slot=slot.slot, card=slot.card, deps=deps)
                frames[slot.slot] = frame
                pins[(slot.card.component_id, frame)] = None
            for component_id, frame in pins:
                deps.frame_store.pin(frame_ref=frame, component_id=component_id)
            current = deps.share_store.snapshot_share(
                share_id=share_id, revision=current.revision, frames=frames
            )
        return JSONResponse(_locations(current, site=deps.site.public_site_url))

    return await _run(deps, job)


async def read_share_history(share_id: str, deps: SharesDepsDep) -> Response:
    require_share_id(share_id)

    def job() -> Response:
        served = deps.share_store.served_frames(share_id=share_id)
        return JSONResponse(
            history_document(share_id=share_id, served=served, deps=deps),
            headers={"Access-Control-Allow-Origin": "*", "Cache-Control": "no-store"},
        )

    return await _run(deps, job)


def _matches(if_none_match: str | None, etag: str) -> bool:
    """Whether `If-None-Match` names `etag`, compared weakly as RFC 9110 says for it."""
    if if_none_match is None:
        return False
    tags = {tag.strip().removeprefix("W/") for tag in if_none_match.split(",")}
    return "*" in tags or etag in tags


async def read_share_slot_data(
    share_id: str,
    deps: SharesDepsDep,
    slot: Annotated[int, Path(ge=1)],
    format_: Annotated[SlotFormat, Query(alias="format")] = "json",
    if_none_match: Annotated[str | None, Header()] = None,
) -> Response:
    require_share_id(share_id)

    def job() -> Response:
        share = deps.share_store.read_share(share_id=share_id)
        data = serve_slot(share=share, slot=slot, format=format_, deps=deps)
        headers = {
            "ETag": data.etag,
            _FRESHNESS_HEADER: data.freshness,
            "Cache-Control": (
                _LIVE_CACHE_CONTROL if share.mode == "live" else _SNAPSHOT_CACHE_CONTROL
            ),
            # Any page may read a share's data across origins.
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Expose-Headers": _SLOT_DATA_EXPOSED,
        }
        if data.checked_at is not None:
            headers[_CHECKED_AT_HEADER] = data.checked_at.isoformat()
        if _matches(if_none_match, data.etag):
            return Response(status_code=304, headers=headers)
        if data.file_name is not None:
            headers["Content-Disposition"] = artifact_content_disposition(data.file_name)
        return Response(content=data.body, media_type=data.media_type, headers=headers)

    return await _run(deps, job)


def classify_shares_exception(exc: BaseException) -> ProblemSpec | None:
    """The share refusals, then the components door's for the card, store and frame failures
    the share routes share with it."""
    if isinstance(exc, WriteTokenRequired):
        return ProblemSpec(
            status=403, code="write_token_required", detail=str(exc), retryable=False
        )
    if isinstance(exc, WriteTokenInvalid):
        return ProblemSpec(status=403, code="write_token_invalid", detail=str(exc), retryable=False)
    if isinstance(exc, ShareNotReplaceable):
        return ProblemSpec(
            status=409, code="share_not_replaceable", detail=str(exc), retryable=False
        )
    if isinstance(exc, ShareNotFound | ShareSlotEmpty):
        return ProblemSpec(status=404, code="not_found", detail=str(exc), retryable=False)
    if isinstance(exc, AllowanceExhausted):
        return ProblemSpec(
            status=429,
            code="allowance_exhausted",
            detail=str(exc),
            retryable=True,
            headers=exc.headers,
        )
    if isinstance(exc, CallerHeaderInvalid):
        return ProblemSpec(status=400, code="request_invalid", detail=str(exc), retryable=False)
    if isinstance(exc, CallerUnresolved):
        # A deployment fault a retry does not clear. This surface's `service_unavailable` is
        # retryable, so it answers `internal_error`.
        logger.error("Shares door cannot name a caller: %s", exc)
        return ProblemSpec(status=500, code="internal_error", detail=str(exc), retryable=False)
    if isinstance(exc, SharesUnavailable):
        return ProblemSpec(status=503, code="service_unavailable", detail=str(exc), retryable=True)
    return classify_components_exception(exc) or classify_component_data_exception(exc)


def build_shares_door(
    *,
    release_registry: ReleaseRegistry | None = None,
    share_store: ShareStore | None = None,
    revision_store: RevisionStore | None = None,
    frame_store: FrameStore | None = None,
    snapshot_reader: SnapshotReader | None = None,
    newest: NewestFrames | None = None,
    allowances: AllowanceLedger | None = None,
    caller: CallerSettings | None = None,
    now: Callable[[], datetime] | None = None,
    pool_size: int | None = None,
    budget: ShutdownBudget | None = None,
) -> DoorContribution:
    """Build the shares door. The lifespan builds each store left out over the live pools: the
    share and revision stores over the presentation pool, the frame store over the datasets
    pool, and the allowance ledger over the datasets pool with `DatasetsBuildSettings`' limits.
    A frame store built here brings its snapshot reader; with one given, `snapshot_reader` is
    the history's, and left out the history states no retention. `newest` answers live slots'
    newer-data checks; left out, every live slot serves its card's frame as stale.
    `release_registry`, which a live slot draws its card on a newer frame under, defaults to the
    baked release. `caller`
    defaults to those settings' `caller`, and the public site to `DFO_PUBLIC_SITE_URL`.
    """
    _budget = budget if budget is not None else ShutdownBudget()
    executor = build_door_executor(
        pool_size if pool_size is not None else PRESENTATION_COMPONENTS_MAX_CONCURRENCY,
        "shares",
        surface="presentation",
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        opened_presentation_pool = False
        # The live frame store and the allowance ledger each borrow the datasets pool once.
        dataset_borrows = 0
        owned_graph = None

        def _close_live_clients_built_here() -> None:
            if owned_graph is not None:
                owned_graph.close()
            for _ in range(dataset_borrows):
                close_dataset_pool()
            if opened_presentation_pool:
                close_presentation_pool()

        try:
            site = PublicSiteSettings.from_env()
            build_settings = (
                DatasetsBuildSettings.from_env() if allowances is None or caller is None else None
            )
            presentation_db: LiveDatasetStore | None = None
            if share_store is None or revision_store is None:
                presentation_db = LiveDatasetStore(connections=presentation_shared_pool())
                opened_presentation_pool = True
            effective_frame_store = frame_store
            effective_snapshot_reader = snapshot_reader
            if effective_frame_store is None:
                live = build_live_frame_store(settings=PresentationSettings.from_env())
                dataset_borrows += 1
                owned_graph = live.graph
                effective_frame_store = live.frame_store
                effective_snapshot_reader = live.snapshot_reader
            effective_allowances = allowances
            effective_caller = caller
            if effective_caller is None:
                assert build_settings is not None
                effective_caller = build_settings.caller
            if effective_allowances is None:
                assert build_settings is not None
                effective_allowances = LiveAllowanceLedger(
                    store=LiveDatasetStore(connections=dataset_shared_pool()),
                    limits=allowance_limits(build_settings),
                )
                dataset_borrows += 1
            deps = SharesDeps(
                share_store=(
                    share_store
                    if share_store is not None
                    else LiveShareStore(store=cast(LiveDatasetStore, presentation_db))
                ),
                revision_store=(
                    revision_store
                    if revision_store is not None
                    else LivePresentationStore(store=cast(LiveDatasetStore, presentation_db))
                ),
                frame_store=effective_frame_store,
                release_registry=(
                    release_registry if release_registry is not None else BakedReleaseRegistry()
                ),
                allowances=effective_allowances,
                caller=effective_caller,
                site=site,
                executor=executor,
                snapshot_reader=effective_snapshot_reader,
                newest=newest,
            )
            if now is not None:
                deps.now = now
        except BaseException:
            _close_live_clients_built_here()
            raise

        SHARES_DEPS_SLOT.set(app, deps)
        try:
            yield
        finally:
            SHARES_DEPS_SLOT.clear(app)
            executor.executor.shutdown(wait=False, cancel_futures=True)
            await asyncio.to_thread(executor.drain, _budget)
            _close_live_clients_built_here()

    router = APIRouter()
    router.add_api_route(
        SHARES_PATH,
        create_share,
        methods=["POST"],
        status_code=201,
        response_model=SharedResponse,
        description="Publish cards as a share: one slot per card, a live link (latest=true) "
        "that follows new data or a snapshot (latest=false) that keeps the versions its cards "
        "draw now. Every card's frame is pinned. Spends one unit of the caller's daily share "
        "allowance. The reply's write_token grants replacing and deleting the share.",
    )
    router.add_api_route(
        SHARE_PATH,
        read_share,
        methods=["GET"],
        response_model=ShareReadResponse,
        description="Read a share: its kind, mode, title, latest revision, slots and the "
        "website's payload. Any origin may read it. It never carries the write token.",
    )
    router.add_api_route(
        SHARE_PATH,
        replace_share,
        methods=["PUT"],
        response_model=ShareWriteResponse,
        description="Give a live share new cards under the same URLs, as its next revision. "
        "Slot n shows the nth card, and slots past the last card stay empty. Needs the share's "
        "write token in x-dfo-write-token. A snapshot refuses share_not_replaceable.",
    )
    router.add_api_route(
        SHARE_PATH,
        delete_share,
        methods=["DELETE"],
        status_code=204,
        response_class=Response,
        description="Take a share down. Needs the share's write token in x-dfo-write-token.",
    )
    router.add_api_route(
        SHARE_SNAPSHOT_PATH,
        post_share_snapshot,
        methods=["POST"],
        response_model=ShareWriteResponse,
        description="Turn a live share into a snapshot under the same URLs: each slot keeps "
        "the frame it serves now, pinned. Needs the share's write token in x-dfo-write-token. "
        "A snapshot answers as it is; no route turns a snapshot live.",
    )
    router.add_api_route(
        SHARE_HISTORY_PATH,
        read_share_history,
        methods=["GET"],
        response_model=ShareHistoryResponse,
        description="Every frame the share's slots served, newest first, each with its "
        "retention counted from when DFO stored it and whether it is still fetchable. Any "
        "origin may read it.",
    )
    router.add_api_route(
        SHARE_SLOT_DATA_PATH,
        read_share_slot_data,
        methods=["GET"],
        response_class=Response,
        responses={
            200: {
                "description": "The rows behind the slot's card. JSON (the default) states "
                "the slot, its card and frame and the freshness beside the rows; csv and "
                "parquet are the card's data file. x-dfo-freshness states current, stale or "
                "pending, and a live slot's x-dfo-checked-at when DFO last checked for newer "
                "data. A strong ETag answers If-None-Match with 304. Any origin may read it.",
                "model": ShareSlotDataResponse,
                "content": {
                    "text/csv": {},
                    "application/vnd.apache.parquet": {},
                },
            },
            304: {"description": "The If-None-Match ETag still names these rows."},
        },
        description="Read the rows a share's slot serves: a snapshot's recorded frame, or a "
        "live slot's newest data, checked at most every 15 minutes.",
    )

    return DoorContribution(
        router=router,
        registry_endpoints=route_registry_endpoints(router, SHARES_ENDPOINT),
        classify=classify_shares_exception,
        exception_types=(
            WriteTokenRequired,
            WriteTokenInvalid,
            ShareNotReplaceable,
            ShareNotFound,
            ShareSlotEmpty,
            ComponentHoldsNoData,
            ComponentExportMismatch,
            SchemaInvalid,
            BindingInvalid,
            ConstraintViolation,
            ResourceExhausted,
            ReleaseMismatch,
            SharesUnavailable,
            AllowanceExhausted,
            CallerHeaderInvalid,
            CallerUnresolved,
            PathParamInvalid,
            DraftNotFound,
            RevisionNotFound,
            EditConflict,
            FrameUnavailable,
            GeneratedIdCollision,
            PresentationStoreUnavailable,
            DatasetsStoreUnavailable,
            SpecRegistryUnavailable,
            SpecRegistryStoreError,
            RequestValidationError,
        ),
        exception_exemplars=(
            WriteTokenRequired("Ab3dEf7h"),
            WriteTokenInvalid("Ab3dEf7h"),
            ShareNotReplaceable("Ab3dEf7h"),
            ShareNotFound("Ab3dEf7h"),
            ShareSlotEmpty(share_id="Ab3dEf7h", slot=2),
            SharesUnavailable(),
            AllowanceExhausted(
                kind=AllowanceKind.share, limit=0, now=datetime(2026, 1, 1, tzinfo=UTC)
            ),
            CallerHeaderInvalid(),
            CallerUnresolved("exemplar"),
            GeneratedIdCollision(subject_kind="share"),
        ),
        lifespan=lifespan,
    )
