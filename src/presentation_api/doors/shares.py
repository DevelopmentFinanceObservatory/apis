"""The shares door publishes cards under one `/s/<id>` id, serves each slot's data, and replaces,
snapshots, deletes and reads shares.

`POST /api/presentation/shares` takes `{cards: [{component_id, revision}], latest, title?, kind?,
payload?, write_token?}`. `latest` has no default. True makes a live share, which follows new
data, and false a snapshot, which keeps the versions its cards drew. `kind` defaults to `card`.
A `card` share takes 1 to 24 cards, a `canvas` 0 to 24, and a `search` or `conversation` share
none, and a `conversation` share is a snapshot. A live share holds at most
`MAX_LIVE_SHARE_DATA_CARDS` (20) data cards, those that draw a frame, on create and on replace
(the website's `LIVE_SHARE_MAX_DATA_CARDS`). More refuses 400 `request_invalid` at `/cards`.
`payload` is website-owned JSON of at most `presentation_store.MAX_SHARE_PAYLOAD_BYTES` (10 MiB)
as compact UTF-8, stored as given and read back on GET. A larger one refuses 413
`payload_too_large`, and one holding NaN or Infinity, which JSON lacks, 400 `request_invalid`
at `/payload`. The create spends one unit of the caller's `share` allowance
(`datasets_api.allowance`, caller named by `dfo_platform.caller.resolve_caller`), refusing 429
`allowance_exhausted` once it is spent. It then pins the frame each card revision draws
(`presentation_store.pin_card_frames`), live or snapshot, since a share first including a card
pins it. A pin or write that then fails gives the unit back (`datasets_api.allowance.spending`),
so a retry spends one unit in all. The reply carries the share's page, embed and per-slot data
URLs under the public site, and the `shw_` write token. The store keeps its SHA-256 digest.

A create may carry its own `shw_` token in `write_token`, in the form the server mints
(`assembler.write_token.require_client_write_token`). The token keys the create. A retry with
the same token and the same body, defaults filled, answers the share the first create made,
201 with its current URLs and revision, spending no allowance and pinning nothing. The
same token with another body refuses 400 `request_invalid` at `/write_token`, and a token whose
share was taken down answers 404 `not_found`. A server-minted token appears only in the
create's reply.

`PUT /api/presentation/shares/{share_id}` takes `{cards}` and writes the share's next revision.
Slot n shows the nth card, and slots past the last card stay empty, so every slot keeps its URLs.
It pins the card revisions the share did not show before. `DELETE` takes the share down. Both
need the share's token in `x-dfo-write-token`, checked before anything else about the share, so
a caller without it learns nothing more: no header refuses 403 `write_token_required`, another
token 403 `write_token_invalid`, and a share stored without a token (a guest's, imported from the
website) refuses every token. A replace of a snapshot then refuses 409 `share_not_replaceable`,
also when the share became one while the replace ran, and a replace of a share replaced
meanwhile refuses 409 `edit_conflict`.
A `DELETE` of a share already taken down answers 204 to the holder of its token, so a client
whose first 204 was lost can retry it, and 404 `not_found` to any other caller.

`POST /api/presentation/shares/{share_id}/snapshot` turns a live share into a snapshot in place,
under the same URLs, with the share's token. Each slot keeps the frame it serves now (its last
newest-data check, else its card's own frame). The door pins those frames, then writes the
share's next revision with them and the snapshot mode. A snapshot answers as it is, and a share
replaced meanwhile refuses 409 `edit_conflict`. No route turns a snapshot live.

The create, replace and snapshot conversion pin as the pin door does for a caller without the
pin grant: each new pinned copy, a card's frame or a derived frame's root, spends one unit of the
daily pin budget, of the open callers' budget and of the caller's `pin` allowance
(`datasets_api.retention_admission.copy_to_pinned`), and a frame pinned already spends nothing.
A spent budget refuses 503 `service_unavailable`, retryable, and a spent allowance 429
`allowance_exhausted`, each with `Retry-After` naming the next UTC midnight. A copy made before
the refusal, or before any later failure of the write, keeps its units, since the pinned bytes
stay and a retry finds them pinned.

`PUT /api/presentation/shares/{share_id}/import` stores a website share under its own id, for
the website's share migration job (`presentation_store.share_import`). It admits one caller, the
value of `PRESENTATION_SHARE_IMPORT_PRINCIPAL` in the principal header, checked before the body
is read; another caller, and every caller while the principal is unset, refuses 403
`share_import_not_permitted`. The body is the share's export line less its `id`, plus
`write_token_sha256`, the hex digest of the write token the website holds, or null for a
guest's share. It pins the frames the share shows as an operator's import does, spending no
allowance or budget. It answers 201 `{id, imported: true}` when it stores the share and 200 with
`imported` false when an earlier import stored it, with the same kind, `created_at` and digest,
taken down since or not; the first import wins, so other content answers 200 too. An id holding
another share, or this one under another digest, and a digest another share holds, refuse 409
`share_id_taken`. Neither pins anything. A body that is not such a line refuses 400 `request_invalid`, a card
revision this service does not hold 404 `not_found`, and a frame that cannot be pinned
`frame_unavailable`, as a create does.

`GET /api/presentation/shares/{share_id}` is public and allows any origin. It returns kind,
mode, title, revision, slots, the website's payload and created_at, and omits the token.

`GET /api/presentation/shares/{share_id}/slots/{slot}/data?format=json|csv|xlsx|parquet&bundle=`
is public and allows any origin. It serves the rows behind the slot's card, JSON by default, on
the frame `presentation_api.share_slots` chooses. That is a snapshot's recorded frame, or a live
slot's newest data, checked again when the last check falls due: after
`newest_frames.FRESHNESS_WINDOW`, or sooner for a `pending` check, after the retry time the
refresh or the run gave. The JSON body and the `x-dfo-freshness` header state `current`, `stale`
or `pending`, and a live slot's `x-dfo-checked-at` says when the check ran. `csv`, `xlsx` and
`parquet` are the card's data file as the components door's data route serves it, unbundled by
default. `bundle=true` serves that route's zip, and with `json` refuses 400 `request_invalid`.
The reply carries a strong `ETag` over what its body is drawn from: the share, its mode and the
slot, the card revision, the frame served and its freshness, the format, and for a card drawn
on a newer frame the catalogue release. An `If-None-Match` naming it answers 304 without
drawing the body while the slot needs no check: a snapshot, or a live slot whose last check is
not yet due. A live slot's reply is `no-cache`, so a cache revalidates it, and a snapshot's is
`public, max-age=300`. A slot number the share has not had, or a slot a
replacement left empty, answers 404 `not_found`, and so does a note's slot.

`GET /api/presentation/shares/{share_id}/slots/{slot}` is public and allows any origin. It
answers the slot's card as `GET /api/presentation/components/{component_id}` reads it, plus
`freshness`, drawn on the frame the data route serves, chosen by the same function
(`share_slots.serve_on_slot_frame`), with that frame's retention. Its `x-dfo-freshness` and
`x-dfo-checked-at` headers, strong `ETag`, 304 and caching are the data route's, and its
`ETag` also covers the frame's retention and, on a newer frame, the vocabulary revision the
read is worded in. A note's card is its stored read. A slot number the share has not had, or an
empty slot, answers 404 `not_found`.

`GET /api/presentation/shares/{share_id}/history` is public and allows any origin. It returns a
timeline of what each slot showed, newest first, each frame with its retention counted from when
DFO stored it and whether it is still fetchable (`share_slots.history_document`). A create, replace
or snapshot conversion records the frame each slot serves in the transaction that writes its
revision, dated with the revision (`ShareStore.create_share`): a live slot whose card kept its
place serves its last check's frame. The history then lists what a share shows before any read,
in the order its writes committed, and a write that cannot record commits nothing. A slot a
replace leaves empty records an entry with `component_id`, `revision`, `frame`, `rendering` and
`retention` null and `fetchable` false, and so does a slot whose card is a note, which draws no
frame. A slot records an entry only when what it shows changes. The route answers a page of
`limit` entries (1 to `HISTORY_PAGE_MAX`, `HISTORY_PAGE_DEFAULT` when left out) and `next`, an
opaque cursor that `before` takes to continue after the page's last entry, `null` on the last
page. A `limit` out of range, or a `before` no page gave, refuses 400 `request_invalid`.

Every `GET` reply of a public route allows any origin, a refusal included, even one sent before
routing, since the platform's CORS layer (`dfo_platform.cors`) adds the header. A share id that
names no share, or one taken down, answers 404 `not_found` on every route. A card's frame that
is gone answers 404 `frame_unavailable`, and a frame read that fails answers retryable 503
`service_unavailable`.

Handlers are module-level functions that read `SHARES_DEPS_SLOT`, which the door's lifespan
sets. Every route runs on one executor of `PRESENTATION_COMPONENTS_MAX_CONCURRENCY` threads. A
request that finds every thread busy waits up to `SHARES_ADMISSION_WAIT_SECONDS` for one, so a
share page reading more slots than the executor holds is served, and past the wait it answers
retryable 503.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import math
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
    field_validator,
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
    VocabularySource,
)
from assembler.write_token import (
    SHARE_WRITE_TOKEN_PREFIX,
    CreateRequestInvalid,
    mint_write_token,
    request_digest,
    require_client_write_token,
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
    SharedAllowanceLedger,
    allowance_limits,
    spending,
)
from datasets_api.artifact_delivery import artifact_content_disposition
from datasets_api.config import DatasetsBuildSettings
from datasets_api.door_executor import DoorExecutor, build_door_executor
from datasets_api.principals import Grant, Principal, Principals, granted_route_class
from datasets_api.retention_admission import (
    CopyClaimUnsettled,
    OpenPinBudgetExhausted,
    PinBudgetExhausted,
    PinUnitHeld,
)
from datasets_api.shutdown import ShutdownBudget
from datasets_api.store_errors import DatasetsStoreUnavailable
from dfo_platform import ProblemSpec, refused
from dfo_platform.caller import Caller, CallerHeaderInvalid, CallerUnresolved, resolve_caller
from dfo_platform.config import CallerSettings, PublicSiteSettings
from dfo_platform.doors import DoorContribution
from dfo_platform.state import StateSlot
from materialise.formats import ExcelIllegalCharacterError
from presentation_api.config import (
    PRESENTATION_COMPONENTS_MAX_CONCURRENCY,
    PRESENTATION_COMPONENTS_TIMEOUT_SECONDS,
    PresentationSettings,
)
from presentation_api.doors.component_data import (
    COMPONENT_DATA_EXCEPTION_EXEMPLARS,
    COMPONENT_DATA_MEDIA_TYPES,
    classify_component_data_exception,
)
from presentation_api.doors.components import (
    DRAW_EXEMPLARS,
    LOOKUP_EXEMPLARS,
    STORE_EXEMPLARS,
    WRITE_TOKEN_HEADER,
    classify_components_exception,
)
from presentation_api.live import build_live_frame_store
from presentation_api.models import (
    SharedResponse,
    ShareHistoryResponse,
    ShareImportResponse,
    ShareReadResponse,
    ShareSlotCardResponse,
    ShareSlotDataResponse,
    ShareWriteResponse,
)
from presentation_api.newest_frames import NewestFrames
from presentation_api.path_params import PathParamInvalid, require_component_id, require_share_id
from presentation_api.refusals import FRAME_UNAVAILABLE_EXEMPLARS
from presentation_api.retention import stated_retention
from presentation_api.routes import (
    SHARE_HISTORY_PATH,
    SHARE_IMPORT_PATH,
    SHARE_PATH,
    SHARE_SLOT_DATA_PATH,
    SHARE_SLOT_PATH,
    SHARE_SNAPSHOT_PATH,
    SHARES_ENDPOINT,
    SHARES_PATH,
    door_registry_endpoints,
)
from presentation_api.share_slots import (
    HISTORY_PAGE_DEFAULT,
    HISTORY_PAGE_MAX,
    ShareSlotEmpty,
    SlotData,
    SlotFormat,
    history_document,
    history_position,
    serve_slot,
    serve_slot_card,
    serving_frame,
    slot_card,
)
from presentation_api.vocabulary import LiveVocabularySource
from presentation_store import (
    MAX_SHARE_PAYLOAD_BYTES,
    Freshness,
    GeneratedIdCollision,
    LivePresentationStore,
    LiveShareStore,
    PresentationStoreUnavailable,
    Renderings,
    ShareCard,
    ShareIdTaken,
    ShareNotFound,
    ShareNotReplaceable,
    ShareRecord,
    ShareStore,
    ShareWriteTokenInUse,
    frame_reads_bounded_by,
    payload_refusal,
    pin_card_frames,
)
from presentation_store import close_shared_pool as close_presentation_pool
from presentation_store import shared_pool as presentation_shared_pool
from presentation_store.share_import import ShareImport, SlotDrawsNoFrame, import_website_share
from spec_registry.errors import SpecRegistryStoreError, SpecRegistryUnavailable

__all__ = [
    "MAX_LIVE_SHARE_DATA_CARDS",
    "MAX_SHARE_CARDS",
    "MAX_SHARE_PAYLOAD_BYTES",
    "SHARES_DEPS_SLOT",
    "SharesDeps",
    "build_shares_door",
    "classify_shares_exception",
]

MAX_SHARE_CARDS = 24
"""The most cards one create or replace takes, and so the most slots a write adds."""

MAX_LIVE_SHARE_DATA_CARDS = 20
"""The most data cards, those that draw a frame, a live share holds. It matches the website's
`LIVE_SHARE_MAX_DATA_CARDS`, since each one's slot checks for newer data on read."""

_MAX_TITLE_LENGTH = 300

# The principals a door built with none admits: nobody, so the import route refuses every caller.
_NO_PRINCIPALS = Principals(header="x-dfo-principal", secrets=())

_FRESHNESS_HEADER = "x-dfo-freshness"
_CHECKED_AT_HEADER = "x-dfo-checked-at"
_SLOT_DATA_EXPOSED = f"ETag, {_FRESHNESS_HEADER}, {_CHECKED_AT_HEADER}"
_LIVE_CACHE_CONTROL = "no-cache"
_SNAPSHOT_CACHE_CONTROL = "public, max-age=300"

# How many cards each kind takes, on create and on replace.
_KIND_CARDS: Mapping[str, tuple[int, int]] = {
    "card": (1, MAX_SHARE_CARDS),
    "canvas": (0, MAX_SHARE_CARDS),
    "search": (0, 0),
    "conversation": (0, 0),
}


def _card_count_refusal(kind: str, count: int) -> str | None:
    """Why a share of `kind` cannot show `count` cards, or `None` when it can. The create
    body and a replace both ask it."""
    least, most = _KIND_CARDS[kind]
    if least <= count <= most:
        return None
    wanted = "none" if most == 0 else f"{least} to {most}"
    return f"a {kind} share takes {wanted} cards, not {count}"


def _holds_non_finite(value: JsonValue) -> bool:
    """Whether `value` holds a NaN or an infinite number anywhere."""
    if isinstance(value, float):
        return not math.isfinite(value)
    if isinstance(value, list):
        return any(_holds_non_finite(item) for item in value)
    if isinstance(value, dict):
        return any(_holds_non_finite(item) for item in value.values())
    return False


class ShareCardBody(BaseModel):
    """One card revision a share shows."""

    model_config = ConfigDict(extra="forbid")

    component_id: str = Field(description="the card's cmp_ id")
    revision: StrictInt = Field(ge=1, description="the card revision the slot shows")


class ShareCardsBody(BaseModel):
    """A replace's body: the cards the share's next revision shows, slot 1 first. The share's
    kind sets how many it takes, as on create."""

    model_config = ConfigDict(extra="forbid")

    cards: list[ShareCardBody] = Field(
        max_length=MAX_SHARE_CARDS,
        description="the cards in slot order. Two entries may name the same revision. A card "
        "share takes at least one, a canvas any number, a search or conversation share none",
    )


class CreateShareBody(BaseModel):
    """A create's body. `latest` has no default, because a live link and a fixed one answer
    different questions under a public URL."""

    model_config = ConfigDict(extra="forbid")

    cards: list[ShareCardBody] = Field(
        default_factory=list,
        max_length=MAX_SHARE_CARDS,
        description="the cards in slot order. Two entries may name the same revision. A card "
        "share takes at least one, a canvas any number, a search or conversation share none",
    )
    latest: StrictBool = Field(
        description="true for a live share that follows new data and takes replacements, "
        "false for a snapshot that keeps the versions its cards draw now, for good"
    )
    title: str | None = Field(
        default=None, min_length=1, max_length=_MAX_TITLE_LENGTH, description="the share's title"
    )
    kind: Literal["card", "canvas", "search", "conversation"] = Field(
        default="card",
        description="what the share publishes. The website renders a canvas, search or "
        "conversation share from its payload. A conversation share is a snapshot",
    )
    payload: JsonValue = Field(
        default=None,
        description="website-owned JSON stored with the share as given and read back on GET, "
        f"at most {MAX_SHARE_PAYLOAD_BYTES} bytes as compact UTF-8",
    )
    write_token: str | None = Field(
        default=None,
        description='a shw_ token this client minted for the new share: "shw_" + '
        "secrets.token_urlsafe(16) or an equivalent CSPRNG, 22 url-safe base64 characters "
        "after the prefix. It becomes the share's write token and keys the create. A retry "
        "with the same write_token and body answers the first share, spending no allowance, "
        "and a different body with it is refused as request_invalid. Mint a new one for each "
        "create.",
    )

    @field_validator("payload")
    @classmethod
    def _finite(cls, payload: JsonValue) -> JsonValue:
        # Python's JSON parser reads NaN and Infinity, which JSON and Postgres's jsonb lack.
        if _holds_non_finite(payload):
            raise ValueError("the payload holds NaN or Infinity, which JSON cannot carry")
        return payload

    @model_validator(mode="after")
    def _fits_its_kind(self) -> Self:
        if refusal := _card_count_refusal(self.kind, len(self.cards)):
            raise ValueError(refusal)
        if self.kind == "conversation" and self.latest:
            raise ValueError("a conversation share is a snapshot: send latest=false")
        return self

    def request_digest(self) -> bytes:
        """The digest a client's write token keys: the body as sent, less the token, with its
        defaults filled.

        Raises:
            CreateRequestInvalid: The payload holds a value RFC 8785 cannot represent.
        """
        return request_digest(
            self.model_dump(mode="json", exclude={"write_token"}), pointer="/payload"
        )


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


class ShareImportNotPermitted(Exception):
    """The caller lacks the `share_import` grant. Maps to 403 `share_import_not_permitted`."""

    def __init__(self, *, presented: bool) -> None:
        self.presented = presented
        detail = "the share import is restricted to its first-party principal"
        if not presented:
            detail = f"no principal header was presented; {detail}"
        super().__init__(detail)


def _share_import_gate(principals: Principals) -> Callable[[Request], Principal]:
    """The check the import route runs before FastAPI reads its body: the caller must hold
    `share_import` under `principals`.

    Raises:
        ShareImportNotPermitted: The caller lacks the grant.
    """

    def check(request: Request) -> Principal:
        principal = principals.parse(request)
        if Grant.share_import not in principal.grants:
            raise ShareImportNotPermitted(presented=principal.presented is not None)
        return principal

    return check


class SharePayloadTooLarge(Exception):
    """A create's payload is over `MAX_SHARE_PAYLOAD_BYTES`. Maps to 413 `payload_too_large`."""


# How long a request waits for a thread of the shares executor before it answers 503.
SHARES_ADMISSION_WAIT_SECONDS = 5.0


class SharesUnavailable(Exception):
    """No executor thread came free within `SHARES_ADMISSION_WAIT_SECONDS`, or the lifespan has
    not set the door's dependencies. Maps to retryable 503 `service_unavailable`."""

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
    # The frame store the share import pins through, spending no allowance or budget.
    import_frame_store: FrameStore
    # The version records history entries take their retention from; `None` states none.
    snapshot_reader: SnapshotReader | None = None
    # Where a live slot asks for newer data; `None` leaves every live slot stale.
    newest: NewestFrames | None = None
    # The words a card drawn on a newer frame is described in; `None` describes none.
    vocabulary_source: VocabularySource | None = None
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
) -> tuple[list[ShareCard], Renderings]:
    """The stored card revisions `cards` name, each with the frame it draws, and the rendering
    each recorded.

    Raises:
        PathParamInvalid: A component id is malformed.
        DraftNotFound: No component exists at an id.
        RevisionNotFound: A component has no such revision.
        EditConflict: A component has no revision at all.
    """
    read: list[ShareCard] = []
    renderings: dict[tuple[str, int], Mapping[str, object] | None] = {}
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
        renderings[(card.component_id, card.revision)] = record.frame_rendering
    return read, renderings


def _require_live_data_cards(cards: Iterable[ShareCard]) -> None:
    """Refuse more than `MAX_LIVE_SHARE_DATA_CARDS` data cards for a live share.

    Raises:
        RequestValidationError: `cards` holds more data cards than a live share takes.
    """
    count = sum(1 for card in cards if card.frame is not None)
    if count > MAX_LIVE_SHARE_DATA_CARDS:
        raise RequestValidationError(
            [
                {
                    "type": "value_error",
                    "loc": ("body", "cards"),
                    "msg": f"a live share holds at most {MAX_LIVE_SHARE_DATA_CARDS} data cards, "
                    f"not {count}",
                    "input": None,
                }
            ]
        )


def _pin(cards: Iterable[ShareCard], *, caller: Caller, deps: SharesDeps) -> None:
    """Pin the frame each distinct card revision in `cards` draws, each new pinned copy
    spending `caller`'s units."""
    for component_id, revision in dict.fromkeys((c.component_id, c.revision) for c in cards):
        pin_card_frames(
            component_id=component_id,
            revision=revision,
            revision_store=deps.revision_store,
            frame_store=deps.frame_store,
            caller=caller,
        )


def _renderings(share: ShareRecord, *, revision_store: RevisionStore) -> Renderings:
    """The rendering each data card revision `share` shows recorded."""
    keys = (
        (slot.card.component_id, slot.card.revision)
        for slot in share.slots
        if slot.card is not None and slot.card.frame is not None
    )
    return {
        (component_id, revision): read_stored_revision(
            component_id=component_id, revision=revision, revision_store=revision_store
        ).frame_rendering
        for component_id, revision in dict.fromkeys(keys)
    }


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
            deps.executor.run(
                job, refuse=SharesUnavailable, wait_seconds=SHARES_ADMISSION_WAIT_SECONDS
            ),
            timeout=PRESENTATION_COMPONENTS_TIMEOUT_SECONDS,
        )


def _created(share: ShareRecord, *, token: str, deps: SharesDeps) -> Response:
    document = {**_locations(share, site=deps.site.public_site_url), "write_token": token}
    return JSONResponse(document, status_code=201, headers={"Cache-Control": "no-store"})


def _replay_create(
    *, token: str, token_sha256: bytes, request_sha256: bytes, deps: SharesDeps
) -> Response | None:
    """The first share a client's token made, answered as its create was, or `None` when no
    share holds the token.

    Raises:
        CreateRequestInvalid: The token's share was made from another request, or from none.
        ShareNotFound: The token's share was taken down.
    """
    prior = deps.share_store.prior_create(write_token_sha256=token_sha256)
    if prior is None:
        return None
    if prior.create_request_sha256 != request_sha256:
        raise CreateRequestInvalid(
            pointer="/write_token",
            detail="write_token already names a share created from a different request; "
            "mint a new write_token for each create",
        )
    share = deps.share_store.read_share(share_id=prior.share_id)
    return _created(share, token=token, deps=deps)


async def create_share(body: CreateShareBody, request: Request, deps: SharesDepsDep) -> Response:
    caller = resolve_caller(request, deps.caller)

    def job() -> Response:
        if refusal := payload_refusal(body.payload):
            raise SharePayloadTooLarge(refusal)
        if body.write_token is not None:
            token = require_client_write_token(body.write_token, prefix=SHARE_WRITE_TOKEN_PREFIX)
            request_sha256: bytes | None = body.request_digest()
        else:
            token = mint_write_token(prefix=SHARE_WRITE_TOKEN_PREFIX)
            request_sha256 = None
        token_sha256 = write_token_digest(token)
        if request_sha256 is not None and (
            replayed := _replay_create(
                token=token, token_sha256=token_sha256, request_sha256=request_sha256, deps=deps
            )
        ):
            return replayed
        cards, renderings = _read_cards(body.cards, revision_store=deps.revision_store)
        if body.latest:
            _require_live_data_cards(cards)
        try:
            with spending(deps.allowances, caller, AllowanceKind.share, deps.now()):
                _pin(cards, caller=caller, deps=deps)
                share = deps.share_store.create_share(
                    kind=body.kind,
                    mode="live" if body.latest else "snapshot",
                    title=body.title,
                    write_token_sha256=token_sha256,
                    cards=cards,
                    renderings=renderings,
                    payload=body.payload,
                    create_request_sha256=request_sha256,
                )
        except ShareWriteTokenInUse:
            # A create with the same token committed first, so this one replays it. A collision of
            # a server-minted token, 128 random bits, means the minter is broken. The unit went
            # back, since the first create spent its own.
            if request_sha256 is None:
                raise GeneratedIdCollision(subject_kind="share write token") from None
            replayed = _replay_create(
                token=token, token_sha256=token_sha256, request_sha256=request_sha256, deps=deps
            )
            if replayed is None:
                raise EditConflict from None
            return replayed
        return _created(share, token=token, deps=deps)

    return await _run(deps, job)


async def replace_share(
    share_id: str,
    body: ShareCardsBody,
    request: Request,
    deps: SharesDepsDep,
    write_token: WriteTokenHeader = None,
) -> Response:
    require_share_id(share_id)

    def job() -> Response:
        current = deps.share_store.read_share(share_id=share_id)
        authorise_share_write(current, presented=write_token)
        if current.mode == "snapshot":
            raise ShareNotReplaceable(share_id)
        caller = resolve_caller(request, deps.caller)
        if refusal := _card_count_refusal(current.kind, len(body.cards)):
            raise RequestValidationError(
                [{"type": "value_error", "loc": ("body", "cards"), "msg": refusal, "input": None}]
            )
        cards, renderings = _read_cards(body.cards, revision_store=deps.revision_store)
        _require_live_data_cards(cards)
        shown = {
            (slot.card.component_id, slot.card.revision)
            for slot in current.slots
            if slot.card is not None
        }
        _pin(
            (card for card in cards if (card.component_id, card.revision) not in shown),
            caller=caller,
            deps=deps,
        )
        share = deps.share_store.replace_share(
            share_id=share_id, revision=current.revision, cards=cards, renderings=renderings
        )
        return JSONResponse(_locations(share, site=deps.site.public_site_url))

    return await _run(deps, job)


def _holds_token(share_id: str, *, presented: str | None, deps: SharesDeps) -> bool:
    """Whether `presented` is the write token of the share `share_id`, taken down or not."""
    if presented is None:
        return False
    held = deps.share_store.prior_create(write_token_sha256=write_token_digest(presented))
    return held is not None and held.share_id == share_id


async def delete_share(
    share_id: str, deps: SharesDepsDep, write_token: WriteTokenHeader = None
) -> Response:
    require_share_id(share_id)

    def job() -> Response:
        try:
            current = deps.share_store.read_share(share_id=share_id)
        except ShareNotFound:
            if not _holds_token(share_id, presented=write_token, deps=deps):
                raise
            return Response(status_code=204)
        authorise_share_write(current, presented=write_token)
        # A concurrent delete by the holder took it down first, so this delete succeeded too.
        with contextlib.suppress(ShareNotFound):
            deps.share_store.delete_share(share_id=share_id)
        return Response(status_code=204)

    return await _run(deps, job)


async def read_share(share_id: str, deps: SharesDepsDep) -> Response:
    require_share_id(share_id)

    def job() -> Response:
        share = deps.share_store.read_share(share_id=share_id)
        return JSONResponse(_read_document(share, site=deps.site.public_site_url))

    return await _run(deps, job)


async def post_share_snapshot(
    share_id: str, request: Request, deps: SharesDepsDep, write_token: WriteTokenHeader = None
) -> Response:
    require_share_id(share_id)

    def job() -> Response:
        current = deps.share_store.read_share(share_id=share_id)
        authorise_share_write(current, presented=write_token)
        if current.mode == "live":
            caller = resolve_caller(request, deps.caller)
            frames: dict[int, FrameRef] = {}
            pins: dict[tuple[str, FrameRef], None] = {}
            for slot in current.slots:
                if slot.card is None or slot.card.frame is None:
                    continue
                frame = serving_frame(share=current, slot=slot.slot, card=slot.card, deps=deps)
                frames[slot.slot] = frame
                pins[(slot.card.component_id, frame)] = None
            for component_id, frame in pins:
                deps.frame_store.pin(frame_ref=frame, component_id=component_id, caller=caller)
            current = deps.share_store.snapshot_share(
                share_id=share_id,
                revision=current.revision,
                frames=frames,
                renderings=_renderings(current, revision_store=deps.revision_store),
            )
        return JSONResponse(_locations(current, site=deps.site.public_site_url))

    return await _run(deps, job)


async def put_share_import(share_id: str, body: ShareImport, deps: SharesDepsDep) -> Response:
    require_share_id(share_id)

    def job() -> Response:
        imported = import_website_share(
            share_id,
            body,
            share_store=deps.share_store,
            revision_store=deps.revision_store,
            frame_store=deps.import_frame_store,
        )
        return JSONResponse(
            {"id": share_id, "imported": imported},
            status_code=201 if imported else 200,
            headers={"Cache-Control": "no-store"},
        )

    return await _run(deps, job)


async def read_share_history(
    share_id: str,
    deps: SharesDepsDep,
    limit: Annotated[
        int,
        Query(
            ge=1,
            le=HISTORY_PAGE_MAX,
            description=f"the most entries the page holds, 1 to {HISTORY_PAGE_MAX}",
        ),
    ] = HISTORY_PAGE_DEFAULT,
    before: Annotated[
        str | None,
        Query(
            description="the `next` cursor of the previous page; the page continues after "
            "that page's last entry"
        ),
    ] = None,
) -> Response:
    require_share_id(share_id)
    position = history_position(before) if before is not None else None
    if before is not None and position is None:
        raise RequestValidationError(
            [
                {
                    "type": "value_error",
                    "loc": ("query", "before"),
                    "msg": "not a cursor a history page gave as next",
                    "input": before,
                }
            ]
        )

    def job() -> Response:
        page = deps.share_store.served_frames(share_id=share_id, limit=limit, before=position)
        return JSONResponse(
            history_document(share_id=share_id, page=page, deps=deps),
            headers={"Cache-Control": "no-store"},
        )

    return await _run(deps, job)


def _matches(if_none_match: str | None, etag: str) -> bool:
    """Whether `If-None-Match` names `etag`, compared weakly as RFC 9110 says for it."""
    if if_none_match is None:
        return False
    tags = {tag.strip().removeprefix("W/") for tag in if_none_match.split(",")}
    return "*" in tags or etag in tags


@dataclass(frozen=True)
class _SlotState:
    """What a slot serves: the frame (`None` for a note's card), its freshness, and when a live
    slot's check ran (`None` for a snapshot or a note)."""

    frame: FrameRef | None
    freshness: Freshness
    checked_at: datetime | None

    @classmethod
    def of(cls, data: SlotData) -> Self:
        return cls(frame=data.frame, freshness=data.freshness, checked_at=data.checked_at)


def _settled_state(
    share: ShareRecord, *, slot: int, card: ShareCard, deps: SharesDeps
) -> _SlotState | None:
    """What slot `slot` of `share`, showing `card`, serves now, when a read need not check for
    newer data first: a note's card, a snapshot's frame, or a live slot's last check before it
    falls due. `None` for a live slot whose check is due or has never run, so the read runs it
    (`share_slots.serve_on_slot_frame`)."""
    if card.frame is None:
        return _SlotState(frame=None, freshness="current", checked_at=None)
    if share.mode == "snapshot":
        return _SlotState(frame=card.frame, freshness="current", checked_at=None)
    check = deps.share_store.slot_check(share_id=share.id, slot=slot, card=card)
    if check is None or check.next_check_at <= deps.now():
        return None
    return _SlotState(frame=check.frame, freshness=check.freshness, checked_at=check.checked_at)


def _redrawn_under(
    card: ShareCard, frame: FrameRef | None, *, worded: bool, deps: SharesDeps
) -> list[str | None] | None:
    """The catalogue release a card drawn on `frame`, a frame other than its own, is drawn
    under, and with `worded` the vocabulary revision its read is worded in. `None` on the card's
    own frame, whose export and read are stored, and for a note."""
    if frame is None or frame == card.frame:
        return None
    vocabulary = (
        deps.vocabulary_source.current().revision
        if worded and deps.vocabulary_source is not None
        else None
    )
    return [deps.release_registry.pinned().catalog_version, vocabulary]


def _slot_etag(
    share: ShareRecord,
    *,
    slot: int,
    card: ShareCard,
    state: _SlotState,
    variant: str,
    drawn_with: object,
) -> str:
    """A slot reply's strong ETag. It hashes what the body is drawn from, so a revalidation
    answers 304 without drawing the body: the share, its mode and the slot, the card revision,
    the frame served and its freshness, `variant` (the route and format), and `drawn_with`,
    whatever else the body states (a card read's retention, a newer frame's release and
    vocabulary). Equal inputs draw equal bodies, so they carry one ETag."""
    validator = {
        "share": share.id,
        "mode": share.mode,
        "slot": slot,
        "card": [card.component_id, card.revision],
        "frame": (
            [state.frame.spec_id, state.frame.content_version] if state.frame is not None else None
        ),
        "freshness": state.freshness,
        "variant": variant,
        "drawn_with": drawn_with,
    }
    encoded = json.dumps(validator, sort_keys=True, separators=(",", ":")).encode()
    return f'"{hashlib.sha256(encoded).hexdigest()[:32]}"'


def _slot_headers(share: ShareRecord, *, state: _SlotState, etag: str) -> dict[str, str]:
    headers = {
        "ETag": etag,
        _FRESHNESS_HEADER: state.freshness,
        "Cache-Control": _LIVE_CACHE_CONTROL if share.mode == "live" else _SNAPSHOT_CACHE_CONTROL,
        "Access-Control-Expose-Headers": _SLOT_DATA_EXPOSED,
    }
    if state.checked_at is not None:
        headers[_CHECKED_AT_HEADER] = state.checked_at.isoformat()
    return headers


def _slot_response(
    data: SlotData, *, share: ShareRecord, etag: str, if_none_match: str | None
) -> Response:
    """A slot read's reply: its freshness, caching and strong ETag, 304 when `If-None-Match`
    names that ETag."""
    headers = _slot_headers(share, state=_SlotState.of(data), etag=etag)
    if _matches(if_none_match, etag):
        return Response(status_code=304, headers=headers)
    if data.file_name is not None:
        headers["Content-Disposition"] = artifact_content_disposition(data.file_name)
    return Response(content=data.body, media_type=data.media_type, headers=headers)


async def read_share_slot_data(
    share_id: str,
    deps: SharesDepsDep,
    slot: Annotated[int, Path(ge=1)],
    format_: Annotated[SlotFormat, Query(alias="format")] = "json",
    bundle: Annotated[bool, Query()] = False,
    if_none_match: Annotated[str | None, Header()] = None,
) -> Response:
    require_share_id(share_id)
    if bundle and format_ == "json":
        raise RequestValidationError(
            [
                {
                    "type": "value_error",
                    "loc": ("query", "bundle"),
                    "msg": "a bundle holds a csv, xlsx or parquet file; json rows have none",
                    "input": "true",
                }
            ]
        )
    variant = f"data:{format_}:bundle" if bundle else f"data:{format_}"

    def etag(share: ShareRecord, card: ShareCard, state: _SlotState) -> str:
        drawn_with = _redrawn_under(card, state.frame, worded=False, deps=deps)
        return _slot_etag(
            share, slot=slot, card=card, state=state, variant=variant, drawn_with=drawn_with
        )

    def job() -> Response:
        share = deps.share_store.read_share(share_id=share_id)
        card = slot_card(share, slot)
        # A note's slot refuses, so only a data card's slot answers 304.
        if if_none_match is not None and card.frame is not None:
            state = _settled_state(share, slot=slot, card=card, deps=deps)
            if state is not None and _matches(if_none_match, tag := etag(share, card, state)):
                return Response(
                    status_code=304, headers=_slot_headers(share, state=state, etag=tag)
                )
        data = serve_slot(share=share, slot=slot, format=format_, bundle=bundle, deps=deps)
        tag = etag(share, card, _SlotState.of(data))
        return _slot_response(data, share=share, etag=tag, if_none_match=if_none_match)

    return await _run(deps, job)


async def read_share_slot_card(
    share_id: str,
    deps: SharesDepsDep,
    slot: Annotated[int, Path(ge=1)],
    if_none_match: Annotated[str | None, Header()] = None,
) -> Response:
    require_share_id(share_id)

    def etag(share: ShareRecord, card: ShareCard, state: _SlotState, retention: object) -> str:
        drawn_with = [retention, _redrawn_under(card, state.frame, worded=True, deps=deps)]
        return _slot_etag(
            share, slot=slot, card=card, state=state, variant="card", drawn_with=drawn_with
        )

    def job() -> Response:
        share = deps.share_store.read_share(share_id=share_id)
        card = slot_card(share, slot)
        if if_none_match is not None:
            state = _settled_state(share, slot=slot, card=card, deps=deps)
            if state is not None:
                retention = (
                    stated_retention(state.frame, snapshot_reader=deps.snapshot_reader)
                    if state.frame is not None
                    else None
                )
                if _matches(if_none_match, tag := etag(share, card, state, retention)):
                    headers = _slot_headers(share, state=state, etag=tag)
                    return Response(status_code=304, headers=headers)
        data = serve_slot_card(share=share, slot=slot, deps=deps)
        # The read states its frame's retention as it read it, so the ETag hashes that.
        retention = json.loads(data.body)["retention"]
        tag = etag(share, card, _SlotState.of(data), retention)
        return _slot_response(data, share=share, etag=tag, if_none_match=if_none_match)

    return await _run(deps, job)


def classify_shares_exception(exc: BaseException) -> ProblemSpec | None:
    """The share refusals, then the components door's for the card, store and frame failures
    the share routes share with it."""
    if isinstance(exc, WriteTokenRequired):
        return refused(
            "write_token_required",
            detail=str(exc),
            retryable=False,
        )
    if isinstance(exc, WriteTokenInvalid):
        return refused(
            "write_token_invalid",
            detail=str(exc),
            retryable=False,
        )
    if isinstance(exc, ShareNotReplaceable):
        return refused(
            "share_not_replaceable",
            detail=str(exc),
            retryable=False,
        )
    if isinstance(exc, ShareIdTaken):
        return refused(
            "share_id_taken",
            detail=str(exc),
            retryable=False,
        )
    if isinstance(exc, ShareImportNotPermitted):
        return refused(
            "share_import_not_permitted",
            detail=str(exc),
            retryable=False,
        )
    if isinstance(exc, SlotDrawsNoFrame):
        return refused(
            "request_invalid",
            detail=str(exc),
            pointer=f"/slots/{exc.index}/frame",
            retryable=False,
        )
    if isinstance(exc, ShareNotFound | ShareSlotEmpty):
        return refused("not_found", detail=str(exc), retryable=False)
    if isinstance(
        exc, PinBudgetExhausted | OpenPinBudgetExhausted | PinUnitHeld | CopyClaimUnsettled
    ):
        return refused(
            "service_unavailable",
            detail=exc.detail or str(exc),
            retryable=True,
            headers=exc.headers,
        )
    if isinstance(exc, SharePayloadTooLarge):
        return refused(
            "payload_too_large",
            detail=str(exc),
            retryable=False,
        )
    if isinstance(exc, SharesUnavailable):
        return refused(
            "service_unavailable",
            detail=str(exc),
            retryable=True,
        )
    return classify_components_exception(exc) or classify_component_data_exception(exc)


def build_shares_door(
    *,
    release_registry: ReleaseRegistry | None = None,
    share_store: ShareStore | None = None,
    revision_store: RevisionStore | None = None,
    frame_store: FrameStore | None = None,
    snapshot_reader: SnapshotReader | None = None,
    newest: NewestFrames | None = None,
    vocabulary_source: VocabularySource | None = None,
    allowances: SharedAllowanceLedger | None = None,
    caller: CallerSettings | None = None,
    principals: Principals | None = None,
    now: Callable[[], datetime] | None = None,
    pool_size: int | None = None,
    budget: ShutdownBudget | None = None,
) -> DoorContribution:
    """Build the shares door. The lifespan builds each store left out over the live pools: the
    share and revision stores over the presentation pool, the frame store over the datasets
    pool, and the allowance ledger over the datasets pool with `DatasetsBuildSettings`' limits.
    The share unit and a frame store built here spend from one ledger, `allowances` when given.
    A frame store built here brings its snapshot reader; with one given, `snapshot_reader` is
    the history's, and left out the history states no retention. `newest` answers live slots'
    newer-data checks; left out, every live slot serves its card's frame as stale.
    `vocabulary_source` gives the words a slot's card drawn on a newer frame is described in,
    and defaults to a `LiveVocabularySource` over the code index.
    `release_registry`, which a live slot draws its card on a newer frame under, defaults to the
    baked release. `caller` defaults to those settings' `caller`, and the public site to
    `DFO_PUBLIC_SITE_URL`. The import route admits the principal `principals` grants
    `share_import`, and with no `principals` refuses every caller. It pins through the frame
    store given, or through a store built here that spends nothing.
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
            effective_frame_store = frame_store
            import_frame_store = frame_store
            effective_snapshot_reader = snapshot_reader
            if effective_frame_store is None:
                live = build_live_frame_store(
                    settings=PresentationSettings.from_env(), allowances=effective_allowances
                )
                dataset_borrows += 1
                owned_graph = live.graph
                effective_frame_store = live.frame_store
                import_frame_store = live.unmetered_frame_store
                effective_snapshot_reader = live.snapshot_reader
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
                import_frame_store=cast(FrameStore, import_frame_store),
                snapshot_reader=effective_snapshot_reader,
                newest=newest,
                vocabulary_source=(
                    vocabulary_source if vocabulary_source is not None else LiveVocabularySource()
                ),
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
        "allowance, and one unit of its daily pin allowance and of the shared pin budgets for "
        "each new pinned copy. The reply's write_token grants replacing and deleting the share.",
    )
    router.add_api_route(
        SHARE_PATH,
        read_share,
        methods=["GET"],
        response_model=ShareReadResponse,
        description="Read a share: its kind, mode, title, latest revision, slots and the "
        "website's payload. Any origin may read it. It omits the write token.",
    )
    router.add_api_route(
        SHARE_PATH,
        replace_share,
        methods=["PUT"],
        response_model=ShareWriteResponse,
        description="Give a live share new cards under the same URLs, as its next revision. "
        "Slot n shows the nth card, and slots past the last card stay empty. Needs the share's "
        "write token in x-dfo-write-token. A snapshot refuses share_not_replaceable. Each new "
        "pinned copy spends one unit of the caller's daily pin allowance and of the shared pin "
        "budgets.",
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
        description="Turn a live share into a snapshot under the same URLs. Each slot keeps "
        "the frame it serves now, pinned. Needs the share's write token in x-dfo-write-token. "
        "A snapshot answers as it is. No route turns a snapshot live. Each new pinned copy "
        "spends one unit of the caller's daily pin allowance and of the shared pin budgets.",
    )
    router.add_api_route(
        SHARE_IMPORT_PATH,
        put_share_import,
        methods=["PUT"],
        status_code=201,
        response_model=ShareImportResponse,
        responses={
            200: {
                "description": "An earlier import stored this share, which is left as it is.",
                "model": ShareImportResponse,
            }
        },
        route_class_override=granted_route_class(
            _share_import_gate(principals if principals is not None else _NO_PRINCIPALS)
        ),
        description="Store one of the website's shares under its own id, for the website's "
        "share migration job. Admits the share import principal alone. The body is the "
        "share's export line less its id, plus write_token_sha256. Pins the frames the share "
        "shows, spending no allowance. 201 when stored, 200 when an earlier import stored it.",
    )
    router.add_api_route(
        SHARE_HISTORY_PATH,
        read_share_history,
        methods=["GET"],
        response_model=ShareHistoryResponse,
        description="What each of the share's slots showed, newest first, a page of up to "
        f"limit entries (default {HISTORY_PAGE_DEFAULT}) at a time: every frame it served, "
        "with its retention counted from when DFO stored it and whether it is still "
        "fetchable, and an entry with a null card and frame from when a replace left the slot "
        "empty. next names the following page's cursor, to send as before, and is null on "
        "the last page. Any origin may read it.",
    )
    router.add_api_route(
        SHARE_SLOT_PATH,
        read_share_slot_card,
        methods=["GET"],
        response_class=Response,
        responses={
            200: {
                "description": "The slot's card as a component read states it, drawn on the "
                "frame the slot's data route serves, with that frame's retention and the "
                "freshness. x-dfo-freshness and a live slot's x-dfo-checked-at are the data "
                "route's. A strong ETag answers If-None-Match with 304. Any origin may read it.",
                "model": ShareSlotCardResponse,
            },
            304: {"description": "The If-None-Match ETag still names this card."},
        },
        description="Read the card a share's slot shows, drawn on the frame the slot serves. "
        "That is a snapshot's recorded frame, or a live slot's newest data, checked at most "
        "every 15 minutes, as the slot's data route chooses it.",
    )
    router.add_api_route(
        SHARE_SLOT_DATA_PATH,
        read_share_slot_data,
        methods=["GET"],
        response_class=Response,
        responses={
            200: {
                "description": "The rows behind the slot's card. JSON (the default) states "
                "the slot, its card and frame and the freshness beside the rows; csv, xlsx and "
                "parquet are the card's data file, and bundle=true serves the components data "
                "route's zip around it. x-dfo-freshness states current, stale or "
                "pending, and a live slot's x-dfo-checked-at when DFO last checked for newer "
                "data. A strong ETag answers If-None-Match with 304. Any origin may read it.",
                "model": ShareSlotDataResponse,
                "content": {media_type: {} for media_type in COMPONENT_DATA_MEDIA_TYPES},
            },
            304: {"description": "The If-None-Match ETag still names these rows."},
        },
        description="Read the rows a share's slot serves. The frame is a snapshot's recorded "
        "frame, or a live slot's newest data, checked at most every 15 minutes.",
    )

    return DoorContribution(
        router=router,
        registry_endpoints=door_registry_endpoints(router),
        public_paths=frozenset(
            (SHARE_PATH, SHARE_HISTORY_PATH, SHARE_SLOT_PATH, SHARE_SLOT_DATA_PATH)
        ),
        classify=classify_shares_exception,
        exception_types=(
            WriteTokenRequired,
            WriteTokenInvalid,
            ShareNotReplaceable,
            ShareIdTaken,
            ShareImportNotPermitted,
            SlotDrawsNoFrame,
            ShareNotFound,
            ShareSlotEmpty,
            SharePayloadTooLarge,
            ComponentHoldsNoData,
            ComponentExportMismatch,
            ExcelIllegalCharacterError,
            SchemaInvalid,
            BindingInvalid,
            ConstraintViolation,
            ResourceExhausted,
            ReleaseMismatch,
            SharesUnavailable,
            AllowanceExhausted,
            PinBudgetExhausted,
            OpenPinBudgetExhausted,
            PinUnitHeld,
            CopyClaimUnsettled,
            CreateRequestInvalid,
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
        endpoint_exemplars={
            SHARES_ENDPOINT: (
                *LOOKUP_EXEMPLARS,
                *STORE_EXEMPLARS,
                *DRAW_EXEMPLARS,
                *FRAME_UNAVAILABLE_EXEMPLARS,
                *COMPONENT_DATA_EXCEPTION_EXEMPLARS,
                WriteTokenRequired("Ab3dEf7h"),
                WriteTokenInvalid("Ab3dEf7h"),
                ShareNotReplaceable("Ab3dEf7h"),
                ShareIdTaken("Ab3dEf7h"),
                ShareImportNotPermitted(presented=False),
                SlotDrawsNoFrame(index=0, component_id="cmp_a", revision=1),
                ShareNotFound("Ab3dEf7h"),
                ShareSlotEmpty(share_id="Ab3dEf7h", slot=2),
                SharePayloadTooLarge("exemplar"),
                SharesUnavailable(),
                AllowanceExhausted(
                    kind=AllowanceKind.share, limit=0, now=datetime(2026, 1, 1, tzinfo=UTC)
                ),
                PinBudgetExhausted(limit=0, now=datetime(2026, 1, 1, tzinfo=UTC)),
                OpenPinBudgetExhausted(limit=0, now=datetime(2026, 1, 1, tzinfo=UTC)),
                PinUnitHeld(retry_after=1),
                CopyClaimUnsettled(),
                CreateRequestInvalid(pointer="/write_token", detail="exemplar"),
                CallerHeaderInvalid(),
                CallerUnresolved("exemplar"),
                GeneratedIdCollision(subject_kind="share"),
            )
        },
        lifespan=lifespan,
    )
