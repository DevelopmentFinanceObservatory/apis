"""The shares door publishes cards under one `/s/<id>` id, and replaces, deletes and reads them.

`POST /api/presentation/shares` takes `{cards: [{component_id, revision}], latest, title?}`.
`latest` has no default: true makes a live share, which follows new data, and false a snapshot,
which keeps the versions its cards drew. The create spends one unit of the caller's `share`
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

`GET /api/presentation/shares/{share_id}` is public and allows any origin: kind, mode, title,
revision, slots, the website's payload and created_at, never the token. A share id that names no
share, or one taken down, answers 404 `not_found` on every route.

Handlers are module-level functions that read `SHARES_DEPS_SLOT`, which the door's lifespan
sets. Every route runs on one executor behind a non-blocking semaphore, and saturation answers
retryable 503.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable, Iterable, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Annotated, cast

from fastapi import APIRouter, Depends, FastAPI, Header, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictInt

from assembler.errors import FrameUnavailable, RevisionNotFound
from assembler.pipeline.stored_revision import read_stored_revision
from assembler.ports import DraftNotFound, EditConflict, FrameRef, FrameStore, RevisionStore
from assembler.write_token import (
    SHARE_WRITE_TOKEN_PREFIX,
    mint_write_token,
    write_token_digest,
    write_token_matches,
)
from dataset_store import LiveDatasetStore
from dataset_store import close_shared_pool as close_dataset_pool
from dataset_store import shared_pool as dataset_shared_pool
from datasets_api.allowance import (
    AllowanceExhausted,
    AllowanceKind,
    AllowanceLedger,
    LiveAllowanceLedger,
    allowance_limits,
)
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
from presentation_api.doors.components import WRITE_TOKEN_HEADER, classify_components_exception
from presentation_api.live import build_live_frame_store
from presentation_api.models import SharedResponse, ShareReadResponse, ShareWriteResponse
from presentation_api.path_params import PathParamInvalid, require_component_id, require_share_id
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

MAX_SHARE_CARDS = 24
"""The most cards one create or replace takes, and so the most slots a write adds."""

_MAX_TITLE_LENGTH = 300


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


class CreateShareBody(ShareCardsBody):
    """A create's body. `latest` has no default, because a live link and a fixed one answer
    different questions under a public URL."""

    latest: StrictBool = Field(
        description="true for a live share that follows new data and takes replacements; "
        "false for a snapshot that keeps the versions its cards draw now, for good"
    )
    title: str | None = Field(
        default=None, min_length=1, max_length=_MAX_TITLE_LENGTH, description="the share's title"
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
    allowances: AllowanceLedger
    caller: CallerSettings
    site: PublicSiteSettings
    executor: DoorExecutor
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
            kind="card",
            mode="live" if body.latest else "snapshot",
            title=body.title,
            write_token_sha256=write_token_digest(token),
            cards=cards,
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
    if isinstance(exc, ShareNotFound):
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
    return classify_components_exception(exc)


def build_shares_door(
    *,
    share_store: ShareStore | None = None,
    revision_store: RevisionStore | None = None,
    frame_store: FrameStore | None = None,
    allowances: AllowanceLedger | None = None,
    caller: CallerSettings | None = None,
    now: Callable[[], datetime] | None = None,
    pool_size: int | None = None,
    budget: ShutdownBudget | None = None,
) -> DoorContribution:
    """Build the shares door. The lifespan builds each store left out over the live pools: the
    share and revision stores over the presentation pool, the frame store over the datasets
    pool, and the allowance ledger over the datasets pool with `DatasetsBuildSettings`' limits.
    `caller` defaults to those settings' `caller`, and the public site to `DFO_PUBLIC_SITE_URL`.
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
            if effective_frame_store is None:
                live = build_live_frame_store(settings=PresentationSettings.from_env())
                dataset_borrows += 1
                owned_graph = live.graph
                effective_frame_store = live.frame_store
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
                allowances=effective_allowances,
                caller=effective_caller,
                site=site,
                executor=executor,
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

    return DoorContribution(
        router=router,
        registry_endpoints=route_registry_endpoints(router, SHARES_ENDPOINT),
        classify=classify_shares_exception,
        exception_types=(
            WriteTokenRequired,
            WriteTokenInvalid,
            ShareNotReplaceable,
            ShareNotFound,
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
