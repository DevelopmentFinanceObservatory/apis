"""The components door adapts create, revise, and read requests to
`assembler.pipeline.write_component` and `get_component`. It validates component IDs, then maps
pipeline refusals to HTTP. The `cmp_` id grants read to any caller. Revising a component needs
the write token its create returned, sent as `x-dfo-write-token`, and a create into an existing
composition needs that composition's token in the same header. A create may carry its own `cmw_`
token in the body's `write_token`; a retry with the same token and request returns the first
create's reply. These routes take no caller or session.

No route pins a frame: a card's frame stays in the working tier until a share first includes
the card (`presentation_store.pin_card_frames`). A create and a read state the stored frame's
retention (`datasets_api.retention.version_retention`): pinned, or held until 30 days after DFO
stored it. A frame whose working copy has aged out refuses `frame_unavailable` on every route that
loads it, naming a fetch with latest=True as the way to current data.

The door also carries `presentation_api.doors.component_data`'s export route and the two render
routes of `presentation_api.doors.render`, which admit the live-refresh principal alone. All six
routes share one executor and a non-blocking semaphore. Saturation returns retryable 503. Each permit
covers the pipeline call and response encoding. A revise's `composition_id` reaches
the pipeline, which rejects composition membership on revise.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Callable, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import cast

from fastapi import APIRouter, FastAPI, Header, Query
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, ConfigDict, Field

from assembler.catalogue import BakedReleaseRegistry
from assembler.errors import (
    BindingInvalid,
    ConstraintViolation,
    DraftNotFound,
    EditConflict,
    FrameUnavailable,
    ReleaseMismatch,
    ResourceExhausted,
    RevisionNotFound,
    SchemaInvalid,
    WriteRefused,
)
from assembler.pipeline import get_component, write_component
from assembler.ports import (
    FrameRef,
    FrameStore,
    ReleaseRegistry,
    RevisionStore,
    VocabularySource,
)
from assembler.write_token import CreateRequestInvalid
from dataset_store import LiveDatasetStore
from dataset_store import close_shared_pool as close_dataset_pool
from dataset_store.snapshot import SnapshotReader
from datasets_api.door_executor import build_door_executor
from datasets_api.live_refresh import LiveRefreshPrincipal
from datasets_api.recipe_store import RecipeReader
from datasets_api.shutdown import ShutdownBudget
from datasets_api.store_errors import DatasetsStoreUnavailable
from dfo_platform import Alternative, ProblemSpec
from dfo_platform.config import PublicSiteSettings
from dfo_platform.doors import DoorContribution, combine_classifiers
from dfo_platform.store_errors import store_failure_is_permanent
from presentation_api.codes import retryable_for
from presentation_api.config import (
    PRESENTATION_COMPONENTS_MAX_CONCURRENCY,
    PRESENTATION_COMPONENTS_TIMEOUT_SECONDS,
    PresentationSettings,
)
from presentation_api.doors.component_data import (
    COMPONENT_DATA_EXCEPTION_EXEMPLARS,
    COMPONENT_DATA_EXCEPTION_TYPES,
    add_component_data_route,
    classify_component_data_exception,
)
from presentation_api.doors.render import (
    RENDER_EXCEPTION_EXEMPLARS,
    RENDER_EXCEPTION_TYPES,
    build_render_router,
    classify_render_exception,
)
from presentation_api.live import build_live_frame_store
from presentation_api.models import ComponentReadResponse, CreateReplyResponse, WriteReplyResponse
from presentation_api.path_params import PathParamInvalid, require_component_id
from presentation_api.refusals import binding_invalid_spec, frame_unavailable_spec
from presentation_api.reply import serialize_create_reply, serialize_write_reply
from presentation_api.retention import stated_retention
from presentation_api.vocabulary import LiveVocabularySource
from presentation_store import (
    GeneratedIdCollision,
    LivePresentationStore,
    PresentationStoreUnavailable,
    frame_reads_bounded_by,
)
from presentation_store import close_shared_pool as close_presentation_pool
from presentation_store import shared_pool as presentation_shared_pool
from spec_registry.errors import SpecRegistryStoreError, SpecRegistryUnavailable

logger = logging.getLogger(__name__)


# Carries the write token a create returned; a revise, or a create into an existing
# composition, sends it back.
WRITE_TOKEN_HEADER = "x-dfo-write-token"


class PresentationBackpressure(Exception):
    """Raised when the shared components semaphore is full."""


class ComponentBody(BaseModel):
    """The create and revise routes' shared request body. `extra="forbid"`
    is what refuses a body carrying `session`/`caller`. An undeclared
    field fails pydantic validation, which
    `presentation_api.refusals.request_validation_spec` classifies as
    `request_invalid` with no door-level code needed, `retryable` false. A
    revise's `composition_id` is passed straight through to
    `write_component`, which refuses it (composition membership is
    create-only); this model does not distinguish the two routes.
    """

    model_config = ConfigDict(extra="forbid")

    intent: dict[str, object]
    composition_id: str | None = None


class CreateComponentBody(ComponentBody):
    """The create route's body. `write_token` is checked by `write_component`, which refuses a
    malformed token, or one sent with `composition_id` "new", as `request_invalid`."""

    write_token: str | None = Field(
        default=None,
        description='a cmw_ token this client minted for the new component: "cmw_" + '
        "secrets.token_urlsafe(16) or an equivalent CSPRNG, 22 url-safe base64 characters "
        "after the prefix. It becomes the component's write token and keys the create: a "
        "retry with the same write_token, intent and composition_id returns the first "
        "create's reply, and a different request with it is refused as request_invalid. Mint "
        'a new one for each create. Not allowed with composition_id "new".',
    )


def classify_components_exception(exc: BaseException) -> ProblemSpec | None:
    if isinstance(exc, PathParamInvalid):
        return ProblemSpec(
            status=400,
            code="request_invalid",
            detail=str(exc),
            retryable=False,
        )
    if isinstance(exc, CreateRequestInvalid):
        return ProblemSpec(
            status=400,
            code="request_invalid",
            detail=exc.detail,
            pointer=exc.pointer,
            retryable=False,
        )
    if isinstance(exc, SchemaInvalid):
        return _schema_invalid_spec(exc)
    if isinstance(exc, BindingInvalid):
        return binding_invalid_spec(exc)
    if isinstance(exc, ConstraintViolation):
        return ProblemSpec(
            status=400,
            code="constraint_violation",
            detail=f"{exc.detail} fix: {exc.fix}",
            pointer=exc.pointer,
            constraint_id=exc.constraint_id,
            retryable=retryable_for("constraint_violation"),
        )
    if isinstance(exc, EditConflict):
        return ProblemSpec(
            status=409,
            code="edit_conflict",
            detail=str(exc) or "the component or composition moved before this write committed",
            retryable=retryable_for("edit_conflict"),
        )
    if isinstance(exc, WriteRefused):
        return ProblemSpec(
            status=403,
            code="write_refused",
            detail=f"{exc.detail} fix: {exc.fix}",
            retryable=retryable_for("write_refused"),
        )
    if isinstance(exc, DraftNotFound):
        return ProblemSpec(
            status=404,
            code="not_found",
            detail=str(exc),
            retryable=retryable_for("not_found"),
        )
    if isinstance(exc, RevisionNotFound):
        return ProblemSpec(
            status=404,
            code="not_found",
            detail=exc.detail,
            retryable=retryable_for("not_found"),
        )
    if isinstance(exc, FrameUnavailable):
        return frame_unavailable_spec(exc)
    if isinstance(exc, ResourceExhausted):
        return ProblemSpec(
            status=400,
            code="resource_exhausted",
            detail=f"{exc.detail} (limit: {exc.limit_name})",
            pointer=exc.pointer,
            retryable=retryable_for("resource_exhausted"),
        )
    if isinstance(exc, ReleaseMismatch):
        return ProblemSpec(
            status=503,
            code="release_mismatch",
            detail=exc.detail,
            retryable=retryable_for("release_mismatch"),
        )
    if isinstance(exc, PresentationBackpressure):
        return ProblemSpec(
            status=503,
            code="service_unavailable",
            detail="the components door is at its concurrent-write capacity right now; retry",
            retryable=retryable_for("service_unavailable"),
        )
    if isinstance(exc, GeneratedIdCollision):
        logger.warning("Fresh presentation %s ID collision. Retry safely", exc.subject_kind)
        return ProblemSpec(
            status=503,
            code="service_unavailable",
            detail=f"A generated presentation {exc.subject_kind} ID collided. Retry the request.",
            retryable=retryable_for("service_unavailable"),
        )
    if isinstance(exc, PresentationStoreUnavailable):
        if store_failure_is_permanent(exc.sqlstate):
            logger.error("Presentation store call failed with a permanent cause: %s", exc)
            return ProblemSpec(
                status=500,
                code="internal_error",
                detail="a presentation store call failed with a non-retryable cause",
                retryable=retryable_for("internal_error"),
            )
        logger.error("Presentation store call failed: %s", exc)
        return ProblemSpec(
            status=503,
            code="service_unavailable",
            detail="a presentation store call failed; retry",
            retryable=retryable_for("service_unavailable"),
        )
    if isinstance(exc, (SpecRegistryUnavailable, SpecRegistryStoreError)):
        # A frame load reads the registry to scope its revocation check, and fails closed.
        logger.error("Spec registry read scoping a frame load's revocation check failed: %s", exc)
        return ProblemSpec(
            status=503,
            code="service_unavailable",
            detail="the spec registry could not scope a frame's revocation check; retry",
            retryable=retryable_for("service_unavailable"),
        )
    if isinstance(exc, DatasetsStoreUnavailable):
        if store_failure_is_permanent(exc.sqlstate):
            logger.error(
                "Datasets store call underlying a frame load failed with a permanent cause: %s",
                exc,
            )
            return ProblemSpec(
                status=500,
                code="internal_error",
                detail="a frame store call failed with a non-retryable cause",
                retryable=retryable_for("internal_error"),
            )
        logger.error("Datasets store call underlying a frame load failed: %s", exc)
        return ProblemSpec(
            status=503,
            code="service_unavailable",
            detail="a frame store call failed; retry",
            retryable=retryable_for("service_unavailable"),
        )
    return None


def _schema_invalid_spec(exc: SchemaInvalid) -> ProblemSpec:
    """`minimal_intent` and `legal_fields` are mutually exclusive. Minimal intents go in
    `detail`, and legal fields become `alternatives`. A missing field can be corrected and
    retried. An illegal field cannot.
    """
    if exc.minimal_intent is not None:
        detail = (
            f"{exc.detail} minimal legal intent: {json.dumps(exc.minimal_intent, sort_keys=True)}"
        )
        return ProblemSpec(
            status=400, code="schema_invalid", detail=detail, pointer=exc.pointer, retryable=True
        )
    legal_fields = cast(tuple[str, ...], exc.legal_fields)
    return ProblemSpec(
        status=400,
        code="schema_invalid",
        detail=exc.detail,
        pointer=exc.pointer,
        alternatives=[Alternative(ref=field, label=field) for field in legal_fields],
        retryable=False,
    )


@dataclass
class ComponentsDeps:
    frame_store: FrameStore
    revision_store: RevisionStore
    release_registry: ReleaseRegistry
    # The public website card exports link to, read from the environment in the lifespan.
    site: PublicSiteSettings
    # What the render routes accept a target frame from, and the reader of the retention a read
    # and a render state. `None` accepts no frame by that rule and states no retention.
    snapshot_reader: SnapshotReader | None = None
    recipe_reader: RecipeReader | None = None
    # The words cards' descriptions are composed from, set in the lifespan.
    vocabulary_source: VocabularySource | None = None


def build_components_door(
    *,
    frame_store: FrameStore | None = None,
    revision_store: RevisionStore | None = None,
    release_registry: ReleaseRegistry | None = None,
    snapshot_reader: SnapshotReader | None = None,
    recipe_reader: RecipeReader | None = None,
    vocabulary_source: VocabularySource | None = None,
    live_refresh: LiveRefreshPrincipal | None = None,
    pool_size: int | None = None,
    budget: ShutdownBudget | None = None,
) -> DoorContribution:
    """Build missing stores in this door's lifespan. The surface passes its resolved release
    registry. The fallback applies when this door is built alone.

    `snapshot_reader` and `recipe_reader` are what the render routes accept a target frame
    from. The lifespan builds both with the live frame store when no `frame_store` is injected.
    With an injected `frame_store` they stay as given, and a missing one accepts no frame by its
    rule. `vocabulary_source` gives the words cards' descriptions are composed from, and the
    lifespan builds a `LiveVocabularySource` when none is injected. `live_refresh` is the one
    caller the render routes admit, and `None` refuses every caller.
    """
    _budget = budget if budget is not None else ShutdownBudget()
    effective_pool_size = (
        pool_size if pool_size is not None else PRESENTATION_COMPONENTS_MAX_CONCURRENCY
    )
    door_executor = build_door_executor(effective_pool_size, "components", surface="presentation")
    executor = door_executor.executor

    deps = ComponentsDeps(
        frame_store=cast(FrameStore, frame_store),
        revision_store=cast(RevisionStore, revision_store),
        release_registry=cast(ReleaseRegistry, release_registry),
        site=cast(PublicSiteSettings, None),
        snapshot_reader=snapshot_reader,
        recipe_reader=recipe_reader,
        vocabulary_source=vocabulary_source,
    )

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        opened_dataset_pool = False
        opened_presentation_pool = False
        owned_graph = None

        def _close_live_clients_built_here() -> None:
            if owned_graph is not None:
                owned_graph.close()
            if opened_dataset_pool:
                close_dataset_pool()
            if opened_presentation_pool:
                close_presentation_pool()

        try:
            deps.site = PublicSiteSettings.from_env()
            if release_registry is None:
                deps.release_registry = BakedReleaseRegistry()
            if vocabulary_source is None:
                deps.vocabulary_source = LiveVocabularySource()
            if revision_store is None:
                connections = presentation_shared_pool()
                opened_presentation_pool = True
                deps.revision_store = LivePresentationStore(
                    store=LiveDatasetStore(connections=connections)
                )
            if frame_store is None:
                settings = PresentationSettings.from_env()
                live = build_live_frame_store(settings=settings)
                opened_dataset_pool = True
                owned_graph = live.graph
                deps.frame_store = live.frame_store
                if snapshot_reader is None:
                    deps.snapshot_reader = live.snapshot_reader
                if recipe_reader is None:
                    deps.recipe_reader = live.recipe_reader
        except BaseException:
            _close_live_clients_built_here()
            raise

        try:
            yield
        finally:
            executor.shutdown(wait=False, cancel_futures=True)
            await asyncio.to_thread(door_executor.drain, _budget)
            _close_live_clients_built_here()

    def _frame_ref_retention(frame_ref: object) -> dict[str, object] | None:
        """The retention of the frame an intent's `frame_ref` names, `None` for a card that
        draws none or a frame DFO holds no version record of."""
        if frame_ref is None:
            return None
        frame = cast("Mapping[str, str]", frame_ref)
        return stated_retention(
            FrameRef(spec_id=frame["spec_id"], content_version=frame["content_version"]),
            snapshot_reader=deps.snapshot_reader,
        )

    async def _run[R: Response](job: Callable[[], R], /) -> R:
        with frame_reads_bounded_by(timeout_seconds=PRESENTATION_COMPONENTS_TIMEOUT_SECONDS):
            return await asyncio.wait_for(
                door_executor.run(job, refuse=PresentationBackpressure),
                timeout=PRESENTATION_COMPONENTS_TIMEOUT_SECONDS,
            )

    router = APIRouter()

    @router.post(
        "/api/presentation/components",
        status_code=201,
        responses={201: {"model": CreateReplyResponse}},
    )
    async def create_component(
        body: CreateComponentBody,
        write_token: str | None = Header(None, alias=WRITE_TOKEN_HEADER),
    ) -> JSONResponse:
        def job() -> JSONResponse:
            reply = write_component(
                intent=body.intent,
                component_id=None,
                composition_id=body.composition_id,
                write_token=write_token,
                create_write_token=body.write_token,
                frame_store=deps.frame_store,
                revision_store=deps.revision_store,
                release_registry=deps.release_registry,
                vocabulary_source=deps.vocabulary_source,
            )
            retention = _frame_ref_retention(body.intent.get("frame_ref"))
            return JSONResponse(
                {**serialize_create_reply(reply), "retention": retention}, status_code=201
            )

        return await _run(job)

    @router.patch(
        "/api/presentation/components/{component_id}",
        responses={200: {"model": WriteReplyResponse}},
    )
    async def revise_component(
        component_id: str,
        body: ComponentBody,
        write_token: str | None = Header(None, alias=WRITE_TOKEN_HEADER),
    ) -> JSONResponse:
        require_component_id(component_id)

        def job() -> JSONResponse:
            reply = write_component(
                intent=body.intent,
                component_id=component_id,
                composition_id=body.composition_id,
                write_token=write_token,
                frame_store=deps.frame_store,
                revision_store=deps.revision_store,
                release_registry=deps.release_registry,
                vocabulary_source=deps.vocabulary_source,
            )
            return JSONResponse(serialize_write_reply(reply), status_code=200)

        return await _run(job)

    @router.get(
        "/api/presentation/components/{component_id}",
        responses={200: {"model": ComponentReadResponse}},
    )
    async def read_component(
        component_id: str,
        revision: int | None = Query(None, ge=1),
    ) -> JSONResponse:
        require_component_id(component_id)

        def job() -> JSONResponse:
            document = get_component(
                component_id=component_id,
                revision=revision,
                revision_store=deps.revision_store,
            )
            stored_intent = cast("Mapping[str, object]", document["stored_intent"])
            retention = _frame_ref_retention(stored_intent.get("frame_ref"))
            return JSONResponse({**document, "retention": retention}, status_code=200)

        return await _run(job)

    add_component_data_route(router, deps=deps, run=_run)
    # Extending the route list keeps each render route's own class, which admits the caller.
    router.routes.extend(
        build_render_router(
            deps=deps,
            run=_run,
            classify=classify_components_exception,
            live_refresh=live_refresh,
        ).routes
    )

    return DoorContribution(
        router=router,
        classify=combine_classifiers(
            classify_components_exception,
            classify_component_data_exception,
            classify_render_exception,
        ),
        exception_types=(
            PathParamInvalid,
            CreateRequestInvalid,
            SchemaInvalid,
            BindingInvalid,
            ConstraintViolation,
            EditConflict,
            WriteRefused,
            DraftNotFound,
            RevisionNotFound,
            FrameUnavailable,
            ResourceExhausted,
            ReleaseMismatch,
            PresentationBackpressure,
            GeneratedIdCollision,
            PresentationStoreUnavailable,
            DatasetsStoreUnavailable,
            SpecRegistryUnavailable,
            SpecRegistryStoreError,
            *COMPONENT_DATA_EXCEPTION_TYPES,
            *RENDER_EXCEPTION_TYPES,
            # The surface classifies validation errors, but this door declares the type raised by
            # `ComponentBody` so platform exception discovery includes it.
            RequestValidationError,
        ),
        exception_exemplars=(
            PathParamInvalid(name="component_id", value="exemplar"),
            CreateRequestInvalid(pointer="/write_token", detail="exemplar"),
            SchemaInvalid(
                pointer="/type", detail="exemplar", legal_fields=("type", "frame_ref", "select")
            ),
            SchemaInvalid(
                pointer="/select", detail="exemplar", minimal_intent={"type": "stat_card"}
            ),
            BindingInvalid(
                pointer="/select",
                dimension="recipient",
                legal_values=("country/USA",),
                truncated=False,
                detail="exemplar",
            ),
            ConstraintViolation(
                constraint_id="log_scale_requires_positive_values",
                pointer="/scale",
                detail="exemplar",
                fix="exemplar fix",
            ),
            EditConflict("exemplar"),
            WriteRefused(
                subject_kind="component",
                subject_id="cmp_exemplar",
                detail="exemplar",
                fix="exemplar fix",
            ),
            DraftNotFound(subject_kind="component", subject_id="cmp_exemplar"),
            RevisionNotFound(component_id="cmp_exemplar", revision=1, detail="exemplar"),
            FrameUnavailable(reason="io", detail="exemplar"),
            ResourceExhausted(pointer="/select", limit_name="MAX_SELECTORS", detail="exemplar"),
            ReleaseMismatch(catalog_version="2020.1.0", detail="exemplar"),
            PresentationBackpressure(),
            GeneratedIdCollision(subject_kind="component"),
            PresentationStoreUnavailable("exemplar"),
            PresentationStoreUnavailable("exemplar", sqlstate="42601"),
            DatasetsStoreUnavailable("exemplar"),
            DatasetsStoreUnavailable("exemplar", sqlstate="23503"),
            SpecRegistryUnavailable(),
            SpecRegistryStoreError("exemplar"),
            RequestValidationError(
                [{"loc": ("body", "intent"), "msg": "Field required", "type": "missing"}],
                body={},
            ),
            *COMPONENT_DATA_EXCEPTION_EXEMPLARS,
            *RENDER_EXCEPTION_EXEMPLARS,
        ),
        lifespan=lifespan,
    )
