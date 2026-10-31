"""The render routes draw one stored component revision on another frame, for a live share.

`POST .../components/{component_id}/revisions/{revision}/render` answers what a read of that
revision answers, built from the revision's stored intent on the frame the body names, through
`assembler.pipeline.render_component`. Its envelope's `provenance.inputs` names that frame and
`provenance.refreshed_from` names the revision. `stored_intent` is the revision's own, so its
`frame_ref` names the frame the revision was stored on, and `retention` is the named frame's
(`presentation_api.retention.stated_retention`), `null` when it cannot be read, so a render
that drew answers 200 whatever the retention read meets. No revision is appended and
nothing is pinned, so a live render leaves the frame it draws in the working tier.

`GET .../render/data` serves the rows behind that rendering as the component data route serves
a stored revision's, through `export_component_data(target_frame=...)`. It pins nothing, and
the response is not cacheable, since the frame a caller names changes with the data.

Both routes belong to the components door, which registers them and lends its executor,
timeout, and stores. They are the only Presentation routes that take a caller: the route class
of `datasets_api.live_refresh.LiveRefreshPrincipal` admits the live-refresh principal alone,
before FastAPI validates the path or reads the body, and refuses every other caller
`refresh_not_permitted` (403). A door built with no principal refuses every caller.

The door accepts the target frame before it renders, in `_accept_target`:

- the frame the revision is stored on is always accepted;
- a dataset frame (`spec_`) is accepted for a revision stored on a frame of the same `spec_id`
  when it is a frame the refresh of the stored frame answers with
  (`datasets_api.doors.refresh.refresh_can_answer`). The stored frame's rendering is the one
  the revision recorded when the card was made (`RevisionRecord.frame_rendering`), so a file of
  another rendering linked later under the stored version changes nothing; a revision stored
  before renderings were recorded is read from the stored frame's records. A plain card accepts any raw version of
  its dataset, whatever files are linked under it. A card on a groupby sum accepts a collapse
  recording that groupby, and a raw version linking a file of it. A stored raw version
  linking a plain file and groupby files accepts only a raw version linking a file of each
  of those groupbys. One linking files of several different groupbys and no plain file,
  and a collapse linking files of several different groupbys, accept no other frame;
- a derived frame (`ops_`) is accepted for a revision stored on a derived frame when the
  target's recipe has the `normal_form_hash` the revision recorded, or, for a revision stored
  before renderings were recorded, the stored frame's recipe holds.

Any other target refuses `render_target_mismatch` (409): another dataset or rendering, a frame
of the other kind, a version or recipe the stores do not hold, and a door with no reader for
the rule. A `note_narration` binds no frame, so no target can stand in for one, and it refuses
`render_target_mismatch` too.

A refusal of the render goes out at the status the code registry gives its code on
`presentation.render` (`RENDER_ENDPOINT`), read through `dfo_platform.codes.registered_status`.
For five codes it differs from the status the write routes serve: `binding_invalid` and
`constraint_violation` answer 422, and `frame_unavailable`, `release_mismatch`, and
`resource_exhausted` 409. The caller replays a stored intent it cannot correct, so
`retryable` is the write routes' value read through `presentation_api.codes.replay_retryable`:
true only for a `retry_after_delay` or `refresh_then_retry` code whose occurrence is retryable.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Iterator, Mapping
from contextlib import contextmanager
from typing import TYPE_CHECKING, Literal, cast

from fastapi import APIRouter, Path, Query
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, ConfigDict

from assembler.pipeline import ComponentHoldsNoFrame, render_component
from assembler.pipeline.export_component import export_component_data
from assembler.pipeline.stored_revision import read_stored_revision
from assembler.ports import FrameRef, RevisionRecord
from dataset_store.snapshot import SnapshotNotFound, SnapshotReader
from datasets_api.artifact_delivery import artifact_content_disposition
from datasets_api.doors.refresh import (
    RenderingNotRecorded,
    StoredRendering,
    refresh_can_answer,
    stored_rendering,
)
from datasets_api.live_refresh import (
    LiveRefreshPrincipal,
    RefreshNotPermitted,
    classify_live_refresh_exception,
)
from datasets_api.recipe_store import RecipeReader
from dfo_platform import ProblemSpec
from dfo_platform.codes import bound_registry_endpoint, registered_status
from presentation_api.codes import replay_retryable, retryable_for
from presentation_api.doors.component_data import COMPONENT_DATA_MEDIA_TYPES
from presentation_api.models import ComponentReadResponse
from presentation_api.path_params import (
    require_component_id,
    require_content_version,
    require_spec_id,
)
from presentation_api.reply import serialize_read_reply
from presentation_api.retention import stated_retention

if TYPE_CHECKING:
    from presentation_api.doors.components import ComponentsDeps

RENDER_PATH = "/api/presentation/components/{component_id}/revisions/{revision}/render"
RENDER_DATA_PATH = f"{RENDER_PATH}/data"

_DATASET_SPEC_PREFIX = "spec_"
_DERIVED_SPEC_PREFIX = "ops_"

# The header a door built with no principal reads. Its empty principal admits nobody, so the
# name decides nothing.
_NO_PRINCIPAL = LiveRefreshPrincipal(header="x-dfo-principal", principal="")

RENDER_ENDPOINT = "presentation.render"


class RenderTargetMismatch(Exception):
    """The target frame may not stand in for the frame the revision is stored on. Maps to
    `render_target_mismatch` (409)."""

    def __init__(self, *, component_id: str, revision: int, detail: str) -> None:
        self.component_id = component_id
        self.revision = revision
        self.detail = detail
        super().__init__(f"{component_id} revision {revision}: {detail}")


class RenderRefused(Exception):
    """An assembler refusal of a render, as the problem the render routes serve for it."""

    def __init__(self, spec: ProblemSpec) -> None:
        self.spec = spec
        super().__init__(spec.detail)


def classify_render_exception(exc: BaseException) -> ProblemSpec | None:
    if isinstance(exc, RenderRefused):
        return exc.spec
    if isinstance(exc, RenderTargetMismatch):
        return ProblemSpec(
            status=409,
            code="render_target_mismatch",
            detail=str(exc),
            retryable=retryable_for("render_target_mismatch"),
        )
    if isinstance(exc, ComponentHoldsNoFrame):
        return ProblemSpec(
            status=409,
            code="render_target_mismatch",
            detail=exc.detail,
            retryable=retryable_for("render_target_mismatch"),
        )
    return classify_live_refresh_exception(exc)


RENDER_EXCEPTION_TYPES: tuple[type[Exception], ...] = (
    RenderRefused,
    RenderTargetMismatch,
    ComponentHoldsNoFrame,
    RefreshNotPermitted,
)
RENDER_EXCEPTION_EXEMPLARS: tuple[BaseException, ...] = (
    RenderRefused(
        ProblemSpec(status=409, code="frame_unavailable", detail="exemplar", retryable=True)
    ),
    RenderTargetMismatch(component_id="cmp_exemplar", revision=1, detail="exemplar"),
    ComponentHoldsNoFrame(component_id="cmp_exemplar", revision=1, detail="exemplar"),
    RefreshNotPermitted(presented=False),
)


class RenderFrame(BaseModel):
    model_config = ConfigDict(extra="forbid")

    spec_id: str
    content_version: str


class RenderBody(BaseModel):
    """The frame to draw the revision on."""

    model_config = ConfigDict(extra="forbid")

    frame: RenderFrame


def _target_frame(*, spec_id: str, content_version: str) -> FrameRef:
    return FrameRef(
        spec_id=require_spec_id(spec_id),
        content_version=require_content_version(content_version),
    )


def _same_dataset_rendering(
    *,
    stored: FrameRef,
    recorded: Mapping[str, object] | None,
    target: FrameRef,
    snapshot_reader: SnapshotReader | None,
) -> bool:
    """Whether `target` is a frame the refresh of `stored`, of the `recorded` rendering, answers
    with: one dataset, and the rendering `datasets_api.doors.refresh.refresh_can_answer`
    accepts. With no `recorded` rendering the stored frame's records give it."""
    if snapshot_reader is None or stored.spec_id != target.spec_id:
        return False
    try:
        stored_as = (
            StoredRendering.from_record(recorded)
            if recorded is not None
            else stored_rendering(
                spec_id=stored.spec_id,
                content_version=stored.content_version,
                snapshot_reader=snapshot_reader,
            )
        )
        target_as = stored_rendering(
            spec_id=target.spec_id,
            content_version=target.content_version,
            snapshot_reader=snapshot_reader,
        )
    except (SnapshotNotFound, RenderingNotRecorded, ValueError):
        return False
    return refresh_can_answer(stored=stored_as, target=target_as)


def _same_recipe(
    *,
    stored: FrameRef,
    recorded: Mapping[str, object] | None,
    target: FrameRef,
    recipe_reader: RecipeReader | None,
) -> bool:
    """Whether `target`'s recipe has the normal form the revision `recorded`, or with no
    `recorded` rendering, the normal form of `stored`'s recipe."""
    if recipe_reader is None:
        return False
    if recorded is not None:
        normal_form_hash = recorded.get("normal_form_hash")
    else:
        stored_recipe = recipe_reader.read_recipe(
            spec_id=stored.spec_id, content_version=stored.content_version
        )
        normal_form_hash = stored_recipe.normal_form_hash if stored_recipe is not None else None
    target_recipe = recipe_reader.read_recipe(
        spec_id=target.spec_id, content_version=target.content_version
    )
    return (
        normal_form_hash is not None
        and target_recipe is not None
        and target_recipe.normal_form_hash == normal_form_hash
    )


def _accept_target(*, record: RevisionRecord, target: FrameRef, deps: ComponentsDeps) -> None:
    """Refuse `target` unless it may stand in for the frame `record` is stored on, by the
    rule the module docstring states.

    Raises:
        ComponentHoldsNoFrame: The revision is a note.
        RenderTargetMismatch: The target is not the stored frame, its dataset rendering, or a
            frame of its recipe.
    """
    stored_body = cast("Mapping[str, str] | None", record.stored_intent.get("frame_ref"))
    if stored_body is None:
        raise ComponentHoldsNoFrame(
            component_id=record.component_id,
            revision=record.revision,
            detail=(
                f"a {record.stored_intent['type']} binds no frame, so it has no rendering on "
                "another one"
            ),
        )
    stored = FrameRef(**stored_body)
    if target == stored:
        return
    prefixes = (stored.spec_id.split("_", 1)[0], target.spec_id.split("_", 1)[0])
    if prefixes == ("spec", "spec"):
        accepted = _same_dataset_rendering(
            stored=stored,
            recorded=record.frame_rendering,
            target=target,
            snapshot_reader=deps.snapshot_reader,
        )
        rule = "the same dataset and rendering as"
    elif prefixes == ("ops", "ops"):
        accepted = _same_recipe(
            stored=stored,
            recorded=record.frame_rendering,
            target=target,
            recipe_reader=deps.recipe_reader,
        )
        rule = "a recipe with the same normal_form_hash as"
    else:
        accepted = False
        rule = "a frame of the same kind as"
    if not accepted:
        raise RenderTargetMismatch(
            component_id=record.component_id,
            revision=record.revision,
            detail=(
                f"frame {target.spec_id!r} at content_version {target.content_version!r} is "
                f"not on record as {rule} the stored frame {stored.spec_id!r} at "
                f"content_version {stored.content_version!r}"
            ),
        )


def build_render_router(
    *,
    deps: ComponentsDeps,
    run: Callable[[Callable[[], Response]], Awaitable[Response]],
    classify: Callable[[BaseException], ProblemSpec | None],
    live_refresh: LiveRefreshPrincipal | None,
) -> APIRouter:
    """The two render routes, run through the components door's `run` on its `deps`.

    `classify` is the components door's classifier for component and component data refusals,
    whose problem the routes restate at its `presentation.render` status and replay
    retryability. `live_refresh=None` refuses every caller.
    """
    principal = live_refresh if live_refresh is not None else _NO_PRINCIPAL
    router = APIRouter(route_class=principal.route_class())

    @contextmanager
    def restated() -> Iterator[None]:
        try:
            yield
        except Exception as exc:
            spec = classify(exc) or classify_render_exception(exc)
            if spec is None:
                raise
            with bound_registry_endpoint(RENDER_ENDPOINT):
                status = registered_status(spec.code)
            raise RenderRefused(
                spec._replace(
                    status=status,
                    retryable=replay_retryable(spec.code, retryable=spec.retryable),
                )
            ) from exc

    def accepted_revision(*, component_id: str, revision: int, target: FrameRef) -> RevisionRecord:
        record = read_stored_revision(
            component_id=component_id, revision=revision, revision_store=deps.revision_store
        )
        _accept_target(record=record, target=target, deps=deps)
        return record

    @router.post(RENDER_PATH, responses={200: {"model": ComponentReadResponse}})
    async def render_component_revision(
        component_id: str, body: RenderBody, revision: int = Path(ge=1)
    ) -> JSONResponse:
        """Draw one stored revision on the frame the body names. Appends no revision."""
        require_component_id(component_id)
        target = _target_frame(
            spec_id=body.frame.spec_id, content_version=body.frame.content_version
        )

        def job() -> JSONResponse:
            with restated():
                record = accepted_revision(
                    component_id=component_id, revision=revision, target=target
                )
                reply = render_component(
                    component_id=component_id,
                    revision=revision,
                    target_frame=target,
                    frame_store=deps.frame_store,
                    revision_store=deps.revision_store,
                    release_registry=deps.release_registry,
                    vocabulary_source=deps.vocabulary_source,
                )
            retention = stated_retention(target, snapshot_reader=deps.snapshot_reader)
            return JSONResponse(
                {
                    **serialize_read_reply(reply, stored_intent=record.stored_intent),
                    "retention": retention,
                },
                headers={"Cache-Control": "no-store"},
            )

        return cast(JSONResponse, await run(job))

    @router.get(
        RENDER_DATA_PATH,
        response_class=Response,
        responses={
            200: {
                "description": "The rows behind every value the revision shows when drawn on "
                "the named frame, as the component data route serves a stored revision's.",
                "content": {media_type: {} for media_type in COMPONENT_DATA_MEDIA_TYPES},
            }
        },
    )
    async def read_rendered_component_data(
        component_id: str,
        spec_id: str,
        content_version: str,
        revision: int = Path(ge=1),
        format_: Literal["csv", "xlsx", "parquet"] = Query("csv", alias="format"),
        bundle: bool = Query(True),
    ) -> Response:
        """The data export of one stored revision drawn on the frame the query names."""
        require_component_id(component_id)
        target = _target_frame(spec_id=spec_id, content_version=content_version)

        def job() -> Response:
            with restated():
                accepted_revision(component_id=component_id, revision=revision, target=target)
                exported = export_component_data(
                    component_id=component_id,
                    revision=revision,
                    format=format_,
                    bundle=bundle,
                    revision_store=deps.revision_store,
                    frame_store=deps.frame_store,
                    site_url=deps.site.public_site_url,
                    target_frame=target,
                    release_registry=deps.release_registry,
                )
            return Response(
                content=exported.data,
                media_type=exported.media_type,
                headers={
                    "Content-Disposition": artifact_content_disposition(exported.file_name),
                    "Cache-Control": "no-store",
                },
            )

        return await run(job)

    return router
