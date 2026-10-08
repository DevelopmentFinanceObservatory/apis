"""The recipe door writes and reads derived frames' recipes, and lists a dataset's versions.

Routes:

- `PUT /api/datasets/frames/{spec_id}/{content_version}/recipe` stores the recipe of a
  published `ops_`/`ocv_` frame and indexes its `replay_key`. It is how a frame published
  without a recipe gains one. A repeat write with the same `normal_form_hash` and `replay_key`
  answers 200 with `created` false.
- `GET /api/datasets/frames/{spec_id}/{content_version}/recipe` returns the stored recipe.
- `GET /api/datasets/recipes/replays/{replay_key}` returns the frame stored under a replay key.
- `GET /api/datasets/{spec_id}/versions` lists a `spec_` dataset's stored versions, newest
  first. A dataset with no stored version, registered or not, answers an empty list. Each
  version states its `rendering`: `plain` for a raw version, `groupby` with the `groupby` a
  collapse was summed by, and `not_recorded` for a collapse with no linked file recording one.
  The groupby is read as the frame refresh reads it (`datasets_api.recorded_rendering`).

Every route requires the publish door's principal. The route class runs
`PublishPrincipals.admit` before FastAPI validates the path or reads a body, so an unauthorized
caller gets `publish_not_permitted` (403) whatever it sends.

`recipe_not_found` (404) answers a read of a pair holding no recipe and a replay key no recipe
carries. `snapshot_not_found` (404) answers a recipe write for a pair with no version row.
`publish_conflict` (409) answers a write the recipe store refuses, with the `constraint_id`
values of `datasets_api.recipe_store`. `request_invalid` (400) answers a malformed path id or
body, and `service_unavailable` (503) a full executor.

The versions route is mounted before the snapshot door's
`GET /api/datasets/{spec_id}/{content_version}`, which would otherwise read `versions` as a
content version (`datasets_api.app`).
"""

from __future__ import annotations

import asyncio
import functools
import logging
from collections.abc import Callable, Coroutine
from contextlib import asynccontextmanager
from typing import Annotated, Any, Literal

from fastapi import APIRouter, FastAPI, Path, Request, Response
from fastapi.responses import JSONResponse
from fastapi.routing import APIRoute
from pydantic import BaseModel, ConfigDict, Field, field_serializer

from dataset_store import LiveDatasetStore, close_shared_pool, shared_pool
from datasets_api.config import DatasetsBuildSettings, DatasetsConfigError
from datasets_api.door_executor import build_door_executor
from datasets_api.doors.publish import (
    PublishNotPermitted,
    PublishPrincipals,
    classify_publish_exception,
    write_frame_recipe,
)
from datasets_api.headers import no_store_headers
from datasets_api.mint_models import GroupbyOptions
from datasets_api.publish_models import OcvContentVersion, OpsSpecId, Recipe, ReplayKey
from datasets_api.recipe_store import (
    LiveRecipeStore,
    RecipeDiverges,
    RecipeFrameNotStored,
    RecipeStore,
    ReplayKeyHeld,
)
from datasets_api.recorded_rendering import recorded_groupby
from datasets_api.shutdown import ShutdownBudget
from datasets_api.timings import door_call, run_with_timings
from datasets_api.version_store import LiveVersionLister
from datasets_api.versions import ListedVersion, VersionLister
from dfo_platform import ProblemSpec
from dfo_platform.doors import DoorContribution

logger = logging.getLogger(__name__)


class RecipeNotFound(Exception):
    """The pair holds no recipe. Maps to `recipe_not_found` (404)."""

    def __init__(self, *, spec_id: str, content_version: str) -> None:
        super().__init__(f"spec_id {spec_id!r} content_version {content_version!r} holds no recipe")


class ReplayNotFound(Exception):
    """No recipe carries the replay key. Maps to `recipe_not_found` (404)."""

    def __init__(self, *, replay_key: str) -> None:
        super().__init__(f"no frame is stored under replay_key {replay_key!r}")


class RecipeBackpressure(Exception):
    """A full recipe semaphore maps to retryable `service_unavailable` (503)."""


def classify_recipe_exception(exc: BaseException) -> ProblemSpec | None:
    if isinstance(exc, RecipeNotFound | ReplayNotFound):
        return ProblemSpec(status=404, code="recipe_not_found", detail=str(exc), retryable=False)
    if isinstance(exc, RecipeBackpressure):
        return ProblemSpec(
            status=503,
            code="service_unavailable",
            detail="the recipe door is at its concurrent-call capacity right now; retry",
            retryable=True,
        )
    # Covers `PublishNotPermitted` and the recipe store's refusals.
    return classify_publish_exception(exc)


class RecipeWriteResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    spec_id: str
    content_version: str
    created: bool = Field(
        description=(
            "true when this call stored the recipe; false when the frame already held a "
            "recipe with the same normal_form_hash and replay_key"
        )
    )


class ReplayedFrameResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    spec_id: str
    content_version: str


class DatasetVersion(BaseModel):
    model_config = ConfigDict(extra="forbid")

    content_version: str
    base_content_version: str | None = Field(
        description=(
            "the raw version a full groupby collapse was rendered from; null on every other version"
        )
    )
    rendering: Literal["plain", "groupby", "not_recorded"] = Field(
        description=(
            "plain: a raw version, which is also what a groupby sum that left the frame "
            "unchanged is stored under. groupby: a collapse, summed by `groupby`. "
            "not_recorded: a collapse with no linked file recording its groupby"
        )
    )
    groupby: GroupbyOptions | None = Field(
        description=(
            "the groupby a collapse was summed by, as its file's rendering records it; null "
            "unless `rendering` is groupby"
        )
    )

    @field_serializer("groupby")
    def _groupby_as_recorded(self, groupby: GroupbyOptions | None):
        """Leave unset members out, as the rendering a mint records does
        (`render.artifact_rendering`), so the value equals the `groupby` the mint was sent."""
        return None if groupby is None else groupby.model_dump(mode="json", exclude_none=True)


class DatasetVersionsResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    spec_id: str
    versions: list[DatasetVersion] = Field(description="stored versions, newest first")


def _read_recipe(
    *, spec_id: str, content_version: str, recipe_store: RecipeStore
) -> dict[str, object]:
    record = recipe_store.read_recipe(spec_id=spec_id, content_version=content_version)
    if record is None:
        raise RecipeNotFound(spec_id=spec_id, content_version=content_version)
    return dict(record.body)


def _read_replay(*, replay_key: str, recipe_store: RecipeStore) -> ReplayedFrameResponse:
    frame = recipe_store.read_replay(replay_key=replay_key)
    if frame is None:
        raise ReplayNotFound(replay_key=replay_key)
    return ReplayedFrameResponse(spec_id=frame.spec_id, content_version=frame.content_version)


def _list_versions(*, spec_id: str, version_lister: VersionLister) -> DatasetVersionsResponse:
    return DatasetVersionsResponse(
        spec_id=spec_id,
        versions=[_dataset_version(version) for version in version_lister.list_versions(spec_id)],
    )


def _dataset_version(version: ListedVersion) -> DatasetVersion:
    groupby = (
        recorded_groupby(version.renderings) if version.base_content_version is not None else None
    )
    if version.base_content_version is None:
        rendering = "plain"
    else:
        rendering = "groupby" if groupby is not None else "not_recorded"
    return DatasetVersion(
        content_version=version.content_version,
        base_content_version=version.base_content_version,
        rendering=rendering,
        groupby=groupby,
    )


def _admitting_route_class(principals: PublishPrincipals) -> type[APIRoute]:
    """Return a route class that checks the caller before FastAPI validates the request."""

    class _AdmittingRecipeRoute(APIRoute):
        def get_route_handler(self) -> Callable[[Request], Coroutine[Any, Any, Response]]:
            handler = super().get_route_handler()

            async def admit_then_handle(request: Request) -> Response:
                principals.admit(request)
                return await handler(request)

            return admit_then_handle

    return _AdmittingRecipeRoute


def build_recipe_door(
    *,
    settings: DatasetsBuildSettings | None = None,
    recipe_store: RecipeStore | None = None,
    version_lister: VersionLister | None = None,
    principal_header: str | None = None,
    allowed_principals: frozenset[str] | None = None,
    pool_size: int | None = None,
    budget: ShutdownBudget | None = None,
) -> DoorContribution:
    """Build the recipe door and its executor.

    Inject `recipe_store` and `version_lister` together, or leave both unset for lifespan to
    build them on one pool borrow. One injected without the other raises
    `DatasetsConfigError`. Inject the recipe store the publish door writes to.

    `settings` defaults from the environment. The principal settings default as the publish
    door's do (`PublishPrincipals.resolve`), so one allowlist admits both doors' callers.
    The two doors serve the same callers, so `pool_size` defaults to
    `settings.publish_max_concurrency`. `budget` is shared across mounted doors. An omitted
    budget creates a private one.
    """
    if (recipe_store is None) != (version_lister is None):
        raise DatasetsConfigError(
            "datasets recipe door built with one of recipe_store/version_lister injected and "
            "the other left unset. Inject both, or leave both None for this door to build "
            "live ones."
        )

    _settings = settings if settings is not None else DatasetsBuildSettings.from_env()
    _budget = budget if budget is not None else ShutdownBudget()
    _recipe_store: RecipeStore | None = recipe_store
    _version_lister: VersionLister | None = version_lister
    principals = PublishPrincipals.resolve(
        settings=_settings,
        principal_header=principal_header,
        allowed_principals=allowed_principals,
    )

    door_executor = build_door_executor(
        pool_size if pool_size is not None else _settings.publish_max_concurrency, "recipe"
    )

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        nonlocal _recipe_store, _version_lister
        pool_opened_here = False
        if recipe_store is None:
            store = LiveDatasetStore(connections=shared_pool())
            pool_opened_here = True
            _recipe_store = LiveRecipeStore(store=store)
            _version_lister = LiveVersionLister(store=store)
        try:
            yield
        finally:
            # Cancel queued calls. Drain running calls before closing the pool.
            door_executor.executor.shutdown(wait=False, cancel_futures=True)
            await asyncio.to_thread(door_executor.drain, _budget)
            if pool_opened_here:
                close_shared_pool()

    async def run[T](fn: Callable[..., T], /, **kwargs: object) -> tuple[T, str]:
        """Run a store call on the door's executor. Return its result and `Server-Timing`."""
        with door_call("recipe", logger=logger) as call:
            result = await door_executor.run(
                functools.partial(run_with_timings, fn, call.timings, **kwargs),
                refuse=RecipeBackpressure,
            )
        return result, call.server_timing

    router = APIRouter(route_class=_admitting_route_class(principals))

    @router.put(
        "/api/datasets/frames/{spec_id}/{content_version}/recipe",
        response_model=RecipeWriteResponse,
    )
    async def put_recipe(
        spec_id: Annotated[OpsSpecId, Path()],
        content_version: Annotated[OcvContentVersion, Path()],
        body: Recipe,
    ) -> JSONResponse:
        created, server_timing = await run(
            write_frame_recipe,
            recipe_store=_recipe_store,
            spec_id=spec_id,
            content_version=content_version,
            recipe=body,
        )
        response = RecipeWriteResponse(
            spec_id=spec_id, content_version=content_version, created=created
        )
        return JSONResponse(
            response.model_dump(mode="json"), headers={"Server-Timing": server_timing}
        )

    @router.get(
        "/api/datasets/frames/{spec_id}/{content_version}/recipe",
        response_model=Recipe,
    )
    async def get_recipe(
        spec_id: Annotated[OpsSpecId, Path()],
        content_version: Annotated[OcvContentVersion, Path()],
    ) -> JSONResponse:
        recipe, server_timing = await run(
            _read_recipe,
            spec_id=spec_id,
            content_version=content_version,
            recipe_store=_recipe_store,
        )
        return JSONResponse(recipe, headers=no_store_headers(server_timing=server_timing))

    @router.get(
        "/api/datasets/recipes/replays/{replay_key}",
        response_model=ReplayedFrameResponse,
    )
    async def get_replay(replay_key: Annotated[ReplayKey, Path()]) -> JSONResponse:
        frame, server_timing = await run(
            _read_replay, replay_key=replay_key, recipe_store=_recipe_store
        )
        return JSONResponse(
            frame.model_dump(mode="json"), headers=no_store_headers(server_timing=server_timing)
        )

    @router.get(
        "/api/datasets/{id:dfo_spec_id}/versions",
        response_model=DatasetVersionsResponse,
    )
    async def get_versions(id: str) -> JSONResponse:
        versions, server_timing = await run(
            _list_versions, spec_id=id, version_lister=_version_lister
        )
        return JSONResponse(
            versions.model_dump(mode="json"),
            headers=no_store_headers(server_timing=server_timing),
        )

    return DoorContribution(
        router=router,
        classify=classify_recipe_exception,
        exception_types=(
            RecipeNotFound,
            ReplayNotFound,
            RecipeBackpressure,
            RecipeFrameNotStored,
            RecipeDiverges,
            ReplayKeyHeld,
            PublishNotPermitted,
        ),
        exception_exemplars=(
            RecipeNotFound(spec_id="ops_exemplar", content_version="ocv_exemplar"),
            ReplayNotFound(replay_key="rpk_exemplar"),
            RecipeBackpressure(),
            RecipeFrameNotStored(spec_id="ops_exemplar", content_version="ocv_exemplar"),
            RecipeDiverges(
                spec_id="ops_exemplar", content_version="ocv_exemplar", field="normal_form_hash"
            ),
            ReplayKeyHeld(
                replay_key="rpk_exemplar",
                spec_id="ops_exemplar",
                content_version="ocv_exemplar",
                holder_spec_id="ops_holder",
                holder_content_version="ocv_holder",
            ),
            PublishNotPermitted(principal=None),
        ),
        lifespan=lifespan,
    )


__all__ = [
    "RecipeBackpressure",
    "RecipeNotFound",
    "ReplayNotFound",
    "build_recipe_door",
    "classify_recipe_exception",
]
