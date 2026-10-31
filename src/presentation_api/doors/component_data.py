"""The component data route serves the frame rows behind one component revision's shown values
as CSV, XLSX, or Parquet, bundled by default, through
`assembler.pipeline.export_component.export_component_data`.

The route belongs to the components door: `build_components_door` registers it on its router,
and it shares that door's executor, semaphore, timeout, and stores. A card's file is bounded by
`assembler.limits` (24 series of 500 points, or 2,000 table rows), so the route serves the bytes
directly. A revision is immutable, and its stored creation time fixes its file name. Its bytes
match from one request to the next while the graph's labels and source documentation behind
its frame are unchanged, so the caller may cache the response for an hour. Bundle links sit
under the public site origin, `DFO_PUBLIC_SITE_URL`, which the door reads in its lifespan.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Literal

from fastapi import APIRouter, Path, Query
from fastapi.responses import Response

from assembler.pipeline.export_component import (
    ComponentExportMismatch,
    ComponentHoldsNoData,
    export_component_data,
)
from datasets_api.artifact_delivery import artifact_content_disposition
from dfo_platform import ProblemSpec, refused
from materialise.formats import ExcelIllegalCharacterError
from presentation_api.codes import retryable_for
from presentation_api.path_params import require_component_id
from presentation_api.routes import COMPONENT_DATA_PATH

if TYPE_CHECKING:
    from presentation_api.doors.components import ComponentsDeps

logger = logging.getLogger(__name__)

COMPONENT_DATA_MEDIA_TYPES = (
    "application/zip",
    "text/csv",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "application/vnd.apache.parquet",
)
_CACHE_CONTROL = "private, max-age=3600"


def classify_component_data_exception(exc: BaseException) -> ProblemSpec | None:
    if isinstance(exc, ComponentHoldsNoData):
        return refused(
            "not_found",
            detail=exc.detail,
            retryable=retryable_for("not_found"),
        )
    if isinstance(exc, ComponentExportMismatch):
        logger.error("Component data export refused: %s", exc)
        return refused(
            "internal_error",
            detail="the frame rows behind this revision do not match the values it shows",
            retryable=retryable_for("internal_error"),
        )
    if isinstance(exc, ExcelIllegalCharacterError):
        # The writer's detail tells the caller to request csv or parquet instead.
        return refused(
            "request_invalid",
            detail=exc.detail or str(exc),
            retryable=False,
        )
    return None


COMPONENT_DATA_EXCEPTION_TYPES: tuple[type[Exception], ...] = (
    ComponentHoldsNoData,
    ComponentExportMismatch,
    ExcelIllegalCharacterError,
)
COMPONENT_DATA_EXCEPTION_EXEMPLARS: tuple[BaseException, ...] = (
    ComponentHoldsNoData(component_id="cmp_exemplar", revision=1, detail="exemplar"),
    ComponentExportMismatch(component_id="cmp_exemplar", revision=1, detail="exemplar"),
    ExcelIllegalCharacterError("exemplar", row=0, column=0),
)


def add_component_data_route(
    router: APIRouter,
    *,
    deps: ComponentsDeps,
    run: Callable[[Callable[[], Response]], Awaitable[Response]],
) -> None:
    """Register the export route on the components door's router, run through its `run`."""

    @router.get(
        COMPONENT_DATA_PATH,
        response_class=Response,
        responses={
            200: {
                "description": "The rows behind every value the revision shows. "
                "`bundle=true` serves a zip holding the file under `data/` with "
                "`data/codes.csv`, `datapackage.json`, `CITATION.cff`, `README.md`, and "
                "`SOURCES.md`.",
                "content": {media_type: {} for media_type in COMPONENT_DATA_MEDIA_TYPES},
            }
        },
    )
    async def read_component_data(
        component_id: str,
        revision: int = Path(ge=1),
        format_: Literal["csv", "xlsx", "parquet"] = Query("csv", alias="format"),
        bundle: bool = Query(True),
    ) -> Response:
        require_component_id(component_id)

        def job() -> Response:
            exported = export_component_data(
                component_id=component_id,
                revision=revision,
                format=format_,
                bundle=bundle,
                revision_store=deps.revision_store,
                frame_store=deps.frame_store,
                site_url=deps.site.public_site_url,
            )
            return Response(
                content=exported.data,
                media_type=exported.media_type,
                headers={
                    "Content-Disposition": artifact_content_disposition(exported.file_name),
                    "Cache-Control": _CACHE_CONTROL,
                },
            )

        return await run(job)
