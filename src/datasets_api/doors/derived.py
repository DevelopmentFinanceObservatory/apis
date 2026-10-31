"""The derived door serves reads of a frame the analysis runner stored, named by its `ops_` id
and `ocv_` content version, as a format 2 ref's member names it.

- `GET /api/datasets/derived/{ops_id}/{ocv}` answers `DerivedManifest` (`derived.read_derived_manifest`):
  the frame's retention, its recipe, its roots with their renderings, the citation of their
  sources, the periods its rolling steps resolved to, and `left_out`, every series a step of
  the recipe dropped for a data gap.
- `GET /api/datasets/derived/{ops_id}/{ocv}/file?format=&bundle=&layout=` answers the frame's rows as
  a file (`derived.render_derived_file`), 200 with the bytes and a `Content-Disposition`
  naming the download. The query is the dataset file selector's (`doors.files.selector_options`)
  with its defaults, a CSV bundle. A derived frame has one layout, the frame's, so `layout`
  takes only `frame`.

A ref's derived member gives both paths, as its `spec.spec_id` and `content_version` are the path
segments, so a client derives the paths from the ref and `POST /api/datasets/analysis` answers no
path of its own. Reading the frame on its datasets' newest versions is `POST /api/datasets/analysis` with
the derived root as the whole tree, `{"tree": {"spec_id": ops_id, "content_version": ocv},
"latest": true}`, which computes the stored recipe again under its own ids.

The manifest binds the `datasets.snapshot` registry endpoint, as a dataset version's manifest
does, and answers its codes (`snapshot_not_found`, `snapshot_revoked`, `artifact_expired`). The
file binds `datasets.files`, whose writer refusals it also answers (`excel_limit`,
`parquet_size_bound`). Both admit every caller. The manifest reads on the read executor, and a
file renders on the work executor under a materialise permit.
"""

from __future__ import annotations

import functools
import logging
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, ConfigDict, Field, JsonValue

from dataset_store.snapshot import SnapshotNotFound, SnapshotRevoked

# Register the route converter before building routes.
from datasets_api import routing  # noqa: F401
from datasets_api.admission import bounded_materialise_slot
from datasets_api.artifact_delivery import artifact_content_disposition
from datasets_api.backpressure import Backpressure
from datasets_api.codes_snapshot import ArtifactExpired
from datasets_api.deps import Deps, DepsCell
from datasets_api.derived import (
    DerivedFile,
    DerivedReads,
    manifest_body,
    read_derived_manifest,
    render_derived_file,
)
from datasets_api.doors.files import (
    SELECTOR_QUERY_PARAMETERS,
    classify_files_exception,
    selector_options,
)
from datasets_api.doors.mint_errors import MintBackpressure
from datasets_api.headers import no_store_headers
from datasets_api.manifest_models import CitationModel, RetentionModel
from datasets_api.mint_models import MintOptions
from datasets_api.publish_models import RecipeAnchor, RecipeDroppedSeries
from datasets_api.query_options import QueryOptionsInvalid
from datasets_api.refusals import classify_datasets_refusal
from datasets_api.timings import door_call, run_with_timings
from dfo_platform.doors import DoorContribution
from materialise.formats import ExcelIllegalCharacterError, ExcelLimitError, ParquetSizeBoundError

logger = logging.getLogger(__name__)

MANIFEST_PATH = "/api/datasets/derived/{id:dfo_derived_id}/{content_version}"
FILE_PATH = "/api/datasets/derived/{id:dfo_derived_id}/{content_version}/file"


class DerivedBackpressure(Backpressure):
    """A full read or work executor maps to retryable `service_unavailable` (503)."""

    code = "service_unavailable"

    detail = "the derived door is at its concurrent-call capacity right now; retry"


class DerivedRecipe(BaseModel):
    model_config = ConfigDict(extra="forbid")

    normal_form_hash: str
    replay_key: str
    execution_identity: str
    tree: dict[str, JsonValue] = Field(description="the recipe's op-spec tree")
    roots: list[JsonValue] = Field(description="how the recipe read each root dataset")


class DerivedManifestRoot(BaseModel):
    model_config = ConfigDict(extra="forbid")

    spec_id: str
    content_version: str
    rendering: dict[str, JsonValue]


class DerivedManifest(BaseModel):
    """What a client builds a derived Dataset from. See the module docstring."""

    model_config = ConfigDict(extra="forbid")

    spec_id: str = Field(pattern=r"^ops_[0-9a-f]{32}$")
    content_version: str = Field(pattern=r"^ocv_[0-9a-f]{32}$")
    retention: RetentionModel
    citation: CitationModel = Field(description="the sources of every root, each once")
    recipe: DerivedRecipe | None = Field(description="null for a frame stored without a recipe")
    roots: list[DerivedManifestRoot] = Field(description="each root dataset version once, in order")
    resolved_periods: list[dict[str, JsonValue]] = Field(
        description="each period the recipe states relative to the latest, with its period"
    )
    anchor: RecipeAnchor | None = Field(
        description="the latest period of the rolling anchor and the series without a row at it"
    )
    left_out: list[RecipeDroppedSeries] | None = Field(
        description=(
            "every series a step of the recipe dropped for a data gap, step by step; null for a "
            "recipe stored without them"
        )
    )
    left_out_complete: bool | None = Field(
        description="false when the steps dropped more series than `left_out` lists"
    )


def derived_reads(current: Deps) -> DerivedReads:
    """The stores a derived read reads, from the surface's `Deps`."""
    return DerivedReads(
        snapshot_reader=current.snapshot_reader,
        artifact_store=current.artifact_store,
        recipe_reader=current.recipe_store,
        enrichment=current.enrichment,
        version_writer=current.version_writer,
        now=current.now,
    )


def _manifest(*, id: str, content_version: str, current: Deps) -> dict[str, Any]:
    facts = read_derived_manifest(
        spec_id=id, content_version=content_version, reads=derived_reads(current)
    )
    return DerivedManifest.model_validate(manifest_body(facts)).model_dump(mode="json")


def derived_file(
    *, id: str, content_version: str, options: MintOptions, current: Deps
) -> DerivedFile:
    """The derived frame's file in `options` (`derived.render_derived_file`), rendered under a
    materialise permit."""
    with bounded_materialise_slot(
        current.materialise_semaphore, wait_seconds=current.settings.materialise_wait_seconds
    ):
        return render_derived_file(
            spec_id=id,
            content_version=content_version,
            options=options,
            reads=derived_reads(current),
            site=current.settings.site,
        )


class _DerivedRoutes:
    """The two handlers over the surface's dependency cell."""

    def __init__(self, cell: DepsCell) -> None:
        self._cell = cell

    async def get_derived(self, id: str, content_version: str) -> JSONResponse:
        """A derived frame's retention, recipe, roots, citation, resolved periods and the
        series its steps left out for missing data."""
        current = self._cell.get()
        with door_call("derived", logger=logger) as call:
            body = await current.read_executor.run(
                functools.partial(
                    run_with_timings,
                    _manifest,
                    call.timings,
                    id=id,
                    content_version=content_version,
                    current=current,
                ),
                refuse=DerivedBackpressure,
            )
        return JSONResponse(body, headers=no_store_headers(server_timing=call.server_timing))

    async def get_derived_file(self, id: str, content_version: str, request: Request) -> Response:
        """The derived frame's rows as a file in the format and bundle the query names."""
        options = selector_options(request.query_params.multi_items())
        current = self._cell.get()
        with door_call("derived_file", logger=logger) as call:
            rendered = await current.work_executor.run(
                functools.partial(
                    run_with_timings,
                    derived_file,
                    call.timings,
                    id=id,
                    content_version=content_version,
                    options=options,
                    current=current,
                ),
                refuse=DerivedBackpressure,
            )
        return Response(
            rendered.data,
            media_type=rendered.media_type,
            headers={
                **no_store_headers(server_timing=call.server_timing),
                "Content-Disposition": artifact_content_disposition(rendered.file_name),
            },
        )


def build_derived_door(*, deps: DepsCell) -> DoorContribution:
    """The manifest route of a derived frame, on the `datasets.snapshot` endpoint."""
    router = APIRouter()
    router.add_api_route(
        MANIFEST_PATH,
        _DerivedRoutes(deps).get_derived,
        methods=["GET"],
        name="get_derived",
        response_model=DerivedManifest,
    )
    return DoorContribution(
        router=router,
        registry_endpoints={MANIFEST_PATH: "datasets.snapshot"},
        classify=classify_datasets_refusal,
        exception_types=(
            SnapshotNotFound,
            SnapshotRevoked,
            ArtifactExpired,
            DerivedBackpressure,
        ),
        exception_exemplars=(
            SnapshotNotFound(id="ops_exemplar", content_version="ocv_exemplar"),
            SnapshotRevoked(id="ops_exemplar", content_version="ocv_exemplar", reason=None),
            ArtifactExpired(id="ops_exemplar", content_version="ocv_exemplar"),
            DerivedBackpressure(),
        ),
    )


_BINARY = {"type": "string", "format": "binary"}

_FILE_EXCEPTION_TYPES: tuple[type[Exception], ...] = (
    ExcelLimitError,
    ExcelIllegalCharacterError,
    ParquetSizeBoundError,
)


def build_derived_files_door(*, deps: DepsCell) -> DoorContribution:
    """The file route of a derived frame, on the `datasets.files` endpoint."""
    router = APIRouter()
    router.add_api_route(
        FILE_PATH,
        _DerivedRoutes(deps).get_derived_file,
        methods=["GET"],
        name="get_derived_file",
        response_class=Response,
        responses={
            200: {
                "description": "the file's bytes",
                "content": {"application/octet-stream": {"schema": _BINARY}},
            }
        },
        openapi_extra={"parameters": SELECTOR_QUERY_PARAMETERS},
    )
    return DoorContribution(
        router=router,
        registry_endpoints={FILE_PATH: "datasets.files"},
        classify=classify_files_exception,
        exception_types=(
            SnapshotNotFound,
            SnapshotRevoked,
            ArtifactExpired,
            QueryOptionsInvalid,
            DerivedBackpressure,
            MintBackpressure,
            *_FILE_EXCEPTION_TYPES,
        ),
        exception_exemplars=(
            SnapshotNotFound(id="ops_exemplar", content_version="ocv_exemplar"),
            SnapshotRevoked(id="ops_exemplar", content_version="ocv_exemplar", reason=None),
            ArtifactExpired(id="ops_exemplar", content_version="ocv_exemplar"),
            QueryOptionsInvalid(pointer="/options/exemplar", detail="exemplar"),
            DerivedBackpressure(),
        ),
    )


__all__ = [
    "DerivedBackpressure",
    "DerivedManifest",
    "build_derived_door",
    "build_derived_files_door",
    "derived_file",
    "derived_reads",
]
