"""The derived door serves reads of a frame the analysis runner stored, named by its `ops_` id
and `ocv_` content version, as a format 2 ref's member names it.

- `GET /api/datasets/derived/{ops_id}/{ocv}` answers `DerivedManifest` (`derived.read_derived_manifest`):
  the frame's retention, its recipe, its roots with their renderings and the group totals
  each root's read left out (`party_totals_left_out`, absent when none), the citation of their
  sources, the periods its rolling steps resolved to, and `left_out`, every series a step of
  the recipe dropped for a data gap.
- `GET /api/datasets/derived/{ops_id}/{ocv}/files/{digest}` serves the file a format 2 ref's
  derived member names by its `artifact.digest`, as the dataset link
  (`doors.files`) serves a dataset version's: 302 to a signed storage URL with
  `x-dfo-artifact-tier`, `snapshot_not_found` for an unknown pair, `snapshot_revoked` for one
  taken down, `artifact_not_found` for a digest the pair never linked, and `artifact_expired`
  for one a re-render replaced or whose bytes and stored frame are both gone. A location past
  working retention renews on the work executor under a materialise permit: its object again
  when it survives, else the frame file rendered again from the stored frame, the same bytes
  (`derived.restore_frame_file`). The digest names the bytes, so the 302 is cacheable:
  `Cache-Control: public, max-age=<n>`, `n` the whole seconds until the signed URL expires
  less 60 (`_LINK_CACHE_MARGIN`), and for a frame in the working tier no more than the seconds
  until its `held_until`; `no-store` when `n` would be under 60, as on every refusal. A pair
  taken down thus stays reachable from a cache for at most `n` seconds.
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
file and the digest link bind `datasets.files`, whose writer refusals the file also answers
(`excel_limit`, `parquet_size_bound`). All three admit every caller. The manifest and the
link read and sign on the read executor, and a file renders on the work executor under a
materialise permit. `POST /api/datasets/analysis` with `include: ["manifest"]` embeds the
manifest this door answers (`derived_manifest`).
"""

from __future__ import annotations

import functools
import logging
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from typing import Any
from urllib.parse import parse_qs, urlsplit

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, RedirectResponse, Response
from pydantic import BaseModel, ConfigDict, Field, JsonValue

from artifact_store import ArtifactSigningError, Tier
from artifact_store.signing import SIGNED_URL_REUSE_WINDOW
from dataset_store.snapshot import (
    ArtifactNotFound,
    ServableVersion,
    SnapshotNotFound,
    SnapshotRevoked,
    StoredArtifactLocation,
    read_servable_version,
)

# Register the route converter before building routes.
from datasets_api import routing  # noqa: F401
from datasets_api.admission import bounded_materialise_slot
from datasets_api.artifact_delivery import artifact_content_disposition
from datasets_api.backpressure import Backpressure
from datasets_api.codes_snapshot import ArtifactExpired, SnapshotSigningFailed
from datasets_api.config import DOWNLOAD_SIGNED_URL_TTL
from datasets_api.current import DefinitionDeps
from datasets_api.delivery import record_derived_file, record_derived_manifest, record_file
from datasets_api.deps import Deps, DepsCell
from datasets_api.derived import (
    DerivedFile,
    DerivedReads,
    locate_derived_frame,
    manifest_body,
    read_derived_manifest,
    recipe_citation,
    render_derived_file,
    restore_frame_file,
)
from datasets_api.doors.definition import _run_retrieval, _sdmx_deadline, request_definition_deps
from datasets_api.doors.files import (
    REDIRECT_RESPONSE,
    SELECTOR_QUERY_PARAMETERS,
    ResolvedLink,
    _renew,
    classify_files_exception,
    selector_options,
    signed_download_url,
)
from datasets_api.doors.mint_errors import MintBackpressure
from datasets_api.headers import ARTIFACT_TIER_HEADER, no_store_headers
from datasets_api.manifest_models import CitationModel, PartyTotalsLeftOutModel, RetentionModel
from datasets_api.mint_models import MintOptions
from datasets_api.publish_models import RecipeAnchor, RecipeDroppedSeries
from datasets_api.query_options import QueryOptionsInvalid
from datasets_api.recipe_store import RecipeReader
from datasets_api.refusals import classify_datasets_refusal
from datasets_api.retention import version_retention
from datasets_api.timings import door_call, run_with_timings
from dfo_platform.doors import DoorContribution, path_registry_endpoints
from materialise.bundle_naming import bare_file_name, sources_slug
from materialise.errors import OverCeilingError
from materialise.formats import ExcelIllegalCharacterError, ExcelLimitError, ParquetSizeBoundError

logger = logging.getLogger(__name__)

MANIFEST_PATH = "/api/datasets/derived/{id:dfo_derived_id}/{content_version}"
FILE_PATH = "/api/datasets/derived/{id:dfo_derived_id}/{content_version}/file"
LINK_PATH = "/api/datasets/derived/{id:dfo_derived_id}/{content_version}/files/{digest}"
# The characters of the `ocv_` version a download name carries, as `derived` names its files.
_TAG_LENGTH = 6

# What a cached redirect keeps of its signed URL's life, so a client that follows it from a
# cache reaches storage before the URL expires. A redirect left with less carries `no-store`.
_LINK_CACHE_MARGIN = timedelta(seconds=60)


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
    party_totals_left_out: list[PartyTotalsLeftOutModel] | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
        description=(
            "the group totals this root version's read left out, its manifest's "
            "`trace.party_totals_left_out` entries. Absent when the read left none out. A ref "
            "root never carries it"
        ),
    )


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
        fold_max_rows=current.settings.groupby_fold_max_rows,
    )


def derived_manifest(*, id: str, content_version: str, current: Deps) -> dict[str, Any]:
    """The body `GET /api/datasets/derived/{id}/{content_version}` answers, read now.

    Raises:
        SnapshotNotFound, SnapshotRevoked, ArtifactExpired: As `read_derived_manifest` does.
    """
    facts = read_derived_manifest(
        spec_id=id, content_version=content_version, reads=derived_reads(current)
    )
    return DerivedManifest.model_validate(manifest_body(facts)).model_dump(mode="json")


def derived_file[T](
    *,
    id: str,
    content_version: str,
    options: MintOptions,
    current: Deps,
    deadline: float,
    then: Callable[[DerivedFile], T],
) -> T:
    """What `then` makes of the derived frame's file in `options` (`derived.render_derived_file`),
    rendered and handed over under one materialise permit, whose wait `deadline` bounds. The
    frame is located first (`derived.locate_derived_frame`), so the refusals its record decides
    come before the permit. Nothing here binds the file, so the permit releases with its bytes
    freed once `then` keeps none."""
    reads = derived_reads(current)
    source = locate_derived_frame(spec_id=id, content_version=content_version, reads=reads)
    with bounded_materialise_slot(
        current.materialise_semaphore,
        wait_seconds=current.settings.materialise_wait_seconds,
        deadline=deadline,
    ):
        return then(
            render_derived_file(source, options=options, reads=reads, site=current.settings.site)
        )


def _read_link(
    *, id: str, content_version: str, digest: str, deps: Deps | DefinitionDeps
) -> ServableVersion:
    """The derived version the link names, with the digest checked linked to it in the same
    transaction. A digest it no longer links, though this deployment stored its bytes, is one a
    re-render replaced, and refuses `ArtifactExpired`.

    Raises:
        SnapshotNotFound, SnapshotRevoked, ArtifactNotFound, ArtifactExpired.
    """
    try:
        return read_servable_version(
            deps.snapshot_reader,
            subject_id=id,
            content_version=content_version,
            artifact_digest=digest,
            now=deps.now(),
        )
    except ArtifactNotFound:
        if deps.snapshot_reader.artifact_inventoried(digest):
            raise ArtifactExpired(id=id, content_version=content_version) from None
        raise


def _signed_expiry(url: str, *, now: datetime) -> datetime:
    """When the signed `url` expires: its V4 `X-Goog-Date` plus `X-Goog-Expires`, else, for a
    URL that states neither, as a local stack's, the earliest a reused signature expires."""
    query = parse_qs(urlsplit(url).query)
    signed, expires = query.get("X-Goog-Date"), query.get("X-Goog-Expires")
    if signed and expires:
        try:
            start = datetime.strptime(signed[0], "%Y%m%dT%H%M%SZ").replace(tzinfo=UTC)
            return start + timedelta(seconds=int(expires[0]))
        except ValueError:
            pass
    return now + DOWNLOAD_SIGNED_URL_TTL - SIGNED_URL_REUSE_WINDOW


def link_cache_control(url: str, *, now: datetime, held_until: datetime | None) -> str:
    """The `Cache-Control` of a derived link's redirect to `url`: `public, max-age=<n>`, `n`
    the whole seconds until the URL expires less `_LINK_CACHE_MARGIN`, no later than
    `held_until` for a working-tier frame, and `no-store` when `n` is under the margin."""
    until = _signed_expiry(url, now=now) - _LINK_CACHE_MARGIN
    if held_until is not None:
        until = min(until, held_until)
    seconds = int((until - now).total_seconds())
    if seconds < _LINK_CACHE_MARGIN.total_seconds():
        return "no-store"
    return f"public, max-age={seconds}"


def _signed_link(
    location: StoredArtifactLocation,
    *,
    stored: ServableVersion,
    deps: Deps | DefinitionDeps,
    recipes: RecipeReader,
) -> ResolvedLink:
    """The signed URL and tier of the derived version's file at `location`, named for download
    as the bare Parquet file route names it, by the sources its roots cite, recorded as the
    call's delivery (`delivery.record_file`)."""
    spec_id, content_version = stored.version.subject_id, stored.version.content_version
    record = recipes.read_recipe(spec_id=spec_id, content_version=content_version)
    citation = recipe_citation(deps.snapshot_reader, record)
    created = location.created_at
    prepared = created if created.tzinfo is not None else created.replace(tzinfo=UTC)
    file_name = bare_file_name(
        slug=sources_slug([source.label for source in citation.sources]),
        prepared=prepared.date(),
        tag=content_version.removeprefix("ocv_")[:_TAG_LENGTH],
        extension="parquet",
    )
    record_file(stored.version, location)
    return ResolvedLink(
        url=signed_download_url(
            location.artifact_digest,
            tier=Tier(location.tier),
            bucket=location.bucket,
            object_name=location.object_name,
            media_type=location.media_type,
            file_name=file_name,
            artifact_store=deps.artifact_store,
        ),
        tier=Tier(location.tier),
    )


def _cacheable(
    link: ResolvedLink, *, id: str, content_version: str, deps: Deps | DefinitionDeps
) -> tuple[ResolvedLink, str]:
    """`link` with the `Cache-Control` its redirect carries (`link_cache_control`), capped by
    the derived frame's retention (`retention.version_retention`)."""
    now = deps.now()
    retention = version_retention(
        deps.snapshot_reader, subject_id=id, content_version=content_version, now=now
    )
    held_until = None if retention.pinned else retention.held_until
    return link, link_cache_control(link.url, now=now, held_until=held_until)


def resolve_derived_link(
    *, id: str, content_version: str, digest: str, deps: Deps
) -> tuple[ResolvedLink, str] | None:
    """The signed URL, tier and `Cache-Control` a servable derived link redirects with, or
    `None` when the digest's location is past working retention and renews first
    (`renew_derived_link`).

    Raises:
        As `_read_link`, before any signing.
    """
    stored = _read_link(id=id, content_version=content_version, digest=digest, deps=deps)
    location = stored.servable_digest(digest)
    if location is None:
        return None
    link = _signed_link(location, stored=stored, deps=deps, recipes=deps.recipe_store)
    return _cacheable(link, id=id, content_version=content_version, deps=deps)


def renew_derived_link(
    *, id: str, content_version: str, digest: str, deps: DefinitionDeps, current: Deps
) -> tuple[ResolvedLink, str]:
    """Renew the digest's object, then resolve the link as `resolve_derived_link` does. The
    object renews when it survives (`files._renew`), else the frame file is rendered again from
    the stored frame (`derived.restore_frame_file`) under a materialise permit, which links the
    same bytes. A digest still not servable refuses `ArtifactExpired`.

    Raises:
        As `_read_link`; `ArtifactExpired` as above or when the stored frame is gone too.
    """
    stored = _read_link(id=id, content_version=content_version, digest=digest, deps=deps)
    if stored.servable_digest(digest) is None:
        clicked = next(loc for loc in stored.aged_out if loc.artifact_digest == digest)
        if not _renew(clicked, deps=deps):
            reads = derived_reads(current)
            source = locate_derived_frame(spec_id=id, content_version=content_version, reads=reads)
            with bounded_materialise_slot(
                deps.materialise_semaphore,
                wait_seconds=deps.settings.materialise_wait_seconds,
                deadline=deps.deadline,
            ):
                restore_frame_file(source, reads=reads, site=current.settings.site)
        stored = _read_link(id=id, content_version=content_version, digest=digest, deps=deps)
    location = stored.servable_digest(digest)
    if location is None:
        raise ArtifactExpired(id=id, content_version=content_version)
    link = _signed_link(location, stored=stored, deps=deps, recipes=current.recipe_store)
    return _cacheable(link, id=id, content_version=content_version, deps=deps)


class _DerivedRoutes:
    """The two handlers over the surface's dependency cell."""

    def __init__(self, cell: DepsCell) -> None:
        self._cell = cell

    async def get_derived(self, id: str, content_version: str) -> JSONResponse:
        """A derived frame's retention, recipe, roots, citation, resolved periods and the
        series its steps left out for missing data."""
        current = self._cell.get()
        with door_call("derived", logger=logger, delivers=True) as call:
            body = await current.read_executor.run(
                functools.partial(
                    run_with_timings,
                    derived_manifest,
                    call.timings,
                    id=id,
                    content_version=content_version,
                    current=current,
                ),
                refuse=DerivedBackpressure,
            )
            record_derived_manifest(spec_id=id, content_version=content_version)
        return call.delivered(
            JSONResponse(body, headers=no_store_headers(server_timing=call.server_timing))
        )

    async def get_derived_link(
        self, id: str, content_version: str, digest: str
    ) -> RedirectResponse:
        """Redirect to the derived member's file its ref names by digest, cacheable for as
        long as the signed URL and the frame's hold allow."""
        current = self._cell.get()
        with door_call("derived_files", logger=logger, delivers=True) as call:
            resolved = await current.read_executor.run(
                functools.partial(
                    run_with_timings,
                    resolve_derived_link,
                    call.timings,
                    id=id,
                    content_version=content_version,
                    digest=digest,
                    deps=current,
                ),
                refuse=DerivedBackpressure,
            )
            if resolved is None:
                resolved = await _run_retrieval(
                    current.work_executor,
                    renew_derived_link,
                    deps=request_definition_deps(current),
                    timings=call.timings,
                    refuse=DerivedBackpressure,
                    id=id,
                    content_version=content_version,
                    digest=digest,
                    current=current,
                )
        link, cache_control = resolved
        return call.delivered(
            RedirectResponse(
                link.url,
                status_code=302,
                headers={
                    **no_store_headers(server_timing=call.server_timing),
                    "cache-control": cache_control,
                    ARTIFACT_TIER_HEADER: link.tier.value,
                },
            )
        )

    async def get_derived_file(self, id: str, content_version: str, request: Request) -> Response:
        """The derived frame's rows as a file in the format and bundle the query names."""
        options = selector_options(request.query_params.multi_items())
        current = self._cell.get()
        with door_call("derived_file", logger=logger, delivers=True) as call:
            rendered = await current.work_executor.run(
                functools.partial(
                    run_with_timings,
                    derived_file,
                    call.timings,
                    id=id,
                    content_version=content_version,
                    options=options,
                    current=current,
                    deadline=_sdmx_deadline(current.settings.definition_timeout_seconds),
                    then=lambda file: file,
                ),
                refuse=DerivedBackpressure,
            )
            record_derived_file(spec_id=id, content_version=content_version, file=rendered)
        return call.delivered(
            Response(
                rendered.data,
                media_type=rendered.media_type,
                headers={
                    **no_store_headers(server_timing=call.server_timing),
                    "Content-Disposition": artifact_content_disposition(rendered.file_name),
                },
            )
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
        registry_endpoints=path_registry_endpoints(router, {MANIFEST_PATH: "datasets.snapshot"}),
        classify=classify_datasets_refusal,
        exception_types=(
            SnapshotNotFound,
            SnapshotRevoked,
            ArtifactExpired,
            DerivedBackpressure,
        ),
        endpoint_exemplars={
            "datasets.snapshot": (
                SnapshotNotFound(id="ops_exemplar", content_version="ocv_exemplar"),
                SnapshotRevoked(id="ops_exemplar", content_version="ocv_exemplar", reason=None),
                ArtifactExpired(id="ops_exemplar", content_version="ocv_exemplar"),
                DerivedBackpressure(),
            )
        },
    )


_BINARY = {"type": "string", "format": "binary"}

# `OverCeilingError` is a stored frame1 document past the decode bound.
_FILE_EXCEPTION_TYPES: tuple[type[Exception], ...] = (
    ExcelLimitError,
    ExcelIllegalCharacterError,
    ParquetSizeBoundError,
    OverCeilingError,
)


def build_derived_files_door(*, deps: DepsCell) -> DoorContribution:
    """The file route and the digest link of a derived frame, on the `datasets.files`
    endpoint."""
    router = APIRouter()
    routes = _DerivedRoutes(deps)
    router.add_api_route(
        LINK_PATH,
        routes.get_derived_link,
        methods=["GET"],
        name="get_derived_link",
        status_code=302,
        response_class=RedirectResponse,
        responses={302: REDIRECT_RESPONSE},
    )
    router.add_api_route(
        FILE_PATH,
        routes.get_derived_file,
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
        registry_endpoints=path_registry_endpoints(
            router, {FILE_PATH: "datasets.files", LINK_PATH: "datasets.files"}
        ),
        classify=classify_files_exception,
        exception_types=(
            SnapshotNotFound,
            SnapshotRevoked,
            ArtifactNotFound,
            ArtifactExpired,
            QueryOptionsInvalid,
            DerivedBackpressure,
            MintBackpressure,
            SnapshotSigningFailed,
            # Wrapped before classification, but signing can raise it.
            ArtifactSigningError,
            *_FILE_EXCEPTION_TYPES,
        ),
        endpoint_exemplars={
            "datasets.files": (
                SnapshotNotFound(id="ops_exemplar", content_version="ocv_exemplar"),
                SnapshotRevoked(id="ops_exemplar", content_version="ocv_exemplar", reason=None),
                ArtifactNotFound(
                    id="ops_exemplar",
                    content_version="ocv_exemplar",
                    artifact_digest="sha256:exemplar",
                ),
                ArtifactExpired(id="ops_exemplar", content_version="ocv_exemplar"),
                SnapshotSigningFailed(
                    bucket="exemplar-bucket",
                    object_name="sha256:exemplar",
                    digest="sha256:exemplar",
                    error_class_name=None,
                    status_code=None,
                ),
                ArtifactSigningError(
                    bucket="exemplar-bucket",
                    object_name="sha256:exemplar",
                    digest="sha256:exemplar",
                ),
                QueryOptionsInvalid(pointer="/options/exemplar", detail="exemplar"),
                DerivedBackpressure(),
                MintBackpressure(),
                OverCeilingError("exemplar", ceiling_kind="bytes", count=1, cap=1),
                ExcelLimitError("exemplar"),
                ExcelIllegalCharacterError("exemplar", row=0, column=0),
                ParquetSizeBoundError("exemplar"),
            )
        },
    )


__all__ = [
    "DerivedBackpressure",
    "DerivedManifest",
    "build_derived_door",
    "build_derived_files_door",
    "derived_file",
    "derived_manifest",
    "derived_reads",
    "link_cache_control",
    "renew_derived_link",
    "resolve_derived_link",
]
