"""The observations door serves `GET /api/datasets/{spec_id}/{content_version}/observations`.

It returns the stored version's observation preview as `SpecObservations`, bare, built from the
version's canonical frame (frame1) by the assembly live resolve uses (`search.observations`):
the same preview mode, labels, sources, conversion block, ranking and caps. The body is
serialised as the live preview's observations event serialises it. A natively satisfied unit
request (one naming the series' own currency, which converts no member) shows each series under
the traced native unit ref with no conversion block, as live does.

Refusals follow the snapshot door (`read_stored_version`): `spec_not_found`,
`snapshot_not_found`, `snapshot_revoked` before expiry, and `artifact_expired` once no listable
rendering is servable. A groupby collapse (`base_content_version` set), a converted version
whose body stores no per-series conversion record for a converted series, and a version whose
preview needs rows but which holds no servable frame1 refuse `observations_unavailable`. A graph
read failure refuses retryable `service_unavailable`. The route takes `spec_` ids alone
(`routing.SpecIdConvertor`), so a collection id matches no route and answers the platform's
`not_found`.

The preview mode comes from the stored version without reading frame bytes: role-bearing when
no resolved StatVar declares `observationAbout`; places defaulted for a standard version that
binds no place and whose mint's default place scope was plural, as the body's
`default_scope_plural` records (such a version minted before bodies stored it refuses
`observations_unavailable`); bound pairs from the members times the entities bound on each
role; the series count from the manifest. A too-large
preview, a converted version's bulk preview and a scope-defaulted version's range read return
live's omissions without looking for frame1, so a version too large for a frame1 answers them
too. Bulk `total_obs` counts frame rows, where live counts coverage rows. Under a latest window
the mint keeps each member's latest period, so a place short of it has no frame row and the read
needs no short-of-latest guard.

Reads run on the door's own executor with `submit_with_context`, so the request's origin
catalogue stays bound, behind a non-blocking semaphore sized like the snapshot door's. The
frame load and decode hold a slot of the materialise semaphore the mint and definition doors
share, since a frame1 holds up to 250,000 rows and decodes whole.
"""

from __future__ import annotations

import asyncio
import functools
import logging
import math
import threading
from collections.abc import Iterable, Mapping, Sequence
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass
from typing import cast

from fastapi import APIRouter, FastAPI
from fastapi.responses import JSONResponse

from artifact_store import ArtifactStore, LiveArtifactInventory, LiveArtifactStore, Tier
from artifact_store.errors import ArtifactNotFoundError
from code_index.labels import normalize_label
from code_index.query import CodeIndex
from code_index.shared import get_index
from dataset_store import LiveDatasetStore, close_shared_pool, shared_pool
from dataset_store.clock import Clock, utc_now
from dataset_store.snapshot import (
    LiveSnapshotReader,
    SnapshotReader,
    SnapshotRevoked,
    servable_location_of_media_type,
)

# Register the route converters before building routes.
from datasets_api import routing  # noqa: F401
from datasets_api.admission import bounded_materialise_slot
from datasets_api.codes_snapshot import (
    ArtifactExpired,
    ObservationsUnavailable,
    SnapshotNotFound,
    SpecNotFound,
)
from datasets_api.config import DatasetsBuildSettings, DatasetsConfigError, DatasetsSettings
from datasets_api.door_executor import build_door_executor
from datasets_api.doors.mint_errors import MintBackpressure
from datasets_api.doors.snapshot import (
    classify_snapshot_exception,
    read_stored_version,
    servable_listable_locations,
)
from datasets_api.generation import enter_shared_generation
from datasets_api.headers import no_store_headers
from datasets_api.manifest import (
    decode_default_scope_plural,
    decode_series_conversions,
    decode_version_body,
)
from datasets_api.manifest_models import Manifest
from datasets_api.render import rendition_token
from datasets_api.shutdown import ShutdownBudget
from datasets_api.timings import door_call, phase, run_with_timings
from dc_sdmx import SdmxClient
from dfo_platform import ProblemSpec
from dfo_platform.doors import DoorContribution
from materialise.formats import FRAME1_MEDIA_TYPE, Table, frame1_entity_columns, read_frame1_table
from materialise.graph import UNRESOLVED_NODE, GraphReader, GraphReadError
from materialise.series_conversion import SeriesConversion, SeriesKey
from origin_catalogue.graph import OriginGraphClient, build_graph_client
from origin_catalogue.service import CatalogueService
from search.models import SpecObservations
from search.observations import (
    PreviewMode,
    SeriesRead,
    assemble_per_series,
    assemble_role_bulk,
    assemble_scope_bulk,
    currency_unit_ref,
    observations_converted_bulk,
    observations_too_large,
    observations_windowed_scope,
    preview_mode,
    role_bulk_shown,
)
from spec_registry.reader import LiveSpecReader, SpecReader

logger = logging.getLogger(__name__)

_STAT_VAR_REF_PREFIX = "dcid:"
_CONSTRAINT_PROPERTIES = "constraintProperties"
_PLACE_ROLE = "observationAbout"
# `search.observations.series_facet_map` keeps this order, the facet axis order live uses.
_FACET_AXES = ("provenance", "measurementMethod", "unit")


class ObservationsBackpressure(Exception):
    """The door's executor or the shared materialise semaphore is full. Maps to retryable
    `service_unavailable` (503)."""


class ObservationsGraphUnavailable(Exception):
    """A graph read for labels or member constraints failed. Maps to retryable
    `service_unavailable` (503)."""


class StoredFrameCorrupt(Exception):
    """A stored frame1 holds one series under two facets, which a mint never writes. Maps to
    `internal_error` (500)."""

    def __init__(self, *, id: str, content_version: str, key: SeriesKey) -> None:
        self.id = id
        self.content_version = content_version
        super().__init__(
            f"{id!r} at content_version {content_version!r} stores series {key!r} under more "
            "than one facet"
        )


def classify_observations_exception(exc: BaseException) -> ProblemSpec | None:
    """Classify the door's own refusals, then the snapshot refusals it shares."""
    if isinstance(exc, ObservationsUnavailable):
        return ProblemSpec(
            status=404, code="observations_unavailable", detail=str(exc), retryable=False
        )
    if isinstance(exc, ObservationsBackpressure):
        return ProblemSpec(
            status=503,
            code="service_unavailable",
            detail="the observations door is at its concurrent-read capacity right now; retry",
            retryable=True,
        )
    if isinstance(exc, ObservationsGraphUnavailable):
        return ProblemSpec(
            status=503,
            code="service_unavailable",
            detail="a graph read these observations need failed; retry",
            retryable=True,
        )
    if isinstance(exc, StoredFrameCorrupt):
        logger.error("Datasets observations door read a corrupt stored frame: %s", exc)
        return ProblemSpec(status=500, code="internal_error", detail=str(exc), retryable=False)
    return classify_snapshot_exception(exc)


@dataclass(frozen=True)
class _IndexLabels:
    """A `search.observations.LabelSource` over the graph's labels, each passed through the
    code index (`code_index.labels.normalize_label`) as live labels are."""

    graph: GraphReader
    index: CodeIndex

    def node_labels_batch(self, dcids: list[str]) -> dict[str, str]:
        raw = self.graph.node_labels_batch(dcids)
        return {dcid: normalize_label(dcid, label, index=self.index) for dcid, label in raw.items()}


@dataclass(frozen=True)
class _StoredPreview:
    """What the stored version says about its preview, read without frame bytes."""

    role_bearing: bool
    places_defaulted: bool
    bound_pairs: int
    series_count: int | None
    range_window: bool
    converted: bool
    target_currency_ref: str | None
    native_unit_ref: str | None

    @classmethod
    def from_manifest(
        cls, manifest: Manifest, *, default_scope_plural: bool | None, id: str
    ) -> _StoredPreview:
        """Raises `ObservationsUnavailable` for a standard version that binds no place and does
        not store whether its default scope was plural (a version minted before bodies stored
        it), since the mode turns on that fact.

        `native_unit_ref` is the traced unit selection's ref when the version requests a unit
        and converts no member (`trace.conversion` absent), so the request named each series'
        own currency. A definition's members share one unit selection
        (`materialise.generation._plan_group_members`), so the request is natively satisfied
        for every member or for none."""
        assert manifest.definition is not None  # A `spec_` id names a definition.
        identity = manifest.definition.identity
        entities = cast("Mapping[str, Sequence[str]]", identity.get("entities", {}))
        resolved = cast("list[Mapping[str, object]]", manifest.trace.get("resolved_stat_vars", []))
        roles = {
            role for entry in resolved for role in cast("Sequence[str]", entry.get("roles", ()))
        }
        role_bearing = _PLACE_ROLE not in roles
        places_defaulted = False
        if not role_bearing and _PLACE_ROLE not in entities:
            if default_scope_plural is None:
                raise ObservationsUnavailable(
                    id=id,
                    content_version=manifest.content_version,
                    reason="does not store whether its default place scope was plural",
                )
            places_defaulted = default_scope_plural
        window = cast("Mapping[str, object]", identity.get("requested_window", {}))
        facets = cast("Mapping[str, Mapping[str, object]]", identity.get("facets", {}))
        request = cast("Mapping[str, object]", facets.get("unit", {}).get("request", {}))
        currency = request.get("currency")
        converted = bool(manifest.trace.get("conversion"))
        traced = cast("list[Mapping[str, object]]", manifest.trace.get("facets", []))
        unit_refs = [entry.get("ref") for entry in traced if entry.get("axis") == "unit"]
        native_unit_ref = (
            unit_refs[0]
            if request and not converted and unit_refs and isinstance(unit_refs[0], str)
            else None
        )
        return cls(
            role_bearing=role_bearing,
            places_defaulted=places_defaulted,
            bound_pairs=len(resolved) * math.prod(len(dcids) for dcids in entities.values()),
            series_count=manifest.series_count,
            range_window=window.get("kind") == "range",
            converted=converted,
            target_currency_ref=currency_unit_ref(currency if isinstance(currency, str) else None),
            native_unit_ref=native_unit_ref,
        )


def _series_reads(
    table: Table,
    *,
    conversions: Mapping[SeriesKey, SeriesConversion],
    native_unit_ref: str | None,
    id: str,
    content_version: str,
) -> dict[SeriesKey, SeriesRead]:
    """Group frame1 rows into series keyed as `materialise.series_conversion.series_key` keys
    them: the bare StatVar and its `(role, entity)` pairs sorted by role.

    Each series' facet refs come from its rows' `provenance`, `measurementMethod` and `unit`
    cells. A unit cell holding a unit request marks a natively satisfied series when
    `native_unit_ref` is set, which takes that ref as its unit, and a converted series
    otherwise, which takes its record from `conversions`. Raises `StoredFrameCorrupt` for a
    series under two facets and `ObservationsUnavailable` for a converted series without a
    record.
    """
    columns = table.columns
    column = {name: index for index, name in enumerate(columns)}
    roles = sorted(frame1_entity_columns(table))
    facet_columns = [column[axis] for axis in _FACET_AXES]
    rows_by_key: dict[SeriesKey, tuple[tuple[object, ...], list[tuple[str, float | None]]]] = {}
    for row in table.rows:
        stat_var = cast(str, row[column["stat_var"]]).removeprefix(_STAT_VAR_REF_PREFIX)
        key = (stat_var, tuple((role, cast(str, row[column[role]])) for role in roles))
        facet = tuple(row[index] for index in facet_columns)
        point = (cast(str, row[column["time_period"]]), cast("float | None", row[column["value"]]))
        held = rows_by_key.setdefault(key, (facet, []))
        if held[0] != facet:
            raise StoredFrameCorrupt(id=id, content_version=content_version, key=key)
        held[1].append(point)

    reads: dict[SeriesKey, SeriesRead] = {}
    for key, (facet, points) in rows_by_key.items():
        refs = dict(zip(_FACET_AXES, facet, strict=True))
        conversion = None
        if refs["unit"] is not None and not isinstance(refs["unit"], str):
            if native_unit_ref is not None:
                refs["unit"] = native_unit_ref
            elif (conversion := conversions.get(key)) is None:
                raise ObservationsUnavailable(
                    id=id,
                    content_version=content_version,
                    reason=f"stores no conversion record for its converted series {key!r}",
                )
        reads[key] = SeriesRead(
            points=sorted(points, key=lambda point: point[0]),
            total_rows=len(points),
            facets={axis: ref for axis, ref in refs.items() if isinstance(ref, str)},
            conversion=conversion,
        )
    return reads


def _member_constraints(
    stat_vars: Iterable[str], *, index: CodeIndex, graph: GraphReader
) -> dict[str, dict[str, str]]:
    """Each StatVar's constraint values, which compose its row label
    (`search.observations.member_row_label`): the code-index row's, else the graph's
    `constraintProperties` and their value arcs. Unresolved values are left out."""
    wanted = set(stat_vars)
    constraints = {
        row.dcid: dict(row.constraint_properties) for row in index.statvars if row.dcid in wanted
    }
    missing = sorted(wanted - constraints.keys())
    if not missing:
        return constraints
    declared = graph.node_arcs_many(missing, properties=[_CONSTRAINT_PROPERTIES])
    properties = {
        sv: [p for p in (arcs or {}).get(_CONSTRAINT_PROPERTIES, ()) if p != UNRESOLVED_NODE]
        for sv, arcs in declared.items()
    }
    constrained = [sv for sv in missing if properties.get(sv)]
    values = (
        graph.node_arcs_many(
            constrained,
            properties=sorted({p for sv in constrained for p in properties[sv]}),
        )
        if constrained
        else {}
    )
    for sv in missing:
        arcs = values.get(sv) or {}
        constraints[sv] = {
            prop: resolved[0]
            for prop in properties.get(sv, ())
            if (resolved := [v for v in arcs.get(prop, ()) if v != UNRESOLVED_NODE])
        }
    return constraints


@dataclass(kw_only=True)
class ObservationsDeps:
    """The door's readers and clients. The lifespan fills the ones left uninjected."""

    spec_reader: SpecReader
    snapshot_reader: SnapshotReader
    artifact_store: ArtifactStore
    graph: GraphReader
    code_index: CodeIndex
    materialise_semaphore: threading.BoundedSemaphore | None
    settings: DatasetsBuildSettings
    now: Clock


def read_snapshot_observations(
    *, id: str, content_version: str, deps: ObservationsDeps
) -> SpecObservations:
    """The stored version's observation preview (see the module docstring).

    Raises the snapshot refusals, `ObservationsUnavailable`, `ObservationsBackpressure` when
    the materialise semaphore stays full, `ObservationsGraphUnavailable` for a failed graph
    read, and `StoredFrameCorrupt`.
    """
    stored_version = read_stored_version(
        id=id,
        content_version=content_version,
        spec_reader=deps.spec_reader,
        snapshot_reader=deps.snapshot_reader,
    )
    if stored_version.base_content_version is not None:
        raise ObservationsUnavailable(
            id=id,
            content_version=content_version,
            reason="is a groupby collapse, which no live preview shows",
        )
    body = stored_version.immutable_body
    fields, _profile = decode_version_body(body)
    manifest = Manifest.model_validate({**fields, "artifacts": []})

    read_at = deps.now()
    locations = deps.snapshot_reader.read_artifact_locations(id, content_version)
    if not servable_listable_locations(
        locations, now=read_at, rendition=rendition_token(deps.settings.site)
    ):
        raise ArtifactExpired(id=id, content_version=content_version)
    stored = _StoredPreview.from_manifest(
        manifest, default_scope_plural=decode_default_scope_plural(body), id=id
    )
    conversions = decode_series_conversions(body)

    @functools.cache
    def reads() -> dict[SeriesKey, SeriesRead]:
        # Looked up only here, so a preview the mode omits answers for a version too large
        # for a frame1.
        frame = servable_location_of_media_type(
            locations, media_type=FRAME1_MEDIA_TYPE, now=read_at
        )
        if frame is None:
            raise ObservationsUnavailable(
                id=id, content_version=content_version, reason="holds no servable frame1"
            )
        try:
            with bounded_materialise_slot(
                deps.materialise_semaphore, wait_seconds=deps.settings.materialise_wait_seconds
            ):
                with phase("frame_load"):
                    data = deps.artifact_store.get(
                        artifact_digest=frame.artifact_digest, tier=Tier(frame.tier)
                    )
                with phase("frame_decode"):
                    return _series_reads(
                        read_frame1_table(data),
                        conversions=conversions,
                        native_unit_ref=stored.native_unit_ref,
                        id=id,
                        content_version=content_version,
                    )
        except MintBackpressure:
            raise ObservationsBackpressure() from None
        except ArtifactNotFoundError:
            raise ObservationsUnavailable(
                id=id, content_version=content_version, reason="has lost its frame1 bytes"
            ) from None

    def series_count() -> int:
        return stored.series_count if stored.series_count is not None else len(reads())

    mode = preview_mode(
        role_bearing=stored.role_bearing,
        places_defaulted=stored.places_defaulted,
        bound_pairs=stored.bound_pairs,
        series_count=series_count,
    )
    if mode is PreviewMode.TOO_LARGE:
        return observations_too_large(series_count())
    if mode is PreviewMode.SCOPE_BULK and stored.range_window:
        return observations_windowed_scope()
    if mode is not PreviewMode.PER_SERIES and stored.converted:
        return observations_converted_bulk(mode)

    series = reads()
    labels = _IndexLabels(graph=deps.graph, index=deps.code_index)
    shown = role_bulk_shown(series) if mode is PreviewMode.ROLE_BULK else series
    try:
        with phase("assemble"):
            constraints = _member_constraints(
                (sv for sv, _roles in shown), index=deps.code_index, graph=deps.graph
            )
            if mode is PreviewMode.ROLE_BULK:
                return assemble_role_bulk(
                    series, label_source=labels, member_constraints=constraints
                )
            if mode is PreviewMode.SCOPE_BULK:
                return assemble_scope_bulk(
                    series, label_source=labels, member_constraints=constraints
                )
            return assemble_per_series(
                series,
                label_source=labels,
                member_constraints=constraints,
                target_currency_ref=stored.target_currency_ref,
            )
    except GraphReadError as exc:
        logger.warning("Datasets observations door's graph read failed: %s", exc)
        raise ObservationsGraphUnavailable() from exc


def observations_body(observations: SpecObservations) -> dict[str, object]:
    """`observations` as the live preview's observations event serialises them."""
    return observations.model_dump(mode="json", exclude_none=True, context={"surface": "sse"})


@dataclass
class _Injected:
    stores: bool
    graph: bool


def build_observations_door(
    *,
    settings: DatasetsBuildSettings | None = None,
    spec_reader: SpecReader | None = None,
    snapshot_reader: SnapshotReader | None = None,
    artifact_store: ArtifactStore | None = None,
    graph: GraphReader | None = None,
    code_index: CodeIndex | None = None,
    materialise_semaphore: threading.BoundedSemaphore | None = None,
    catalogue: CatalogueService | None = None,
    now: Clock = utc_now,
    pool_size: int | None = None,
    budget: ShutdownBudget | None = None,
) -> DoorContribution:
    """Build the observations door and its executor.

    Inject `spec_reader`, `snapshot_reader` and `artifact_store` together, or leave all three
    for the lifespan to build over one pool borrow; a partial set raises `DatasetsConfigError`.
    An uninjected `graph` is a live origin graph client keyed on the shared graph generation.
    `code_index` defaults to the process-wide index. `materialise_semaphore` is the one the
    mint and definition doors share. `pool_size` defaults to
    `settings.snapshot_max_concurrency`. `budget` bounds the shutdown drain.
    """
    stores = (spec_reader, snapshot_reader, artifact_store)
    injected_stores = sum(store is not None for store in stores)
    if injected_stores not in (0, len(stores)):
        raise DatasetsConfigError(
            "datasets observations door built with some of spec_reader/snapshot_reader/"
            "artifact_store injected; inject every one of them or none"
        )
    _settings = settings if settings is not None else DatasetsBuildSettings.from_env()
    _budget = budget if budget is not None else ShutdownBudget()
    injected = _Injected(stores=injected_stores == len(stores), graph=graph is not None)
    deps = ObservationsDeps(
        spec_reader=cast(SpecReader, spec_reader),
        snapshot_reader=cast(SnapshotReader, snapshot_reader),
        artifact_store=cast(ArtifactStore, artifact_store),
        graph=cast(GraphReader, graph),
        code_index=code_index if code_index is not None else get_index(),
        materialise_semaphore=materialise_semaphore,
        settings=_settings,
        now=now,
    )
    effective_pool_size = pool_size if pool_size is not None else _settings.snapshot_max_concurrency
    door_executor = build_door_executor(effective_pool_size, "observations")
    executor = door_executor.executor

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        pool_opened_here = False
        async with AsyncExitStack() as stack:
            try:
                if not injected.stores:
                    live_settings = DatasetsSettings.from_env()
                    store = LiveDatasetStore(connections=shared_pool())
                    pool_opened_here = True
                    deps.spec_reader = LiveSpecReader(store=store)
                    deps.snapshot_reader = LiveSnapshotReader(store=store)
                    deps.artifact_store = LiveArtifactStore(
                        buckets={
                            Tier.working: live_settings.artifact_bucket_working,
                            Tier.pinned: live_settings.artifact_bucket_pinned,
                        },
                        inventory=LiveArtifactInventory(store=store),
                    )
                if not injected.graph:
                    live_settings = DatasetsSettings.from_env()
                    generation = await enter_shared_generation(
                        stack, catalogue=catalogue, graph_base_url=live_settings.graph_base_url
                    )
                    deps.graph = build_graph_client(
                        base_url=live_settings.graph_base_url,
                        token=_settings.graph_token,
                        generation=generation,
                        un_base_url=live_settings.un_graph_base_url,
                        un_availability=SdmxClient(
                            base_url=live_settings.un_graph_base_url,
                            timeout=_settings.sdmx_timeout_seconds,
                        ),
                    )
            except BaseException:
                if pool_opened_here:
                    close_shared_pool()
                raise
            try:
                yield
            finally:
                executor.shutdown(wait=False, cancel_futures=True)
                await asyncio.to_thread(door_executor.drain, _budget)
                if not injected.graph:
                    cast(OriginGraphClient, deps.graph).close()
                if pool_opened_here:
                    close_shared_pool()

    router = APIRouter()

    @router.get(
        "/api/datasets/{id:dfo_spec_id}/{content_version}/observations",
        response_model=SpecObservations,
    )
    async def get_observations(id: str, content_version: str) -> JSONResponse:
        with door_call("observations", logger=logger) as call:
            observations = await door_executor.run(
                functools.partial(
                    run_with_timings,
                    read_snapshot_observations,
                    call.timings,
                    id=id,
                    content_version=content_version,
                    deps=deps,
                ),
                refuse=ObservationsBackpressure,
            )
        return JSONResponse(
            observations_body(observations),
            headers=no_store_headers(server_timing=call.server_timing),
        )

    return DoorContribution(
        router=router,
        classify=classify_observations_exception,
        exception_types=(
            SpecNotFound,
            SnapshotNotFound,
            SnapshotRevoked,
            ArtifactExpired,
            ObservationsUnavailable,
            ObservationsBackpressure,
            ObservationsGraphUnavailable,
            StoredFrameCorrupt,
        ),
        exception_exemplars=(
            ObservationsUnavailable(
                id="spec_exemplar", content_version="cv_exemplar", reason="holds no frame1"
            ),
            ObservationsBackpressure(),
            ObservationsGraphUnavailable(),
            StoredFrameCorrupt(
                id="spec_exemplar", content_version="cv_exemplar", key=("Count_Person", ())
            ),
        ),
        lifespan=lifespan,
    )
