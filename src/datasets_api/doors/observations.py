"""The observations door serves `GET /api/datasets/{spec_id}/{content_version}/observations`.

It returns the stored version's observation preview as `SpecObservations`, bare, built from the
version's stored frame by the assembly live resolve uses (`search.observations`):
the same preview mode, labels, sources, conversion block, ranking and caps. The body is
serialised as the live preview's observations event serialises it. A natively satisfied unit
request (one naming the series' own currency, which converts no member) shows each series under
the traced native unit ref with no conversion block, as live does.

The path id resolves to the served ids the registry binds to it, and a version recorded under
the posted id itself answers too (`doors.snapshot.read_under_served_id`). Refusals follow the
snapshot door (`read_stored_version`): `spec_not_found`, `identity_quarantined`,
`snapshot_not_found`, `snapshot_revoked` before expiry, and `artifact_expired` once no listable
rendering is servable. A derived version (`base_content_version` set: a groupby collapse or a
party-totals cut), a converted version that stores no per-series conversion record for a
converted series, and a version whose preview needs rows but which holds no servable stored
frame refuse `observations_unavailable`. A
graph read failure refuses retryable `service_unavailable`. The route takes `spec_` ids alone
(`routing.SpecIdConvertor`), as every Datasets route does, so any other id matches no route and
answers the platform's `not_found`.

The preview mode comes from the stored version without reading frame bytes: role-bearing when
no resolved StatVar declares `observationAbout`; places defaulted for a standard version that
binds no place and whose mint's default place scope was plural, as the version's stored
`default_scope_plural` records (such a version minted before versions stored it refuses
`observations_unavailable`); bound pairs from the members times the entities bound on each
role; the series count from the manifest. A too-large preview, a converted version's bulk
preview and a scope-defaulted version's range read return live's omissions without reading the
stored frame. Bulk `total_obs` counts frame rows, where live counts coverage rows. Under a latest
window the mint keeps each member's latest period, so a place short of it has no frame row and
the read needs no short-of-latest guard.

Reads run on the surface's read executor with `submit_with_context`, so the request's origin
catalogue stays bound. The frame load and decode hold a permit of the surface's materialise
semaphore, since the frame loads whole. A canonical frame is read without its `status` column,
which the preview never reads.
"""

from __future__ import annotations

import functools
import logging
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import cast

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from artifact_store import Tier
from artifact_store.errors import ArtifactNotFoundError
from code_index.labels import normalize_label
from code_index.query import CodeIndex
from dataset_store.snapshot import (
    SnapshotNotFound,
    SnapshotRevoked,
    StoredArtifactLocation,
)

# Register the route converters before building routes.
from datasets_api import routing  # noqa: F401
from datasets_api.admission import bounded_materialise_slot
from datasets_api.backpressure import Backpressure
from datasets_api.codes_snapshot import (
    ArtifactExpired,
    ObservationsUnavailable,
    SpecNotFound,
)
from datasets_api.deps import Deps, DepsCell
from datasets_api.doors.mint_errors import MintBackpressure
from datasets_api.doors.snapshot import (
    read_stored_version,
    read_under_served_id,
    servable_listable_locations,
)
from datasets_api.headers import no_store_headers
from datasets_api.manifest import (
    stored_manifest,
    stored_manifest_facts,
    stored_series_conversions,
)
from datasets_api.manifest_models import Manifest
from datasets_api.refusals import classify_datasets_refusal
from datasets_api.render import (
    refuse_frame1_past_decode_bound,
    rendition_token,
    servable_frame_location,
)
from datasets_api.store_errors import stored_frame_rows
from datasets_api.timings import door_call, phase, run_with_timings
from dfo_platform import ProblemSpec
from dfo_platform.doors import DoorContribution, route_registry_endpoints
from dfo_platform.problem import Refusal
from materialise.errors import OverCeilingError
from materialise.formats import (
    frame_entity_columns,
)
from materialise.graph import UNRESOLVED_NODE, ArcTarget, GraphReader, GraphReadError
from materialise.series_conversion import SeriesConversion, SeriesKey
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
    role_bulk_order,
    role_bulk_shown,
)
from spec_registry import IdentityQuarantined

logger = logging.getLogger(__name__)

_STAT_VAR_REF_PREFIX = "dcid:"
_CONSTRAINT_PROPERTIES = "constraintProperties"
_PLACE_ROLE = "observationAbout"
# `search.observations.series_facet_map` keeps this order, the facet axis order live uses.
_FACET_AXES = ("provenance", "measurementMethod", "unit")


class ObservationsBackpressure(Backpressure):
    """The door's executor or the shared materialise semaphore is full. Maps to retryable
    `service_unavailable` (503)."""

    code = "service_unavailable"

    detail = "the observations door is at its concurrent-read capacity right now; retry"


class ObservationsGraphUnavailable(Refusal):
    """A graph read for labels or member constraints failed. Maps to retryable
    `service_unavailable` (503)."""

    code = "service_unavailable"
    retryable = True
    detail = "a graph read these observations need failed; retry"


class StoredFrameCorrupt(Refusal):
    """A stored frame holds one series under two facets, which a mint never writes. Maps to
    `internal_error` (500)."""

    code = "internal_error"

    def __init__(self, *, id: str, content_version: str, key: SeriesKey) -> None:
        self.id = id
        self.content_version = content_version
        super().__init__(
            f"{id!r} at content_version {content_version!r} stores series {key!r} under more "
            "than one facet"
        )


def classify_observations_exception(exc: BaseException) -> ProblemSpec | None:
    """Classify the door's own refusals and the snapshot refusals it shares."""
    return classify_datasets_refusal(exc)


@dataclass(frozen=True)
class _IndexLabels:
    """A `search.observations.LabelSource` over the graph's labels, each passed through the
    code index (`code_index.labels.normalize_label`) as live labels are, and its node reads."""

    graph: GraphReader
    index: CodeIndex

    def node_labels_batch(self, dcids: list[str]) -> dict[str, str]:
        raw = self.graph.node_labels_batch(dcids)
        return {dcid: normalize_label(dcid, label, index=self.index) for dcid, label in raw.items()}

    def node_arc_targets_many(
        self, dcids: Sequence[str], *, properties: Sequence[str], deadline: float | None = None
    ) -> dict[str, dict[str, tuple[ArcTarget, ...]] | None]:
        return self.graph.node_arc_targets_many(dcids, properties=properties, deadline=deadline)


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
    columns: Sequence[str],
    rows: Iterable[tuple[object, ...]],
    *,
    conversions: Mapping[SeriesKey, SeriesConversion],
    native_unit_ref: str | None,
    id: str,
    content_version: str,
) -> dict[SeriesKey, SeriesRead]:
    """Group stored frame rows into series keyed as `materialise.series_conversion.series_key` keys
    them: the bare StatVar and its `(role, entity)` pairs sorted by role.

    Each series' facet refs come from its rows' `provenance`, `measurementMethod` and `unit`
    cells. A unit cell holding a unit request marks a natively satisfied series when
    `native_unit_ref` is set, which takes that ref as its unit, and a converted series
    otherwise, which takes its record from `conversions`. Raises `StoredFrameCorrupt` for a
    series under two facets and `ObservationsUnavailable` for a converted series without a
    record.
    """
    column = {name: index for index, name in enumerate(columns)}
    roles = sorted(frame_entity_columns(columns))
    facet_columns = [column[axis] for axis in _FACET_AXES]
    rows_by_key: dict[SeriesKey, tuple[tuple[object, ...], list[tuple[str, float | None]]]] = {}
    for row in rows:
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


def _load_series_reads(
    frame: StoredArtifactLocation,
    *,
    deps: Deps,
    conversions: Mapping[SeriesKey, SeriesConversion],
    native_unit_ref: str | None,
    id: str,
    content_version: str,
) -> dict[SeriesKey, SeriesRead]:
    """Read the stored frame at `frame`, decode it and group its rows (`_series_reads`). The
    caller holds a materialise permit and binds neither the bytes nor the decoded rows, so
    both are freed when this returns, before the permit is released. A frame1 document past
    `render.frame1_decode_max_bytes` refuses on its recorded size before it is read."""
    refuse_frame1_past_decode_bound(frame, fold_max_rows=deps.settings.groupby_fold_max_rows)
    with phase("frame_load"):
        data = deps.artifact_store.get(artifact_digest=frame.artifact_digest, tier=Tier(frame.tier))
    with phase("frame_decode"):
        columns, rows = stored_frame_rows(
            data, media_type=frame.media_type, digest=frame.artifact_digest
        )
        return _series_reads(
            columns,
            rows,
            conversions=conversions,
            native_unit_ref=native_unit_ref,
            id=id,
            content_version=content_version,
        )


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


def read_snapshot_observations(*, id: str, content_version: str, deps: Deps) -> SpecObservations:
    """The stored version's observation preview (see the module docstring).

    Raises the snapshot refusals, `ObservationsUnavailable`, `ObservationsBackpressure` when
    the materialise semaphore stays full, `ObservationsGraphUnavailable` for a failed graph
    read, and `StoredFrameCorrupt`.
    """
    read_at = deps.now()
    stored = read_under_served_id(
        id=id,
        spec_reader=deps.spec_reader,
        read=lambda subject_id, subject_ids: read_stored_version(
            id=subject_id,
            content_version=content_version,
            spec_reader=None,
            snapshot_reader=deps.snapshot_reader,
            now=read_at,
            revoked_under=subject_ids,
        ),
    )
    if stored.version.base_content_version is not None:
        raise ObservationsUnavailable(
            id=id,
            content_version=content_version,
            reason="is a derived version (a groupby collapse or a party-totals cut), which no "
            "live preview shows",
        )
    facts = stored_manifest_facts(stored.version)
    manifest = stored_manifest(stored.version)

    locations = stored.servable
    if not servable_listable_locations(
        locations,
        now=read_at,
        rendition=rendition_token(deps.settings.site),
        # Only asks whether any file is listed, so which digest rendered it does not matter.
        enrichment_digest=None,
    ):
        raise ArtifactExpired(id=id, content_version=content_version)
    stored = _StoredPreview.from_manifest(
        manifest, default_scope_plural=facts.default_scope_plural, id=id
    )

    @functools.cache
    def reads() -> dict[SeriesKey, SeriesRead]:
        # Looked up and decoded only here, so a preview the mode omits reads no frame and
        # decodes no conversion records.
        frame = servable_frame_location(locations, now=read_at)
        if frame is None:
            raise ObservationsUnavailable(
                id=id, content_version=content_version, reason="holds no servable stored frame"
            )
        conversions = stored_series_conversions(facts)
        try:
            with bounded_materialise_slot(
                deps.materialise_semaphore, wait_seconds=deps.settings.materialise_wait_seconds
            ):
                return _load_series_reads(
                    frame,
                    deps=deps,
                    conversions=conversions,
                    native_unit_ref=stored.native_unit_ref,
                    id=id,
                    content_version=content_version,
                )
        except MintBackpressure:
            raise ObservationsBackpressure() from None
        except ArtifactNotFoundError:
            raise ObservationsUnavailable(
                id=id, content_version=content_version, reason="has lost its stored frame bytes"
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
    # The frame holds the rows the answer keeps, the count the live read gates on. A
    # converted bulk version was omitted above.
    order = (
        role_bulk_order(
            rows=sum(read.total_rows for read in series.values()), converts_to_role_currency=False
        )
        if mode is PreviewMode.ROLE_BULK
        else None
    )
    shown = series if order is None else role_bulk_shown(series, order=order)
    try:
        with phase("assemble"):
            constraints = _member_constraints(
                (sv for sv, _roles in shown), index=deps.code_index, graph=deps.graph
            )
            if order is not None:
                return assemble_role_bulk(
                    series,
                    label_source=labels,
                    on_names_unread=_names_unread,
                    index=deps.code_index,
                    member_constraints=constraints,
                    order=order,
                )
            if mode is PreviewMode.SCOPE_BULK:
                return assemble_scope_bulk(
                    series,
                    label_source=labels,
                    on_names_unread=_names_unread,
                    index=deps.code_index,
                    member_constraints=constraints,
                )
            return assemble_per_series(
                series,
                label_source=labels,
                on_names_unread=_names_unread,
                index=deps.code_index,
                member_constraints=constraints,
                target_currency_ref=stored.target_currency_ref,
            )
    except GraphReadError as exc:
        logger.warning("Datasets observations door's graph read failed: %s", exc)
        raise ObservationsGraphUnavailable() from exc


def _names_unread() -> None:
    """A provenance name read the assembly could not make refuses as a failed label read
    does."""
    logger.warning("Datasets observations door's provenance name read failed")
    raise ObservationsGraphUnavailable()


def observations_body(observations: SpecObservations) -> dict[str, object]:
    """`observations` as the live preview's observations event serialises them."""
    return observations.model_dump(mode="json", exclude_none=True, context={"surface": "sse"})


def build_observations_door(*, deps: DepsCell) -> DoorContribution:
    """Build the observations door over the surface's dependency cell. Reads run on the read
    executor."""
    router = APIRouter()

    @router.get(
        "/api/datasets/{id:dfo_spec_id}/{content_version}/observations",
        response_model=SpecObservations,
    )
    async def get_observations(id: str, content_version: str) -> JSONResponse:
        current = deps.get()
        with door_call("observations", logger=logger) as call:
            observations = await current.read_executor.run(
                functools.partial(
                    run_with_timings,
                    read_snapshot_observations,
                    call.timings,
                    id=id,
                    content_version=content_version,
                    deps=current,
                ),
                refuse=ObservationsBackpressure,
            )
        return JSONResponse(
            observations_body(observations),
            headers=no_store_headers(server_timing=call.server_timing),
        )

    return DoorContribution(
        router=router,
        registry_endpoints=route_registry_endpoints(router, "datasets.observations"),
        classify=classify_observations_exception,
        exception_types=(
            SpecNotFound,
            # The path id resolves through the spec registry, which raises it for a quarantined
            # path id or served id bound to it.
            IdentityQuarantined,
            SnapshotNotFound,
            SnapshotRevoked,
            ArtifactExpired,
            ObservationsUnavailable,
            ObservationsBackpressure,
            ObservationsGraphUnavailable,
            StoredFrameCorrupt,
            # A stored frame1 document past the decode bound refuses before it is read.
            OverCeilingError,
        ),
        exception_exemplars=(
            IdentityQuarantined(subject_id="spec_exemplar", id_kind="spec"),
            ObservationsUnavailable(
                id="spec_exemplar", content_version="cv_exemplar", reason="holds no frame1"
            ),
            ObservationsBackpressure(),
            ObservationsGraphUnavailable(),
            StoredFrameCorrupt(
                id="spec_exemplar", content_version="cv_exemplar", key=("Count_Person", ())
            ),
            OverCeilingError("exemplar", ceiling_kind="bytes", count=1, cap=1),
        ),
    )
