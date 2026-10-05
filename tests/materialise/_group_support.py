"""Shared fixtures and builders for `materialise_group` tests."""

from __future__ import annotations

from collections.abc import Sequence

import httpx

from dc_sdmx import SdmxClient
from materialise import MaterialisationRequest
from materialise.facets import FacetBinding
from materialise.graph import Facet
from materialise.revision import RevisionSnapshot, SourceRevision
from materialise.window import RequestedWindow
from series_coverage.index import CoverageIndex
from series_coverage.schema import FacetMetadata, RoleNames
from tests.engine._fake_coverage import build_coverage_index

_ENTITY = "country/XYZ"
# A second entity, distinct from `_ENTITY`, for a test that needs two
# members to land in separate `_ShapeChunk`s (`facets.entity_filter_key`
# differs) rather than sharing one combined SDMX request.
_OTHER_ENTITY = "country/OTHER"
_ROLE = "recipient"
_ALL_WINDOW = RequestedWindow(kind="all")

_STAT_VAR_A = "dcid:Stat_A"
_STAT_VAR_B = "dcid:Stat_B"
_STAT_VAR_C = "dcid:Stat_C"
_BARE_A = "Stat_A"
_BARE_B = "Stat_B"
_BARE_C = "Stat_C"

_SNAPSHOT = RevisionSnapshot(
    graph_revision="graph-rev-2024-01",
    source_revisions=(SourceRevision(source="OECD_DAC_CRS", revision="rev-7"),),
)

_CSV_HEADER = (
    "STRUCTURE,STRUCTURE_ID,ACTION,variableMeasured,recipient,unit,measurementMethod,"
    "observationPeriod,provenance,TIME_PERIOD,OBS_VALUE,scalingFactor,facetId\r\n"
)

_BOUND_FACETS = {
    "provenance": FacetBinding(state="bound", ref="PROV1"),
    "measurementMethod": FacetBinding(state="bound", ref="METH1"),
    "unit": FacetBinding(state="bound", ref="UNIT1"),
}

_SHARED_FACET = Facet(
    facet_id="facet1",
    provenance_id="PROV1",
    measurement_method="METH1",
    unit="UNIT1",
    obs_count=1,
    earliest_date="2015",
    latest_date="2017",
)


def _csv_row(
    *, stat_var: str, year: str, value: str, unit: str, method: str, provenance: str, facet_id: str
) -> str:
    method_cell = method if method != "" else "NotApplicable"
    bare_stat_var = stat_var.removeprefix("dcid:")
    return (
        f"dataflow,DC:DF_OBS(1.0.0),I,{bare_stat_var},{_ENTITY},{unit},{method_cell},"
        f"NotApplicable,{provenance},{year},{value},,{facet_id}\r\n"
    )


def _data_rows(body: str) -> str:
    """Remove the shared CSV header before combining member rows."""
    return body.removeprefix(_CSV_HEADER)


def _sdmx_client_for(
    bodies_by_bare_stat_var: dict[str, str],
    *,
    probe_bodies_by_bare_stat_var: dict[str, str] | None = None,
) -> tuple[SdmxClient, list[httpx.Request]]:
    """Route requests by `c[variableMeasured]` and combine comma-joined members
    under one CSV header. Role-bearing requests also issue `LATEST` facet
    probes. `probe_bodies_by_bare_stat_var` can give those probes separate
    responses, such as a populated probe and an empty data fetch.
    """
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        stat_vars = str(request.url.params.get("c[variableMeasured]")).split(",")
        is_latest_probe = request.url.params.get("c[TIME_PERIOD]") == "LATEST"
        rows = []
        for stat_var in stat_vars:
            body = bodies_by_bare_stat_var[stat_var]
            if is_latest_probe and probe_bodies_by_bare_stat_var is not None:
                body = probe_bodies_by_bare_stat_var.get(stat_var, body)
            rows.append(_data_rows(body))
        combined_body = _CSV_HEADER + "".join(rows)
        return httpx.Response(
            200,
            text=combined_body,
            headers={"content-type": "application/vnd.sdmx.data+csv;version=2.0.0"},
        )

    return SdmxClient(
        base_url="https://datacommons.one.org/core/api", transport=httpx.MockTransport(handler)
    ), requests


def _request(
    *,
    stat_var: str,
    facet_bindings: dict[str, FacetBinding],
    declared_roles: tuple[str, ...] = (_ROLE,),
    window: RequestedWindow = _ALL_WINDOW,
    entity: str = _ENTITY,
    index_sourced: bool = False,
) -> MaterialisationRequest:
    return MaterialisationRequest(
        stat_var=stat_var,
        role_bindings={_ROLE: [entity]},
        declared_roles=declared_roles,
        facet_bindings=facet_bindings,
        window=window,
        index_sourced=index_sourced,
    )


def _coverage_for(
    entries: Sequence[tuple[str, str]],
    *,
    entity: str = _ENTITY,
    donor: str = "country/DONOR",
    provenance: str = "PROV1",
    method: str = "METH1",
    unit: str = "UNIT1",
    year: int = 2015,
) -> CoverageIndex:
    """Build coverage for each stat-var/facet pair and bind `entity` as the
    recipient. Default facet metadata matches `_BOUND_FACETS`.
    """
    role_names = {bare: RoleNames(donor="donor", recipient=_ROLE) for bare, _ in entries}
    facet_metadata = {
        facet_id: FacetMetadata(provenance_id=provenance, measurement_method=method, unit=unit)
        for _, facet_id in entries
    }
    return build_coverage_index(
        [(bare, facet_id, donor, entity, [year]) for bare, facet_id in entries],
        role_names=role_names,
        facet_metadata=facet_metadata,
    )
