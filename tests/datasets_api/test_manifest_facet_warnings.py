"""Tests when `facet_default_selected` appears, including bound axes and collection aggregation.
The composed app and materialiser are real. Graph, SDMX, and artifact storage are fakes.
"""

from __future__ import annotations

import httpx

from datasets_api.revision import SentinelRevisionReader
from dc_sdmx import SdmxClient
from materialise.graph import Facet
from series_coverage.schema import FacetMetadata, RoleNames
from spec_registry import LiveSpecRegistry
from spec_registry.reader import LiveSpecReader
from tests.engine._fake_coverage import FakeCoverageService, build_coverage_index

from ._app_factories import make_client
from ._artifact_doubles import FakeArtifactStore
from ._graph_doubles import FakeMintGraphReader
from ._snapshot_doubles import FakeSnapshotReader
from .conftest import FakeDatasetStore
from .test_pin_door import FakeArtifactReferenceWriter

_STAT_VAR_A = "FacetWarningGrantA"
_REF_A = f"dcid:{_STAT_VAR_A}"
_BASE_URL = "https://datacommons.one.org/core/api"

_CSV_HEADER = (
    "STRUCTURE,STRUCTURE_ID,ACTION,variableMeasured,recipient,unit,measurementMethod,"
    "observationPeriod,provenance,TIME_PERIOD,OBS_VALUE,scalingFactor,facetId\r\n"
)

# Highest `obs_count` -- always the top-ranked candidate.
_WINNING_FACET = Facet(
    facet_id="facet1",
    provenance_id="OECD_DAC2A",
    measurement_method="GrantEquivalent",
    unit="USD",
    obs_count=402,
    earliest_date="2015",
    latest_date="2015",
)
# Agrees with `_WINNING_FACET` on every axis -- more than one candidate
# survives narrowing, but none of them disagree with the winner.
_AGREEING_FACET = Facet(
    facet_id="facet3",
    provenance_id="OECD_DAC2A",
    measurement_method="GrantEquivalent",
    unit="USD",
    obs_count=10,
    earliest_date="2015",
    latest_date="2015",
)
# Disagrees with `_WINNING_FACET` on `provenance` only.
_DISAGREEING_FACET = Facet(
    facet_id="facet2",
    provenance_id="IMF_WEO",
    measurement_method="GrantEquivalent",
    unit="USD",
    obs_count=10,
    earliest_date="2015",
    latest_date="2015",
)


def _arcs() -> dict[str, tuple[str, ...]]:
    return {
        "typeOf": ("StatisticalVariable",),
        "observationProperties": ("recipient",),
        "measuredProperty": ("amount",),
    }


def _graph(facets_by_stat_var: dict[str, list[Facet]]) -> FakeMintGraphReader:
    return FakeMintGraphReader(
        arcs={stat_var: _arcs() for stat_var in facets_by_stat_var},
        facets={
            (stat_var, "country/KEN"): facets for stat_var, facets in facets_by_stat_var.items()
        },
        labels={"OECD_DAC2A": "OECD DAC2a", "IMF_WEO": "IMF WEO", "country/KEN": "Kenya"},
    )


def _csv_by_stat_var(facets_by_stat_var: dict[str, list[Facet]]) -> dict[str, str]:
    """Builds SDMX rows from facets using bare StatVar ids and their facet metadata.
    Role-bearing candidate probes use coverage. This transport serves selected-facet fetches.
    """
    return {
        stat_var: _CSV_HEADER
        + "".join(
            f"dataflow,DC:DF_OBS(1.0.0),I,{stat_var},country/KEN,{facet.unit},"
            f"{facet.measurement_method},NotApplicable,{facet.provenance_id},2015,1000,,"
            f"{facet.facet_id}\r\n"
            for facet in facets
        )
        for stat_var, facets in facets_by_stat_var.items()
    }


def _sdmx_transport(csv_by_stat_var: dict[str, str]):
    def handler(request: httpx.Request) -> httpx.Response:
        # Collection requests join member ids in one `c[variableMeasured]` OR-list.
        # Return every requested member's rows under one header.
        requested = request.url.params.get("c[variableMeasured]", "").split(",")
        rows = "".join(
            csv_by_stat_var[stat_var].removeprefix(_CSV_HEADER)
            for stat_var in requested
            if stat_var in csv_by_stat_var
        )
        return httpx.Response(
            200,
            text=_CSV_HEADER + rows,
            headers={"content-type": "application/vnd.sdmx.data+csv;version=2.0.0"},
        )

    return httpx.MockTransport(handler)


def _coverage_years(obs_count: int) -> list[int]:
    """Returns up to 20 consecutive years from 2015. Coverage starts in 1960.
    Fixture counts 402 and 10 exceed that range. Assertions compare relative rank.
    """
    return list(range(2015, 2015 + min(obs_count, 20)))


def _coverage_service(facets_by_stat_var: dict[str, list[Facet]]) -> FakeCoverageService:
    """Builds coverage rows and facet metadata from each StatVar's facets."""
    rows = [
        (stat_var, facet.facet_id, "country/USA", "country/KEN", _coverage_years(facet.obs_count))
        for stat_var, facets in facets_by_stat_var.items()
        for facet in facets
    ]
    role_names = {
        stat_var: RoleNames(donor="provider", recipient="recipient")
        for stat_var in facets_by_stat_var
    }
    facet_metadata = {
        facet.facet_id: FacetMetadata(
            provenance_id=facet.provenance_id,
            measurement_method=facet.measurement_method,
            unit=facet.unit,
        )
        for facets in facets_by_stat_var.values()
        for facet in facets
    }
    index = build_coverage_index(rows, role_names=role_names, facet_metadata=facet_metadata)
    return FakeCoverageService(index)


def _client(*, graph: FakeMintGraphReader, csv_by_stat_var: dict[str, str], coverage=None):
    store = FakeDatasetStore()
    sdmx = SdmxClient(base_url=_BASE_URL, transport=_sdmx_transport(csv_by_stat_var))
    return make_client(
        registry=LiveSpecRegistry(store=store),
        spec_reader=LiveSpecReader(store=store),
        graph=graph,
        sdmx=sdmx,
        revision_reader=SentinelRevisionReader(),
        artifact_store=FakeArtifactStore(),
        snapshot_reader=FakeSnapshotReader(),
        reference_writer=FakeArtifactReferenceWriter(),
        coverage=coverage,
    )


def _body(*, statvars: list[str], facets: dict[str, dict] | None = None) -> dict:
    scope: dict = {"entities": {"recipient": ["country/KEN"]}}
    if facets is not None:
        scope["facets"] = facets
    return {
        "schema_version": 1,
        "input": {"kind": "statvar_set", "statvars": statvars, "scope": scope},
        "options": {"bundle": False},
    }


def _facet_warnings(manifest: dict) -> list[dict]:
    return [w for w in manifest["warnings"] if w["code"] == "facet_default_selected"]


def _facet_entry(manifest: dict, axis: str) -> dict:
    return next(entry for entry in manifest["trace"]["facets"] if entry["axis"] == axis)


def test_disagreeing_candidates_carry_exactly_one_warning_naming_the_axis():
    facets_by_stat_var = {_STAT_VAR_A: [_WINNING_FACET, _DISAGREEING_FACET]}
    graph = _graph(facets_by_stat_var)
    client = _client(
        graph=graph,
        csv_by_stat_var=_csv_by_stat_var(facets_by_stat_var),
        coverage=_coverage_service(facets_by_stat_var),
    )

    resp = client.post("/api/datasets", json=_body(statvars=[_REF_A]))

    assert resp.status_code == 200, resp.text
    manifest = resp.json()
    assert _facet_entry(manifest, "provenance")["basis"] == "deployment_fallback"
    warnings = _facet_warnings(manifest)
    assert len(warnings) == 1
    assert "'provenance'" in warnings[0]["message"]


def test_agreeing_candidates_carry_no_warning():
    # Agreeing candidates select by observation count but trigger no warning.
    facets_by_stat_var = {_STAT_VAR_A: [_WINNING_FACET, _AGREEING_FACET]}
    graph = _graph(facets_by_stat_var)
    client = _client(
        graph=graph,
        csv_by_stat_var=_csv_by_stat_var(facets_by_stat_var),
        coverage=_coverage_service(facets_by_stat_var),
    )

    resp = client.post("/api/datasets", json=_body(statvars=[_REF_A]))

    assert resp.status_code == 200, resp.text
    manifest = resp.json()
    assert _facet_entry(manifest, "provenance")["basis"] == "deployment_fallback"
    assert _facet_entry(manifest, "provenance")["selection_reason"] == "most_observations"
    assert _facet_warnings(manifest) == []


def test_single_candidate_carries_no_warning():
    # A sole candidate uses `sole_candidate` and triggers no warning.
    facets_by_stat_var = {_STAT_VAR_A: [_WINNING_FACET]}
    graph = _graph(facets_by_stat_var)
    client = _client(
        graph=graph,
        csv_by_stat_var=_csv_by_stat_var(facets_by_stat_var),
        coverage=_coverage_service(facets_by_stat_var),
    )

    resp = client.post("/api/datasets", json=_body(statvars=[_REF_A]))

    assert resp.status_code == 200, resp.text
    manifest = resp.json()
    assert _facet_entry(manifest, "provenance")["basis"] == "deployment_fallback"
    assert _facet_entry(manifest, "provenance")["selection_reason"] == "sole_candidate"
    assert _facet_warnings(manifest) == []


def test_caller_bound_axis_carries_no_warning_for_that_axis():
    # Binding provenance removes the disagreeing candidate before default selection.
    facets_by_stat_var = {_STAT_VAR_A: [_WINNING_FACET, _DISAGREEING_FACET]}
    graph = _graph(facets_by_stat_var)
    client = _client(
        graph=graph,
        csv_by_stat_var=_csv_by_stat_var(facets_by_stat_var),
        coverage=_coverage_service(facets_by_stat_var),
    )

    resp = client.post(
        "/api/datasets",
        json=_body(statvars=[_REF_A], facets={"provenance": {"ref": "OECD_DAC2A"}}),
    )

    assert resp.status_code == 200, resp.text
    manifest = resp.json()
    assert _facet_entry(manifest, "provenance")["basis"] == "bound"
    assert _facet_warnings(manifest) == []
