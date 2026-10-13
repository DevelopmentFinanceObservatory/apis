"""Facet-binding traces and aggregated refusal errors from `materialise_spec`."""

from __future__ import annotations

import httpx
import pytest

from dc_sdmx import SdmxClient
from materialise import EnrichmentLookup, MaterialisationRequest, materialise_spec
from materialise.errors import MaterialisationError
from materialise.facets import (
    DEPLOYMENT_FACET_POLICY,
    DEPLOYMENT_FACET_POLICY_VERSION,
    FacetBinding,
)
from materialise.graph import Facet, FacetDefaultPolicy
from materialise.window import RequestedWindow
from series_coverage.index import CoverageIndex
from series_coverage.schema import FacetMetadata, RoleNames
from tests.engine._fake_coverage import build_coverage_index

from .conftest import FakeGraphReader, only_chunk

_STAT_VAR = "dcid:MalariaGrantDisbursements"
# The graph and SDMX wires key on the bare dcid -- `probe_facets`/
# `sdmx.data` receive this form from `generation._generate_member`, never the
# `dcid:`-prefixed `MaterialisationRequest.stat_var`. `FakeGraphReader` is
# keyed by exactly what it's called with, so these fixtures use that bare
# form to match.
_BARE_STAT_VAR = "MalariaGrantDisbursements"

_CSV_BODY = (
    "STRUCTURE,STRUCTURE_ID,ACTION,variableMeasured,recipient,unit,measurementMethod,"
    "observationPeriod,provenance,TIME_PERIOD,OBS_VALUE,scalingFactor,facetId\r\n"
    f"dataflow,DC:DF_OBS(1.0.0),I,{_STAT_VAR},country/TGO,USD,GrantEquivalent,"
    "NotApplicable,OECD_DAC2A,2015,1000,,facet1\r\n"
)


_FACET = Facet(
    facet_id="facet1",
    provenance_id="OECD_DAC2A",
    measurement_method="GrantEquivalent",
    unit="USD",
    obs_count=1,
    earliest_date="2015",
    latest_date="2015",
)


def _sdmx_client(*, requests: list[httpx.Request] | None = None) -> SdmxClient:
    calls = requests if requests is not None else []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(
            200,
            text=_CSV_BODY,
            headers={"content-type": "application/vnd.sdmx.data+csv;version=2.0.0"},
        )

    return SdmxClient(
        base_url="https://datacommons.one.org/core/api", transport=httpx.MockTransport(handler)
    )


def _coverage_for(candidates: list[Facet], *, entity: str = "country/TGO") -> CoverageIndex:
    """Build role-bearing coverage for each facet's id, metadata, and date.
    The recipient is bound; donor is a placeholder because it is unbound.
    """
    role_names = {_BARE_STAT_VAR: RoleNames(donor="donor", recipient="recipient")}
    facet_metadata = {
        candidate.facet_id: FacetMetadata(
            provenance_id=candidate.provenance_id,
            measurement_method=candidate.measurement_method,
            unit=candidate.unit,
        )
        for candidate in candidates
        if candidate.facet_id is not None
    }
    rows = [
        (
            _BARE_STAT_VAR,
            candidate.facet_id,
            "country/DONOR",
            entity,
            [int(candidate.latest_date or "2015")],
        )
        for candidate in candidates
        if candidate.facet_id is not None
    ]
    return build_coverage_index(rows, role_names=role_names, facet_metadata=facet_metadata)


def _request(facet_bindings: dict[str, FacetBinding]) -> MaterialisationRequest:
    return MaterialisationRequest(
        stat_var=_STAT_VAR,
        role_bindings={"recipient": ["country/TGO"]},
        declared_roles=("recipient",),
        facet_bindings=facet_bindings,
        window=RequestedWindow(kind="all"),
    )


def test_all_three_axes_bound():
    graph = FakeGraphReader(facets={(_BARE_STAT_VAR, "country/TGO"): [_FACET]})
    facet_bindings = {
        "provenance": FacetBinding(state="bound", ref="OECD_DAC2A"),
        "measurementMethod": FacetBinding(state="bound", ref="GrantEquivalent"),
        "unit": FacetBinding(state="bound", ref="USD"),
    }

    coverage = _coverage_for([_FACET])
    result = materialise_spec(
        _request(facet_bindings),
        sdmx=_sdmx_client(),
        graph=graph,
        enrichment=EnrichmentLookup(graph, parties={}),
        materialiser_build="build-123",
        capability_version="cap-v1",
        coverage=coverage,
    )

    facets_by_axis = {entry["axis"]: entry for entry in result.trace.facets}
    assert set(facets_by_axis) == {"provenance", "measurementMethod", "unit"}
    for entry in facets_by_axis.values():
        assert entry["basis"] == "bound"
        assert entry["default_policy"] is None
        assert entry["default_policy_version"] is None
        assert entry["selection_reason"] is None
        assert entry["dispatch_facet_id"] == "facet1"


def test_declared_default_availability_confirmed():
    policy = FacetDefaultPolicy(
        resolved_ref="OECD_DAC2A",
        policy_node="policy/provenance-default",
        policy_version="v1",
        selection_reason="most-widely-reported source",
    )
    graph = FakeGraphReader(
        facets={(_BARE_STAT_VAR, "country/TGO"): [_FACET]},
        policies={(_STAT_VAR, "provenance"): policy},
    )
    facet_bindings = {
        "provenance": FacetBinding(state="unbound"),
        "measurementMethod": FacetBinding(state="bound", ref="GrantEquivalent"),
        "unit": FacetBinding(state="bound", ref="USD"),
    }

    coverage = _coverage_for([_FACET])
    result = materialise_spec(
        _request(facet_bindings),
        sdmx=_sdmx_client(),
        graph=graph,
        enrichment=EnrichmentLookup(graph, parties={}),
        materialiser_build="build-123",
        capability_version="cap-v1",
        coverage=coverage,
    )

    facets_by_axis = {entry["axis"]: entry for entry in result.trace.facets}
    provenance_entry = facets_by_axis["provenance"]
    assert provenance_entry["basis"] == "declared_default"
    assert provenance_entry["ref"] == "OECD_DAC2A"
    assert provenance_entry["default_policy"] == "policy/provenance-default"
    assert provenance_entry["default_policy_version"] == "v1"
    assert provenance_entry["selection_reason"] == "most-widely-reported source"
    assert provenance_entry["dispatch_facet_id"] == "facet1"


def test_undeclared_default_raises_after_the_facet_probe_and_before_any_data_fetch():
    # Each candidate matches one bound axis, but none matches both. The
    # narrowed provenance pool is empty, so the unbound axis has no default.
    # Coverage supplies candidates in memory; no SDMX fetch runs.
    matches_measurement_method = Facet(
        facet_id="facetA",
        provenance_id="OECD_DAC2A",
        measurement_method="GrantEquivalent",
        unit="EUR",
        obs_count=1,
        earliest_date="2015",
        latest_date="2015",
    )
    matches_unit = Facet(
        facet_id="facetB",
        provenance_id="IMF_WEO",
        measurement_method="Other",
        unit="USD",
        obs_count=1,
        earliest_date="2015",
        latest_date="2015",
    )
    graph = FakeGraphReader(
        facets={(_BARE_STAT_VAR, "country/TGO"): [matches_measurement_method, matches_unit]}
    )
    facet_bindings = {
        "provenance": FacetBinding(state="unbound"),
        "measurementMethod": FacetBinding(state="bound", ref="GrantEquivalent"),
        "unit": FacetBinding(state="bound", ref="USD"),
    }
    requests: list[httpx.Request] = []
    coverage = _coverage_for([matches_measurement_method, matches_unit])
    sdmx = _sdmx_client(requests=requests)

    with pytest.raises(MaterialisationError) as excinfo:
        materialise_spec(
            _request(facet_bindings),
            sdmx=sdmx,
            graph=graph,
            enrichment=EnrichmentLookup(graph, parties={}),
            materialiser_build="build-123",
            capability_version="cap-v1",
            coverage=coverage,
        )

    assert excinfo.value.code == "default_facet_undefined"
    assert excinfo.value.issues is not None
    assert {issue.axis for issue in excinfo.value.issues} == {"provenance"}
    (provenance_issue,) = excinfo.value.issues
    # `default_facet_undefined` issues always leave `alternatives` unset.
    assert provenance_issue.alternatives is None
    assert provenance_issue.truncated is None
    # Candidates come from `coverage`, an in-memory lookup: the failed
    # default-facet check raises before any network call at all.
    assert requests == []


def test_mixed_failure_aggregates_into_one_error_prioritising_undefined():
    # Provenance has no default; the bound unit has no matching observation.
    graph = FakeGraphReader(facets={(_BARE_STAT_VAR, "country/TGO"): [_FACET]})
    facet_bindings = {
        "provenance": FacetBinding(state="unbound"),
        "measurementMethod": FacetBinding(state="bound", ref="GrantEquivalent"),
        "unit": FacetBinding(state="bound", ref="EUR"),
    }

    coverage = _coverage_for([_FACET])
    with pytest.raises(MaterialisationError) as excinfo:
        materialise_spec(
            _request(facet_bindings),
            sdmx=_sdmx_client(),
            graph=graph,
            enrichment=EnrichmentLookup(graph, parties={}),
            materialiser_build="build-123",
            capability_version="cap-v1",
            coverage=coverage,
        )

    assert excinfo.value.code == "default_facet_undefined"
    assert excinfo.value.issues is not None
    issues_by_axis = {issue.axis: issue for issue in excinfo.value.issues}
    assert set(issues_by_axis) == {"provenance", "unit"}
    # The single-facet `FakeGraphReader` fixture yields exactly 1 alternative
    # for `unit` -- well below `_MAX_ALTERNATIVES_PER_ISSUE` -- so this axis's
    # issue is not truncated.
    assert issues_by_axis["unit"].truncated is False


def test_all_three_axes_unbound_bind_via_deployment_default_policy():
    # No graph-declared policy, one probed candidate, every axis unbound --
    # the deployment default policy binds all three axes off that one
    # candidate.
    graph = FakeGraphReader(facets={(_BARE_STAT_VAR, "country/TGO"): [_FACET]})
    coverage = _coverage_for([_FACET])

    result = materialise_spec(
        _request({}),
        sdmx=_sdmx_client(),
        graph=graph,
        enrichment=EnrichmentLookup(graph, parties={}),
        materialiser_build="build-123",
        capability_version="cap-v1",
        coverage=coverage,
    )

    facets_by_axis = {entry["axis"]: entry for entry in result.trace.facets}
    assert set(facets_by_axis) == {"provenance", "measurementMethod", "unit"}
    for axis, expected_ref in (
        ("provenance", "OECD_DAC2A"),
        ("measurementMethod", "GrantEquivalent"),
        ("unit", "USD"),
    ):
        entry = facets_by_axis[axis]
        assert entry["ref"] == expected_ref
        assert entry["basis"] == "deployment_fallback"
        assert entry["default_policy"] == DEPLOYMENT_FACET_POLICY
        assert entry["default_policy_version"] == DEPLOYMENT_FACET_POLICY_VERSION
        assert entry["selection_reason"] == "sole_candidate"
        assert entry["dispatch_facet_id"] == "facet1"


def test_unbound_axis_absent_on_the_selected_facet_binds_null():
    facet_without_measurement_method = Facet(
        facet_id="facet1",
        provenance_id="OECD_DAC2A",
        measurement_method=None,
        unit="USD",
        obs_count=1,
        earliest_date="2015",
        latest_date="2015",
    )
    graph = FakeGraphReader(
        facets={(_BARE_STAT_VAR, "country/TGO"): [facet_without_measurement_method]}
    )
    csv_body = (
        "STRUCTURE,STRUCTURE_ID,ACTION,variableMeasured,recipient,unit,measurementMethod,"
        "observationPeriod,provenance,TIME_PERIOD,OBS_VALUE,scalingFactor,facetId\r\n"
        # `measurementMethod`'s own cell is the `NotApplicable` sentinel, which
        # `dc_sdmx.results` parses to `None` -- matching the selected facet's
        # own absent value on this axis, so the row survives the AND-filter.
        f"dataflow,DC:DF_OBS(1.0.0),I,{_STAT_VAR},country/TGO,USD,NotApplicable,"
        "NotApplicable,OECD_DAC2A,2015,1000,,facet1\r\n"
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            text=csv_body,
            headers={"content-type": "application/vnd.sdmx.data+csv;version=2.0.0"},
        )

    sdmx = SdmxClient(
        base_url="https://datacommons.one.org/core/api", transport=httpx.MockTransport(handler)
    )
    coverage = _coverage_for([facet_without_measurement_method])

    result = materialise_spec(
        _request({}),
        sdmx=sdmx,
        graph=graph,
        enrichment=EnrichmentLookup(graph, parties={}),
        materialiser_build="build-123",
        capability_version="cap-v1",
        coverage=coverage,
    )

    facets_by_axis = {entry["axis"]: entry for entry in result.trace.facets}
    # The trace carries all three axes, `measurementMethod` with a null
    # ref.
    assert len(result.trace.facets) == 3
    measurement_method_entry = facets_by_axis["measurementMethod"]
    assert measurement_method_entry["ref"] is None
    assert measurement_method_entry["basis"] == "deployment_fallback"
    assert measurement_method_entry["selection_reason"] == "absent_on_selected_facet"


def test_dispatch_facet_ids_include_a_sole_joint_match_not_just_the_witnesses():
    # A/B/C each match one axis first; D matches all three. Dispatch must
    # include D as well as the per-axis witnesses or its rows go unfetched.
    candidate_a = Facet(
        facet_id="facet1",
        provenance_id="OECD_DAC2A",
        measurement_method="Other",
        unit="EUR",
        obs_count=1,
        earliest_date="2015",
        latest_date="2015",
    )
    candidate_b = Facet(
        facet_id="facet2",
        provenance_id="IMF_WEO",
        measurement_method="GrantEquivalent",
        unit="GBP",
        obs_count=1,
        earliest_date="2015",
        latest_date="2015",
    )
    candidate_c = Facet(
        facet_id="facet3",
        provenance_id="WorldBank",
        measurement_method="Amount",
        unit="USD",
        obs_count=1,
        earliest_date="2015",
        latest_date="2015",
    )
    candidate_d = Facet(
        facet_id="facet4",
        provenance_id="OECD_DAC2A",
        measurement_method="GrantEquivalent",
        unit="USD",
        obs_count=1,
        earliest_date="2015",
        latest_date="2015",
    )
    graph = FakeGraphReader(
        facets={
            (_BARE_STAT_VAR, "country/TGO"): [candidate_a, candidate_b, candidate_c, candidate_d]
        }
    )
    facet_bindings = {
        "provenance": FacetBinding(state="bound", ref="OECD_DAC2A"),
        "measurementMethod": FacetBinding(state="bound", ref="GrantEquivalent"),
        "unit": FacetBinding(state="bound", ref="USD"),
    }
    csv_body = (
        "STRUCTURE,STRUCTURE_ID,ACTION,variableMeasured,recipient,unit,measurementMethod,"
        "observationPeriod,provenance,TIME_PERIOD,OBS_VALUE,scalingFactor,facetId\r\n"
        f"dataflow,DC:DF_OBS(1.0.0),I,{_STAT_VAR},country/TGO,USD,GrantEquivalent,"
        "NotApplicable,OECD_DAC2A,2015,1000,,facet4\r\n"
    )
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            text=csv_body,
            headers={"content-type": "application/vnd.sdmx.data+csv;version=2.0.0"},
        )

    sdmx = SdmxClient(
        base_url="https://datacommons.one.org/core/api", transport=httpx.MockTransport(handler)
    )
    coverage = _coverage_for([candidate_a, candidate_b, candidate_c, candidate_d])

    result = materialise_spec(
        _request(facet_bindings),
        sdmx=sdmx,
        graph=graph,
        enrichment=EnrichmentLookup(graph, parties={}),
        materialiser_build="build-123",
        capability_version="cap-v1",
        coverage=coverage,
    )

    # Candidates come from `coverage`, an in-memory lookup: the one
    # dispatched fetch is the only SDMX call this test issues.
    assert len(requests) == 1
    assert "c%5BfacetId%5D=facet4" in str(requests[-1].url)
    assert len(only_chunk(result).rows) == 1
    assert result.warnings == ()


def test_dispatch_skips_the_fetch_when_no_candidate_jointly_matches(caplog):
    # No candidate matches all axes. Empty `facet_ids` disables SDMX
    # filtering, so the fetch must be skipped and `empty_result` logged.
    # The log distinguishes this branch from an empty upstream response.
    candidate_a = Facet(
        facet_id="facet1",
        provenance_id="OECD_DAC2A",
        measurement_method="Other",
        unit="EUR",
        obs_count=1,
        earliest_date="2015",
        latest_date="2015",
    )
    candidate_b = Facet(
        facet_id="facet2",
        provenance_id="IMF_WEO",
        measurement_method="GrantEquivalent",
        unit="GBP",
        obs_count=1,
        earliest_date="2015",
        latest_date="2015",
    )
    candidate_c = Facet(
        facet_id="facet3",
        provenance_id="WorldBank",
        measurement_method="Amount",
        unit="USD",
        obs_count=1,
        earliest_date="2015",
        latest_date="2015",
    )
    graph = FakeGraphReader(
        facets={(_BARE_STAT_VAR, "country/TGO"): [candidate_a, candidate_b, candidate_c]}
    )
    facet_bindings = {
        "provenance": FacetBinding(state="bound", ref="OECD_DAC2A"),
        "measurementMethod": FacetBinding(state="bound", ref="GrantEquivalent"),
        "unit": FacetBinding(state="bound", ref="USD"),
    }
    requests: list[httpx.Request] = []
    sdmx = _sdmx_client(requests=requests)
    coverage = _coverage_for([candidate_a, candidate_b, candidate_c])

    with caplog.at_level("DEBUG", logger="materialise.generation"):
        result = materialise_spec(
            _request(facet_bindings),
            sdmx=sdmx,
            graph=graph,
            enrichment=EnrichmentLookup(graph, parties={}),
            materialiser_build="build-123",
            capability_version="cap-v1",
            coverage=coverage,
        )

    # Candidates come from `coverage`, an in-memory lookup: no candidate
    # jointly matches every bound axis, so the data fetch itself is
    # skipped, and no SDMX call happens at all.
    assert requests == []
    assert only_chunk(result).rows == ()
    assert result.warnings[0]["code"] == "empty_result"
    assert any(
        "jointly matched" in record.getMessage() and _STAT_VAR in record.getMessage()
        for record in caplog.records
    )


def test_one_frame_minted_under_two_facet_bases_gets_one_token():
    """The same rows bound explicitly on every axis and chosen by the deployment default
    policy are one frame and so one `content_version`. The token hashes the canonical frame
    alone, and each run's trace still records its own facet bases."""
    graph = FakeGraphReader(facets={(_BARE_STAT_VAR, "country/TGO"): [_FACET]})
    bound = {
        "provenance": FacetBinding(state="bound", ref="OECD_DAC2A"),
        "measurementMethod": FacetBinding(state="bound", ref="GrantEquivalent"),
        "unit": FacetBinding(state="bound", ref="USD"),
    }

    results = [
        materialise_spec(
            _request(bindings),
            sdmx=_sdmx_client(),
            graph=graph,
            enrichment=EnrichmentLookup(graph, parties={}),
            materialiser_build="build-123",
            capability_version="cap-v1",
            coverage=_coverage_for([_FACET]),
        )
        for bindings in (bound, {})
    ]

    assert {entry["basis"] for entry in results[0].trace.facets} == {"bound"}
    assert {entry["basis"] for entry in results[1].trace.facets} == {"deployment_fallback"}
    assert only_chunk(results[0]).table.equals(only_chunk(results[1]).table)
    assert results[0].content_version == results[1].content_version
