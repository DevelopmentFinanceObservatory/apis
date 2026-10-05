"""Tests for role-keyed observation series, entity mapping, and pair dispatch.
They also pin error handling for preview reads, which differs from primary
resolve handling of role divergence.
"""

from __future__ import annotations

import asyncio
from unittest.mock import patch

import pydantic
import pytest

from materialise import (
    DefaultFacetUndefinedError,
    FacetNotAvailableError,
    UpstreamResponseError,
)
from search.engine.bind import _UniversalBindOutput
from search.engine.errors import EngineInputError, GraphCapExceededError
from search.engine.region import resolve_spec_resubmit, resolve_variable
from search.engine.region.codes import OBSERVATIONS_NO_POINTS, OBSERVATIONS_PROBE_FAILED
from search.engine.region.observations import fetch as region_fetch
from search.engine.region.observations.entities import _flat_obs_entities, _observation_entities
from search.engine.shape import ShapeDraft
from search.models import (
    ObservationPoint,
    ObservationSeries,
    ResubmitIdentityPayload,
    SeriesEntityRef,
    SeriesStatVarRef,
)
from series_coverage.schema import FacetMetadata, RoleNames
from tests.engine._base_identity import make_identity
from tests.engine._fake_coverage import FakeCoverageService, empty_coverage_snapshot, patch_coverage
from tests.engine._fake_sdmx import FakeSdmxClient
from tests.engine._harness import (
    ObsFacetFixture,
    build_materialise_double,
    build_observations_planned,
    harness_reader,
    materialise_double_from_obs,
    stub_preview_result,
)
from tests.fixtures import FakeGraph, make_resolver
from tests.materialise.conftest import FakeGraphReader

# A subject shape: declared_roles defaults to None (non-declaring), so
# entities_by_role reads the `subject` argument alone.
_SUBJECT_SHAPE = ShapeDraft(
    shape_id="test_subject_shape",
    label="test flow",
    pop_type_dcid="Person",
    meas_prop_dcid="count",
    stat_type_dcid="measuredValue",
    meas_qual_dcid=None,
    meas_denom_dcid=None,
    slot_keys=(),
)


def test_observation_series_round_trips_role_tuple_shape():
    series = ObservationSeries(
        stat_var=SeriesStatVarRef(ref="sdg/DC_ODA_BDVDL", label="ODA basic drinking water"),
        entities=[
            SeriesEntityRef(role="origin", ref="country/USA", label="United States"),
            SeriesEntityRef(role="destination", ref="country/KEN", label="Kenya"),
        ],
        points=[ObservationPoint(date="2020", value=1.0)],
        total_obs=1,
    )
    dumped = series.model_dump(mode="json")
    assert dumped["stat_var"] == {
        "ref": "sdg/DC_ODA_BDVDL",
        "label": "ODA basic drinking water",
    }
    assert dumped["entities"] == [
        {"role": "origin", "ref": "country/USA", "label": "United States"},
        {"role": "destination", "ref": "country/KEN", "label": "Kenya"},
    ]
    assert dumped["total_obs"] == 1
    assert ObservationSeries.model_validate(dumped) == series


def test_observation_series_labels_exclude_none_not_null():
    series = ObservationSeries(
        stat_var=SeriesStatVarRef(ref="sv1"),
        entities=[SeriesEntityRef(role="observationAbout", ref="country/IND")],
        points=[],
        total_obs=0,
    )
    dumped = series.model_dump(mode="json", exclude_none=True)
    assert "label" not in dumped["stat_var"]
    assert "label" not in dumped["entities"][0]


def test_series_entity_ref_role_rejects_a_role_outside_the_closed_vocabulary():
    with pytest.raises(pydantic.ValidationError):
        SeriesEntityRef(role="donor", ref="country/USA")  # type: ignore[arg-type]


def test_series_stat_var_ref_and_entity_ref_forbid_unknown_fields():
    with pytest.raises(pydantic.ValidationError):
        SeriesStatVarRef(ref="sv1", stat_var_dcid="sv1")  # type: ignore[call-arg]
    with pytest.raises(pydantic.ValidationError):
        SeriesEntityRef(role="observationAbout", ref="country/IND", entity_dcid="country/IND")  # type: ignore[call-arg]


def test_observation_entities_standard_returns_observation_about_map():
    result, warn = _observation_entities(
        _SUBJECT_SHAPE, destination_dcids=["a", "b"], destination_dcid="c"
    )
    assert result == {"observationAbout": ["a", "b"]}
    assert warn is None


def test_observation_entities_standard_empty_dict_when_no_entity():
    """With no entity, variable disambiguation builds an empty role map and
    skips the preview probe without a warning.
    """
    result, warn = _observation_entities(
        _SUBJECT_SHAPE, destination_dcids=None, destination_dcid=None
    )
    assert result == {}
    assert warn is None


def test_observation_entities_unplaceable_role_returns_a_warning():
    """An unplaceable declared role returns an empty map with
    `OBSERVATIONS_PROBE_FAILED`.
    """
    unplaceable_shape = ShapeDraft(
        shape_id="test_unplaceable_role_shape",
        label="test flow",
        pop_type_dcid="EconomicActivity",
        meas_prop_dcid="amount",
        stat_type_dcid="measuredValue",
        meas_qual_dcid=None,
        meas_denom_dcid=None,
        slot_keys=(),
        declared_roles=("contributor",),
    )
    result, warn = _observation_entities(
        unplaceable_shape, destination_dcids=None, destination_dcid="country/KEN"
    )
    assert result == {}
    assert warn is not None
    assert warn.code == OBSERVATIONS_PROBE_FAILED


def test_flat_obs_entities_reconstructs_standard_flat_list():
    assert _flat_obs_entities({"observationAbout": ["a", "b"]}) == ["a", "b"]


def test_flat_obs_entities_empty_when_map_empty():
    assert _flat_obs_entities({}) == []


def _empty_fake_graph(obs: dict | None = None) -> FakeGraph:
    return FakeGraph(nodes={}, obs=obs or {}, detect={}, resolve={})


def test_build_observations_standard_routes_through_a_single_pair():
    """An `observationAbout` map dispatches one standard pair through
    `materialise_preview`.
    """
    materialise_graph, materialise_sdmx = materialise_double_from_obs(
        {
            "Count_Person|country/IND": [
                {
                    "facetId": "f1",
                    "earliestDate": "2020",
                    "latestDate": "2020",
                    "obsCount": 1,
                    "provenanceId": "src/1",
                    "observations": [{"date": "2020", "value": 5.0}],
                }
            ]
        }
    )
    obs_entities, _obs_entities_warn = _observation_entities(
        _SUBJECT_SHAPE, destination_dcids=None, destination_dcid="country/IND"
    )
    assert _obs_entities_warn is None
    result, warn = build_observations_planned(
        ["Count_Person"],
        obs_entities,
        graph=FakeGraph(nodes={}),
        materialise_graph=materialise_graph,
        materialise_sdmx=materialise_sdmx,
        coverage=empty_coverage_snapshot(),
    )

    assert warn is None
    assert result is not None
    assert len(result.series) == 1
    series = result.series[0]
    assert series.stat_var.ref == "Count_Person"
    assert series.entities == [SeriesEntityRef(role="observationAbout", ref="country/IND")]
    assert series.total_obs == 1


def test_build_observations_role_bearing_routes_through_a_role_tuple_pair(monkeypatch):
    """A role map dispatches one pair containing the declared role tuple."""
    sv = "sdg/DC_ODA_BDVDL"
    materialise_graph, materialise_sdmx, _requests = build_materialise_double(
        {
            (sv, (("provider", "country/USA"), ("recipient", "country/KEN"))): [
                ObsFacetFixture(
                    facet_id="f1",
                    points=(("2020", 7.0),),
                    provenance_id="provenance/OECDDACCreditorReportingSystem",
                )
            ]
        }
    )
    _auto_coverage_index_1 = patch_coverage(
        monkeypatch,
        [(sv, "f1", "country/USA", "country/KEN", [2020])],
        role_names={sv: RoleNames(donor="provider", recipient="recipient")},
        facet_metadata={
            "f1": FacetMetadata(provenance_id="provenance/OECDDACCreditorReportingSystem")
        },
    )
    obs_entities = {"provider": ["country/USA"], "recipient": ["country/KEN"]}

    result, warn = build_observations_planned(
        [sv],
        obs_entities,
        graph=FakeGraph(nodes={}),
        materialise_graph=materialise_graph,
        materialise_sdmx=materialise_sdmx,
        coverage=FakeCoverageService(_auto_coverage_index_1).snapshot(),
    )

    assert warn is None
    assert result is not None
    assert len(result.series) == 1
    series = result.series[0]
    assert series.stat_var.ref == sv
    assert {e.role for e in series.entities} == {"origin", "destination"}
    assert {e.ref for e in series.entities} == {"country/USA", "country/KEN"}
    assert series.total_obs == 1


@pytest.mark.parametrize(
    "exc,expected_warning_code,expected_reason",
    [
        (FacetNotAvailableError(), None, OBSERVATIONS_NO_POINTS),
        (DefaultFacetUndefinedError(), OBSERVATIONS_PROBE_FAILED, OBSERVATIONS_PROBE_FAILED),
        (
            UpstreamResponseError("role divergence"),
            OBSERVATIONS_PROBE_FAILED,
            OBSERVATIONS_PROBE_FAILED,
        ),
    ],
)
def test_build_observations_role_errors_degrade_fail_open_never_no_data(
    exc, expected_warning_code, expected_reason
):
    """A missing facet yields a silent no-points omission. Undefined defaults
    and role divergence yield `OBSERVATIONS_PROBE_FAILED` warnings.
    """

    def _spy(
        request,
        *,
        sdmx,
        graph,
        limit,
        ceiling,
        deadline=None,
        rate_cache=None,
        coverage=None,
        precomputed=None,
        series_roles=(),
        series_limit=None,
        refuse_when_every_row_omitted=True,
    ):
        raise exc

    with patch.object(region_fetch, "materialise_preview", _spy):
        result, warn = build_observations_planned(
            ["sv1"],
            {"provider": ["country/USA"], "recipient": ["country/KEN"]},
            graph=FakeGraph(nodes={}),
            materialise_graph=FakeGraphReader(),
            materialise_sdmx=FakeSdmxClient(),
            coverage=empty_coverage_snapshot(),
        )

    assert result.series == []
    assert result.omitted is not None
    assert result.omitted.reason == expected_reason
    if expected_warning_code is None:
        assert warn is None
    else:
        assert warn is not None
        assert warn.code == expected_warning_code


def test_build_observations_standard_pairs_never_build_a_role_tuple():
    """Standard dispatch declares only the `observationAbout` role."""
    captured = []

    def _spy(
        request,
        *,
        sdmx,
        graph,
        limit,
        ceiling,
        deadline=None,
        rate_cache=None,
        coverage=None,
        precomputed=None,
        series_roles=(),
        series_limit=None,
        refuse_when_every_row_omitted=True,
    ):
        captured.append(request)
        return stub_preview_result(sv="Count_Person", entity="country/IND")

    with patch.object(region_fetch, "materialise_preview", _spy):
        result, warn = build_observations_planned(
            ["Count_Person"],
            {"observationAbout": ["country/IND"]},
            graph=FakeGraph(nodes={}),
            materialise_graph=harness_reader(["country/IND"]),
            materialise_sdmx=FakeSdmxClient(),
            coverage=empty_coverage_snapshot(),
        )

    assert warn is None
    assert result is not None
    assert len(captured) == 1
    assert captured[0].declared_roles == ("observationAbout",)


def test_build_observations_forwards_deadline_to_materialise_preview():
    """Per-pair dispatch receives the deadline passed to
    `_build_observations`.
    """
    captured_deadlines = []

    def _spy(
        request,
        *,
        sdmx,
        graph,
        limit,
        ceiling,
        deadline=None,
        rate_cache=None,
        coverage=None,
        precomputed=None,
        series_roles=(),
        series_limit=None,
        refuse_when_every_row_omitted=True,
    ):
        captured_deadlines.append(deadline)
        return stub_preview_result(sv="Count_Person", entity="country/IND")

    with patch.object(region_fetch, "materialise_preview", _spy):
        result, warn = build_observations_planned(
            ["Count_Person"],
            {"observationAbout": ["country/IND"]},
            graph=FakeGraph(nodes={}),
            materialise_graph=harness_reader(["country/IND"]),
            materialise_sdmx=FakeSdmxClient(),
            deadline=123.0,
            coverage=empty_coverage_snapshot(),
        )

    assert warn is None
    assert result is not None
    assert captured_deadlines == [123.0]


def test_build_observations_non_materialisation_graph_error_now_propagates_uncaught():
    """`GraphCapExceededError` from the materialisation graph propagates. The
    per-pair handler catches only documented materialisation errors.
    """

    class _CapExceededGraphReader(FakeGraphReader):
        def variable_facets_batch(self, *, stat_var, entities, deadline: float | None = None):
            raise GraphCapExceededError(
                "too many bytes", cap_kind="bytes", cap=20_000_000, observed=50_000_000
            )

    with pytest.raises(GraphCapExceededError):
        build_observations_planned(
            ["Count_Person"],
            {"observationAbout": ["country/IND"]},
            graph=FakeGraph(nodes={}),
            materialise_graph=_CapExceededGraphReader(),
            materialise_sdmx=FakeSdmxClient(),
            coverage=empty_coverage_snapshot(),
        )


def test_build_observations_rows_over_search_obs_preview_max_rows_now_truncates_not_fails(
    search_settings,
):
    """`SEARCH_OBS_PREVIEW_MAX_ROWS` limits each pair's delivered rows. Exceeding
    the limit truncates the preview and preserves the full `total_obs` count.
    """
    search_settings(obs_preview_max_rows=1)
    materialise_graph, materialise_sdmx = materialise_double_from_obs(
        {
            "Count_Person|country/IND": [
                {
                    "facetId": "f1",
                    "earliestDate": "2020",
                    "latestDate": "2021",
                    "obsCount": 2,
                    "provenanceId": "dc/base/Src",
                    "observations": [
                        {"date": "2020", "value": 1.0},
                        {"date": "2021", "value": 2.0},
                    ],
                }
            ]
        }
    )
    result, warn = build_observations_planned(
        ["Count_Person"],
        {"observationAbout": ["country/IND"]},
        graph=FakeGraph(nodes={}),
        materialise_graph=materialise_graph,
        materialise_sdmx=materialise_sdmx,
        coverage=empty_coverage_snapshot(),
    )

    assert warn is None
    assert result is not None
    # `total_obs` reports `PreviewResult.total_rows`; the delivered frame is bounded.
    assert result.total_obs == 2


# Structured resubmits and text queries check role vocabulary before resolve.


def test_resolve_spec_resubmit_standard_without_origins_is_unaffected():
    """A standard resubmit without origin refs fails on its unresolvable shape."""
    inp = ResubmitIdentityPayload(
        identity=make_identity({"kind": "statvar_schema", "population_type": "doesnt_matter"})
    )

    # Shape resolution supplies the input error for this fixture.
    with pytest.raises(EngineInputError) as exc_info:
        resolve_spec_resubmit(
            inp=inp,
            graph=_empty_fake_graph(),
            sdmx=FakeSdmxClient(),
            coverage=empty_coverage_snapshot(),
        )

    assert exc_info.value.code != "role_incompatible"


# This standard-family fixture has no directional role declaration.
_TEXT_DOOR_SV = "Count_Person"
_TEXT_DOOR_QUERY = "aid statistics"
_TEXT_DOOR_NODES: dict = {
    _TEXT_DOOR_SV: {
        "label": "Count Person",
        "type": "StatisticalVariable",
        "arcs": {
            "typeOf": {"nodes": [{"dcid": "StatisticalVariable"}]},
            "populationType": {"nodes": [{"dcid": "Person"}]},
            "measuredProperty": {"nodes": [{"dcid": "count"}]},
            "statType": {"nodes": [{"dcid": "measuredValue"}]},
        },
    },
    "country/KEN": {"label": "Kenya", "type": "Country"},
}
_TEXT_DOOR_DETECT: dict = {_TEXT_DOOR_QUERY: {"svs": [_TEXT_DOOR_SV], "entities": []}}
_TEXT_DOOR_RESOLVE: dict = {"Kenya": "country/KEN"}


class _TextDoorFixedBindLLM:
    """Elects the sole candidate shape with no constraint bindings, unconditionally."""

    async def generate_structured(self, *, prompt, system, schema):
        return _UniversalBindOutput(chosen_shape_index=0, bindings=[]), None


def test_directional_wording_with_no_declaring_shape_in_the_recall_does_not_refuse():
    """Without a role-bearing candidate, "aid to Kenya" resolves as a subject
    query despite its directional wording.
    """
    graph = FakeGraph(
        nodes=_TEXT_DOOR_NODES,
        detect=_TEXT_DOOR_DETECT,
        resolve=_TEXT_DOOR_RESOLVE,
        obs={},
    )
    result = asyncio.run(
        resolve_variable(
            _TEXT_DOOR_QUERY,
            entities=["Kenya"],
            date_request=None,
            detect_query=_TEXT_DOOR_QUERY,
            role_query="aid to Kenya",
            pac=True,
            graph=graph,
            resolver=make_resolver(graph),
            llm=_TextDoorFixedBindLLM(),
            sdmx=FakeSdmxClient(),
            base_steps=[],
            base_timing={},
            coverage=empty_coverage_snapshot(),
        )
    )

    assert result.no_data_reason != "role_incompatible"


def test_standard_shape_without_directional_wording_is_unaffected():
    """A standard shape handles the query as a subject query when no
    directional preposition appears near the place.
    """
    graph = FakeGraph(
        nodes=_TEXT_DOOR_NODES,
        detect=_TEXT_DOOR_DETECT,
        resolve=_TEXT_DOOR_RESOLVE,
        obs={},
    )
    result = asyncio.run(
        resolve_variable(
            _TEXT_DOOR_QUERY,
            entities=["Kenya"],
            date_request=None,
            detect_query=_TEXT_DOOR_QUERY,
            role_query="aid statistics Kenya",
            pac=True,
            graph=graph,
            resolver=make_resolver(graph),
            llm=_TextDoorFixedBindLLM(),
            sdmx=FakeSdmxClient(),
            base_steps=[],
            base_timing={},
            coverage=empty_coverage_snapshot(),
        )
    )

    assert result.no_data_reason != "role_incompatible"
