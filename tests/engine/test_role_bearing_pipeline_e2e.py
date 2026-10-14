"""End-to-end tests for role-bearing resolution through resolve_variable.
Synthetic fixtures use role names absent from the deployed corpus."""

from __future__ import annotations

import asyncio
from dataclasses import replace

import pytest

from search.engine.bind import SlotBindingDraft, _UniversalBindOutput
from search.engine.clients.graph import Facet
from search.engine.conjoin import _build_part
from search.engine.extract import DateRequest
from search.engine.quantity_read import QuantityOutput
from search.engine.region import candidate_grounding as candidate_grounding_module
from search.engine.region import candidates as candidates_module
from search.engine.region import materialise_step as materialise_step_module
from search.engine.region import phase_answer as phase_answer_module
from search.engine.region.codes import NO_MEMBER_HAS_DATA_AT_PLACE
from search.engine.region.pipeline import resolve_variable
from search.engine.region.state import EngineOptions
from search.engine.retrieval.retrieve import Materialised
from search.engine.source_scope import SourceScope
from search.models import (
    ObservationPoint,
    ObservationSeries,
    SeriesStatVarRef,
    SpecObservations,
    TimeWindow,
)
from series_coverage.schema import FacetMetadata, RoleNames
from series_coverage.shared import CoverageSnapshot
from tests.engine._fake_coverage import (
    FakeCoverageService,
    build_coverage_index,
    empty_coverage_snapshot,
    patch_coverage,
)
from tests.engine._fake_sdmx import FakeSdmxClient
from tests.engine._harness import (
    ObsFacetFixture,
    _csv_header,
    _csv_row,
    build_materialise_double,
    build_materialise_double_union_candidates,
    door_roles_for_test_query,
    sdmx_double,
)
from tests.engine._quantity import NO_QUANTITY
from tests.fixtures import FakeGraph, make_resolver
from tests.materialise.conftest import FakeGraphReader

_SV_DCID = "TESTFIXTURE/RoleBearingPipelineE2E/Flow"
_UNPLACEABLE_SV_DCID = "TESTFIXTURE/RoleBearingPipelineE2E/UnplaceableFlow"
_ALT_SV_DCID = "TESTFIXTURE/RoleBearingPipelineE2E/FlowAlt"
_NOR = "country/NOR"
_KEN = "country/KEN"
_USA = "country/USA"
_TWN = "country/TWN"
_MHL = "country/MHL"

# Coverage rows use donor/recipient columns. RoleNames stores each fixture's declared names.
_CREDITOR_DEBTOR_ROLE_NAMES = RoleNames(donor="creditor", recipient="debtor")
_PROVIDER_RECIPIENT_ROLE_NAMES = RoleNames(donor="provider", recipient="recipient")


def _years(n: int, *, start: int = 2018) -> list[int]:
    """n distinct years for a coverage row, starting at `start`."""
    return list(range(start, start + n))


def _facet(
    *,
    obs_count: int = 3,
    facet_id: str = "f1",
    provenance_id: str | None = None,
    unit: str | None = None,
) -> Facet:
    """Build a summary facet with one 2020 point and an indexable facet_id."""
    return Facet(
        earliest_date="2015",
        latest_date="2020",
        obs_count=obs_count,
        dates=["2020"],
        values=[0.0],
        facet_id=facet_id,
        provenance_id=provenance_id,
        unit=unit,
    )


def _dated_facet(dates: list[str], *, facet_id: str = "f1") -> Facet:
    """Build a facet with dated points for observation-series counts."""
    return Facet(
        earliest_date=dates[0],
        latest_date=dates[-1],
        obs_count=len(dates),
        dates=dates,
        values=[float(i) for i in range(len(dates))],
        facet_id=facet_id,
    )


def _materialise_points(dates: list[str]) -> tuple[tuple[str, float], ...]:
    """Convert dated facet points to the materialisation fixture format."""
    return tuple((d, float(i)) for i, d in enumerate(dates))


def _declaring_arcs(*, observation_properties: list[str]) -> dict:
    return {
        "populationType": {"nodes": [{"dcid": "EconomicActivity"}]},
        "measuredProperty": {"nodes": [{"dcid": "amount"}]},
        "statType": {"nodes": [{"dcid": "measuredValue"}]},
        "observationProperties": {"nodes": [{"dcid": p} for p in observation_properties]},
    }


class _FixedBindLLM:
    """Elects shape index 0 unconditionally, with no constraint bindings."""

    async def generate_structured(self, *, prompt, system, schema):
        if schema is QuantityOutput:
            return NO_QUANTITY
        assert schema.__name__ == "_UniversalBindOutput"
        return _UniversalBindOutput(chosen_shape_index=0, bindings=[]), None


class _CandidatesBindLLM:
    """Elects the first shape while returning the second as plausible."""

    async def generate_structured(self, *, prompt, system, schema):
        if schema is QuantityOutput:
            return NO_QUANTITY
        assert schema.__name__ == "_UniversalBindOutput"
        return _UniversalBindOutput(chosen_shape_index=0, bindings=[], also_plausible=[1]), None


def _run(
    *,
    entities: list[str],
    sv_dcid: str,
    sv_arcs: dict,
    resolve: dict[str, str],
    sv_score: float,
    role_query: str,
    sdmx: FakeSdmxClient,
    include_observations: bool = False,
    materialise_graph=None,
    materialise_sdmx=None,
    provenance_labels: dict[str, str] | None = None,
    unit_text: str | None = None,
    date_request: DateRequest | None = None,
    coverage: CoverageSnapshot | None = None,
    source_scope: SourceScope | None = None,
):
    graph = FakeGraph(
        nodes={
            **{dcid: {"label": name, "type": "Country"} for name, dcid in resolve.items()},
            sv_dcid: {"label": "Test flow", "arcs": sv_arcs},
            **{dcid: {"label": label} for dcid, label in (provenance_labels or {}).items()},
        },
        detect={role_query: {"svs": [sv_dcid], "cosine_scores": [sv_score]}},
        resolve=resolve,
        obs={},
    )
    return asyncio.run(
        resolve_variable(
            "aid",
            entities=entities,
            date_request=date_request,
            detect_query=role_query,
            role_query=role_query,
            entity_roles=door_roles_for_test_query(role_query, entities),
            pac=True,
            graph=graph,
            resolver=make_resolver(graph),
            llm=_FixedBindLLM(),
            sdmx=sdmx,
            base_timing={},
            unit_text=unit_text,
            source_scope=source_scope,
            coverage=coverage if coverage is not None else empty_coverage_snapshot(),
            options=EngineOptions(
                include_observations=include_observations,
                materialise_graph=materialise_graph,
                materialise_sdmx=materialise_sdmx,
            ),
        )
    )


def test_destination_wording_resolves_on_the_destination_axis(monkeypatch):
    """A lone place named with "to" binds the destination role.

    The scope-default hook stays inert for a role-bearing shape.
    The coverage-backed confirmation makes no SDMX call."""
    captured: dict = {}
    real_materialise = materialise_step_module.materialise

    def _spy(*args, **kwargs):
        captured["place_dcids"] = kwargs.get("place_dcids")
        return real_materialise(*args, **kwargs)

    monkeypatch.setattr(materialise_step_module, "materialise", _spy)

    _auto_coverage_index_1 = patch_coverage(
        monkeypatch,
        [(_SV_DCID, "f1", "country/FRA", _KEN, _years(3))],
        role_names={_SV_DCID: _CREDITOR_DEBTOR_ROLE_NAMES},
    )
    sdmx = FakeSdmxClient(observations={})
    result = _run(
        entities=["Kenya"],
        sv_dcid=_SV_DCID,
        sv_arcs=_declaring_arcs(observation_properties=["creditor", "debtor"]),
        resolve={"Kenya": _KEN},
        sv_score=0.9,
        role_query="aid to Kenya",
        sdmx=sdmx,
        coverage=FakeCoverageService(_auto_coverage_index_1).snapshot(),
    )
    assert result.status == "definite", result
    assert sdmx.series_calls == []
    assert captured["place_dcids"] is None


def test_origin_wording_resolves_on_the_origin_axis(monkeypatch):
    """ "From Norway to Kenya" binds each place to its own role and excludes the self-flow row.

    The where slot carries Kenya, so resubmission preserves the destination."""
    _auto_coverage_index_1 = patch_coverage(
        monkeypatch,
        [
            (_SV_DCID, "f1", _NOR, _KEN, _years(3)),
            (_SV_DCID, "f2", _NOR, _NOR, _years(50, start=1973)),
        ],
        role_names={_SV_DCID: _CREDITOR_DEBTOR_ROLE_NAMES},
    )
    sdmx = FakeSdmxClient(observations={})
    result = _run(
        entities=["Norway", "Kenya"],
        sv_dcid=_SV_DCID,
        sv_arcs=_declaring_arcs(observation_properties=["creditor", "debtor"]),
        resolve={"Norway": _NOR, "Kenya": _KEN},
        sv_score=0.9,
        role_query="aid from Norway to Kenya",
        sdmx=sdmx,
        coverage=FakeCoverageService(_auto_coverage_index_1).snapshot(),
    )
    assert result.status == "definite", result
    assert sdmx.series_calls == []
    where_slot = next(slot for slot in result.specs[0].resolved.slots if slot.key.axis == "where")
    assert where_slot.binding.value.ref.ref == _KEN
    assert result.specs[0].definition.identity.entities == {
        "origin": [_NOR],
        "destination": [_KEN],
    }
    coverage = result.specs[0].resolved.coverage
    assert coverage.kind == "exact"
    assert coverage.observation_count == 3, (
        "the Norway-to-Norway row's obs_count leaked into the answer"
    )


def test_bound_unit_can_confirm_only_the_second_provider(monkeypatch):
    """A plural provider filter keeps the second provider's USD facet.

    The first-scalar path would report facet_not_available. The read plan confirms the unit,
    so an answer without a preview reads no observations to decide it.
    """
    _auto_coverage_index_1 = patch_coverage(
        monkeypatch,
        [
            (_SV_DCID, "f1", _NOR, _KEN, _years(3)),
            (_SV_DCID, "f2", _USA, _KEN, _years(3)),
        ],
        role_names={_SV_DCID: _CREDITOR_DEBTOR_ROLE_NAMES},
        # f2 sorts first and carries USD. The matching facet belongs to provider two.
        facet_metadata={
            "f1": FacetMetadata(provenance_id="dc/base/RBSrcOther", unit="CurrencyEUR"),
            "f2": FacetMetadata(provenance_id="dc/base/RBSrc", unit="CurrencyUSD"),
        },
    )
    sdmx = FakeSdmxClient(observations={})
    captured: list[dict[str, list[str]]] = []

    def _unit_match(*_args, **kwargs):
        captured.append(_args[2])
        return (
            SpecObservations(
                series=[
                    ObservationSeries(
                        stat_var=SeriesStatVarRef(ref=_SV_DCID),
                        entities=[],
                        points=[ObservationPoint(date="2020", value=0.0)],
                        total_obs=1,
                    )
                ],
                total_obs=1,
                truncated=False,
            ),
            None,
        )

    monkeypatch.setattr(phase_answer_module, "_build_observations", _unit_match)
    # Conversion uses a separate materialise client.
    # This test checks the entity list in the fetch.
    materialise_body = _csv_header(["creditor", "debtor"]) + _csv_row(
        stat_var=_SV_DCID,
        role_dcids=[_USA, _KEN],
        fixture=ObsFacetFixture(
            facet_id="f1",
            points=(("2020", 1.0),),
            provenance_id="dc/base/RBSrc",
            unit="CurrencyUSD",
        ),
        date_str="2020",
        value=1.0,
    )
    materialise_graph = FakeGraphReader(
        arcs={
            _SV_DCID: {"measurementQualifier": ("Nominal",)},
            "CurrencyUSD": {"typeOf": ("CurrencyUnit",)},
        }
    )
    materialise_sdmx = sdmx_double(csv_body=materialise_body)
    result = _run(
        entities=["Norway", "USA", "Kenya"],
        sv_dcid=_SV_DCID,
        sv_arcs=_declaring_arcs(observation_properties=["creditor", "debtor"]),
        resolve={"Norway": _NOR, "USA": _USA, "Kenya": _KEN},
        sv_score=0.9,
        role_query="aid from Norway and USA to Kenya",
        sdmx=sdmx,
        unit_text="USD",
        date_request=DateRequest(window=None, latest=True),
        materialise_graph=materialise_graph,
        materialise_sdmx=materialise_sdmx,
        coverage=FakeCoverageService(_auto_coverage_index_1).snapshot(),
    )

    assert result.status == "definite", result
    assert sdmx.series_calls == []
    assert captured == []


def test_candidate_filtering_keeps_every_named_provider(monkeypatch):
    """Both candidate shapes depend on the second provider. Each filter keeps both origins."""
    captured: list[list[str] | None] = []
    real_observation_entities = candidate_grounding_module._observation_entities

    def _record_observation_entities(*args, **kwargs):
        captured.append(kwargs.get("origin_dcids"))
        return real_observation_entities(*args, **kwargs)

    monkeypatch.setattr(
        candidate_grounding_module, "_observation_entities", _record_observation_entities
    )
    alt_arcs = _declaring_arcs(observation_properties=["creditor", "debtor"])
    alt_arcs["measuredProperty"] = {"nodes": [{"dcid": "amountAlt"}]}
    graph = FakeGraph(
        nodes={
            _NOR: {"label": "Norway", "type": "Country"},
            _USA: {"label": "USA", "type": "Country"},
            _KEN: {"label": "Kenya", "type": "Country"},
            _SV_DCID: {
                "label": "Test flow",
                "arcs": _declaring_arcs(observation_properties=["creditor", "debtor"]),
            },
            _ALT_SV_DCID: {"label": "Alternate flow", "arcs": alt_arcs},
        },
        detect={
            "aid from Norway and USA to Kenya": {
                "svs": [_SV_DCID, _ALT_SV_DCID],
                "cosine_scores": [0.9, 0.85],
            }
        },
        resolve={"Norway": _NOR, "USA": _USA, "Kenya": _KEN},
        obs={},
    )
    _auto_coverage_index_1 = patch_coverage(
        monkeypatch,
        [
            (_SV_DCID, "f1", _USA, _KEN, _years(3)),
            (_ALT_SV_DCID, "f1", _USA, _KEN, _years(3)),
        ],
        role_names={
            _SV_DCID: _CREDITOR_DEBTOR_ROLE_NAMES,
            _ALT_SV_DCID: _CREDITOR_DEBTOR_ROLE_NAMES,
        },
    )
    sdmx = FakeSdmxClient(observations={})

    result = asyncio.run(
        resolve_variable(
            "aid",
            entities=["Norway", "USA", "Kenya"],
            date_request=None,
            detect_query="aid from Norway and USA to Kenya",
            role_query="aid from Norway and USA to Kenya",
            entity_roles=door_roles_for_test_query(
                "aid from Norway and USA to Kenya", ["Norway", "USA", "Kenya"]
            ),
            pac=True,
            graph=graph,
            resolver=make_resolver(graph),
            llm=_CandidatesBindLLM(),
            sdmx=sdmx,
            base_timing={},
            coverage=FakeCoverageService(_auto_coverage_index_1).snapshot(),
        )
    )

    assert result.status == "candidates", result
    assert len(result.specs) == 2
    assert captured == [[_NOR, _USA], [_NOR, _USA]]


def test_self_flow_preserves_both_directional_roles(monkeypatch):
    """One resolved DCID can occupy both ends of a fresh role-bearing flow."""
    _auto_coverage_index_1 = patch_coverage(
        monkeypatch,
        [(_SV_DCID, "f1", _NOR, _NOR, _years(3))],
        role_names={_SV_DCID: _CREDITOR_DEBTOR_ROLE_NAMES},
    )
    sdmx = FakeSdmxClient(observations={})
    result = _run(
        entities=["Norway", "Norwegian recipient"],
        sv_dcid=_SV_DCID,
        sv_arcs=_declaring_arcs(observation_properties=["creditor", "debtor"]),
        resolve={"Norway": _NOR, "Norwegian recipient": _NOR},
        sv_score=0.9,
        role_query="aid from Norway to Norwegian recipient",
        sdmx=sdmx,
        coverage=FakeCoverageService(_auto_coverage_index_1).snapshot(),
    )
    assert result.status == "definite", result
    spec = result.specs[0]
    assert {(entity.ref, entity.role) for entity in spec.resolved.entities} == {
        (_NOR, "origin"),
        (_NOR, "destination"),
    }
    assert spec.definition.identity.entities == {"origin": [_NOR], "destination": [_NOR]}


def test_origin_wording_observation_series_excludes_the_origin_self_flow(monkeypatch):
    """The series excludes the Norway-to-Norway row.

    Observation fetching builds a separate role map.
    """
    _auto_coverage_index_1 = patch_coverage(
        monkeypatch,
        [
            (_SV_DCID, "f1", _NOR, _KEN, [2018, 2019, 2020]),
            (_SV_DCID, "f2", _NOR, _NOR, [2010, 2011, 2012, 2013, 2014]),
        ],
        role_names={_SV_DCID: _PROVIDER_RECIPIENT_ROLE_NAMES},
        # Facet provenance must match the materialisation fixture for selection.
        facet_metadata={
            "f1": FacetMetadata(provenance_id="dc/base/RoleBearingE2E"),
            "f2": FacetMetadata(provenance_id="dc/base/RoleBearingE2EPoisoned"),
        },
    )
    sdmx = FakeSdmxClient(observations={})
    materialise_graph, materialise_sdmx, _requests = build_materialise_double_union_candidates(
        {
            (_SV_DCID, (("provider", _NOR), ("recipient", _KEN))): [
                ObsFacetFixture(
                    facet_id="f1",
                    points=_materialise_points(["2018", "2019", "2020"]),
                    provenance_id="dc/base/RoleBearingE2E",
                )
            ],
            (_SV_DCID, (("provider", _NOR), ("recipient", _NOR))): [
                ObsFacetFixture(
                    facet_id="f2",
                    points=_materialise_points(["2010", "2011", "2012", "2013", "2014"]),
                    provenance_id="dc/base/RoleBearingE2EPoisoned",
                )
            ],
        }
    )
    result = _run(
        entities=["Norway", "Kenya"],
        sv_dcid=_SV_DCID,
        sv_arcs=_declaring_arcs(observation_properties=["provider", "recipient"]),
        resolve={"Norway": _NOR, "Kenya": _KEN},
        sv_score=0.9,
        role_query="aid from Norway to Kenya",
        sdmx=sdmx,
        include_observations=True,
        materialise_graph=materialise_graph,
        materialise_sdmx=materialise_sdmx,
        coverage=FakeCoverageService(_auto_coverage_index_1).snapshot(),
    )
    assert result.status == "definite", result
    observations = result.specs[0].resolved.observations
    assert observations is not None
    assert observations.total_obs == 3, (
        "the Norway-to-Norway row's obs_count leaked into the returned series"
    )
    assert len(observations.series) == 1
    series_entities = {(e.role, e.ref) for e in observations.series[0].entities}
    assert series_entities == {("origin", _NOR), ("destination", _KEN)}


def test_no_directional_wording_refuses_role_not_resolved():
    """A lone place without directional wording returns role_not_resolved.

    No observation read runs.
    """
    sdmx = FakeSdmxClient(
        observations={_SV_DCID: {(("debtor", _KEN),): [_facet()]}},
    )
    result = _run(
        entities=["Kenya"],
        sv_dcid=_SV_DCID,
        sv_arcs=_declaring_arcs(observation_properties=["creditor", "debtor"]),
        resolve={"Kenya": _KEN},
        sv_score=0.9,
        role_query="aid Kenya",
        sdmx=sdmx,
        coverage=empty_coverage_snapshot(),
    )
    assert result.status == "no_data", result
    assert result.no_data_reason == "role_not_resolved"
    assert result.unresolved_entity_names == ("Kenya",)
    assert sdmx.series_calls == []


def test_unplaceable_declared_role_refuses_role_incompatible_before_bind():
    """A shape whose one role is on no flow axis cannot place Kenya as the recipient "aid
    to Kenya" asks for, so no reading is admissible: the part refuses role_incompatible
    without a bind call and flags nothing, on the region result and on the wire part."""
    sdmx = FakeSdmxClient(observations={})
    result = _run(
        entities=["Kenya"],
        sv_dcid=_UNPLACEABLE_SV_DCID,
        sv_arcs=_declaring_arcs(observation_properties=["contributor"]),
        resolve={"Kenya": _KEN},
        sv_score=0.9,
        role_query="aid to Kenya",
        sdmx=sdmx,
        coverage=empty_coverage_snapshot(),
    )
    assert result.status == "no_data"
    assert result.no_data_reason == "role_incompatible"
    assert not result.flagged_match
    assert "bind" not in result.timing_by_step

    part = _build_part(
        result,
        representative=result,
        folded=False,
        max_candidates=5,
        include_observations=False,
        include_skills=False,
        skills_index=None,
        envelope_warnings=[],
    )
    assert part.status == "no_data"
    assert part.no_data.reason == "role_incompatible"
    assert part.no_data.flagged_match is None


def test_unresolved_second_entity_refuses_role_not_resolved_instead_of_guessing():
    """Taiwan has no role preposition while the Marshall Islands has "to".

    Return role_not_resolved for this ambiguity. role_incompatible would misstate the schema."""
    sdmx = FakeSdmxClient(observations={})
    result = _run(
        entities=["Taiwan", "Marshall Islands"],
        sv_dcid=_SV_DCID,
        sv_arcs=_declaring_arcs(observation_properties=["creditor", "debtor"]),
        resolve={"Taiwan": _TWN, "Marshall Islands": _MHL},
        sv_score=0.9,
        role_query="Taiwan ODA to the Marshall Islands",
        sdmx=sdmx,
        coverage=empty_coverage_snapshot(),
    )
    assert result.status == "no_data", result
    assert result.no_data_reason == "role_not_resolved"
    # crs-010: the refusal names the place whose direction the query does not tell.
    assert result.unresolved_entity_names == ("Taiwan",)
    assert len(sdmx.series_calls) == 0


def test_role_bearing_shape_the_index_does_not_carry_is_held_to_the_unknown_score_floor(
    monkeypatch,
):
    """A role-bearing shape whose members the code index does not hold has only its recall
    score to claim the question with, so a score of 0.5 falls under the 0.65 floor and the
    variable is not resolved. The exemption belongs to shapes the index carries."""
    _auto_coverage_index_1 = patch_coverage(
        monkeypatch,
        [(_SV_DCID, "f1", "country/FRA", _KEN, _years(3))],
        role_names={_SV_DCID: _CREDITOR_DEBTOR_ROLE_NAMES},
    )
    sdmx = FakeSdmxClient(observations={})
    result = _run(
        entities=["Kenya"],
        sv_dcid=_SV_DCID,
        sv_arcs=_declaring_arcs(observation_properties=["creditor", "debtor"]),
        resolve={"Kenya": _KEN},
        sv_score=0.5,
        role_query="aid to Kenya",
        sdmx=sdmx,
        coverage=FakeCoverageService(_auto_coverage_index_1).snapshot(),
    )
    assert result.status == "no_data", result
    assert result.no_data_reason == "variable_not_resolved"


def test_role_axis_placement_binds_every_named_place(monkeypatch):
    """Ten origins and one destination all bind on their axes. No place is dropped, so the
    destination is read and the answer has its data."""
    donor_names = [f"Donor{i}" for i in range(10)]
    donor_dcids = [f"country/D{i}" for i in range(10)]
    resolve = dict(zip(donor_names, donor_dcids, strict=True))
    resolve["Kenya"] = _KEN
    entities = [*donor_names, "Kenya"]
    role_query = "aid " + " ".join(f"from {name}" for name in donor_names) + " to Kenya"

    index = patch_coverage(
        monkeypatch,
        [(_SV_DCID, "f1", donor_dcids[0], _KEN, _years(3))],
        role_names={_SV_DCID: _CREDITOR_DEBTOR_ROLE_NAMES},
    )
    result = _run(
        entities=entities,
        sv_dcid=_SV_DCID,
        sv_arcs=_declaring_arcs(observation_properties=["creditor", "debtor"]),
        resolve=resolve,
        sv_score=0.9,
        role_query=role_query,
        sdmx=FakeSdmxClient(observations={}),
        coverage=FakeCoverageService(index).snapshot(),
    )
    assert result.status == "definite", result
    # The fixture runs no member walk, so no warning concerns the places.
    assert [w.code for w in result.warnings] == ["membership_unverified"]
    assert sorted((e.ref, e.role) for e in result.specs[0].resolved.entities) == sorted(
        [(_KEN, "destination"), *((dcid, "origin") for dcid in donor_dcids)]
    )


_RELAX_PROP = "TestRelaxPurpose"
_RELAX_VAL_MATCHED = "PurposeMatched"
_RELAX_VAL_OTHER = "PurposeOther"
_RELAX_NODATA_SV = "TESTFIXTURE/RoleBearingPipelineE2E/RelaxNoData"
_RELAX_HASDATA_SV = "TESTFIXTURE/RoleBearingPipelineE2E/RelaxHasData"


def _relax_arcs(*, purpose_value: str) -> dict:
    return {
        "populationType": {"nodes": [{"dcid": "EconomicActivity"}]},
        "measuredProperty": {"nodes": [{"dcid": "amount"}]},
        "statType": {"nodes": [{"dcid": "measuredValue"}]},
        "observationProperties": {"nodes": [{"dcid": "provider"}, {"dcid": "recipient"}]},
        "constraintProperties": {"nodes": [{"dcid": _RELAX_PROP}]},
        _RELAX_PROP: {"nodes": [{"dcid": purpose_value}]},
    }


class _BindLLMWithConstraint:
    """Elects shape index 0, drafting one bound what-slot constraint that
    matches only the no-data member. The primary resolve confirms a
    real StatVar with no observations, exactly the no_observations state
    _suggest_nearest_real's relaxation runs on."""

    async def generate_structured(self, *, prompt, system, schema):
        if schema is QuantityOutput:
            return NO_QUANTITY
        assert schema.__name__ == "_UniversalBindOutput"
        binding = SlotBindingDraft(
            axis="what",
            property_dcid=_RELAX_PROP,
            kind="value",
            value_dcids=[_RELAX_VAL_MATCHED],
        )
        return _UniversalBindOutput(chosen_shape_index=0, bindings=[binding]), None


def _relaxed_usa_to_kenya(monkeypatch, date_request: DateRequest | None):
    """Resolve "aid from USA to Kenya", whose bound purpose has no data, so the answer
    carries a nearest-real suggestion relaxing it. A USA-to-USA row poisons a misbound
    destination."""
    role_query = "aid from USA to Kenya"
    graph = FakeGraph(
        nodes={
            _USA: {"label": "USA", "type": "Country"},
            _KEN: {"label": "Kenya", "type": "Country"},
            _RELAX_NODATA_SV: {
                "label": "No data",
                "arcs": _relax_arcs(purpose_value=_RELAX_VAL_MATCHED),
            },
            _RELAX_HASDATA_SV: {
                "label": "Has data",
                "arcs": _relax_arcs(purpose_value=_RELAX_VAL_OTHER),
            },
        },
        detect={
            role_query: {
                "svs": [_RELAX_NODATA_SV, _RELAX_HASDATA_SV],
                "cosine_scores": [0.9, 0.9],
            }
        },
        resolve={"USA": _USA, "Kenya": _KEN},
        obs={},
    )
    _auto_coverage_index_1 = patch_coverage(
        monkeypatch,
        [
            (_RELAX_HASDATA_SV, "f1", _USA, _KEN, [2018, 2019, 2020]),
            (_RELAX_HASDATA_SV, "f2", _USA, _USA, [2010, 2011, 2012, 2013, 2014]),
        ],
        role_names={_RELAX_HASDATA_SV: _PROVIDER_RECIPIENT_ROLE_NAMES},
        # Facet provenance must match the materialisation fixture for selection.
        facet_metadata={
            "f1": FacetMetadata(provenance_id="dc/base/RelaxHasData"),
            "f2": FacetMetadata(provenance_id="dc/base/RelaxHasDataPoisoned"),
        },
    )
    sdmx = FakeSdmxClient(observations={})
    materialise_graph, materialise_sdmx, _requests = build_materialise_double_union_candidates(
        {
            (_RELAX_HASDATA_SV, (("provider", _USA), ("recipient", _KEN))): [
                ObsFacetFixture(
                    facet_id="f1",
                    points=_materialise_points(["2018", "2019", "2020"]),
                    provenance_id="dc/base/RelaxHasData",
                )
            ],
            (_RELAX_HASDATA_SV, (("provider", _USA), ("recipient", _USA))): [
                ObsFacetFixture(
                    facet_id="f2",
                    points=_materialise_points(["2010", "2011", "2012", "2013", "2014"]),
                    provenance_id="dc/base/RelaxHasDataPoisoned",
                )
            ],
        }
    )
    return asyncio.run(
        resolve_variable(
            "aid",
            entities=["USA", "Kenya"],
            date_request=date_request,
            detect_query=role_query,
            role_query=role_query,
            entity_roles=door_roles_for_test_query(role_query, ["USA", "Kenya"]),
            pac=True,
            graph=graph,
            resolver=make_resolver(graph),
            llm=_BindLLMWithConstraint(),
            sdmx=sdmx,
            base_timing={},
            coverage=FakeCoverageService(_auto_coverage_index_1).snapshot(),
            options=EngineOptions(
                include_observations=True,
                materialise_graph=materialise_graph,
                materialise_sdmx=materialise_sdmx,
            ),
        )
    )


def test_relaxation_binds_the_correct_destination_when_origin_named_first(monkeypatch):
    """Nearest-real coverage for "aid from USA to Kenya" must bind Kenya as destination.

    The higher-count USA-to-USA poison row tests coverage and series independently.
    """
    result = _relaxed_usa_to_kenya(monkeypatch, None)
    assert result.status == "no_data", result
    assert result.no_data_reason == "no_observations"
    assert result.nearest_real
    for nr in result.nearest_real:
        # Coverage and observations use separate reads, so check both role splits.
        assert nr.resolved.coverage.kind == "exact"
        assert nr.resolved.coverage.observation_count == 3, (
            "the USA-to-USA self-flow row confirmed the suggestion's coverage "
            "instead of the real USA-to-Kenya one"
        )
        assert nr.resolved.observations is not None
        assert nr.resolved.observations.total_obs == 3, (
            "the USA-to-USA self-flow row leaked into the returned series"
        )
        series_entities = {(e.role, e.ref) for e in nr.resolved.observations.series[0].entities}
        assert series_entities == {("origin", _USA), ("destination", _KEN)}


@pytest.mark.parametrize(
    ("date_request", "served"),
    [
        (DateRequest(window=None, latest=True), ["2020"]),
        (
            DateRequest(window=TimeWindow(start_year=2019, end_year=2020), latest=False),
            ["2019", "2020"],
        ),
        (
            DateRequest(window=TimeWindow(start_year=2019, end_year=None), latest=False),
            ["2019", "2020"],
        ),
        (
            DateRequest(window=TimeWindow(start_year=None, end_year=2019), latest=False),
            ["2018", "2019"],
        ),
        (None, ["2018", "2019", "2020"]),
    ],
    ids=["latest", "closed", "since", "until", "dateless"],
)
def test_a_nearest_real_suggestion_counts_the_points_its_preview_serves(
    monkeypatch, date_request, served
):
    """The suggestion's preview reads the plan its relaxation built: coverage counts and
    dates exactly the points it serves under each kind of request."""
    result = _relaxed_usa_to_kenya(monkeypatch, date_request)
    assert result.nearest_real
    for nr in result.nearest_real:
        observations = nr.resolved.observations
        assert observations is not None
        assert [p.date for s in observations.series for p in s.points] == served
        assert nr.resolved.coverage.observation_count == len(served)
        if date_request is not None:
            assert nr.resolved.coverage.window == TimeWindow(
                start_year=int(served[0]), end_year=int(served[-1])
            )


def test_resolved_entities_carry_their_directional_role(monkeypatch):
    """Resolved entities use origin/destination labels.

    This holds when the shape declares creditor/debtor."""
    _auto_coverage_index_1 = patch_coverage(
        monkeypatch,
        [(_SV_DCID, "f1", _NOR, _KEN, _years(3))],
        role_names={_SV_DCID: _CREDITOR_DEBTOR_ROLE_NAMES},
    )
    sdmx = FakeSdmxClient(observations={})
    result = _run(
        entities=["Norway", "Kenya"],
        sv_dcid=_SV_DCID,
        sv_arcs=_declaring_arcs(observation_properties=["creditor", "debtor"]),
        resolve={"Norway": _NOR, "Kenya": _KEN},
        sv_score=0.9,
        role_query="aid from Norway to Kenya",
        sdmx=sdmx,
        coverage=FakeCoverageService(_auto_coverage_index_1).snapshot(),
    )
    assert result.status == "definite", result
    roles_by_ref = {e.ref: e.role for e in result.specs[0].resolved.entities}
    assert roles_by_ref == {_NOR: "origin", _KEN: "destination"}


def test_declared_role_outside_the_wire_vocabulary_still_names_the_entity_via_its_axis(
    monkeypatch,
):
    """The series uses the destination wire role for a shape's debtor entity, and the origin
    wire role for the creditor the open origin's series binds.

    EntityRoleKey exposes observationAbout, origin, and destination."""
    _auto_coverage_index_1 = patch_coverage(
        monkeypatch,
        [(_SV_DCID, "f1", "country/FRA", _KEN, [2018, 2019, 2020])],
        role_names={_SV_DCID: _CREDITOR_DEBTOR_ROLE_NAMES},
        # Match facet provenance to the materialisation fixture for selection.
        facet_metadata={"f1": FacetMetadata(provenance_id="dc/base/RoleBearingE2EDebtor")},
    )
    sdmx = FakeSdmxClient(observations={})
    # The open origin reads one series per creditor, binding both roles.
    materialise_graph, materialise_sdmx, _requests = build_materialise_double(
        {
            (_SV_DCID, (("creditor", "country/FRA"), ("debtor", _KEN))): [
                ObsFacetFixture(
                    facet_id="f1",
                    points=_materialise_points(["2018", "2019", "2020"]),
                    provenance_id="dc/base/RoleBearingE2EDebtor",
                )
            ],
        },
        declared_roles={_SV_DCID: ("creditor", "debtor")},
    )
    result = _run(
        entities=["Kenya"],
        sv_dcid=_SV_DCID,
        sv_arcs=_declaring_arcs(observation_properties=["creditor", "debtor"]),
        resolve={"Kenya": _KEN},
        sv_score=0.9,
        role_query="aid to Kenya",
        sdmx=sdmx,
        include_observations=True,
        materialise_graph=materialise_graph,
        materialise_sdmx=materialise_sdmx,
        coverage=FakeCoverageService(_auto_coverage_index_1).snapshot(),
    )
    assert result.status == "definite", result
    observations = result.specs[0].resolved.observations
    assert observations is not None
    assert observations.total_obs == 3
    assert len(observations.series) == 1
    series_entities = {(e.role, e.ref) for e in observations.series[0].entities}
    assert series_entities == {("destination", _KEN), ("origin", "country/FRA")}


def test_origin_only_data_never_confirms_as_a_definite_destination_answer(monkeypatch):
    """Norway has 8 observations for other destinations but none for Kenya.

    Coverage binds both roles, so the request returns no_observations.
    """
    origin_only_rows = [
        (_SV_DCID, "f1", _NOR, "country/OTHER_A", [2018, 2019, 2020, 2021, 2022]),
        (_SV_DCID, "f2", _NOR, "country/OTHER_B", [2015, 2016, 2017]),
    ]

    # With the recipient unbound, Norway has 8 observations.
    origin_only_index = build_coverage_index(
        origin_only_rows, role_names={_SV_DCID: _CREDITOR_DEBTOR_ROLE_NAMES}
    )
    origin_only_candidates = origin_only_index.filter_candidates(
        variables=[_SV_DCID], bound={"donor": frozenset({_NOR})}
    )
    assert sum(c.n_obs for c in origin_only_candidates) == 8

    _auto_coverage_index_1 = patch_coverage(
        monkeypatch, origin_only_rows, role_names={_SV_DCID: _CREDITOR_DEBTOR_ROLE_NAMES}
    )
    sdmx = FakeSdmxClient(observations={})
    result = _run(
        entities=["Norway", "Kenya"],
        sv_dcid=_SV_DCID,
        sv_arcs=_declaring_arcs(observation_properties=["creditor", "debtor"]),
        resolve={"Norway": _NOR, "Kenya": _KEN},
        sv_score=0.9,
        role_query="aid from Norway to Kenya",
        sdmx=sdmx,
        include_observations=True,
        coverage=FakeCoverageService(_auto_coverage_index_1).snapshot(),
    )
    assert result.status == "no_data", result
    assert result.no_data_reason == "no_observations"
    assert not result.nearest_real
    # Every nearest-real probe keeps the recipient bound through coverage.
    assert sdmx.series_calls == []


# Cite and narrow sources from discovered role indexes.


def test_a_named_source_removes_a_broader_role_coverage_series(monkeypatch):
    """Role-bearing coverage admits source B before ranking source A's larger series."""
    _auto_coverage_index_1 = patch_coverage(
        monkeypatch,
        [
            (_SV_DCID, "f-source-a", _NOR, _KEN, _years(8)),
            (_SV_DCID, "f-source-b", _USA, _KEN, _years(1)),
        ],
        role_names={_SV_DCID: _CREDITOR_DEBTOR_ROLE_NAMES},
        facet_metadata={
            "f-source-a": FacetMetadata(provenance_id="source/A"),
            "f-source-b": FacetMetadata(provenance_id="source/B"),
        },
    )

    result = _run(
        entities=["Kenya"],
        sv_dcid=_SV_DCID,
        sv_arcs=_declaring_arcs(observation_properties=["creditor", "debtor"]),
        resolve={"Kenya": _KEN, "Norway": _NOR, "USA": _USA},
        sv_score=0.9,
        role_query="aid to Kenya",
        sdmx=FakeSdmxClient(observations={}),
        provenance_labels={"source/A": "Source A", "source/B": "Source B"},
        coverage=FakeCoverageService(_auto_coverage_index_1).snapshot(),
        source_scope=SourceScope(allowed=frozenset({"source/B"}), named=True),
    )

    assert result.status == "definite", result
    assert [ref.ref for ref in result.specs[0].resolved.trace.resolved_sources] == ["source/B"]
    assert result.specs[0].resolved.coverage.observation_count == 1


def test_discovered_scope_cites_the_one_source_its_variable_serves(monkeypatch):
    """Two discovered recipients hold facets from two sources. The variable serves one group
    across them, so resolved_sources cites that source alone and the scope lists the
    recipient it covers."""
    _auto_coverage_index_1 = patch_coverage(
        monkeypatch,
        [
            (_SV_DCID, "fA", "TESTFIXTURE/RoleBearingPipelineE2E/UnboundDonor", _KEN, [2020]),
            (_SV_DCID, "fB", "TESTFIXTURE/RoleBearingPipelineE2E/UnboundDonor", _NOR, [2020]),
        ],
        role_names={_SV_DCID: _CREDITOR_DEBTOR_ROLE_NAMES},
        facet_metadata={
            "fA": FacetMetadata(provenance_id="source/A"),
            "fB": FacetMetadata(provenance_id="source/B"),
        },
    )
    sdmx = FakeSdmxClient(observations={}, availability={})

    result = _run(
        entities=[],
        sv_dcid=_SV_DCID,
        sv_arcs=_declaring_arcs(observation_properties=["creditor", "debtor"]),
        resolve={"Kenya": _KEN, "Norway": _NOR},
        sv_score=0.9,
        role_query="aid disbursed",
        sdmx=sdmx,
        provenance_labels={"source/A": "Source A", "source/B": "Source B"},
        coverage=FakeCoverageService(_auto_coverage_index_1).snapshot(),
    )

    assert result.status == "definite", result
    spec = result.specs[0]
    assert [ref.ref for ref in spec.resolved.trace.resolved_sources] == ["source/A"]
    assert spec.resolved.scope is not None
    assert [member.ref for member in spec.resolved.scope.members] == [_KEN]


def test_narrowing_a_discovered_scope_drops_sources_and_citations_together(
    monkeypatch, search_settings
):
    """A USD binding removes source/B/EUR from the role index and citations.

    At pair cap 1, the preview reads points separately from the coverage index.
    """
    search_settings(obs_preview_max_pairs=1)
    _donor = "TESTFIXTURE/RoleBearingPipelineE2E/UnboundDonor"
    _auto_coverage_index_1 = patch_coverage(
        monkeypatch,
        [
            (_SV_DCID, "fA", _donor, _KEN, [2020]),
            (_SV_DCID, "fB", _donor, _NOR, [2020]),
        ],
        role_names={_SV_DCID: _CREDITOR_DEBTOR_ROLE_NAMES},
        facet_metadata={
            "fA": FacetMetadata(provenance_id="source/A", unit="CurrencyUSD"),
            "fB": FacetMetadata(provenance_id="source/B", unit="CurrencyEUR"),
        },
    )
    observations = {
        _SV_DCID: {
            (("creditor", _donor), ("debtor", _KEN)): [
                _facet(facet_id="fA", provenance_id="source/A", unit="CurrencyUSD")
            ],
            (("creditor", _donor), ("debtor", _NOR)): [
                _facet(facet_id="fB", provenance_id="source/B", unit="CurrencyEUR")
            ],
        }
    }
    sdmx = FakeSdmxClient(observations=observations, availability={})

    # Conversion uses a separate materialise client with a fixed native-USD response.
    # This test measures source narrowing.
    materialise_body = _csv_header(["debtor"]) + _csv_row(
        stat_var=_SV_DCID,
        role_dcids=[_KEN],
        fixture=ObsFacetFixture(
            facet_id="fA", points=(("2020", 1.0),), provenance_id="source/A", unit="CurrencyUSD"
        ),
        date_str="2020",
        value=1.0,
    )
    materialise_graph = FakeGraphReader(
        arcs={
            _SV_DCID: {"measurementQualifier": ("Nominal",)},
            "CurrencyUSD": {"typeOf": ("CurrencyUnit",)},
        }
    )
    materialise_sdmx = sdmx_double(csv_body=materialise_body)
    result = _run(
        entities=[],
        sv_dcid=_SV_DCID,
        sv_arcs=_declaring_arcs(observation_properties=["creditor", "debtor"]),
        resolve={"Kenya": _KEN, "Norway": _NOR},
        sv_score=0.9,
        role_query="aid disbursed",
        sdmx=sdmx,
        provenance_labels={"source/A": "Source A", "source/B": "Source B"},
        unit_text="USD",
        date_request=DateRequest(window=None, latest=True),
        materialise_graph=materialise_graph,
        materialise_sdmx=materialise_sdmx,
        coverage=FakeCoverageService(_auto_coverage_index_1).snapshot(),
    )

    assert result.status == "definite", result
    spec = result.specs[0]
    resolved_source_dcids = [r.ref for r in spec.resolved.trace.resolved_sources]
    assert resolved_source_dcids == ["source/A"]
    assert "source/B" not in resolved_source_dcids
    dims = {d.label: d.count for d in spec.resolved.coverage.dimensions}
    assert dims["sources"] == 1


def test_zero_observation_resolve_returns_an_editable_flagged_carrier(monkeypatch):
    """An empty bound pair returns an editable no-observations carrier.

    A stocked control confirms the lookup succeeds for the same shape.
    """
    entities = ["Norway", "Kenya"]
    arcs = _declaring_arcs(observation_properties=["creditor", "debtor"])

    _auto_coverage_index_1 = patch_coverage(
        monkeypatch,
        [(_SV_DCID, "f1", _NOR, _KEN, [2020])],
        role_names={_SV_DCID: _CREDITOR_DEBTOR_ROLE_NAMES},
    )
    control = _run(
        entities=entities,
        sv_dcid=_SV_DCID,
        sv_arcs=arcs,
        resolve={"Norway": _NOR, "Kenya": _KEN},
        sv_score=0.9,
        role_query="aid from Norway to Kenya",
        sdmx=FakeSdmxClient(observations={}),
        coverage=FakeCoverageService(_auto_coverage_index_1).snapshot(),
    )
    assert control.status == "definite", control.no_data_reason

    _auto_coverage_index_2 = patch_coverage(monkeypatch, [], role_names={})
    result = _run(
        entities=entities,
        sv_dcid=_SV_DCID,
        sv_arcs=arcs,
        resolve={"Norway": _NOR, "Kenya": _KEN},
        sv_score=0.9,
        role_query="aid from Norway to Kenya",
        sdmx=FakeSdmxClient(observations={}),
        coverage=FakeCoverageService(_auto_coverage_index_2).snapshot(),
    )

    assert result.status == "no_data", result
    assert result.no_data_reason == "no_observations"
    assert result.gate == NO_MEMBER_HAS_DATA_AT_PLACE
    assert result.flagged_match is not None
    assert len(result.flagged_match) == 1
    flagged = result.flagged_match[0]
    assert flagged.resolved.available_at_place is False
    assert [sv.ref.ref for sv in flagged.resolved.members] == [_SV_DCID]
    assert result.elected_stat_vars is not None
    assert [ref.ref for ref in result.elected_stat_vars] == [_SV_DCID]


def test_bound_destination_open_origin_warns_about_the_open_origin_axis(monkeypatch):
    """A bound destination with an open origin reports the two-provider scope."""
    role_query = "aid to Kenya"

    _auto_coverage_index_1 = patch_coverage(
        monkeypatch,
        [
            (_SV_DCID, "f1", "country/FRA", _KEN, _years(3)),
            (_SV_DCID, "f2", "country/DEU", _KEN, _years(3)),
        ],
        role_names={_SV_DCID: _CREDITOR_DEBTOR_ROLE_NAMES},
    )
    result = _run(
        entities=["Kenya"],
        sv_dcid=_SV_DCID,
        sv_arcs=_declaring_arcs(observation_properties=["creditor", "debtor"]),
        resolve={"Kenya": _KEN, "France": "country/FRA", "Germany": "country/DEU"},
        sv_score=0.9,
        role_query=role_query,
        sdmx=FakeSdmxClient(observations={}),
        coverage=FakeCoverageService(_auto_coverage_index_1).snapshot(),
    )
    assert result.status == "definite", result
    defaulted = [w for w in result.warnings if w.code == "role_scope_defaulted"]
    assert [w.message for w in defaulted] == [
        "no provider was named; the answer was resolved across every provider with data."
    ]
    facts = result.specs[0].resolved.role_axis_facts
    assert facts is not None
    assert [f.model_dump(mode="json") for f in facts] == [
        {"role": "origin", "member_count": 2, "members_available": True}
    ]


def _candidates_with_flagged_alt(
    monkeypatch,
    *,
    unverified: bool,
    open_origin: bool = False,
):
    """Run two candidate shapes and set flags on the alternate's materialisation."""
    role_query = "aid from Norway to Kenya"
    alt_arcs = _declaring_arcs(observation_properties=["creditor", "debtor"])
    alt_arcs["measuredProperty"] = {"nodes": [{"dcid": "amountAlt"}]}
    graph = FakeGraph(
        nodes={
            _NOR: {"label": "Norway", "type": "Country"},
            _KEN: {"label": "Kenya", "type": "Country"},
            _SV_DCID: {
                "label": "Test flow",
                "arcs": _declaring_arcs(observation_properties=["creditor", "debtor"]),
            },
            _ALT_SV_DCID: {"label": "Alternate flow", "arcs": alt_arcs},
        },
        detect={role_query: {"svs": [_SV_DCID, _ALT_SV_DCID], "cosine_scores": [0.9, 0.85]}},
        resolve={"Norway": _NOR, "Kenya": _KEN},
        obs={},
    )
    _auto_coverage_index_1 = patch_coverage(
        monkeypatch,
        [
            (_SV_DCID, "f1", _NOR, _KEN, _years(3)),
            (_ALT_SV_DCID, "f1", _NOR, _KEN, _years(3)),
        ],
        role_names={
            _SV_DCID: _CREDITOR_DEBTOR_ROLE_NAMES,
            _ALT_SV_DCID: _CREDITOR_DEBTOR_ROLE_NAMES,
        },
    )
    sdmx = FakeSdmxClient(observations={})

    real_materialise = candidates_module.materialise
    alt_shapes_seen: list[object] = []

    def _flagged(shape, bindings, entities, **kwargs):
        result = real_materialise(shape, bindings, entities, **kwargs)
        if (
            isinstance(result, Materialised)
            and shape.sv_arc_facts
            and (_ALT_SV_DCID in shape.sv_arc_facts)
        ):
            alt_shapes_seen.append(shape)
            return replace(
                result,
                role_declaration_unverified=unverified,
                open_origin_dcids=(frozenset({_NOR}) if open_origin else result.open_origin_dcids),
            )
        return result

    monkeypatch.setattr(candidates_module, "materialise", _flagged)
    result = asyncio.run(
        resolve_variable(
            "aid",
            entities=["Norway", "Kenya"],
            date_request=None,
            detect_query=role_query,
            role_query=role_query,
            entity_roles=door_roles_for_test_query(role_query, ["Norway", "Kenya"]),
            pac=True,
            graph=graph,
            resolver=make_resolver(graph),
            llm=_CandidatesBindLLM(),
            sdmx=sdmx,
            base_timing={},
            coverage=FakeCoverageService(_auto_coverage_index_1).snapshot(),
        )
    )
    assert alt_shapes_seen, "the alt shape never reached materialise in this fixture"
    return result


def test_an_alt_shape_with_unverified_roles_warns_on_a_candidates_part(monkeypatch):
    """An unverified alternate role declaration adds a warning to the candidates response.

    A verified control emits no such warning.
    """
    control = _candidates_with_flagged_alt(monkeypatch, unverified=False)
    assert control.status == "candidates", control
    assert [w.code for w in control.warnings if w.code == "role_declaration_unverified"] == []

    result = _candidates_with_flagged_alt(monkeypatch, unverified=True)
    assert result.status == "candidates", result
    assert [w.code for w in result.warnings if w.code == "role_declaration_unverified"] == [
        "role_declaration_unverified"
    ]


def test_an_alt_shape_with_an_open_origin_warns_on_candidates(monkeypatch):
    """#664. A bound-destination/open-origin alternate adds a candidates warning."""
    control = _candidates_with_flagged_alt(monkeypatch, unverified=False, open_origin=False)
    assert [w.code for w in control.warnings if w.code == "role_scope_defaulted"] == []

    result = _candidates_with_flagged_alt(monkeypatch, unverified=False, open_origin=True)
    defaulted = [w for w in result.warnings if w.code == "role_scope_defaulted"]
    assert [w.message for w in defaulted] == [
        "no provider was named; the answer was resolved across every provider with data."
    ]


def test_zero_observation_bound_destination_open_origin_still_warns(monkeypatch):
    """Open-origin warnings accompany zero-observation carriers and definite answers.

    The stocked control emits the same warning.
    """
    role_query = "aid to Kenya"
    arcs = _declaring_arcs(observation_properties=["creditor", "debtor"])

    _auto_coverage_index_1 = patch_coverage(
        monkeypatch,
        [(_SV_DCID, "f1", "country/FRA", _KEN, _years(3))],
        role_names={_SV_DCID: _CREDITOR_DEBTOR_ROLE_NAMES},
    )
    control = _run(
        entities=["Kenya"],
        sv_dcid=_SV_DCID,
        sv_arcs=arcs,
        resolve={"Kenya": _KEN, "France": "country/FRA"},
        sv_score=0.9,
        role_query=role_query,
        sdmx=FakeSdmxClient(observations={}),
        coverage=FakeCoverageService(_auto_coverage_index_1).snapshot(),
    )
    assert control.status == "definite", control
    assert [w.code for w in control.warnings if w.code == "role_scope_defaulted"] == [
        "role_scope_defaulted"
    ]

    _auto_coverage_index_2 = patch_coverage(monkeypatch, [], role_names={})
    result = _run(
        entities=["Kenya"],
        sv_dcid=_SV_DCID,
        sv_arcs=arcs,
        resolve={"Kenya": _KEN},
        sv_score=0.9,
        role_query=role_query,
        sdmx=FakeSdmxClient(observations={}),
        coverage=FakeCoverageService(_auto_coverage_index_2).snapshot(),
    )

    assert result.status == "no_data", result
    assert result.no_data_reason == "no_observations"
    assert result.gate == NO_MEMBER_HAS_DATA_AT_PLACE
    defaulted = [w for w in result.warnings if w.code == "role_scope_defaulted"]
    assert [w.message for w in defaulted] == [
        "no provider was named; the answer was resolved across every provider with data."
    ]


def test_both_axes_bound_emits_no_role_scope_defaulted_warning(monkeypatch):
    """Binding both axes emits no open-origin warning or role-axis fact."""
    role_query = "aid from Norway to Kenya"

    _auto_coverage_index_1 = patch_coverage(
        monkeypatch,
        [(_SV_DCID, "f1", _NOR, _KEN, _years(3))],
        role_names={_SV_DCID: _CREDITOR_DEBTOR_ROLE_NAMES},
    )
    result = _run(
        entities=["Norway", "Kenya"],
        sv_dcid=_SV_DCID,
        sv_arcs=_declaring_arcs(observation_properties=["creditor", "debtor"]),
        resolve={"Norway": _NOR, "Kenya": _KEN},
        sv_score=0.9,
        role_query=role_query,
        sdmx=FakeSdmxClient(observations={}),
        coverage=FakeCoverageService(_auto_coverage_index_1).snapshot(),
    )

    assert result.status == "definite", result
    assert [w for w in result.warnings if w.code == "role_scope_defaulted"] == []
    assert result.specs[0].resolved.role_axis_facts is None
