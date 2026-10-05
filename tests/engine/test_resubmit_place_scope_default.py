"""Entity-less resubmits use the same Earth/Country default scope as fresh
queries. Tests cover bulk observations, empty country coverage, and Earth-only
data; split ownership is covered by the no-entity gate tests.
"""

from __future__ import annotations

import dataclasses

import pytest

from code_index.shared import get_index
from datasets_api.adapter import resolve_definition_members
from search.engine.region import (
    NO_MEMBER_HAS_DATA_AT_PLACE,
    PLACE_SCOPE_DEFAULTED,
    resolve_spec_resubmit,
)
from search.engine.region.codes import MEMBERSHIP_UNVERIFIED
from search.engine.region.state import EngineOptions, identity_served_stat_vars
from search.engine.retrieval.bound_values import synthetic_shape_for_identity
from search.models import BaseIdentity, CoverageExact, IdentityShapeSchema, ResubmitIdentityPayload
from statvar_members import MemberEnumerator
from tests.code_index.aid_identity import (
    EDUCATION_AID_SPEC_ID,
    IMPUTED_PROVENANCE,
    education_aid_identity,
)
from tests.datasets_api._graph_doubles import FakeMintGraphReader
from tests.engine._base_identity import make_base, make_identity
from tests.engine._builders import unbound_slot
from tests.engine._fake_coverage import empty_coverage_snapshot
from tests.engine._fake_sdmx import FakeSdmxClient
from tests.engine._harness import materialise_double_from_obs, member_walk_reader
from tests.fixtures import FakeGraph

pytestmark = pytest.mark.usefixtures("request_log")

_SV = "TEST/ResubmitPlaceScopeDefault_Population"
_SHAPE: dict = {
    "kind": "statvar_schema",
    "population_type": "TestResubmitPlaceScopePopulation",
    "measured_property": "testResubmitCount",
    "stat_type": "measuredValue",
}
_COUNTRIES = [f"country/RPS{i}" for i in range(1, 6)]  # 5-country default scope
_EARTH_CHILDREN = {"Earth": {"Country": _COUNTRIES}}
_NODES: dict = {
    _SV: {
        "label": "Test Resubmit Place Scope Population",
        "type": "StatisticalVariable",
        "arcs": {
            "typeOf": {"nodes": [{"dcid": "StatisticalVariable"}]},
            "name": {"nodes": [{"value": "Test Resubmit Place Scope Population"}]},
            "populationType": {"nodes": [{"dcid": "TestResubmitPlaceScopePopulation"}]},
            "measuredProperty": {"nodes": [{"dcid": "testResubmitCount"}]},
            "statType": {"nodes": [{"dcid": "measuredValue"}]},
        },
    },
}


def _entity_less_input() -> ResubmitIdentityPayload:
    """An unedited resubmit of the SAME identity a fresh unbound query would
    have minted: no where-slot posted at all (BaseIdentity.entities == {}),
    resolved_stat_var_dcids echoing the single confirmed member back."""
    base_spec_id, base_identity = make_base(_SHAPE, entities=[])
    return ResubmitIdentityPayload(
        identity=make_identity(_SHAPE),
        base_spec_id=base_spec_id,
        base_identity=base_identity,
        resolved_stat_var_dcids=[_SV],
    )


# ---------------------------------------------------------------------------
# (a) real data across the default scope -> definite, CoverageExact,
#     place_scope_defaulted, bulk series (never per-pair previews)
# ---------------------------------------------------------------------------


def test_no_where_slot_resubmit_is_definite_with_bulk_series():
    """Two of five countries carry data, sharing one facet_id, so availability lists those
    two: definite, CoverageExact naming the real place count, place_scope_defaulted
    fires, and the observations preview comes from ONE subject_observations
    call across the whole scope, never a per-(stat_var, entity) preview."""
    obs = {
        f"{_SV}|{_COUNTRIES[0]}": [
            {"earliestDate": "2020", "latestDate": "2020", "obsCount": 1, "facetId": "F1"}
        ],
        f"{_SV}|{_COUNTRIES[1]}": [
            {"earliestDate": "2021", "latestDate": "2021", "obsCount": 1, "facetId": "F1"}
        ],
    }
    graph = FakeGraph(nodes=_NODES, obs=obs, children=_EARTH_CHILDREN)
    sdmx = FakeSdmxClient(
        observations={},
        availability={},
        subject_observations={
            _SV: {
                _COUNTRIES[0]: [],
                _COUNTRIES[1]: [],
            }
        },
        scope_availability=graph.scope_availability,
    )
    region = resolve_spec_resubmit(
        inp=_entity_less_input(),
        graph=graph,
        sdmx=sdmx,
        coverage=empty_coverage_snapshot(),
        options=EngineOptions(include_observations=True),
    )

    assert region.status == "definite", region.no_data_reason
    spec = region.specs[0]
    assert spec.resolved.coverage.kind == "exact"
    assert spec.resolved.coverage.has_data is True
    places_dim = next(d for d in spec.resolved.coverage.dimensions if d.label == "places")
    assert places_dim.count == 2  # only the two data-bearing countries

    defaulted = [w for w in region.warnings if w.code == PLACE_SCOPE_DEFAULTED]
    assert len(defaulted) == 1
    assert "every place with data" in defaulted[0].message

    # The bulk-series branch dispatched, never the per-pair preview path.
    assert len(sdmx.subject_observations_calls) == 1
    call = sdmx.subject_observations_calls[0]
    assert call["variables"] == [_SV]
    assert call["places"] == [_COUNTRIES[0], _COUNTRIES[1]]
    assert call["facet_ids"] == ["F1"]


# ---------------------------------------------------------------------------
# (c) no country in the default scope has data -> no_observations
# ---------------------------------------------------------------------------


def test_no_where_slot_resubmit_no_country_has_data_reports_no_observations():
    """No where-slot posted, the SV re-reads fine, but no default-scope
    country carries any observation: honest no_data (genuine data gap), never
    a fabricated definite from StatVar existence alone."""
    graph = FakeGraph(nodes=_NODES, obs={}, children=_EARTH_CHILDREN)
    region = resolve_spec_resubmit(
        inp=_entity_less_input(),
        graph=graph,
        sdmx=FakeSdmxClient(
            observations={}, availability={}, scope_availability=graph.scope_availability
        ),
        coverage=empty_coverage_snapshot(),
    )

    assert region.status == "no_data"
    assert region.no_data_reason == "no_observations"
    assert region.flagged_match is not None and len(region.flagged_match) == 1
    assert region.gate == NO_MEMBER_HAS_DATA_AT_PLACE
    assert any(w.code == PLACE_SCOPE_DEFAULTED for w in region.warnings)


# ---------------------------------------------------------------------------
# (iv) earth-only SV -> same worldwide fallback as the fresh-query door
# ---------------------------------------------------------------------------


def test_no_where_slot_resubmit_earth_only_data_resolves_worldwide():
    """No where-slot posted, no default-scope country carries any data for
    the SV, but Earth itself does: the resubmit door reaches the same
    definite/worldwide outcome the fresh-query door does for the identical
    identity -- countries and resubmit must never disagree on one identity."""
    obs = {
        f"{_SV}|Earth": [
            {
                "earliestDate": "2020",
                "latestDate": "2020",
                "obsCount": 1,
                "provenanceId": "dc/base/Src",
                "observations": [{"date": "2020", "value": 7.0}],
            }
        ],
    }
    graph = FakeGraph(nodes=_NODES, obs=obs, children=_EARTH_CHILDREN)
    materialise_graph, materialise_sdmx = materialise_double_from_obs(obs)
    sdmx = FakeSdmxClient(
        observations={}, availability={}, scope_availability=graph.scope_availability
    )
    region = resolve_spec_resubmit(
        inp=_entity_less_input(),
        graph=graph,
        sdmx=sdmx,
        coverage=empty_coverage_snapshot(),
        options=EngineOptions(
            include_observations=True,
            materialise_graph=materialise_graph,
            materialise_sdmx=materialise_sdmx,
        ),
    )

    assert region.status == "definite", region.no_data_reason
    spec = region.specs[0]
    assert isinstance(spec.resolved.coverage, CoverageExact)
    assert spec.resolved.coverage.has_data is True
    assert all(d.label not in {"countries", "places"} for d in spec.resolved.coverage.dimensions)

    assert sdmx.subject_observations_calls == []
    assert spec.resolved.observations is not None
    assert len(spec.resolved.observations.series) == 1
    assert spec.resolved.observations.series[0].points

    defaulted = [w for w in region.warnings if w.code == PLACE_SCOPE_DEFAULTED]
    assert len(defaulted) == 1
    assert defaulted[0].message == (
        "no place was named; the answer was resolved as a worldwide total."
    )


# ---------------------------------------------------------------------------
# (v) the scope covers the StatVars the identity's mint serves, not the posted ones
# ---------------------------------------------------------------------------

_SV_FEMALE = "TEST/ResubmitPlaceScopeDefault_PopulationFemale"
_TOTAL_AND_PART_NODES: dict = {
    **_NODES,
    _SV_FEMALE: {
        "label": "Test Resubmit Place Scope Population, female",
        "type": "StatisticalVariable",
        "arcs": {
            **_NODES[_SV]["arcs"],
            "name": {"nodes": [{"value": "Test Resubmit Place Scope Population, female"}]},
            "constraintProperties": {"nodes": [{"dcid": "gender"}]},
            "gender": {"nodes": [{"dcid": "Female"}]},
        },
    },
}
_TOTAL_AND_PART_OBS = {
    f"{_SV}|{_COUNTRIES[0]}": [{"earliestDate": "2020", "latestDate": "2020", "obsCount": 1}],
    f"{_SV_FEMALE}|{_COUNTRIES[1]}": [
        {"earliestDate": "2021", "latestDate": "2021", "obsCount": 1}
    ],
}


def _scope_members(region) -> list[str]:
    (spec,) = region.specs
    assert spec.resolved.scope is not None
    return [entity.ref for entity in spec.resolved.scope.members]


def test_a_resubmitted_total_and_its_part_are_scoped_over_the_total_their_mint_serves():
    """The posted total and its female part share the identity, and `gender` reads
    `aggregate`, so the mint serves the total alone. The part's data in the second country
    stays out of the answer's scope, which is the total's own country."""
    graph = FakeGraph(
        nodes=_TOTAL_AND_PART_NODES, obs=_TOTAL_AND_PART_OBS, children=_EARTH_CHILDREN
    )
    base_spec_id, base_identity = make_base(
        _SHAPE, [unbound_slot("how", "gender", "gender")], entities=[]
    )

    region = resolve_spec_resubmit(
        inp=ResubmitIdentityPayload(
            identity=base_identity,
            base_spec_id=base_spec_id,
            base_identity=base_identity,
            resolved_stat_var_dcids=[_SV, _SV_FEMALE],
        ),
        graph=graph,
        sdmx=FakeSdmxClient(
            observations={}, availability={}, scope_availability=graph.scope_availability
        ),
        coverage=empty_coverage_snapshot(),
    )

    assert region.status == "definite", region.no_data_reason
    assert [sv.ref for sv in region.specs[0].resolved.stat_vars.items] == [_SV]
    assert _scope_members(region) == [_COUNTRIES[0]]
    (defaulted,) = [w for w in region.warnings if w.code == PLACE_SCOPE_DEFAULTED]
    assert "every place with data" in defaulted.message


def test_a_resubmitted_part_answers_as_the_total_its_mint_serves():
    """The resubmit posts the female part alone, with `gender` unbound. The mint seeds its walk
    with the part, finds the total beside it, and serves the total, so the answer lists the
    total under the posted identity and is scoped over the total's own country."""
    graph = FakeGraph(
        nodes=_TOTAL_AND_PART_NODES, obs=_TOTAL_AND_PART_OBS, children=_EARTH_CHILDREN
    )
    member_graph = member_walk_reader(graph)
    base_spec_id, base_identity = make_base(
        _SHAPE, [unbound_slot("how", "gender", "gender")], entities=[]
    )

    region = resolve_spec_resubmit(
        inp=ResubmitIdentityPayload(
            identity=base_identity,
            base_spec_id=base_spec_id,
            base_identity=base_identity,
            resolved_stat_var_dcids=[_SV_FEMALE],
        ),
        graph=graph,
        sdmx=FakeSdmxClient(
            observations={}, availability={}, scope_availability=graph.scope_availability
        ),
        coverage=empty_coverage_snapshot(),
        options=EngineOptions(materialise_graph=member_graph),
    )

    assert region.status == "definite", region.no_data_reason
    (spec,) = region.specs
    assert spec.spec_id == base_spec_id
    minted = resolve_definition_members(
        base_identity.model_dump(mode="json", exclude_none=True),
        spec_id=base_spec_id,
        graph=member_graph,
        enumerator=MemberEnumerator(graph=member_graph),
        code_index=get_index(),
        seed=[_SV_FEMALE],
    )
    answered = [sv.ref for sv in spec.resolved.stat_vars.items]
    assert answered == [shape.stat_var.removeprefix("dcid:") for shape in minted.shapes]
    assert answered == [_SV]
    assert MEMBERSHIP_UNVERIFIED not in [w.code for w in region.warnings]
    assert _scope_members(region) == [_COUNTRIES[0]]


def test_a_resubmit_the_index_decides_is_scoped_over_the_family_its_mint_serves():
    """The code index decides the education aid identity, so the mint resolves it through
    the index and ignores the posted member, an imputed multilateral series of another
    dataset. The served StatVars are the mint's own."""
    document = education_aid_identity()
    identity = BaseIdentity.model_validate(document)
    imputed = next(
        row.dcid for row in get_index().statvars if IMPUTED_PROVENANCE in row.provenances
    )
    posted = dataclasses.replace(
        synthetic_shape_for_identity(IdentityShapeSchema.model_validate(document["shape"])),
        sv_arc_facts={imputed: {}},
        sv_arc_facts_complete=True,
    )
    graph = FakeMintGraphReader(arcs={})
    minted = resolve_definition_members(
        document,
        spec_id=EDUCATION_AID_SPEC_ID,
        graph=graph,
        enumerator=MemberEnumerator(graph=graph),
        code_index=get_index(),
    )

    served = identity_served_stat_vars(identity, posted)

    assert imputed not in served
    assert sorted(served) == sorted(shape.stat_var.removeprefix("dcid:") for shape in minted.shapes)


# ---------------------------------------------------------------------------
# (vi) members with data in different countries -> each member at its own
# ---------------------------------------------------------------------------

_SV_OVER65 = "TEST/ResubmitPlaceScopeDefault_PopulationOver65"
_FEMALE_AND_OVER65_NODES: dict = {
    _SV_FEMALE: _TOTAL_AND_PART_NODES[_SV_FEMALE],
    _SV_OVER65: {
        "label": "Test Resubmit Place Scope Population, 65 and over",
        "type": "StatisticalVariable",
        "arcs": {
            **_NODES[_SV]["arcs"],
            "name": {"nodes": [{"value": "Test Resubmit Place Scope Population, 65 and over"}]},
            "constraintProperties": {"nodes": [{"dcid": "age"}]},
            "age": {"nodes": [{"dcid": "Years65Onwards"}]},
        },
    },
}


class _ProbeLogGraph(FakeGraph):
    """Records the StatVars and places of every observation probe."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.probes: list[tuple[list[str], list[str]]] = []

    def observation_facets_batch(self, stat_vars, entities, needs_dates=False, **kwargs):
        self.probes.append((list(stat_vars), list(entities)))
        return super().observation_facets_batch(stat_vars, entities, needs_dates=needs_dates)


def test_a_resubmitted_member_is_counted_and_grounded_at_its_own_scope():
    """Neither posted member dominates the other: the female member holds data in the first
    country and the 65-and-over member in the second. One batch reads both at the union of
    their scopes; the answer counts the female member's one observation, is scoped over its
    own country, and previews its series there alone."""
    obs = {
        f"{_SV_FEMALE}|{_COUNTRIES[0]}": [
            {
                "earliestDate": "2020",
                "latestDate": "2020",
                "obsCount": 1,
                "facetId": "F1",
                "observations": [{"date": "2020", "value": 1.0}],
            }
        ],
        f"{_SV_OVER65}|{_COUNTRIES[1]}": [
            {
                "earliestDate": "2021",
                "latestDate": "2021",
                "obsCount": 1,
                "facetId": "F1",
                "observations": [{"date": "2021", "value": 1.0}],
            }
        ],
    }
    graph = _ProbeLogGraph(nodes=_FEMALE_AND_OVER65_NODES, obs=obs, children=_EARTH_CHILDREN)
    materialise_graph, materialise_sdmx = materialise_double_from_obs(obs)
    base_spec_id, base_identity = make_base(
        _SHAPE,
        [unbound_slot("how", "gender", "gender"), unbound_slot("how", "age", "age")],
        entities=[],
    )

    region = resolve_spec_resubmit(
        inp=ResubmitIdentityPayload(
            identity=base_identity,
            base_spec_id=base_spec_id,
            base_identity=base_identity,
            resolved_stat_var_dcids=[_SV_FEMALE, _SV_OVER65],
        ),
        graph=graph,
        sdmx=FakeSdmxClient(
            observations={}, availability={}, scope_availability=graph.scope_availability
        ),
        coverage=empty_coverage_snapshot(),
        options=EngineOptions(
            include_observations=True,
            materialise_graph=materialise_graph,
            materialise_sdmx=materialise_sdmx,
        ),
    )

    assert region.status == "definite", region.no_data_reason
    spec = region.specs[0]
    assert [sv.ref for sv in spec.resolved.stat_vars.items] == [_SV_FEMALE]
    assert spec.resolved.coverage.observation_count == 1
    assert _scope_members(region) == [_COUNTRIES[0]]
    assert ([_SV_FEMALE, _SV_OVER65], _COUNTRIES[:2]) in graph.probes
    observations = spec.resolved.observations
    assert observations is not None and observations.series
    assert {entity.ref for series in observations.series for entity in series.entities} == {
        _COUNTRIES[0]
    }
