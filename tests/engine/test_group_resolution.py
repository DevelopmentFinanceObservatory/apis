"""Groups resolve to their members through the place-set path, subject to `max_child_places`.

Most tests inject resolutions; `TestG7RealResolverSettlement` uses the shared resolver and real G7 members.
"""

from __future__ import annotations

import asyncio

import pytest
from dfo_spec_kit.identity import spec_id_from_value

from entity_resolution import Entity, Group, Resolved
from search.engine.bind import _UniversalBindOutput
from search.engine.clients.graph import Facet
from search.engine.config import settings
from search.engine.core import resolve_async
from search.engine.departures import spec_departure_facts
from search.engine.region.pipeline import resolve_variable
from search.engine.role_text import DoorRole
from search.models import IdentityEntityGroup, RawTextInput, ResolveRequest
from series_coverage.schema import RoleNames
from series_coverage.shared import CoverageSnapshot
from tests.engine._fake_coverage import (
    FakeCoverageService,
    empty_coverage_snapshot,
    patch_coverage,
)
from tests.engine._fake_sdmx import FakeSdmxClient
from tests.fixtures import FakeGraph, FakeLLM, make_resolver

_FINANCING_ROLE_NAMES = RoleNames(donor="financingProvider", recipient="financingRecipient")

_GDP_SV = "Amount_EconomicActivity_GrossDomesticProduction_Nominal"
_GDP_NODE = {
    "label": "Nominal gross domestic product",
    "type": "StatisticalVariable",
    "arcs": {
        "constraintProperties": {
            "nodes": [{"name": "activitySource", "types": ["Property"], "dcid": "activitySource"}]
        },
        "activitySource": {
            "nodes": [{"name": "GrossDomesticProduction", "dcid": "GrossDomesticProduction"}]
        },
        "measuredProperty": {"nodes": [{"name": "amount", "dcid": "amount"}]},
        "measurementQualifier": {"nodes": [{"name": "Nominal", "dcid": "Nominal"}]},
        "statType": {"nodes": [{"name": "measuredValue", "dcid": "measuredValue"}]},
        "typeOf": {"nodes": [{"name": "StatisticalVariable", "dcid": "StatisticalVariable"}]},
        "name": {"nodes": [{"value": "Nominal gross domestic product"}]},
        "populationType": {"nodes": [{"name": "EconomicActivity", "dcid": "EconomicActivity"}]},
    },
}

# The graph names a provenance on every facet it serves; `select_facet` refuses one without.
_SOURCE = "dc/base/PlaceFixture"


def _country(dcid: str, name: str) -> Entity:
    return Entity(dcid=dcid, name=name, kind="country", graph_type="Country")


class _FixedBindLLM:
    """Elects the sole shape without a recorded LLM response."""

    async def generate_structured(self, *, prompt, system, schema):
        assert schema.__name__ == "_UniversalBindOutput"
        return _UniversalBindOutput(
            chosen_shape_index=0, bindings=[], also_plausible=None, ask=None
        ), None


def _run(
    *,
    entities,
    pre_resolved,
    graph,
    entity_roles=None,
    sdmx=None,
    llm=None,
    coverage: CoverageSnapshot | None = None,
):
    return asyncio.run(
        resolve_variable(
            "gdp",
            entities=entities,
            date_request=None,
            detect_query="gdp",
            role_query="gdp " + " ".join(entities),
            entity_roles=entity_roles,
            pac=True,
            graph=graph,
            resolver=make_resolver(graph),
            sdmx=sdmx if sdmx is not None else FakeSdmxClient(observations={}, availability={}),
            llm=llm if llm is not None else FakeLLM(),
            base_steps=[],
            base_timing={},
            pre_resolved=pre_resolved,
            coverage=coverage if coverage is not None else empty_coverage_snapshot(),
        )
    )


class TestGroupExpansion:
    """A Group expands like a continent and uses `max_child_places`, not the 10-place standard cap."""

    _MEMBERS = (
        _country("country/T1", "Testland One"),
        _country("country/T2", "Testland Two"),
        _country("country/T3", "Testland Three"),
    )

    def _graph(self, obs: dict) -> FakeGraph:
        return FakeGraph(
            nodes={_GDP_SV: _GDP_NODE},
            obs=obs,
            detect={"gdp": {"svs": [_GDP_SV], "cosine_scores": [0.95]}},
            resolve={},
            places={},
            children={},
        )

    def test_group_with_no_entity_expands_to_its_members(self):
        obs = {
            f"{_GDP_SV}|country/T1": [
                {
                    "earliestDate": "2020",
                    "latestDate": "2020",
                    "obsCount": 1,
                    "provenanceId": _SOURCE,
                }
            ],
            f"{_GDP_SV}|country/T2": [
                {
                    "earliestDate": "2021",
                    "latestDate": "2021",
                    "obsCount": 1,
                    "provenanceId": _SOURCE,
                }
            ],
        }
        result = _run(
            entities=["G7"],
            pre_resolved={"G7": Group(members=self._MEMBERS, entity=None)},
            graph=self._graph(obs),
        )

        assert result.status == "definite", result
        spec = result.specs[0]
        # Groups have no dcid, so each member binds as its own entity.
        assert [e.ref for e in spec.resolved.entities] == ["country/T1", "country/T2", "country/T3"]
        countries_dim = next(
            d for d in (spec.resolved.coverage.dimensions or []) if d.label == "countries"
        )
        assert countries_dim.count == 2

    def test_group_member_count_rides_the_child_place_cap_not_the_standard_cap(
        self, search_settings
    ):
        """A Group uses the child-place cap, here 2, rather than the standard 10-place cap."""
        search_settings(max_child_places=2)
        obs = {
            f"{_GDP_SV}|country/T1": [
                {
                    "earliestDate": "2020",
                    "latestDate": "2020",
                    "obsCount": 1,
                    "provenanceId": _SOURCE,
                }
            ],
        }
        result = _run(
            entities=["G7"],
            pre_resolved={"G7": Group(members=self._MEMBERS, entity=None)},
            graph=self._graph(obs),
        )
        # Three members exceed the child cap of 2, so resolution refuses instead of truncating.
        assert result.status == "no_data"
        assert result.no_data_reason == "entity_not_resolved"


class TestGroupTooLarge:
    """Groups above `max_child_places` return entity_not_resolved, with no truncated answer."""

    def test_oversized_group_refuses_entity_not_resolved(self):
        members = tuple(
            _country(f"country/T{i}", f"Testland {i}")
            for i in range(settings().max_child_places + 1)
        )
        graph = FakeGraph(
            nodes={_GDP_SV: _GDP_NODE},
            obs={},
            detect={"gdp": {"svs": [_GDP_SV], "cosine_scores": [0.95]}},
            resolve={},
            places={},
            children={},
        )
        result = _run(
            entities=["G77"],
            pre_resolved={"G77": Group(members=members, entity=None)},
            graph=graph,
        )
        assert result.status == "no_data"
        assert result.no_data_reason == "entity_not_resolved"
        assert result.unresolved_entity_names == ("G77",)


class TestGroupRoleBearingExpansion:
    """Every member binds on its role axis; ambiguity is checked per group name."""

    _MEMBERS = (
        _country("country/T1", "Testland One"),
        _country("country/T2", "Testland Two"),
    )

    def _graph(self, *, group_label: str | None = None) -> FakeGraph:
        """`group_label` is the graph's label for the group's own node, which the graph
        holds only when it is set."""
        sv_arcs = {
            "populationType": {"nodes": [{"dcid": "EconomicActivity"}]},
            "measuredProperty": {"nodes": [{"dcid": "amount"}]},
            "statType": {"nodes": [{"dcid": "measuredValue"}]},
            "observationProperties": {
                "nodes": [{"dcid": "financingProvider"}, {"dcid": "financingRecipient"}]
            },
        }
        return FakeGraph(
            nodes={
                _GDP_SV: {"label": "Test flow", "arcs": sv_arcs},
                # Admission.admits reads rangeIncludes off the role
                # property's own node -- both axes admit Country here, since
                # every group member this test resolves is Country-kind.
                "financingProvider": {"arcs": {"rangeIncludes": {"nodes": [{"dcid": "Country"}]}}},
                "financingRecipient": {"arcs": {"rangeIncludes": {"nodes": [{"dcid": "Country"}]}}},
                **({"TestlandUnion": {"label": group_label}} if group_label is not None else {}),
            },
            obs={},
            detect={"gdp": {"svs": [_GDP_SV], "cosine_scores": [0.95]}},
            resolve={},
            places={},
            children={},
        )

    def _facet(self) -> Facet:
        return Facet(
            earliest_date="2020",
            latest_date="2020",
            obs_count=1,
            dates=["2020"],
            values=[1.0],
            facet_id="f1",
        )

    def test_axis_less_solitary_group_refuses_role_not_resolved(self):
        """A group name without an origin or destination role returns role_not_resolved."""
        sdmx = FakeSdmxClient(
            observations={
                _GDP_SV: {
                    (("financingRecipient", "country/T1"),): [self._facet()],
                    (("financingRecipient", "country/T2"),): [self._facet()],
                }
            },
        )
        result = _run(
            entities=["Testland Union"],
            pre_resolved={"Testland Union": Group(members=self._MEMBERS, entity=None)},
            graph=self._graph(),
            sdmx=sdmx,
            llm=_FixedBindLLM(),
        )
        assert result.status == "no_data", result
        assert result.no_data_reason == "role_not_resolved"

    _GROUP_NODE = Entity(
        dcid="TestlandUnion", name="Testland Union", kind="aggregate", graph_type="Place"
    )

    @pytest.mark.parametrize(
        ("group_node", "graph_label", "label"),
        [
            (None, None, "Testland Union"),
            (_GROUP_NODE, None, "Testland Union"),
            (_GROUP_NODE, "Union of Testland", "Union of Testland"),
        ],
        ids=["no_group_node", "group_node_unlabelled", "group_node_labelled"],
    )
    def test_group_as_origin_binds_every_member_alongside_a_named_destination(
        self, monkeypatch, group_node, graph_label, label
    ):
        """The where-slot binds Kenya; each origin member belongs in resolved entities, and the
        origin role carries the group outside the spec_id hash, with its node when the graph
        holds one. The label is the graph's for that node, else the query's name for the group.
        Its departure asks for it as the query named it."""
        _auto_coverage_index_1 = patch_coverage(
            monkeypatch,
            [
                (_GDP_SV, "f1", "country/T1", "country/KEN", [2020]),
                (_GDP_SV, "f1", "country/T2", "country/KEN", [2020]),
            ],
            role_names={_GDP_SV: _FINANCING_ROLE_NAMES},
        )
        sdmx = FakeSdmxClient(observations={})  # never read on this branch
        result = _run(
            entities=["Testland Union", "Kenya"],
            pre_resolved={
                "Testland Union": Group(members=self._MEMBERS, entity=group_node),
                "Kenya": Resolved(entity=_country("country/KEN", "Kenya")),
            },
            entity_roles={
                "Testland Union": [DoorRole(axis="origin", span="from")],
                "Kenya": [DoorRole(axis="destination", span="to")],
            },
            graph=self._graph(group_label=graph_label),
            sdmx=sdmx,
            llm=_FixedBindLLM(),
            coverage=FakeCoverageService(_auto_coverage_index_1).snapshot(),
        )
        assert result.status == "definite", result
        spec = result.specs[0]
        entities_by_ref = {e.ref: e.role for e in spec.resolved.entities}
        assert entities_by_ref["country/T1"] == "origin"
        assert entities_by_ref["country/T2"] == "origin"
        assert entities_by_ref["country/KEN"] == "destination"
        group = IdentityEntityGroup(
            dcid=group_node.dcid if group_node is not None else None, label=label
        )
        assert spec.resolved.entity_groups == {"origin": group}
        assert spec.definition.identity.entity_groups == {"origin": group}
        assert spec.spec_id == spec_id_from_value(spec.definition.identity.core_document())
        asked = [
            fact.asked
            for fact in spec_departure_facts(spec, operations=(), entity_words=result.entity_words)
            if fact.code == "group_read_as_members"
        ]
        assert asked == ["Testland Union"]


def _gdp_graph(obs: dict) -> FakeGraph:
    return FakeGraph(
        nodes={_GDP_SV: _GDP_NODE},
        obs=obs,
        detect={"gdp": {"svs": [_GDP_SV], "cosine_scores": [0.95]}},
        resolve={},
        places={},
        children={},
    )


class TestTwoGroupsNamedInOneQuery:
    """Each resolved group name contributes its members to the place set."""

    _G7_MEMBERS = (
        _country("country/T1", "Testland One"),
        _country("country/T2", "Testland Two"),
    )
    _G20_MEMBERS = (
        _country("country/T3", "Testland Three"),
        _country("country/T4", "Testland Four"),
    )

    def test_two_named_groups_both_settle_and_union(self):
        obs = {
            f"{_GDP_SV}|country/T1": [
                {
                    "earliestDate": "2020",
                    "latestDate": "2020",
                    "obsCount": 1,
                    "provenanceId": _SOURCE,
                }
            ],
            f"{_GDP_SV}|country/T3": [
                {
                    "earliestDate": "2020",
                    "latestDate": "2020",
                    "obsCount": 1,
                    "provenanceId": _SOURCE,
                }
            ],
        }
        result = _run(
            entities=["G7", "G20"],
            pre_resolved={
                "G7": Group(members=self._G7_MEMBERS, entity=None),
                "G20": Group(members=self._G20_MEMBERS, entity=None),
            },
            graph=_gdp_graph(obs),
        )
        assert result.status == "definite", result
        spec = result.specs[0]
        assert {e.ref for e in spec.resolved.entities} == {
            "country/T1",
            "country/T2",
            "country/T3",
            "country/T4",
        }


class TestGroupWithEmptyMembers:
    """An injected empty group resolves as entity_not_resolved instead of raising IndexError."""

    def test_empty_members_no_entity_reads_as_unresolved_not_a_crash(self):
        result = _run(
            entities=["Empty"],
            pre_resolved={"Empty": Group(members=(), entity=None)},
            graph=_gdp_graph({}),
        )
        assert result.status == "no_data"
        assert result.no_data_reason == "entity_not_resolved"
        assert result.unresolved_entity_names == ("Empty",)


class TestDuplicateMemberDcidWithinAGroup:
    """Injected duplicate member dcids bind once; deduplication happens in `_phase_place_bindings`."""

    _MEMBERS_WITH_A_DUPLICATE = (
        _country("country/T1", "Testland One"),
        _country("country/T1", "Testland One"),
        _country("country/T2", "Testland Two"),
    )

    def test_duplicate_member_dcid_binds_once(self):
        obs = {
            f"{_GDP_SV}|country/T1": [
                {
                    "earliestDate": "2020",
                    "latestDate": "2020",
                    "obsCount": 1,
                    "provenanceId": _SOURCE,
                }
            ],
            f"{_GDP_SV}|country/T2": [
                {
                    "earliestDate": "2021",
                    "latestDate": "2021",
                    "obsCount": 1,
                    "provenanceId": _SOURCE,
                }
            ],
        }
        result = _run(
            entities=["G7"],
            pre_resolved={"G7": Group(members=self._MEMBERS_WITH_A_DUPLICATE, entity=None)},
            graph=_gdp_graph(obs),
        )
        assert result.status == "definite", result
        spec = result.specs[0]
        assert [e.ref for e in spec.resolved.entities] == ["country/T1", "country/T2"]


class TestGroupCapBoundary:
    """A group at the 256-member cap settles; one member above it returns entity_not_resolved."""

    @staticmethod
    def _members(n: int) -> tuple[Entity, ...]:
        return tuple(_country(f"country/T{i}", f"Testland {i}") for i in range(n))

    def test_group_at_the_cap_settles(self):
        result = _run(
            entities=["G7"],
            pre_resolved={
                "G7": Group(members=self._members(settings().max_child_places), entity=None)
            },
            graph=_gdp_graph({}),
        )
        # No observations are supplied; this assertion only checks the cap refusal is absent.
        assert not (result.status == "no_data" and result.no_data_reason == "entity_not_resolved")

    def test_group_one_past_the_cap_refuses(self):
        result = _run(
            entities=["G7"],
            pre_resolved={
                "G7": Group(members=self._members(settings().max_child_places + 1), entity=None)
            },
            graph=_gdp_graph({}),
        )
        assert result.status == "no_data"
        assert result.no_data_reason == "entity_not_resolved"
        assert result.unresolved_entity_names == ("G7",)


class TestG7RealResolverSettlement:
    """Resolves G7 through the shared resolver and binds its seven real member countries in dcid order."""

    _G7_DCIDS = (
        "country/CAN",
        "country/DEU",
        "country/FRA",
        "country/GBR",
        "country/ITA",
        "country/JPN",
        "country/USA",
    )

    def test_g7_resolves_and_settles_to_its_real_member_countries(self):
        graph = FakeGraph(
            nodes={
                _GDP_SV: _GDP_NODE,
                **{dcid: {"label": dcid, "type": "Country"} for dcid in self._G7_DCIDS},
            },
            obs={
                f"{_GDP_SV}|country/CAN": [
                    {
                        "earliestDate": "2020",
                        "latestDate": "2020",
                        "obsCount": 1,
                        "provenanceId": _SOURCE,
                    }
                ],
                f"{_GDP_SV}|country/USA": [
                    {
                        "earliestDate": "2020",
                        "latestDate": "2020",
                        "obsCount": 1,
                        "provenanceId": _SOURCE,
                    }
                ],
            },
            detect={"gdp": {"svs": [_GDP_SV], "cosine_scores": [0.95]}},
            resolve={},
            places={},
            children={},
        )
        result = _run(entities=["G7"], pre_resolved=None, graph=graph)

        assert result.status == "definite", result
        spec = result.specs[0]
        assert [e.ref for e in spec.resolved.entities] == list(self._G7_DCIDS)


class TestG7AidQueryEndToEnd:
    """The raw-text path resolves G7 without an entity refusal.

    This query has no resolved variable, so it does not test group settlement; that is covered above.
    """

    def test_g7_aid_query(self):
        req = ResolveRequest(
            input=RawTextInput(query="How much do G7 countries contribute to international aid?")
        )
        result = asyncio.run(
            resolve_async(req, graph=FakeGraph(), llm=FakeLLM(), sdmx=FakeSdmxClient())
        )
        # See the class docstring for what this assertion does and does not
        # prove.
        assert result.parts, result
        part = result.parts[0]
        assert part.status == "no_data", part
        assert part.no_data.reason == "variable_not_resolved", part.no_data
        assert part.no_data.entity_terms == []
