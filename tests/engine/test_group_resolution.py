"""Groups resolve to their members through the place-set path, under the bound on the places
a Spec binds (`max_child_places`).

Most tests inject resolutions; `TestG7RealResolverSettlement` uses the shared resolver and real G7 members.
"""

from __future__ import annotations

import asyncio
import json

import pytest
from dfo_spec_kit.identity import spec_id_from_value

from entity_resolution import Ambiguous, Entity, Group, Resolved
from search.engine.clients.graph import Facet
from search.engine.config import settings
from search.engine.core import resolve_async
from search.engine.departures import spec_departure_facts
from search.engine.quantity_read import QuantityOutput
from search.engine.region import resolve_spec_resubmit
from search.engine.region.pipeline import resolve_variable
from search.engine.role_text import DoorRole
from search.models import (
    EntityGroup,
    RawTextInput,
    ResolvedFacet,
    ResolveRequest,
    ResubmitIdentityPayload,
)
from series_coverage.schema import RoleNames
from series_coverage.shared import CoverageSnapshot
from tests.engine._binders import elected_reply
from tests.engine._fake_coverage import (
    FakeCoverageService,
    empty_coverage_snapshot,
    patch_coverage,
)
from tests.engine._fake_sdmx import FakeSdmxClient
from tests.engine._harness import walk_options
from tests.engine._quantity import NO_QUANTITY
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
        if schema is QuantityOutput:
            return NO_QUANTITY
        assert schema.__name__ == "_UniversalBindOutput"
        return elected_reply(0, bindings=[], also_plausible=None), None


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
            base_timing={},
            pre_resolved=pre_resolved,
            coverage=coverage if coverage is not None else empty_coverage_snapshot(),
        )
    )


class TestGroupExpansion:
    """A Group expands like a continent, under the bound on the places a Spec binds."""

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

    def test_group_over_a_lowered_place_bound_answers_entity_set_too_large(self, search_settings):
        """A Group's members count against the bound on the places a Spec binds, here 2."""
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
        # Three members exceed the bound of 2, so resolution refuses instead of truncating.
        assert result.status == "no_data"
        assert result.no_data_reason == "entity_set_too_large"
        assert result.specs == ()


class TestGroupTooLarge:
    """Groups above `max_child_places` return entity_set_too_large, with no truncated answer."""

    def test_oversized_group_answers_entity_set_too_large(self):
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
        assert result.no_data_reason == "entity_set_too_large"
        # The group resolved. Only its membership is too large.
        assert result.unresolved_entity_names == ()


class TestGroupRoleBearingExpansion:
    """Every member binds on its role axis; ambiguity is checked per group name."""

    _MEMBERS = (
        _country("country/T1", "Testland One"),
        _country("country/T2", "Testland Two"),
    )

    def _graph(
        self,
        *,
        group_label: str | None = None,
        children: dict | None = None,
        countries: tuple[str, ...] = (),
    ) -> FakeGraph:
        """`group_label` is the graph's label for the group's own node, which the graph
        holds only when it is set. `children` are a region's member places by type, and
        `countries` the country nodes the graph holds."""
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
                **{dcid: {"label": dcid, "type": "Country"} for dcid in countries},
            },
            obs={},
            detect={"gdp": {"svs": [_GDP_SV], "cosine_scores": [0.95]}},
            resolve={},
            places={},
            children=children or {},
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
        origin role's group is served on `resolved.entity_groups`, with its node and the node's
        graph type when the graph holds one. The label is the graph's for that node, else the
        query's name for the group.
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
        group = EntityGroup(
            ref=group_node.dcid if group_node is not None else None,
            label=label,
            entity_type=group_node.graph_type if group_node is not None else None,
        )
        assert spec.resolved.entity_groups == {"origin": group}
        assert spec.spec_id == spec_id_from_value(
            spec.definition.identity.model_dump(exclude_none=True)
        )
        asked = [
            fact.asked
            for fact in spec_departure_facts(spec, operations=(), entity_words=result.entity_words)
            if fact.code == "group_read_as_members"
        ]
        assert asked == ["Testland Union"]

    @pytest.mark.parametrize(
        ("name", "group", "party"),
        [
            pytest.param("G7", "GroupOfSeven", "GroupOfSeven", id="a-summing-total"),
            # "DAC countries, total" can differ from the sum over the DAC's members.
            pytest.param("DAC", "groups/OECD.DAC", "DACCountriesTotal", id="a-published-total"),
        ],
    )
    @pytest.mark.parametrize(
        ("published", "members"),
        [
            pytest.param(True, False, id="the-dataset-publishes-it"),
            pytest.param(False, True, id="the-dataset-does-not"),
        ],
    )
    def test_a_group_binds_the_total_its_dataset_publishes(
        self, monkeypatch, name, group, party, published, members
    ):
        """The code index declares GroupOfSeven the total of the resolver's G7, and
        DACCountriesTotal that of the DAC (`CodeIndex.totals_by_group`). Where the elected
        series publishes it as a donor, the origin binds that one row and keeps the group,
        naming the total, with no `group_read_as_members` departure. Otherwise the origin binds
        the members."""
        origins = ["country/T1", "country/T2"] if members else [party]
        total = None if members else party
        rows = [
            (_GDP_SV, "f1", "country/T1", "country/KEN", [2020]),
            (_GDP_SV, "f1", "country/T2", "country/KEN", [2020]),
        ]
        if published:
            rows.append((_GDP_SV, "f1", party, "country/KEN", [2020]))
        index = patch_coverage(monkeypatch, rows, role_names={_GDP_SV: _FINANCING_ROLE_NAMES})
        result = _run(
            entities=[name, "Kenya"],
            pre_resolved={
                name: Group(members=self._MEMBERS, entity=None, dcid=group),
                "Kenya": Resolved(entity=_country("country/KEN", "Kenya")),
            },
            entity_roles={
                name: [DoorRole(axis="origin", span="from")],
                "Kenya": [DoorRole(axis="destination", span="to")],
            },
            graph=self._graph(),
            sdmx=FakeSdmxClient(observations={}),
            llm=_FixedBindLLM(),
            coverage=FakeCoverageService(index).snapshot(),
        )

        assert result.status == "definite", result
        (spec,) = result.specs
        assert sorted(e.ref for e in spec.resolved.entities if e.role == "origin") == origins
        assert spec.resolved.entity_groups == {
            "origin": EntityGroup(label=name, total=total, group_id=group)
        }
        assert spec.resolved.group_totals == ({"origin": party} if members else {})
        codes = [fact.code for fact in spec_departure_facts(spec, operations=())]
        assert ("group_read_as_members" in codes) is members

    @pytest.mark.parametrize(
        ("published", "destinations", "total", "kept"),
        [
            pytest.param(True, ["DACAsia"], "DACAsia", {}, id="the-dataset-publishes-it"),
            pytest.param(
                False,
                ["country/T1", "country/T2"],
                None,
                {"destination": "DACAsia"},
                id="the-dataset-does-not",
            ),
        ],
    )
    def test_a_dac_region_binds_its_total_or_keeps_it_beside_the_members(
        self, monkeypatch, published, destinations, total, kept
    ):
        """The code index declares DACAsia the total of resolvekit's "Asia (DAC)", a group with
        no node of its own. Where the elected series publishes it as a recipient, the
        destination binds that row. Otherwise it binds the members and keeps the total aside
        (`ResolvedSpec.group_totals`), for the departure naming what the members leave out."""
        rows = [
            (_GDP_SV, "f1", "country/KEN", "country/T1", [2020]),
            (_GDP_SV, "f1", "country/KEN", "country/T2", [2020]),
        ]
        if published:
            rows.append((_GDP_SV, "f1", "country/KEN", "DACAsia", [2020]))
        index = patch_coverage(monkeypatch, rows, role_names={_GDP_SV: _FINANCING_ROLE_NAMES})
        result = _run(
            entities=["Kenya", "Asia (DAC)"],
            pre_resolved={
                "Kenya": Resolved(entity=_country("country/KEN", "Kenya")),
                "Asia (DAC)": Group(
                    members=self._MEMBERS, entity=None, dcid="groups/OECD.DAC.Region.Asia"
                ),
            },
            entity_roles={
                "Kenya": [DoorRole(axis="origin", span="from")],
                "Asia (DAC)": [DoorRole(axis="destination", span="to")],
            },
            graph=self._graph(),
            sdmx=FakeSdmxClient(observations={}),
            llm=_FixedBindLLM(),
            coverage=FakeCoverageService(index).snapshot(),
        )

        assert result.status == "definite", result
        (spec,) = result.specs
        bound = sorted(e.ref for e in spec.resolved.entities if e.role == "destination")
        assert bound == destinations
        assert spec.resolved.entity_groups == {
            "destination": EntityGroup(
                label="Asia (DAC)", total=total, group_id="groups/OECD.DAC.Region.Asia"
            )
        }
        assert spec.resolved.group_totals == kept
        if kept:
            # Served from the CRS, the members' departure names the regional amounts the kept
            # total counts beside them.
            crs = ResolvedFacet(
                basis="fallback", ref="provenance/OECDCreditorReportingSystem", label="CRS"
            )
            facets = {**(spec.resolved.facets or {}), "provenance": crs}
            served = spec.model_copy(
                update={"resolved": spec.resolved.model_copy(update={"facets": facets})}
            )
            (fact,) = (
                f
                for f in spec_departure_facts(served, operations=())
                if f.code == "group_read_as_members"
            )
            assert "Asia, unallocated by country (DAC)" in fact.excluded

    def test_africa_dac_binds_the_total_its_dataset_publishes(self, monkeypatch):
        """The shared resolver gives "Africa (DAC)" the id of the graph place `africa`
        (`group_ids.json`), whose members are the DAC region's, so the code index's total
        for that place, DACAfrica, holds it, and the destination binds that row."""
        graph = self._graph()
        (africa,) = asyncio.run(make_resolver(graph).resolve(["Africa (DAC)"]))
        assert isinstance(africa, Group)
        members = [member.dcid for member in africa.members]
        rows = [(_GDP_SV, "f1", "country/FRA", dcid, [2016]) for dcid in members]
        rows.append((_GDP_SV, "f1", "country/FRA", "DACAfrica", [2016]))
        index = patch_coverage(monkeypatch, rows, role_names={_GDP_SV: _FINANCING_ROLE_NAMES})
        result = _run(
            entities=["France", "Africa (DAC)"],
            pre_resolved={
                "France": Resolved(entity=_country("country/FRA", "France")),
                "Africa (DAC)": africa,
            },
            entity_roles={
                "France": [DoorRole(axis="origin", span="from")],
                "Africa (DAC)": [DoorRole(axis="destination", span="to")],
            },
            graph=graph,
            sdmx=FakeSdmxClient(observations={}),
            llm=_FixedBindLLM(),
            coverage=FakeCoverageService(index).snapshot(),
        )

        assert result.status == "definite", result
        (spec,) = result.specs
        assert spec.definition.identity.entities["destination"] == ["DACAfrica"]
        assert spec.resolved.entity_groups["destination"].total == "DACAfrica"

    @pytest.mark.parametrize(
        ("name", "group_role", "other", "total"),
        [
            pytest.param("G7", "origin", ("Kenya", "country/KEN"), "GroupOfSeven", id="g7"),
            pytest.param(
                "Asia (DAC)",
                "destination",
                ("France", "country/FRA"),
                "DACAsia",
                id="asia-dac",
            ),
        ],
    )
    def test_a_members_read_of_a_nodeless_group_survives_a_resubmit(
        self, monkeypatch, name, group_role, other, total
    ):
        """A first answer reads a group the graph holds no node for as its members, on a
        dataset that does not publish the group's total. The served Spec, serialised and
        posted back, resubmits to the same `spec_id` with the same group, the total kept
        aside, and the same departures."""
        (group,) = asyncio.run(make_resolver(self._graph()).resolve([name]))
        assert isinstance(group, Group)
        assert group.entity is None
        other_name, other_dcid = other
        graph = self._graph(countries=(other_dcid, *(m.dcid for m in group.members)))
        other_role = "destination" if group_role == "origin" else "origin"
        rows = [
            (
                _GDP_SV,
                "f1",
                *((m.dcid, other_dcid) if group_role == "origin" else (other_dcid, m.dcid)),
                [2020],
            )
            for m in group.members
        ]
        index = patch_coverage(monkeypatch, rows, role_names={_GDP_SV: _FINANCING_ROLE_NAMES})
        coverage = FakeCoverageService(index).snapshot()
        first = _run(
            entities=[name, other_name],
            pre_resolved={
                name: group,
                other_name: Resolved(entity=_country(other_dcid, other_name)),
            },
            entity_roles={
                name: [DoorRole(axis=group_role, span="from")],
                other_name: [DoorRole(axis=other_role, span="to")],
            },
            graph=graph,
            sdmx=FakeSdmxClient(observations={}),
            llm=_FixedBindLLM(),
            coverage=coverage,
        )
        assert first.status == "definite", first
        (served,) = first.specs
        assert served.resolved.entity_groups[group_role].group_id == group.dcid
        assert served.resolved.group_totals == {group_role: total}

        wire = json.loads(served.model_dump_json(exclude_none=True))
        region = resolve_spec_resubmit(
            inp=ResubmitIdentityPayload.model_validate(
                {
                    "identity": wire["definition"]["identity"],
                    "entity_groups": wire["resolved"]["entity_groups"],
                }
            ),
            graph=graph,
            sdmx=FakeSdmxClient(observations={}),
            coverage=coverage,
            options=walk_options(graph),
        )

        assert region.status == "definite", region
        (spec,) = region.specs
        assert spec.spec_id == served.spec_id
        assert spec.resolved.entity_groups == served.resolved.entity_groups
        assert spec.resolved.group_totals == {group_role: total}
        codes = [fact.code for fact in spec_departure_facts(spec, operations=())]
        assert "group_read_as_members" in codes
        assert codes == [fact.code for fact in spec_departure_facts(served, operations=())]

    def test_a_region_on_a_role_that_admits_only_its_members_binds_them(
        self, monkeypatch, declare_regions
    ):
        """The recipient role admits countries and refuses a continent. The series holds
        rows to the continent's node and to one member, and the Spec binds the members."""
        region = Entity(dcid="testonia", name="Testonia", kind="continent", graph_type="Continent")
        declare_regions({region.dcid: ["country/T1", "country/T2"]})
        coverage = patch_coverage(
            monkeypatch,
            [
                (_GDP_SV, "f1", "country/KEN", region.dcid, [2020]),
                (_GDP_SV, "f1", "country/KEN", "country/T1", [2020]),
            ],
            role_names={_GDP_SV: _FINANCING_ROLE_NAMES},
        )
        graph = self._graph(children={region.dcid: {"Country": ["country/T1", "country/T2"]}})
        result = _run(
            entities=["Kenya", "Testonia"],
            pre_resolved={
                "Kenya": Resolved(entity=_country("country/KEN", "Kenya")),
                "Testonia": Resolved(entity=region),
            },
            entity_roles={
                "Kenya": [DoorRole(axis="origin", span="from")],
                "Testonia": [DoorRole(axis="destination", span="to")],
            },
            graph=graph,
            sdmx=FakeSdmxClient(observations={}),
            llm=_FixedBindLLM(),
            coverage=FakeCoverageService(coverage).snapshot(),
        )

        assert result.status == "definite", result
        spec = result.specs[0]
        assert spec.definition.identity.entities == {
            "origin": ["country/KEN"],
            "destination": ["country/T1", "country/T2"],
        }
        assert spec.resolved.entity_groups == {
            "destination": EntityGroup(ref="testonia", label="Testonia", entity_type="Continent")
        }


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


class TestGroupNamedBesideAnAmbiguousEntity:
    """The answer carries the dcid each resolved name pins, and a group pins none."""

    _MEMBERS = (
        _country("country/T1", "Testland One"),
        _country("country/T2", "Testland Two"),
    )
    _NODE = Entity(
        dcid="TestlandUnion", name="Testland Union", kind="aggregate", graph_type="Place"
    )

    @pytest.mark.parametrize("node", [None, _NODE], ids=["no-node", "with-node"])
    def test_a_group_is_pinned_to_neither_a_member_nor_its_node(self, node):
        """Pinning the group to a member's dcid, or to its node, would resubmit one place
        for the several the query named."""
        result = _run(
            entities=["Testland Union", "Kenya", "Congo"],
            pre_resolved={
                "Testland Union": Group(members=self._MEMBERS, entity=node),
                "Kenya": Resolved(entity=_country("country/KEN", "Kenya")),
                "Congo": Ambiguous(
                    candidates=(
                        _country("country/COD", "Congo (DRC)"),
                        _country("country/COG", "Congo (Republic)"),
                    )
                ),
            },
            graph=_gdp_graph({}),
        )

        assert result.no_data_reason == "entity_ambiguous"
        assert result.resolved_entity_dcids == {"Kenya": "country/KEN"}


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
    """A group at the 256-member bound settles. One member above it returns
    entity_set_too_large."""

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
        # No observations are supplied, so this only checks that the bound refusal is absent.
        assert result.no_data_reason not in {"entity_set_too_large", "entity_not_resolved"}

    def test_group_one_past_the_cap_refuses(self):
        result = _run(
            entities=["G7"],
            pre_resolved={
                "G7": Group(members=self._members(settings().max_child_places + 1), entity=None)
            },
            graph=_gdp_graph({}),
        )
        assert result.status == "no_data"
        assert result.no_data_reason == "entity_set_too_large"
        assert result.unresolved_entity_names == ()

    _CONGO = Ambiguous(
        candidates=(
            _country("country/COD", "Congo (DRC)"),
            _country("country/COG", "Congo (Republic)"),
        )
    )

    @pytest.mark.parametrize(
        ("entities", "reason", "ambiguous", "unresolved"),
        [
            (
                ["G7", "Congo"],
                "entity_ambiguous",
                {"Congo": ["country/COD", "country/COG"]},
                ("Congo",),
            ),
            (
                ["Congo", "Atlantis"],
                "entity_not_resolved",
                {"Congo": ["country/COD", "country/COG"]},
                ("Congo", "Atlantis"),
            ),
            (["G7", "Atlantis"], "entity_not_resolved", {}, ("Atlantis",)),
        ],
        ids=["ambiguous-and-too-large", "ambiguous-and-not-found", "too-large-and-not-found"],
    )
    def test_mixed_entity_refusals_elect_one_reason_and_name_every_entity(
        self, entities, reason, ambiguous, unresolved
    ):
        """`NO_DATA_RESOLUTION_ORDER` elects the reason over the ones the names mint, and
        the refusal names an ambiguous name with its candidates and every name that resolved
        to no single node as unresolved. A group past the cap resolved, so it is never among
        the unresolved."""
        group = Group(members=self._members(settings().max_child_places + 1), entity=None)
        result = _run(
            entities=entities,
            pre_resolved={"G7": group, "Congo": self._CONGO},
            graph=_gdp_graph({}),
        )
        assert result.status == "no_data"
        assert result.no_data_reason == reason
        assert {
            entity.text: [candidate.dcid for candidate in entity.candidates]
            for entity in result.ambiguous_entities
        } == ambiguous
        assert result.unresolved_entity_names == unresolved


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
