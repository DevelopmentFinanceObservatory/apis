"""End-to-end tests for a recipient region read as one sum.

France's education aid to Africa goes through the parsed door. Bound to a sector, it is
counted in gross CRS disbursements (`code_index.scope_conventions`), elects the CRS's
member, and finds no Africa total there (DAC2a publishes it), so Africa binds its members.
Read as one sum it binds the seven residual buckets DAC's Africa total counts beside them,
which the CRS publishes as recipients, and states the region choice. A ranking and a
request for "African countries" bind the members alone. France's ODA to Africa with no
sector is a recipient's total, counted on net DAC2a receipts, which publish Africa's
total, so it reads that total and settles no region axis. The graph holds Africa, its
members and its residuals, as the dev graph does; coverage holds France's CRS series to
every member and its DAC2a net series to Africa's total.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from code_index.scope_conventions import REGION_READINGS
from code_index.shared import get_index
from entity_resolution.shared import shared_membership_policy
from search.engine.departures import spec_departure_facts
from search.engine.entry import dataset_resubmit
from search.engine.region import scope_decision
from search.models import (
    ChoiceOrigin,
    DatasetResubmitInput,
    DefinitePart,
    Entry,
    ParsedInput,
    ResolveRequest,
    ResubmitIdentityEntry,
    Spec,
)
from series_coverage.schema import FacetMetadata, RoleNames
from tests.code_index.aid_identity import CRS_PROVENANCE
from tests.engine._fake_coverage import patch_coverage
from tests.engine._fake_sdmx import FakeSdmxClient
from tests.engine._harness import offline_resolve
from tests.engine.test_aid_dataset_bind_e2e import _CRS_MEMBER, _DAC2A, _ScriptedLLM
from tests.engine.test_whole_total_bind_e2e import _FRA, _VARIABLE, _graph, _parsed
from tests.fixtures import FakeGraph

# The CRS member binds CRS 11330, under education (110), which "education" names.
_SECTOR_VARIABLE = "education aid"
# DAC2a's net ODA receipts, the convention's basis for a recipient's total.
_DAC2A_NET = next(
    row.dcid
    for row in get_index().statvars
    if row.provenances == (_DAC2A,)
    and (row.population_type, row.measured_property) == ("FinancialFlow", "amount")
    and row.measurement_denominator is None
    and row.constraint_properties.get("dac2aAidType") == "DAC2aAidType206"
    and row.constraint_properties.get("nettingBasis") == "Net"
)
_DAC_AFRICA = get_index().totals_by_group["africa"].dcid
_AFRICA_MEMBERS = shared_membership_policy().members("africa").members
_AFRICA_RESIDUALS = get_index().totals_by_group["africa"].residuals
_ALL = "region:members_and_residuals"
_MEMBERS = "region:members_only"
_FROM_FRANCE = {"role": "from", "text": "France"}


@pytest.fixture(autouse=True)
def _coverage(monkeypatch: pytest.MonkeyPatch) -> None:
    years = [2019, 2020, 2021]
    patch_coverage(
        monkeypatch,
        [
            *((_CRS_MEMBER, "crs", _FRA, member, years) for member in _AFRICA_MEMBERS),
            (_DAC2A_NET, "dac2a", _FRA, _DAC_AFRICA, years),
        ],
        role_names=dict.fromkeys(
            (_CRS_MEMBER, _DAC2A_NET),
            RoleNames(donor="financingProvider", recipient="financingRecipient"),
        ),
        facet_metadata={
            "crs": FacetMetadata(provenance_id=CRS_PROVENANCE, unit="USD"),
            "dac2a": FacetMetadata(provenance_id=_DAC2A, unit="USD"),
        },
    )


def _africa_graph(variable: str = _SECTOR_VARIABLE, member: str = _CRS_MEMBER) -> FakeGraph:
    graph = _graph(variable=variable, members=(member,))
    nodes = graph._nodes
    nodes["africa"] = {"label": "Africa", "type": "Continent"}
    for member in _AFRICA_MEMBERS:
        nodes.setdefault(member, {"label": member, "type": "Country"})
    for residual in _AFRICA_RESIDUALS:
        nodes[residual] = {"label": get_index().parties[residual].label, "type": "Organization"}
    graph._resolve["Africa"] = "africa"
    return graph


def _run(
    request: ResolveRequest, variable: str = _SECTOR_VARIABLE, member: str = _CRS_MEMBER
) -> DefinitePart:
    response = offline_resolve(
        request,
        graph=_africa_graph(variable, member),
        llm=_ScriptedLLM(variable),
        sdmx=FakeSdmxClient(observations={}, availability={}),
    )
    (part,) = response.parts
    assert isinstance(part, DefinitePart), part
    return part


def _ask(africa: dict[str, Any], *operations: dict[str, str]) -> DefinitePart:
    return _run(
        _parsed(
            {
                "variable": _SECTOR_VARIABLE,
                "entities": [_FROM_FRANCE, {"role": "to", "text": "Africa", **africa}],
                "operations": list(operations),
                "time": "2021",
            }
        )
    )


def _destination(spec: Spec) -> list[str]:
    return spec.definition.identity.entities["destination"]


def _region(spec: Spec) -> tuple[str, str, str | None]:
    """The region choice's active option, provenance and the residual option's code."""
    scope = spec.resolved.scope_choices
    assert scope is not None
    (choice,) = [c for c in scope.choices if c.axis == "region"]
    residuals = next(o for o in choice.options if o.option_id == _ALL)
    return choice.active, choice.provenance, residuals.availability.code


@pytest.mark.parametrize(
    "operations",
    [(), ({"phrase": "total aid", "verb": "aggregate", "over": "destination"},)],
    ids=["no-operation", "aggregate-over-the-recipients"],
)
def test_a_summed_region_binds_its_residuals_and_states_the_choice(operations):
    part = _ask({}, *operations)

    spec = part.spec
    assert _destination(spec) == sorted((*_AFRICA_MEMBERS, *_AFRICA_RESIDUALS))
    assert _region(spec) == (_ALL, "default", None)
    assert spec.resolved.entity_groups is not None
    assert spec.resolved.entity_groups["destination"].ref == "africa"
    # The residuals hold no series and depart as no member; the group counts its members.
    codes = [d.code for d in part.departures]
    assert "group_read_as_members" in codes
    assert "entity_unavailable" not in codes
    (fact,) = spec_departure_facts(spec, operations=())
    assert (fact.group_size, fact.excluded, len(fact.included)) == (
        len(_AFRICA_MEMBERS),
        (),
        len(_AFRICA_RESIDUALS),
    )


@pytest.mark.parametrize(
    ("africa", "operation", "region"),
    [
        pytest.param(
            {},
            {"phrase": "top recipients", "verb": "rank"},
            (_MEMBERS, "default", "residuals_need_summed_region"),
            id="a-ranking",
        ),
        pytest.param(
            {"members": True},
            {"phrase": "which African countries received the most", "verb": "rank"},
            (_MEMBERS, "explicit", "residuals_need_summed_region"),
            id="a-ranking-of-african-countries",
        ),
        pytest.param(
            {"members": True},
            {"phrase": "total aid", "verb": "aggregate"},
            (_MEMBERS, "explicit", None),
            id="african-countries-summed",
        ),
    ],
)
def test_a_ranking_or_a_members_ask_binds_the_members_alone(africa, operation, region):
    part = _ask(africa, {**operation, "over": "destination"})

    spec = part.spec
    assert _destination(spec) == sorted(_AFRICA_MEMBERS)
    assert _region(spec) == region
    entries = spec.resolved.scope_choices.entries if spec.resolved.scope_choices else []
    assert [(e.option_id, e.origin.kind) for e in entries if e.axis == "region"] == (
        [(_MEMBERS, "text")] if region[1] == "explicit" else []
    )


def test_a_carried_members_only_entry_reaches_the_region_binder():
    """A members-only choice the conversation settled earlier binds a summed region's
    members alone on a new part, and the scope states it as carried."""
    carried = Entry(
        axis="region",
        option_id=_MEMBERS,
        provenance="carried",
        origin=ChoiceOrigin(choice_id="earlier:region"),
    )
    request = _parsed(
        {
            "variable": _SECTOR_VARIABLE,
            "entities": [_FROM_FRANCE, {"role": "to", "text": "Africa"}],
        }
    )
    assert isinstance(request.input, ParsedInput)
    request.input.carried_entries = [carried]

    spec = _run(request).spec

    assert _destination(spec) == sorted(_AFRICA_MEMBERS)
    assert _region(spec) == (_MEMBERS, "carried", None)
    assert spec.resolved.scope_choices is not None
    assert [e for e in spec.resolved.scope_choices.entries if e.axis == "region"] == [carried]


def _without_left_out(spec: Spec) -> dict[str, dict[str, object]]:
    """`spec`'s entity groups by role, each less what its members leave out of its total."""
    return {
        role: group.model_dump(exclude={"left_out", "sum_differs"})
        for role, group in (spec.resolved.entity_groups or {}).items()
    }


def test_a_switch_round_trips_through_dataset_resubmit():
    """Switching the summed region to its members and back, each posted over the scope's
    base identity with its entries, returns the first answer's spec_id."""
    first = _ask({}).spec
    scope = first.resolved.scope_choices
    assert scope is not None

    def switched(option: str) -> Spec:
        switch = Entry(
            axis="region",
            option_id=option,
            provenance="explicit",
            origin=ChoiceOrigin(choice_id=f"{scope.id}:region"),
        )
        entry = ResubmitIdentityEntry(
            variable_text=_SECTOR_VARIABLE,
            identity=scope.base.identity,
            entity_groups=first.resolved.entity_groups,
            scope_entries=[*(e for e in scope.entries if e.axis != "region"), switch],
        )
        return _run(ResolveRequest(input=DatasetResubmitInput(entries=[entry]))).spec

    members = switched(_MEMBERS)
    back = switched(_ALL)

    assert _destination(members) == sorted(_AFRICA_MEMBERS)
    assert _region(members)[:2] == (_MEMBERS, "explicit")
    assert members.spec_id != first.spec_id
    assert back.spec_id == first.spec_id
    # The members alone leave the residuals out (`EntityGroup.left_out`), which the summed
    # answer binds, so the groups agree on all but the fields derived from that.
    assert _without_left_out(back) == _without_left_out(members)
    assert back.resolved.entity_groups == first.resolved.entity_groups
    assert _region(back)[:2] == (_ALL, "explicit")


def test_oda_to_africa_with_no_sector_reads_the_dac2a_total_on_net():
    """A recipient's total over every sector is counted on net DAC2a receipts, which
    publish Africa's total, so the part reads the total itself and settles no region
    axis."""
    part = _run(
        _parsed(
            {
                "variable": _VARIABLE,
                "entities": [_FROM_FRANCE, {"role": "to", "text": "Africa"}],
                "time": "2021",
            }
        ),
        variable=_VARIABLE,
        member=_DAC2A_NET,
    )

    spec = part.spec
    assert _destination(spec) == [_DAC_AFRICA]
    netting = {s.property: s.binding for s in spec.definition.identity.slots}["nettingBasis"]
    assert netting.model_dump() == {"state": "value", "ref": "Net"}
    scope = spec.resolved.scope_choices
    assert scope is not None
    assert "region" not in [c.axis for c in scope.choices]
    (basis,) = (c for c in scope.choices if c.axis == "measure_basis")
    assert (basis.active, basis.provenance) == ("measure_basis:net", "default")


_SSA = "SubSaharanAfrica"
_SSA_ROW = next(row for row in REGION_READINGS if row.region == _SSA)
_DAC_SSA = get_index().parties[_SSA_ROW.total]
assert _DAC_SSA.totals is not None
_OECD_SSA_MEMBERS = shared_membership_policy().members(_DAC_SSA.totals).members
_GRAPH_SSA_MEMBERS = shared_membership_policy().members(_SSA).members


def test_a_switch_between_a_regions_two_readings_names_the_group_each_reads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Sub-Saharan Africa summed on the CRS reads as the OECD counts it, named by the
    registry group. A switch to the graph region's members names the graph region, and a
    switch back names the registry group again, so each answer states its group and the
    members it was read as."""
    members = sorted({*_OECD_SSA_MEMBERS, *_GRAPH_SSA_MEMBERS})
    patch_coverage(
        monkeypatch,
        [(_CRS_MEMBER, "crs", _FRA, member, [2019, 2020, 2021]) for member in members],
        role_names={
            _CRS_MEMBER: RoleNames(donor="financingProvider", recipient="financingRecipient")
        },
        facet_metadata={"crs": FacetMetadata(provenance_id=CRS_PROVENANCE, unit="USD")},
    )
    graph = _africa_graph()
    graph._nodes[_SSA] = {"label": "Sub-Saharan Africa", "type": "GeoRegion"}
    for member in members:
        graph._nodes.setdefault(member, {"label": member, "type": "Country"})
    for residual in _DAC_SSA.residuals:
        graph._nodes.setdefault(
            residual, {"label": get_index().parties[residual].label, "type": "Organization"}
        )
    graph._resolve["Sub-Saharan Africa"] = _SSA

    def run(request: ResolveRequest) -> DefinitePart:
        response = offline_resolve(
            request,
            graph=graph,
            llm=_ScriptedLLM(_SECTOR_VARIABLE),
            sdmx=FakeSdmxClient(observations={}, availability={}),
        )
        (part,) = response.parts
        assert isinstance(part, DefinitePart), part
        return part

    first = run(
        _parsed(
            {
                "variable": _SECTOR_VARIABLE,
                "entities": [_FROM_FRANCE, {"role": "to", "text": "Sub-Saharan Africa"}],
                "time": "2021",
            }
        )
    ).spec
    scope = first.resolved.scope_choices
    assert scope is not None
    assert first.resolved.entity_groups is not None
    assert first.resolved.entity_groups["destination"].group_id == _DAC_SSA.totals

    def switched(option: str) -> DefinitePart:
        switch = Entry(
            axis="region",
            option_id=option,
            provenance="explicit",
            origin=ChoiceOrigin(choice_id=f"{scope.id}:region"),
        )
        entry = ResubmitIdentityEntry(
            variable_text=_SECTOR_VARIABLE,
            identity=scope.base.identity,
            entity_groups=first.resolved.entity_groups,
            scope_entries=[*(e for e in scope.entries if e.axis != "region"), switch],
        )
        return run(ResolveRequest(input=DatasetResubmitInput(entries=[entry])))

    to_members = switched(_MEMBERS)
    groups = to_members.spec.resolved.entity_groups

    assert _destination(to_members.spec) == sorted(_GRAPH_SSA_MEMBERS)
    assert groups is not None
    assert groups["destination"].ref == _SSA
    assert "group_read_as_members" in [d.code for d in to_members.departures]

    back = switched(_ALL)

    assert back.spec.spec_id == first.spec_id
    assert back.spec.resolved.entity_groups == first.resolved.entity_groups


def test_the_scope_builds_off_the_event_loop(monkeypatch: pytest.MonkeyPatch) -> None:
    """A search and a switch build their scope in a worker thread, so a scope build's
    coverage reads never block the event loop."""
    on_loop: list[bool] = []

    def recording(build):
        def wrapper(**kwargs):
            try:
                asyncio.get_running_loop()
            except RuntimeError:
                on_loop.append(False)
            else:
                on_loop.append(True)
            return build(**kwargs)

        return wrapper

    for module in (scope_decision, dataset_resubmit):
        monkeypatch.setattr(module, "build_part_scope", recording(module.build_part_scope))

    first = _ask({}).spec
    scope = first.resolved.scope_choices
    assert scope is not None
    switch = Entry(
        axis="region",
        option_id=_MEMBERS,
        provenance="explicit",
        origin=ChoiceOrigin(choice_id=f"{scope.id}:region"),
    )
    _run(
        ResolveRequest(
            input=DatasetResubmitInput(
                entries=[
                    ResubmitIdentityEntry(
                        variable_text=_SECTOR_VARIABLE,
                        identity=scope.base.identity,
                        entity_groups=first.resolved.entity_groups,
                        scope_entries=[switch],
                    )
                ]
            )
        )
    )

    assert on_loop == [False, False]
