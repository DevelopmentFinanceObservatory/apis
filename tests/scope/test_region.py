"""The region axis rule over the packaged code index: a summed recipient region offers its
residual buckets beside its members, a ranking offers them unavailable, and a switch round
trips to the base spec_id."""

from __future__ import annotations

import pytest

from code_index.scope_conventions import OPTION_LABELS, REGION_READINGS, RegionReading
from code_index.shared import get_index
from entity_resolution.shared import shared_membership_policy
from search.engine.spec_id import identity_spec_id
from search.models import BaseIdentity, EntityGroup, Operation, PartScope
from search.scope.build import build_part_scope
from search.scope.compile import Refusal
from search.scope.region import is_residual, published_residuals, without_residuals
from search.scope.tables import PartFacts, ScopeFacts
from tests.engine._fake_coverage import empty_coverage_snapshot
from tests.scope._axes import explicit, switch

_CRS = "provenance/OECDCreditorReportingSystem"
_ALL = "region:members_and_residuals"
_MEMBERS = "region:members_only"
_AFRICA_MEMBERS = ["country/GHA", "country/KEN", "country/TZA"]
_AFRICA_RESIDUALS = [
    "DACAfricaUnallocated",
    "DACEasternAfricaUnallocated",
    "DACMiddleAfricaUnallocated",
    "DACNorthOfSaharaUnallocated",
    "DACSouthOfSaharaUnallocated",
    "DACSouthernAfricaUnallocated",
    "DACWesternAfricaUnallocated",
]
_AFRICA = EntityGroup(ref="africa", label="Africa", entity_type="Continent")
_RANK = Operation(verb="rank", phrase="top recipients", over="destination")


def _identity(destination: list[str]) -> BaseIdentity:
    return BaseIdentity.model_validate(
        {
            "identity_version": "spec1",
            "shape": {
                "kind": "statvar_schema",
                "population_type": "FinancialFlow",
                "measured_property": "amount",
            },
            "slots": [
                {
                    "property": "crsPurpose",
                    "binding": {"state": "value", "ref": "CRSPurposeCode120"},
                }
            ],
            "entities": {"origin": ["country/FRA"], "destination": sorted(destination)},
            "requested_window": {"kind": "all"},
        }
    )


_SUMMED = _identity([*_AFRICA_MEMBERS, *_AFRICA_RESIDUALS])
_LISTED = _identity(_AFRICA_MEMBERS)


def _facts(
    *, operations: tuple[Operation, ...] = (), group: EntityGroup | None = _AFRICA
) -> ScopeFacts:
    return ScopeFacts(
        index=get_index(),
        coverage=empty_coverage_snapshot(),
        part=PartFacts(
            operations=operations,
            entity_groups={} if group is None else {"destination": group},
            datasets=(_CRS,),
        ),
    )


def _scope(base: BaseIdentity, entries=(), **facts) -> PartScope:
    scope = build_part_scope(base=base, entries=list(entries), records={}, facts=_facts(**facts))
    assert isinstance(scope, PartScope), scope
    return scope


def _region(scope: PartScope):
    (choice,) = [c for c in scope.choices if c.axis == "region"]
    return choice


def test_residuals_are_kept_out_of_member_lists_by_party_kind():
    index = get_index()

    assert all(is_residual(dcid, index) for dcid in _AFRICA_RESIDUALS)
    assert not any(is_residual(dcid, index) for dcid in (*_AFRICA_MEMBERS, "DACAfrica"))
    assert without_residuals(_SUMMED.entities["destination"], index) == _AFRICA_MEMBERS


@pytest.mark.parametrize("group", [_AFRICA, None], ids=["by-group", "by-residuals-held"])
def test_a_summed_region_holds_its_residuals_by_default(group):
    """A recipient role holding Africa's members and residuals takes the residual option by
    default, found by the role's group or, with none posted, by the residuals it holds."""
    scope = _scope(_SUMMED, group=group)

    choice = _region(scope)
    assert (choice.active, choice.provenance, choice.rule) == (_ALL, "default", "region_residuals")
    assert [(o.option_id, o.availability.status) for o in choice.options] == [
        (_ALL, "available"),
        (_MEMBERS, "available"),
    ]
    assert choice.options[0].binding.model_dump() == {
        "kind": "entities",
        "role": "destination",
        "add": _AFRICA_RESIDUALS,
    }
    assert scope.compiled.spec_id == scope.id == identity_spec_id(_SUMMED)


def test_a_switch_to_the_members_round_trips_to_the_base_spec_id():
    """Switching a summed region to its members drops the residuals; switching back from
    the same base returns the base's own spec_id."""
    base_id = identity_spec_id(_SUMMED)

    members = _scope(_SUMMED, [switch(_MEMBERS)])
    back = _scope(_SUMMED, [switch(_ALL)])

    assert members.compiled.identity.entities["destination"] == _AFRICA_MEMBERS
    assert members.compiled.spec_id == identity_spec_id(_LISTED)
    assert _region(members).provenance == "explicit"
    assert back.compiled.spec_id == base_id


def test_a_ranking_offers_the_residuals_unavailable():
    """A ranking over the recipients reads Africa's members alone. The residual option is
    unavailable, `residuals_need_summed_region`, and a switch to it refuses."""
    scope = _scope(_LISTED, operations=(_RANK,))
    refused = build_part_scope(
        base=_LISTED, entries=[switch(_ALL)], records={}, facts=_facts(operations=(_RANK,))
    )

    choice = _region(scope)
    assert (choice.active, choice.provenance) == (_MEMBERS, "default")
    residuals = choice.options[0].availability
    assert (residuals.status, residuals.code) == ("unavailable", "residuals_need_summed_region")
    assert scope.compiled.spec_id == identity_spec_id(_LISTED)
    assert refused == Refusal(
        code="requested_option_unavailable", entry=0, cause="residuals_need_summed_region"
    )


def test_members_asked_for_by_name_keep_them_and_can_switch_to_the_sum():
    """ "African countries" summed: the explicit members-only entry holds the members; a
    switch to the residuals from the same base adds those the served dataset publishes."""
    words = explicit(_MEMBERS)

    kept = _scope(_LISTED, [words])
    summed = _scope(_LISTED, [switch(_ALL)])

    assert (_region(kept).active, _region(kept).provenance) == (_MEMBERS, "explicit")
    assert kept.compiled.spec_id == identity_spec_id(_LISTED)
    assert summed.compiled.spec_id == identity_spec_id(_SUMMED)


@pytest.mark.parametrize(
    "destination",
    [["DACAfrica"], ["country/GHA"], list(_AFRICA_RESIDUALS)],
    ids=["the-total-itself", "one-country", "residuals-alone"],
)
def test_no_region_axis_where_the_role_reads_no_summed_members(destination):
    """A role binding the region's total, a lone country, or only residuals settles no
    region axis."""
    scope = _scope(_identity(destination), group=None)

    assert "region" not in [c.axis for c in scope.choices]


def test_a_resolver_group_finds_its_total_by_registry_id():
    """Asia (DAC), a registry group with no graph node, finds its total by group id."""
    asia = EntityGroup(label="Asia (DAC)", group_id="groups/OECD.DAC.Region.Asia")
    base = _identity(["country/IDN", "country/IND"])

    scope = _scope(base, group=asia)

    add = _region(scope).options[0].binding.model_dump()["add"]
    assert "DACAsiaUnallocated" in add
    assert all(is_residual(dcid, get_index()) for dcid in add)


# A graph region the OECD counts with other members reads both ways.


def _members(ref: str) -> list[str]:
    held = shared_membership_policy().members(ref).members
    assert isinstance(held, tuple), ref
    return sorted(held)


def _oecd_case(row: RegionReading) -> tuple[BaseIdentity, BaseIdentity, EntityGroup]:
    """The part summed as the OECD counts `row`'s region on the CRS, its group as the answer
    names it, and the part holding the graph region's members."""
    total = get_index().parties[row.total]
    assert total.totals is not None
    residuals = published_residuals(total, datasets=(_CRS,), index=get_index())
    summed = _identity([*_members(total.totals), *residuals])
    group = EntityGroup(label=row.name, group_id=total.totals)
    return summed, _identity(_members(row.region)), group


@pytest.mark.parametrize("row", REGION_READINGS, ids=[row.region for row in REGION_READINGS])
def test_a_region_the_oecd_counts_otherwise_offers_both_readings_and_round_trips(row):
    """Summed on the CRS, a graph region the OECD counts with other members holds the OECD's
    region by default, labelled with the OECD's member count. The other option is the graph
    region's member countries; a switch to it binds them and a switch back returns the base
    spec_id. A region whose members the two count alike differs by its residuals alone."""
    summed, listed, group = _oecd_case(row)
    total = get_index().parties[row.total]
    assert total.totals is not None

    scope = _scope(summed, group=group)
    choice = _region(scope)
    to_members = _scope(summed, [switch(_MEMBERS)], group=group)
    back = _scope(summed, [switch(_ALL)], group=group)

    assert (choice.active, choice.provenance) == (_ALL, "default")
    assert [o.label for o in choice.options] == [
        OPTION_LABELS.region_as_oecd_counts.format(
            region=row.name, count=len(_members(total.totals))
        ),
        OPTION_LABELS.region_place_members.format(
            region_of=f"{row.name}'s", count=len(_members(row.region))
        ),
    ]
    assert to_members.compiled.identity.entities["destination"] == _members(row.region)
    assert to_members.compiled.spec_id == identity_spec_id(listed)
    assert back.compiled.spec_id == identity_spec_id(summed)


def test_sub_saharan_africa_as_the_oecd_counts_it_adds_sudan_and_the_residuals():
    """The OECD's sub-Saharan Africa holds Sudan, which the graph region does not."""
    row = next(r for r in REGION_READINGS if r.region == "SubSaharanAfrica")
    summed, _listed, group = _oecd_case(row)

    choice = _region(_scope(summed, group=group))

    assert choice.options[0].binding.model_dump()["add"][0] == "country/SDN"
    assert choice.options[1].binding.model_dump()["add"] == []


@pytest.mark.parametrize("row", REGION_READINGS, ids=[row.region for row in REGION_READINGS])
def test_a_ranking_over_a_region_the_oecd_counts_otherwise_reads_the_graphs_members(row):
    """Ranked, the part holds the graph region's members, named by the graph node, and the
    OECD's region is unavailable."""
    _summed, listed, _group = _oecd_case(row)
    node = EntityGroup(ref=row.region, label=row.name)

    choice = _region(_scope(listed, operations=(_RANK,), group=node))

    assert (choice.active, choice.provenance) == (_MEMBERS, "default")
    assert choice.options[0].availability.code == "residuals_need_summed_region"


def test_a_region_the_oecd_counts_otherwise_has_no_axis_off_dac_data():
    """A dataset counting no DAC region leaves the graph region as it is: no region axis."""
    row = next(r for r in REGION_READINGS if r.region == "SubSaharanAfrica")
    _summed, listed, _group = _oecd_case(row)
    facts = ScopeFacts(
        index=get_index(),
        coverage=empty_coverage_snapshot(),
        part=PartFacts(
            entity_groups={"destination": EntityGroup(ref=row.region, label=row.name)},
            datasets=("provenance/WorldBankWDI",),
        ),
    )

    scope = build_part_scope(base=listed, entries=[], records={}, facts=facts)

    assert isinstance(scope, PartScope)
    assert "region" not in [c.axis for c in scope.choices]


def test_every_region_reading_names_a_summing_dac_total_and_a_graph_region():
    """Each declared reading pairs a graph region the membership policy holds with a DAC
    total that sums the registry group it holds (`TOTAL_GROUPS`), and no graph region
    already keys a total of its own."""
    index = get_index()
    policy = shared_membership_policy()
    for row in REGION_READINGS:
        total = index.parties[row.total]
        assert total.sums_members and total.residuals and total.totals is not None, row
        assert total.totals in policy and policy.get(total.totals).kind == "group", row
        assert policy.holds_region(row.region), row
        assert row.region not in index.totals_by_group, row
