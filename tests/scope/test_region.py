"""The region axis rule over the packaged code index: a summed recipient region offers its
residual buckets beside its members, a ranking offers them unavailable, and a switch round
trips to the base spec_id."""

from __future__ import annotations

import pytest

from code_index.shared import get_index
from search.engine.spec_id import identity_spec_id
from search.models import BaseIdentity, EntityGroup, Operation, PartScope
from search.scope.build import build_part_scope
from search.scope.compile import Refusal
from search.scope.region import is_residual, without_residuals
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

    assert [c.axis for c in scope.choices] == []


def test_a_resolver_group_finds_its_total_by_registry_id():
    """Asia (DAC), a registry group with no graph node, finds its total by group id."""
    asia = EntityGroup(label="Asia (DAC)", group_id="groups/OECD.DAC.Region.Asia")
    base = _identity(["country/IDN", "country/IND"])

    scope = _scope(base, group=asia)

    add = _region(scope).options[0].binding.model_dump()["add"]
    assert "DACAsiaUnallocated" in add
    assert all(is_residual(dcid, get_index()) for dcid in add)
