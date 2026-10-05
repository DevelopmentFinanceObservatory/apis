"""Pins agreement of `declared_roles` among members that survive antichain reduction.

`datasets_api.adapter` builds each shape from the enumerated members' roles without re-reading,
so this check inside enumeration is the only place a role-drifted group is refused.
"""

from __future__ import annotations

import pytest

from statvar_members.enumerate import MemberRoleDisagreement, enumerate_members
from statvar_members.types import FiveTuple

from .conftest import FakeMemberArcReader

_FIVE_TUPLE = FiveTuple(
    population_type="Person", measured_property="count", stat_type="measuredValue"
)
_UNBOUND = ("unbound",)


def _member_arcs(*, gender: str, roles: tuple[str, ...]) -> dict[str, tuple[str, ...]]:
    return {
        "typeOf": ("StatisticalVariable",),
        "populationType": ("Person",),
        "measuredProperty": ("count",),
        "statType": ("measuredValue",),
        "measurementQualifier": (),
        "measurementDenominator": (),
        "constraintProperties": ("gender",),
        "observationProperties": roles,
        "gender": (gender,),
    }


def test_a_single_surviving_member_trivially_agrees_with_itself():
    """One member (or an empty group) has nothing to disagree with --
    `_check_role_agreement` never raises regardless of what it declares."""
    reader = FakeMemberArcReader(
        arcs={"Count_Person_Male": _member_arcs(gender="Male", roles=("recipient",))}
    )

    group = enumerate_members(
        five_tuple=_FIVE_TUPLE, slots={"gender": ("value", "Male")}, graph=reader
    )

    assert [member.stat_var for member in group.members] == ["Count_Person_Male"]


def test_members_agreeing_on_declared_roles_return_the_group_unrefused():
    """Two members surviving the antichain reduction with the same
    `declared_roles` set -- this is the graph state every described-arm
    mint depends on `_check_role_agreement` to confirm before
    `datasets_api.adapter.resolve_definition_members` ever turns each
    member into a `StatVarShape` straight from its own enumerated fields."""
    reader = FakeMemberArcReader(
        arcs={
            "Count_Person_Male": _member_arcs(gender="Male", roles=("recipient",)),
            "Count_Person_Female": _member_arcs(gender="Female", roles=("recipient",)),
        }
    )

    group = enumerate_members(five_tuple=_FIVE_TUPLE, slots={"gender": _UNBOUND}, graph=reader)

    assert {member.stat_var for member in group.members} == {
        "Count_Person_Male",
        "Count_Person_Female",
    }
    assert {member.declared_roles for member in group.members} == {("recipient",)}


def test_members_disagreeing_on_declared_roles_refuse_the_whole_group():
    """A `MemberGroup` is only ever all-agree or an exception, with no
    partial result in between: `_check_role_agreement` refuses the whole
    group rather than returning it with the disagreeing member dropped."""
    reader = FakeMemberArcReader(
        arcs={
            "Count_Person_Male": _member_arcs(gender="Male", roles=("recipient",)),
            "Count_Person_Female": _member_arcs(gender="Female", roles=("provider",)),
        }
    )

    with pytest.raises(MemberRoleDisagreement) as exc_info:
        enumerate_members(five_tuple=_FIVE_TUPLE, slots={"gender": _UNBOUND}, graph=reader)

    assert exc_info.value.roles_by_stat_var == {
        "Count_Person_Male": ("recipient",),
        "Count_Person_Female": ("provider",),
    }
