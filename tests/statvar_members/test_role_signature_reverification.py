"""Pins recovery of dropped `observationProperties` during antichain reverification."""

from __future__ import annotations

import pytest

from statvar_members.enumerate import MemberRoleDisagreement, enumerate_members
from statvar_members.types import FiveTuple

from .conftest import FlakyMemberArcReader

_FIVE_TUPLE = FiveTuple(
    population_type="Person", measured_property="count", stat_type="measuredValue"
)
_UNBOUND = ("unbound",)
_SLOTS = {"gender": _UNBOUND, "age": _UNBOUND}

_ARCS: dict[str, dict[str, tuple[str, ...]]] = {
    "Count_Person_Gendered": {
        "typeOf": ("StatisticalVariable",),
        "populationType": ("Person",),
        "measuredProperty": ("count",),
        "statType": ("measuredValue",),
        "measurementQualifier": (),
        "measurementDenominator": (),
        "constraintProperties": ("gender",),
        "observationProperties": ("unit",),
        "gender": ("Male",),
    },
    "Count_Person_Aged": {
        "typeOf": ("StatisticalVariable",),
        "populationType": ("Person",),
        "measuredProperty": ("count",),
        "statType": ("measuredValue",),
        "measurementQualifier": (),
        "measurementDenominator": (),
        "constraintProperties": ("age",),
        "observationProperties": ("unit",),
        "age": ("Years0",),
    },
}
"""The incomparable `{gender}` and `{age}` members both declare the `unit` role."""


def _reader(**kwargs) -> FlakyMemberArcReader:
    return FlakyMemberArcReader(arcs=_ARCS, **kwargs)


def test_dropped_role_on_the_reverify_reread_is_recovered_not_refused():
    """The second batch drops `Gendered`'s role arc.
    Re-probing restores `unit` and avoids a false role disagreement."""
    reader = _reader(drop_second_batch={"Count_Person_Gendered": "observationProperties"})

    group = enumerate_members(five_tuple=_FIVE_TUPLE, slots=_SLOTS, graph=reader)

    assert {member.stat_var: member.declared_roles for member in group.members} == {
        "Count_Person_Gendered": ("unit",),
        "Count_Person_Aged": ("unit",),
    }
    assert group.inconsistent_candidates == ()


_DISAGREEING_ARCS: dict[str, dict[str, tuple[str, ...]]] = {
    "Count_Person_Gendered": {
        **_ARCS["Count_Person_Gendered"],
        "observationProperties": ("unit",),
    },
    "Count_Person_Aged": {
        **_ARCS["Count_Person_Aged"],
        "observationProperties": ("consecutiveValue",),
    },
}
"""These members genuinely declare different roles, `unit` and `consecutiveValue`."""


def test_genuine_role_disagreement_still_raises():
    """A genuine role mismatch between surviving members still raises
    `MemberRoleDisagreement`."""
    reader = FlakyMemberArcReader(arcs=_DISAGREEING_ARCS)

    with pytest.raises(MemberRoleDisagreement):
        enumerate_members(five_tuple=_FIVE_TUPLE, slots=_SLOTS, graph=reader)


_ROLE_BUDGET_ARCS: dict[str, dict[str, tuple[str, ...]]] = {
    f"Count_Person_Bucket{i:02d}": {
        "typeOf": ("StatisticalVariable",),
        "populationType": ("Person",),
        "measuredProperty": ("count",),
        "statType": ("measuredValue",),
        "measurementQualifier": (),
        "measurementDenominator": (),
        "constraintProperties": ("bucket",),
        "observationProperties": ("unit",),
        "bucket": (f"Bucket{i:02d}",),
    }
    for i in range(40)
}
"""All 40 distinct bucket members survive reduction and declare `unit`."""


def test_role_reverification_respects_the_shared_reprobe_budget():
    """Of 40 candidates, 32 fit the re-probe budget.
    Their second batch drops `observationProperties`, so single-dcid probes restore `unit`.
    The other 8 are inconsistent."""
    reader = FlakyMemberArcReader(
        arcs=_ROLE_BUDGET_ARCS,
        drop_second_batch=dict.fromkeys(_ROLE_BUDGET_ARCS, "observationProperties"),
    )

    group = enumerate_members(
        five_tuple=_FIVE_TUPLE, slots={"bucket": _UNBOUND}, graph=reader, member_cap=40
    )

    assert len(group.members) == 32
    assert all(member.declared_roles == ("unit",) for member in group.members)
    assert len(group.inconsistent_candidates) == 8
    # One initial batch call, one batched re-verify round covering the 32
    # budgeted winners, and one individual fallback call per one of those
    # 32 -- every budgeted winner's role-signature check disagrees with the
    # batched re-read, unlike the constraint-only budget test where a
    # recovered (grown) constraint set never counts as a shrink.
    assert reader.calls["node_arcs_many"] == 1 + 1 + 32


def _root_twin_arcs(roles: tuple[str, ...]) -> dict[str, tuple[str, ...]]:
    return {
        **_ROLE_BUDGET_ARCS["Count_Person_Bucket00"],
        "constraintProperties": (),
        "observationProperties": roles,
    }


_ROOT_TWIN_ARCS: dict[str, dict[str, tuple[str, ...]]] = {
    **{f"Count_Person_Root{i:02d}": _root_twin_arcs(("recipient",)) for i in range(39)},
    "Count_Person_RootZ": _root_twin_arcs(("target",)),
}
"""Forty same-keyset roots that each place the destination. The last in antichain order
declares a different role, so a stable read refuses the group with `MemberRoleDisagreement`."""


def test_a_root_twin_left_unverified_by_the_budget_is_read_alone():
    """The re-verify batch drops every root's `typeOf`, so all 40 need a single-dcid
    fallback. The budget covers 32, and `Count_Person_RootZ` is among the 8 left unverified.
    The 32 confirmed roots agree, and the root-twin read finds the one that does not."""
    reader = FlakyMemberArcReader(
        arcs=_ROOT_TWIN_ARCS, drop_second_batch=dict.fromkeys(_ROOT_TWIN_ARCS, "typeOf")
    )

    with pytest.raises(MemberRoleDisagreement) as exc_info:
        enumerate_members(
            five_tuple=_FIVE_TUPLE,
            slots={},
            graph=reader,
            member_cap=40,
            role_keys=("destination",),
        )

    assert len(exc_info.value.roles_by_stat_var) == 40
    assert exc_info.value.roles_by_stat_var["Count_Person_RootZ"] == ("target",)


def test_a_stable_read_of_the_same_roots_refuses_the_role_disagreement():
    reader = FlakyMemberArcReader(arcs=_ROOT_TWIN_ARCS)

    with pytest.raises(MemberRoleDisagreement):
        enumerate_members(
            five_tuple=_FIVE_TUPLE,
            slots={},
            graph=reader,
            member_cap=40,
            role_keys=("destination",),
        )
