"""Tests how unresolved markers differ from absent arcs in member enumeration.

A concrete-dcid filter excludes candidates before enumeration, so its case calls
`_matches_five_tuple` directly. A denominator target typed exactly `ProvisionalNode`
reads as unresolved, as Search reads it; `PerCapita` is such a target on the graph.
"""

from __future__ import annotations

import pytest

from statvar_members.enumerate import MemberRoleDisagreement, enumerate_members
from statvar_members.types import UNRESOLVED_NODE, FiveTuple

from .conftest import FakeMemberArcReader, FlakyMemberArcReader

_UNBOUND = ("unbound",)


def _arcs(**overrides: tuple[str, ...]) -> dict[str, tuple[str, ...]]:
    base: dict[str, tuple[str, ...]] = {
        "typeOf": ("StatisticalVariable",),
        "populationType": (),
        "measuredProperty": ("amount",),
        "statType": (),
        "measurementQualifier": (),
        "measurementDenominator": (),
        "constraintProperties": (),
        "observationProperties": (),
    }
    base.update(overrides)
    return base


def test_denominator_unresolved_ask_admits_a_marker_denominator_member():
    reader = FakeMemberArcReader(
        arcs={"Amount_WithDenom": _arcs(measurementDenominator=(UNRESOLVED_NODE,))}
    )

    group = enumerate_members(
        five_tuple=FiveTuple(measured_property="amount", denominator_unresolved=True),
        slots={},
        graph=reader,
    )

    assert [member.stat_var for member in group.members] == ["Amount_WithDenom"]


def test_denominator_unresolved_ask_rejects_a_resolved_dcid_denominator_member():
    reader = FakeMemberArcReader(
        arcs={"Amount_WithDenom": _arcs(measurementDenominator=("Count_Person",))}
    )

    group = enumerate_members(
        five_tuple=FiveTuple(measured_property="amount", denominator_unresolved=True),
        slots={},
        graph=reader,
    )

    assert group.members == ()


def test_denominator_unresolved_ask_rejects_a_no_arc_member():
    reader = FakeMemberArcReader(arcs={"Amount_NoDenom": _arcs()})

    group = enumerate_members(
        five_tuple=FiveTuple(measured_property="amount", denominator_unresolved=True),
        slots={},
        graph=reader,
    )

    assert group.members == ()


def test_no_denominator_ask_rejects_a_marker_denominator_member():
    reader = FakeMemberArcReader(
        arcs={"Amount_WithDenom": _arcs(measurementDenominator=(UNRESOLVED_NODE,))}
    )

    group = enumerate_members(
        five_tuple=FiveTuple(measured_property="amount"), slots={}, graph=reader
    )

    assert group.members == ()


def test_marker_constraint_value_under_an_unbound_slot_rejects_the_candidate():
    """A marker value has no ref form for `Member.constraints`, so an `unbound` slot excludes it."""
    reader = FakeMemberArcReader(
        arcs={"Amount_Gendered": _arcs(constraintProperties=("gender",), gender=(UNRESOLVED_NODE,))}
    )

    group = enumerate_members(
        five_tuple=FiveTuple(measured_property="amount"),
        slots={"gender": _UNBOUND},
        graph=reader,
    )

    assert group.members == ()


_PER_CAPITA_NODE = {"typeOf": ("ProvisionalNode",)}
_UNRESOLVED_ASK = FiveTuple(measured_property="amount", denominator_unresolved=True)


def test_denominator_unresolved_ask_admits_a_provisional_node_denominator_member():
    reader = FakeMemberArcReader(
        arcs={
            "Amount_PerCapita": _arcs(measurementDenominator=("PerCapita",)),
            "PerCapita": _PER_CAPITA_NODE,
        }
    )

    group = enumerate_members(five_tuple=_UNRESOLVED_ASK, slots={}, graph=reader)

    assert [member.stat_var for member in group.members] == ["Amount_PerCapita"]


def test_denominator_unresolved_ask_rejects_a_target_with_a_type_beside_provisional_node():
    reader = FakeMemberArcReader(
        arcs={
            "Amount_PerCapita": _arcs(measurementDenominator=("PerCapita",)),
            "PerCapita": {"typeOf": ("ProvisionalNode", "StatisticalVariable")},
        }
    )

    group = enumerate_members(five_tuple=_UNRESOLVED_ASK, slots={}, graph=reader)

    assert group.members == ()


def test_denominator_unresolved_ask_admits_marker_and_provisional_members_and_no_resolved_one():
    reader = FakeMemberArcReader(
        arcs={
            "Amount_PerCapita": _arcs(measurementDenominator=("PerCapita",)),
            "Amount_Marker": _arcs(measurementDenominator=(UNRESOLVED_NODE,)),
            "Amount_PerPerson": _arcs(measurementDenominator=("Count_Person",)),
            "PerCapita": _PER_CAPITA_NODE,
            "Count_Person": {"typeOf": ("StatisticalVariable",)},
        }
    )

    group = enumerate_members(five_tuple=_UNRESOLVED_ASK, slots={}, graph=reader)

    assert [member.stat_var for member in group.members] == ["Amount_Marker", "Amount_PerCapita"]


@pytest.mark.parametrize(
    "five_tuple",
    [
        pytest.param(FiveTuple(measured_property="amount"), id="no-denominator-ask"),
        pytest.param(
            FiveTuple(measured_property="amount", measurement_denominator="Count_Person"),
            id="concrete-ask",
        ),
    ],
)
def test_a_provisional_node_denominator_is_not_a_member_of_a_resolved_ask(five_tuple):
    reader = FakeMemberArcReader(
        arcs={
            "Amount_PerCapita": _arcs(measurementDenominator=("PerCapita",)),
            "PerCapita": _PER_CAPITA_NODE,
        }
    )

    group = enumerate_members(five_tuple=five_tuple, slots={}, graph=reader)

    assert group.members == ()


def test_a_concrete_ask_naming_a_provisional_node_still_matches_its_dcid():
    """Only an unresolved ask reads target types, so a stored `ref: PerCapita` keeps its members."""
    reader = FakeMemberArcReader(
        arcs={
            "Amount_PerCapita": _arcs(measurementDenominator=("PerCapita",)),
            "PerCapita": _PER_CAPITA_NODE,
        }
    )

    group = enumerate_members(
        five_tuple=FiveTuple(measured_property="amount", measurement_denominator="PerCapita"),
        slots={},
        graph=reader,
    )

    assert [member.stat_var for member in group.members] == ["Amount_PerCapita"]


def test_a_target_type_dropped_from_a_batch_read_is_read_again_alone():
    reader = FlakyMemberArcReader(
        arcs={
            "Amount_PerCapita": _arcs(measurementDenominator=("PerCapita",)),
            "Amount_PerPerson": _arcs(measurementDenominator=("Count_Person",)),
            "PerCapita": _PER_CAPITA_NODE,
            "Count_Person": {"typeOf": ("StatisticalVariable",)},
        },
        drop_first_batch={"PerCapita": "typeOf"},
    )

    group = enumerate_members(five_tuple=_UNRESOLVED_ASK, slots={}, graph=reader)

    assert [member.stat_var for member in group.members] == ["Amount_PerCapita"]


def _per_capita_family(no_denominator_siblings: int) -> dict[str, dict[str, tuple[str, ...]]]:
    """A root per-capita member beside siblings whose denominator arc reads empty. Each
    sibling is an absence suspect, and those past the 32-dcid re-probe budget stay unread."""
    arcs: dict[str, dict[str, tuple[str, ...]]] = {
        "Amount_GDP_PerCapita": _arcs(
            measurementDenominator=("PerCapita",),
            constraintProperties=("activitySource",),
            activitySource=("GrossDomesticProduction",),
        ),
        "PerCapita": _PER_CAPITA_NODE,
    }
    for i in range(no_denominator_siblings):
        arcs[f"Amount_GDP_Sector{i:02d}"] = _arcs(
            constraintProperties=("activitySource", "sector"),
            activitySource=("GrossDomesticProduction",),
            sector=(f"Sector{i:02d}",),
        )
    return arcs


def test_a_confirmed_root_leaves_unread_candidates_unable_to_change_members():
    reader = FakeMemberArcReader(arcs=_per_capita_family(40))

    group = enumerate_members(
        five_tuple=_UNRESOLVED_ASK,
        slots={"activitySource": ("value", "GrossDomesticProduction"), "sector": ("unbound",)},
        graph=reader,
    )

    assert [member.stat_var for member in group.members] == ["Amount_GDP_PerCapita"]
    assert len(group.inconsistent_candidates) == 8
    assert group.root_confirmed is True
    assert group.undismissed_candidates == ()


def test_without_a_root_unread_candidates_leave_the_group_unconfirmed():
    arcs = _per_capita_family(40)
    del arcs["Amount_GDP_PerCapita"]
    reader = FakeMemberArcReader(arcs=arcs)

    group = enumerate_members(
        five_tuple=_UNRESOLVED_ASK,
        slots={"activitySource": ("value", "GrossDomesticProduction"), "sector": ("unbound",)},
        graph=reader,
    )

    assert group.members == ()
    assert len(group.inconsistent_candidates) == 8
    assert group.root_confirmed is False
    assert group.undismissed_candidates == group.inconsistent_candidates


def test_a_confirmed_root_still_checks_role_agreement_beside_unread_candidates():
    arcs = _per_capita_family(40)
    arcs["Amount_GDP_PerCapita"] = {
        **arcs["Amount_GDP_PerCapita"],
        "observationProperties": ("recipient",),
    }
    arcs["Amount_GDP_PerCapita_Twin"] = {
        **arcs["Amount_GDP_PerCapita"],
        "observationProperties": ("target",),
    }
    reader = FakeMemberArcReader(arcs=arcs)

    # Both roots place the destination with a different role, so the election cannot order
    # them.
    with pytest.raises(MemberRoleDisagreement):
        enumerate_members(
            five_tuple=_UNRESOLVED_ASK,
            slots={"activitySource": ("value", "GrossDomesticProduction"), "sector": ("unbound",)},
            graph=reader,
            role_keys=("destination",),
        )
