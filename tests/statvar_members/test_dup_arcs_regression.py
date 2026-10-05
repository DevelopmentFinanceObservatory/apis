"""Regression for #698. Identical provenance arcs collapse. Distinct values remain multivalued.

Raw wire arcs pass through `materialise.graph.parse_arcs` before the echo-style reader returns them.
"""

from __future__ import annotations

import pytest

from materialise.graph import UNRESOLVED_NODE, parse_arcs
from statvar_members.enumerate import MultiValuedConstraint, enumerate_members
from statvar_members.types import FiveTuple
from tests.materialise._dup_arcs_fixtures import doubled_wire_arcs

from .conftest import FakeMemberArcReader

_KENYA_FIVE_TUPLE = FiveTuple(
    population_type="Person", measured_property="lifeExpectancy", stat_type="measuredValue"
)

_LIFE_EXPECTANCY_FEMALE_VALUES: dict[str, tuple[str, ...]] = {
    "typeOf": ("StatisticalVariable",),
    "populationType": ("Person",),
    "measuredProperty": ("lifeExpectancy",),
    "statType": ("measuredValue",),
    "measurementQualifier": (),
    "measurementDenominator": (),
    "constraintProperties": ("gender",),
    "observationProperties": (),
    "gender": ("Female",),
}


def test_dual_provenance_described_arm_resolves_rather_than_refusing_at_filter_covered():
    # The parser collapses each property duplicated across two provenances before matching.
    parsed = parse_arcs(
        doubled_wire_arcs(_LIFE_EXPECTANCY_FEMALE_VALUES), tuple(_LIFE_EXPECTANCY_FEMALE_VALUES)
    )
    reader = FakeMemberArcReader(arcs={"LifeExpectancy_Person_Female": parsed})

    group = enumerate_members(
        five_tuple=_KENYA_FIVE_TUPLE,
        slots={"gender": ("value", "Female")},
        graph=reader,
    )

    assert [member.stat_var for member in group.members] == ["LifeExpectancy_Person_Female"]
    assert group.inconsistent_candidates == ()


def test_dual_provenance_genuinely_conflicting_constraint_values_still_raise():
    # Distinct constraint values survive parsing and raise `MultiValuedConstraint`.
    arcs = {
        "gender": {
            "nodes": [
                {"dcid": "Male", "provenanceId": "dc/base/India_LifeExpectancy"},
                {"dcid": "Female", "provenanceId": "dc/base/Mongolia_Health"},
            ]
        }
    }
    parsed = {**_LIFE_EXPECTANCY_FEMALE_VALUES, **parse_arcs(arcs, ["gender"])}
    reader = FakeMemberArcReader(arcs={"LifeExpectancy_Person_Ambiguous": parsed})

    with pytest.raises(MultiValuedConstraint) as excinfo:
        enumerate_members(
            five_tuple=_KENYA_FIVE_TUPLE,
            slots={"gender": ("value", "Male")},
            graph=reader,
        )

    assert excinfo.value.stat_var == "LifeExpectancy_Person_Ambiguous"
    assert excinfo.value.values == ("Male", "Female")


def test_two_unresolvable_provenances_on_an_unbound_slot_are_excluded_not_raised():
    # Duplicate unresolved targets collapse to one marker, avoiding a false multi-value error.
    # An unbound slot then excludes the candidate without raising.
    raw = {
        "nodes": [
            {"provenanceId": "dc/base/India_LifeExpectancy"},
            {"provenanceId": "dc/base/Schema"},
        ]
    }
    parsed_gender = parse_arcs({"gender": raw}, ["gender"], mark_unresolved=True)
    assert parsed_gender == {"gender": (UNRESOLVED_NODE,)}
    arcs = {**_LIFE_EXPECTANCY_FEMALE_VALUES, **parsed_gender}
    reader = FakeMemberArcReader(arcs={"LifeExpectancy_Person_UnresolvedGender": arcs})

    group = enumerate_members(
        five_tuple=_KENYA_FIVE_TUPLE,
        slots={"gender": ("unbound",)},
        graph=reader,
    )

    assert "LifeExpectancy_Person_UnresolvedGender" not in {
        member.stat_var for member in group.members
    }
    assert group.inconsistent_candidates == ()


def test_a_resolved_value_and_a_distinct_unresolvable_marker_still_raise():
    # A resolved value and an unresolved marker remain distinct and raise on a value-bound ask.
    raw = {
        "nodes": [
            {"dcid": "Male", "provenanceId": "dc/base/India_LifeExpectancy"},
            {"provenanceId": "dc/base/Schema"},
        ]
    }
    parsed_gender = parse_arcs({"gender": raw}, ["gender"], mark_unresolved=True)
    assert parsed_gender == {"gender": ("Male", UNRESOLVED_NODE)}
    arcs = {**_LIFE_EXPECTANCY_FEMALE_VALUES, **parsed_gender}
    reader = FakeMemberArcReader(arcs={"LifeExpectancy_Person_PartlyUnresolved": arcs})

    with pytest.raises(MultiValuedConstraint) as excinfo:
        enumerate_members(
            five_tuple=_KENYA_FIVE_TUPLE,
            slots={"gender": ("value", "Male")},
            graph=reader,
        )

    assert excinfo.value.values == ("Male", UNRESOLVED_NODE)


def test_a_member_unresolved_before_a_multivalued_constraint_is_excluded_not_raised():
    """The walk checks a member's constraints in their listed order and excludes it at the
    first value that does not resolve, so a later multi-valued constraint never raises and
    the rest of the family still resolves."""
    arcs = {
        "LifeExpectancy_Person_Female": _LIFE_EXPECTANCY_FEMALE_VALUES,
        "LifeExpectancy_Person_Malformed": {
            **_LIFE_EXPECTANCY_FEMALE_VALUES,
            "constraintProperties": ("age", "gender"),
            "age": (UNRESOLVED_NODE,),
            "gender": ("Female", "Male"),
        },
    }

    group = enumerate_members(
        five_tuple=_KENYA_FIVE_TUPLE,
        slots={"gender": ("unbound",), "age": ("unbound",)},
        graph=FakeMemberArcReader(arcs=arcs),
    )

    assert [member.stat_var for member in group.members] == ["LifeExpectancy_Person_Female"]
