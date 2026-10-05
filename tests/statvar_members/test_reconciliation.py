"""Tests recovery when batch forward reads omit filter-covered properties."""

from __future__ import annotations

import pytest

from statvar_members.enumerate import enumerate_members
from statvar_members.types import FiveTuple

from .conftest import FlakyMemberArcReader

_FIVE_TUPLE = FiveTuple(
    population_type="Person", measured_property="lifeExpectancy", stat_type="measuredValue"
)
_COUNT_FIVE_TUPLE = FiveTuple(
    population_type="Person", measured_property="count", stat_type="measuredValue"
)
_UNBOUND = ("unbound",)

_ARCS: dict[str, dict[str, tuple[str, ...]]] = {
    "LifeExpectancy_Person": {
        "typeOf": ("StatisticalVariable",),
        "populationType": ("Person",),
        "measuredProperty": ("lifeExpectancy",),
        "statType": ("measuredValue",),
        "measurementQualifier": (),
        "measurementDenominator": (),
        "constraintProperties": (),
        "observationProperties": (),
    },
    "LifeExpectancy_Person_Male": {
        "typeOf": ("StatisticalVariable",),
        "populationType": ("Person",),
        "measuredProperty": ("lifeExpectancy",),
        "statType": ("measuredValue",),
        "measurementQualifier": (),
        "measurementDenominator": (),
        "constraintProperties": ("gender",),
        "observationProperties": (),
        "gender": ("Male",),
    },
    "LifeExpectancy_Person_Female": {
        "typeOf": ("StatisticalVariable",),
        "populationType": ("Person",),
        "measuredProperty": ("lifeExpectancy",),
        "statType": ("measuredValue",),
        "measurementQualifier": (),
        "measurementDenominator": (),
        "constraintProperties": ("gender",),
        "observationProperties": (),
        "gender": ("Female",),
    },
}


def _reader(**kwargs) -> FlakyMemberArcReader:
    return FlakyMemberArcReader(arcs=_ARCS, **kwargs)


@pytest.mark.parametrize(
    ("reader_kwargs", "expected_read_count"),
    [
        pytest.param({}, 2, id="no-drop"),
        pytest.param(
            {"drop_first_batch": {"LifeExpectancy_Person_Male": "statType"}},
            3,
            id="filter-covered-stat-type-drop",
        ),
    ],
)
def test_no_drop_or_filter_covered_drop_reconciles_members(reader_kwargs, expected_read_count: int):
    """Re-verification preserves membership after a recoverable batch drop."""
    reader = _reader(**reader_kwargs)

    group = enumerate_members(five_tuple=_FIVE_TUPLE, slots={"gender": _UNBOUND}, graph=reader)

    assert [member.stat_var for member in group.members] == ["LifeExpectancy_Person"]
    assert group.inconsistent_candidates == ()
    assert reader.calls["node_arcs_many"] == expected_read_count


def test_recoverable_drop_recovers_the_member_a_value_bound_slot_selects():
    """Recover the Male member when its batch response drops `statType`."""
    reader = _reader(drop_first_batch={"LifeExpectancy_Person_Male": "statType"})

    group = enumerate_members(
        five_tuple=_FIVE_TUPLE, slots={"gender": ("value", "Male")}, graph=reader
    )

    assert [member.stat_var for member in group.members] == ["LifeExpectancy_Person_Male"]
    assert group.inconsistent_candidates == ()


@pytest.mark.parametrize(
    ("slots", "degradations", "expected_stat_vars"),
    [
        pytest.param(
            {"gender": _UNBOUND},
            (
                {"drop_first_batch": {"LifeExpectancy_Person": "populationType"}},
                {"drop_first_batch": {"LifeExpectancy_Person_Male": "measuredProperty"}},
                {"drop_first_batch": {"LifeExpectancy_Person_Female": "statType"}},
            ),
            ["LifeExpectancy_Person"],
            id="filter_covered_property_drop",
        ),
        pytest.param(
            {"gender": ("set", ("Female",))},
            ({"empty_first_batch": ["LifeExpectancy_Person_Female"]},),
            ["LifeExpectancy_Person_Female"],
            id="whole_candidate_map_drop_on_necessary_member",
        ),
    ],
)
def test_filter_covered_batch_degradation_is_reconciled(slots, degradations, expected_stat_vars):
    """Reprobing restores a needed member after a property or whole-map batch drop."""
    reference = enumerate_members(five_tuple=_FIVE_TUPLE, slots=slots, graph=_reader())
    assert [member.stat_var for member in reference.members] == expected_stat_vars

    for degradation in degradations:
        group = enumerate_members(
            five_tuple=_FIVE_TUPLE,
            slots=slots,
            graph=_reader(**degradation),
        )

        assert [member.stat_var for member in group.members] == expected_stat_vars
        assert group.inconsistent_candidates == ()


def test_recoverable_drop_on_the_pushed_slot_filter_itself_is_reconciled():
    """A value-bound gender filter is pushed server-side, so a dropped property must be restored.

    Two Male candidates keep the request multi-dcid, triggering the fake's degradation.
    """
    arcs = {
        "LifeExpectancy_Person_Male": _ARCS["LifeExpectancy_Person_Male"],
        "LifeExpectancy_Person_Male_Other": _ARCS["LifeExpectancy_Person_Male"],
    }
    reader = FlakyMemberArcReader(
        arcs=arcs, drop_first_batch={"LifeExpectancy_Person_Male": "gender"}
    )

    group = enumerate_members(
        five_tuple=_FIVE_TUPLE, slots={"gender": ("value", "Male")}, graph=reader
    )

    assert {member.stat_var for member in group.members} == {
        "LifeExpectancy_Person_Male",
        "LifeExpectancy_Person_Male_Other",
    }
    assert group.inconsistent_candidates == ()


def test_candidate_still_contradicting_the_filter_after_reprobe_is_excluded_and_reported():
    """Contradictory reverse and forward reads exclude and report the candidate."""
    reader = _reader(drop_always={"LifeExpectancy_Person_Male": "statType"})

    group = enumerate_members(
        five_tuple=_FIVE_TUPLE, slots={"gender": ("value", "Male")}, graph=reader
    )

    assert group.members == ()
    assert group.inconsistent_candidates == ("LifeExpectancy_Person_Male",)


def test_candidate_with_an_entirely_empty_forward_read_is_reprobed_and_recovered():
    """The reverse query finds the member, but its batch forward read is empty.

    A batched re-probe restores the no-drop member set.
    """
    reader = _reader(empty_first_batch=["LifeExpectancy_Person_Male"])

    group = enumerate_members(five_tuple=_FIVE_TUPLE, slots={"gender": _UNBOUND}, graph=reader)

    assert [member.stat_var for member in group.members] == ["LifeExpectancy_Person"]
    assert group.inconsistent_candidates == ()


def test_candidate_still_empty_after_reprobe_is_excluded_and_reported():
    """An empty forward read after reprobe excludes and reports the candidate."""
    reader = _reader(empty_always=["LifeExpectancy_Person_Male"])

    group = enumerate_members(
        five_tuple=_FIVE_TUPLE, slots={"gender": ("value", "Male")}, graph=reader
    )

    assert group.members == ()
    assert group.inconsistent_candidates == ("LifeExpectancy_Person_Male",)


def test_reprobe_budget_is_respected():
    """The 32-candidate budget reprobes the first 32 of 40 candidates in dcid order.

    The remaining 8 are excluded and reported without a re-probe. Each candidate binds
    its own `bucket`, so the group has no root and no root-twin read.
    """
    arcs = {
        f"Count_Person_Bucket{i:02d}": {
            "typeOf": ("StatisticalVariable",),
            "populationType": ("Person",),
            "measuredProperty": ("count",),
            "statType": ("measuredValue",),
            "measurementQualifier": (),
            "measurementDenominator": (),
            "constraintProperties": ("bucket",),
            "observationProperties": (),
            "bucket": (f"Bucket{i:02d}",),
        }
        for i in range(40)
    }
    reader = FlakyMemberArcReader(arcs=arcs, drop_first_batch=dict.fromkeys(arcs, "statType"))

    group = enumerate_members(
        five_tuple=_COUNT_FIVE_TUPLE, slots={"bucket": _UNBOUND}, graph=reader, member_cap=32
    )

    assert len(group.members) == 32
    assert group.inconsistent_candidates == tuple(
        f"Count_Person_Bucket{i:02d}" for i in range(32, 40)
    )
    assert group.undismissed_candidates == group.inconsistent_candidates
    # Calls are the initial batch, 32-candidate re-probe, and 32-winner verification. The
    # incomparable members have no subset relation or single-dcid fallback.
    assert reader.calls["node_arcs_many"] == 3
