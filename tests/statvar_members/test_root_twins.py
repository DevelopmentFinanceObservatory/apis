"""Tests how a confirmed root's possible twins are ordered, read, and refused."""

from __future__ import annotations

import pytest

from statvar_members.enumerate import (
    _REPROBE_BUDGET,
    _TWIN_PROBE_BUDGET,
    MemberRoleDisagreement,
    _ProvisionalDenominatorReader,
    _read_root_twins,
    enumerate_members,
)
from statvar_members.types import FiveTuple

from .conftest import FakeMemberArcReader, FlakyMemberArcReader

_COUNT = FiveTuple(population_type="Person", measured_property="count", stat_type="measuredValue")
_DESTINATION = ("destination",)
"""The role key every root here covers, `recipient` and `target` each placing it. A root
that covers none confirms nothing, since an unread member that covers would replace it."""


def _root(roles: tuple[str, ...] = ()) -> dict[str, tuple[str, ...]]:
    return {
        "typeOf": ("StatisticalVariable",),
        "populationType": ("Person",),
        "measuredProperty": ("count",),
        "statType": ("measuredValue",),
        "measurementQualifier": (),
        "measurementDenominator": (),
        "constraintProperties": (),
        "observationProperties": roles,
    }


_DISAGREEING_ROOTS = {
    "Count_Person_A": _root(("recipient",)),
    "Count_Person_B": _root(("target",)),
    "Count_Person_Z": _root(("recipient",)),
}


def test_confirmed_roots_that_disagree_refuse_beside_an_unreadable_twin():
    """`Count_Person_Z` reads empty every time. It cannot remove A or B, whose roles
    disagree whatever Z declares: each declares the one role that places the destination."""
    reader = FlakyMemberArcReader(arcs=_DISAGREEING_ROOTS, empty_always=["Count_Person_Z"])

    with pytest.raises(MemberRoleDisagreement) as exc_info:
        enumerate_members(five_tuple=_COUNT, slots={}, graph=reader, role_keys=_DESTINATION)

    assert exc_info.value.roles_by_stat_var == {
        "Count_Person_A": ("recipient",),
        "Count_Person_B": ("target",),
    }


def test_an_unreadable_twin_beside_agreeing_roots_stays_undismissed():
    arcs = {**_DISAGREEING_ROOTS, "Count_Person_B": _root(("recipient",))}
    reader = FlakyMemberArcReader(arcs=arcs, empty_always=["Count_Person_Z"])

    group = enumerate_members(five_tuple=_COUNT, slots={}, graph=reader, role_keys=_DESTINATION)

    assert [member.stat_var for member in group.members] == ["Count_Person_A", "Count_Person_B"]
    assert group.root_confirmed is True
    assert group.inconsistent_candidates == ("Count_Person_Z",)
    assert group.undismissed_candidates == ("Count_Person_Z",)


_CONTRADICTED_TWIN = {
    "Count_Person_A": _root(("recipient",)),
    "Count_Person_B": _root(("recipient",)),
    "Count_Person_Z": _root(("target",)),
}


def test_a_stable_read_of_the_contradicted_twin_family_refuses_the_role_disagreement():
    with pytest.raises(MemberRoleDisagreement):
        enumerate_members(
            five_tuple=_COUNT,
            slots={},
            graph=FakeMemberArcReader(arcs=_CONTRADICTED_TWIN),
            role_keys=_DESTINATION,
        )


def test_a_walked_twin_whose_read_contradicts_the_reverse_index_stays_undismissed():
    """Every read of `Count_Person_Z` drops `statType`, which the reverse query filtered
    on. The twin read cannot dismiss a candidate the index lists as a member."""
    reader = FlakyMemberArcReader(
        arcs=_CONTRADICTED_TWIN, drop_always={"Count_Person_Z": "statType"}
    )

    group = enumerate_members(five_tuple=_COUNT, slots={}, graph=reader, role_keys=_DESTINATION)

    assert [member.stat_var for member in group.members] == ["Count_Person_A", "Count_Person_B"]
    assert group.root_confirmed is True
    assert group.inconsistent_candidates == ("Count_Person_Z",)
    assert group.undismissed_candidates == ("Count_Person_Z",)


@pytest.mark.parametrize(
    ("walked", "unsettled"),
    [(frozenset({"Count_Person_Z"}), ("Count_Person_Z",)), (frozenset(), ())],
    ids=["walked", "caller_supplied"],
)
def test_a_twin_read_missing_a_filtered_arc_settles_only_a_caller_supplied_twin(
    walked: frozenset[str], unsettled: tuple[str, ...]
):
    """The reverse query vouches for a walked twin's `statType`, so a read without it
    settles nothing. No query vouches for a caller-supplied twin, so the read dismisses it."""
    reader = FlakyMemberArcReader(
        arcs=_CONTRADICTED_TWIN, drop_always={"Count_Person_Z": "statType"}
    )

    confirmed, still_unsettled = _read_root_twins(
        ["Count_Person_Z"],
        walked=walked,
        five_tuple=_COUNT,
        slots={},
        graph=reader,
        properties=tuple(_root()),
    )

    assert confirmed == ()
    assert still_unsettled == unsettled


def test_roots_past_the_reprobe_budget_are_read_alone_and_join_the_members():
    """The first batch drops `statType` from 40 roots. The re-probe budget recovers 32, and
    the root-twin pass reads the other 8 alone."""
    arcs = {f"Count_Person_Root{i:02d}": _root() for i in range(_REPROBE_BUDGET + 8)}
    reader = FlakyMemberArcReader(arcs=arcs, drop_first_batch=dict.fromkeys(arcs, "statType"))

    group = enumerate_members(five_tuple=_COUNT, slots={}, graph=reader)

    assert [member.stat_var for member in group.members] == sorted(arcs)
    assert set(group.family) == set(arcs)
    assert group.inconsistent_candidates == ()
    assert group.undismissed_candidates == ()
    # The initial batch, the 32-candidate re-probe, the 32-winner verification, and one
    # read per twin.
    assert reader.calls["node_arcs_many"] == 3 + 8


def test_the_twin_pass_adds_at_most_its_budget_of_reads():
    """Twelve possible twins read empty every time. The pass reads `_TWIN_PROBE_BUDGET` of
    them, and all twelve stay undismissed."""
    twins = [f"Count_Person_Twin{i:02d}" for i in range(_TWIN_PROBE_BUDGET + 4)]
    arcs = {"Count_Person": _root(), **dict.fromkeys(twins, _root())}
    reader = FlakyMemberArcReader(arcs=arcs, empty_always=twins)

    group = enumerate_members(five_tuple=_COUNT, slots={}, graph=reader)

    assert [member.stat_var for member in group.members] == ["Count_Person"]
    assert group.undismissed_candidates == tuple(twins)
    # Before the pass: the initial batch, one re-probe batch of the 12 empty reads, one
    # single-dcid retry for each, and the root's verification.
    reads_before_the_pass = 1 + 1 + len(twins) + 1
    assert reader.calls["node_arcs_many"] == reads_before_the_pass + _TWIN_PROBE_BUDGET


def test_the_twin_pass_adds_a_type_read_for_each_unseen_denominator_target():
    """Twelve twins each name their own provisional denominator. The pass reads eight of
    them alone, and each read costs one more for its target's `typeOf`."""
    ask = FiveTuple(
        population_type="Person",
        measured_property="count",
        stat_type="measuredValue",
        denominator_unresolved=True,
    )
    twins = [f"Count_Person_Twin{i:02d}" for i in range(_TWIN_PROBE_BUDGET + 4)]
    arcs: dict[str, dict[str, tuple[str, ...]]] = {}
    for i, twin in enumerate(twins):
        arcs[twin] = {**_root(), "measurementDenominator": (f"Denominator{i:02d}",)}
        arcs[f"Denominator{i:02d}"] = {"typeOf": ("ProvisionalNode",)}
    reader = FakeMemberArcReader(arcs=arcs)

    confirmed, unsettled = _read_root_twins(
        twins,
        walked=frozenset(twins),
        five_tuple=ask,
        slots={},
        graph=_ProvisionalDenominatorReader(reader),
        properties=tuple(_root()),
    )

    assert [member.stat_var for member in confirmed] == twins[:_TWIN_PROBE_BUDGET]
    assert unsettled == tuple(twins[_TWIN_PROBE_BUDGET:])
    assert reader.calls["node_arcs_many"] == 2 * _TWIN_PROBE_BUDGET


_PER_CAPITA_ASK = FiveTuple(measured_property="amount", denominator_unresolved=True)
_GDP_SLOTS = {
    "activitySource": ("value", "GrossDomesticProduction"),
    "economicSector": ("unbound",),
}


def _amount(**overrides: tuple[str, ...]) -> dict[str, tuple[str, ...]]:
    return {
        "typeOf": ("StatisticalVariable",),
        "populationType": (),
        "measuredProperty": ("amount",),
        "statType": (),
        "measurementQualifier": (),
        "measurementDenominator": (),
        "constraintProperties": (),
        "observationProperties": (),
        **overrides,
    }


def _gdp_family(bare_total: str) -> dict[str, dict[str, tuple[str, ...]]]:
    """A per-capita root, the no-denominator total named `bare_total`, and 40
    no-denominator sector siblings. Under an unresolved-denominator ask, the total and
    the siblings are all absence suspects, and only the total is root-shaped."""
    arcs = {
        "Amount_GDP_PerCapita": _amount(
            measurementDenominator=("PerCapita",),
            constraintProperties=("activitySource",),
            activitySource=("GrossDomesticProduction",),
        ),
        "PerCapita": {"typeOf": ("ProvisionalNode",)},
        bare_total: _amount(
            constraintProperties=("activitySource",),
            activitySource=("GrossDomesticProduction",),
        ),
    }
    for i in range(40):
        arcs[f"Amount_GDP_Sector{i:02d}"] = _amount(
            constraintProperties=("activitySource", "economicSector"),
            activitySource=("GrossDomesticProduction",),
            economicSector=(f"S{i:02d}",),
        )
    return arcs


@pytest.mark.parametrize("bare_total", ["Amount_GDP", "Nominal_Amount_GDP"])
def test_a_root_shaped_suspect_is_reprobed_wherever_its_dcid_sorts(bare_total: str):
    """`Amount_GDP` sorts before the 40 sector siblings and `Nominal_Amount_GDP` after
    them. Either takes the first re-probe slot, so the unread excess is nine siblings."""
    reader = FakeMemberArcReader(arcs=_gdp_family(bare_total))

    group = enumerate_members(five_tuple=_PER_CAPITA_ASK, slots=_GDP_SLOTS, graph=reader)

    assert [member.stat_var for member in group.members] == ["Amount_GDP_PerCapita"]
    assert group.root_confirmed is True
    assert group.inconsistent_candidates == tuple(
        f"Amount_GDP_Sector{i:02d}" for i in range(31, 40)
    )
    assert group.undismissed_candidates == ()
