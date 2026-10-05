"""Pins recovery when batch reads drop constraint facts from antichain winners."""

from __future__ import annotations

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


def test_determinism_regardless_of_which_member_the_drop_lands_on():
    """The same ask, against a reader that drops `constraintProperties` on
    the genuinely unconstrained member instead of a constrained one, still
    yields the same member set as the undropped reference."""
    reference = enumerate_members(
        five_tuple=_FIVE_TUPLE, slots={"gender": _UNBOUND}, graph=_reader()
    )
    reference_stat_vars = [member.stat_var for member in reference.members]

    for drop_first_batch in (
        {"LifeExpectancy_Person_Female": "constraintProperties"},
        {"LifeExpectancy_Person": "constraintProperties"},
    ):
        reader = _reader(drop_first_batch=drop_first_batch)
        group = enumerate_members(five_tuple=_FIVE_TUPLE, slots={"gender": _UNBOUND}, graph=reader)
        assert [member.stat_var for member in group.members] == reference_stat_vars
        assert group.inconsistent_candidates == ()


_CHAIN_ARCS: dict[str, dict[str, tuple[str, ...]]] = {
    "Count_Person_Gendered": {
        "typeOf": ("StatisticalVariable",),
        "populationType": ("Person",),
        "measuredProperty": ("count",),
        "statType": ("measuredValue",),
        "measurementQualifier": (),
        "measurementDenominator": (),
        "constraintProperties": ("gender",),
        "observationProperties": (),
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
        "observationProperties": (),
        "age": ("Years0",),
    },
}
"""True constraint sets `{gender}` and `{age}`: incomparable, so both belong
in the true antichain. Neither's own true set is a subset of the other's,
which is what makes the drop below a genuine wrongful exclusion rather than
a correct one that merely looks suspicious."""

_CHAIN_SLOTS = {"gender": _UNBOUND, "age": _UNBOUND}


def _chain_reader(**kwargs) -> FlakyMemberArcReader:
    return FlakyMemberArcReader(arcs=_CHAIN_ARCS, **kwargs)


def test_reverification_converges_across_multiple_rounds():
    """The first batch makes Gendered look unconstrained, so it seems to dominate Aged.
    Re-reading Gendered restores `{gender}`, leaving both members in the antichain after
    three rounds."""
    reader = _chain_reader(drop_first_batch={"Count_Person_Gendered": "constraintProperties"})

    group = enumerate_members(five_tuple=_COUNT_FIVE_TUPLE, slots=_CHAIN_SLOTS, graph=reader)

    assert [member.stat_var for member in group.members] == [
        "Count_Person_Aged",
        "Count_Person_Gendered",
    ]
    assert group.inconsistent_candidates == ()
    # One batch call, one re-verify round covering `Count_Person_Gendered`
    # (round one, the sole apparent winner), one re-verify round covering
    # `Count_Person_Aged` once it's released into the winner set (round
    # two). Each round's batch happens to hold one dcid here, so the count
    # is the same whether the round is batched or issued individually.
    assert reader.calls["node_arcs_many"] == 3


def test_dropped_type_of_on_the_reverify_reread_is_recovered_not_refused():
    """The second batch drops `typeOf`; a single-dcid fallback restores the
    `StatisticalVariable` type and keeps both members."""
    reader = _chain_reader(drop_second_batch={"Count_Person_Gendered": "typeOf"})

    group = enumerate_members(five_tuple=_COUNT_FIVE_TUPLE, slots=_CHAIN_SLOTS, graph=reader)

    assert {member.stat_var for member in group.members} == {
        "Count_Person_Gendered",
        "Count_Person_Aged",
    }
    assert group.inconsistent_candidates == ()


def test_dropped_five_tuple_arc_on_the_reverify_reread_is_recovered():
    """A dropped five-tuple arc makes a genuine winner look like a
    non-member during the delayed re-verify batch. That rejection is a batch
    artifact, so the bounded single-dcid fallback recovers it."""
    reader = _chain_reader(drop_second_batch={"Count_Person_Gendered": "measuredProperty"})

    group = enumerate_members(five_tuple=_COUNT_FIVE_TUPLE, slots=_CHAIN_SLOTS, graph=reader)

    assert {member.stat_var for member in group.members} == {
        "Count_Person_Gendered",
        "Count_Person_Aged",
    }
    assert group.inconsistent_candidates == ()
    # Initial confirmation, a re-verify batch for the tied winners, and
    # one bounded single-dcid fallback for the dropped five-tuple arc.
    assert reader.calls["node_arcs_many"] == 3


def test_dropped_constraint_value_on_the_reverify_reread_is_recovered():
    """Keeping `constraintProperties` while dropping its value arc makes
    `_confirm_member` absence-reject a genuine winner. The same bounded
    single-dcid fallback recovers the value and preserves the winner."""
    reader = _chain_reader(drop_second_batch={"Count_Person_Gendered": "gender"})

    group = enumerate_members(five_tuple=_COUNT_FIVE_TUPLE, slots=_CHAIN_SLOTS, graph=reader)

    assert {member.stat_var for member in group.members} == {
        "Count_Person_Gendered",
        "Count_Person_Aged",
    }
    assert group.inconsistent_candidates == ()
    # Initial confirmation, a re-verify batch for the tied winners, and
    # one bounded single-dcid fallback for the dropped constraint value.
    assert reader.calls["node_arcs_many"] == 3


def test_dropped_type_of_on_the_initial_batch_read_has_no_effect():
    """An initial-batch `typeOf` drop does not exclude a candidate. The winner re-read
    confirms the type before returning the member."""
    reader = _chain_reader(drop_first_batch={"Count_Person_Gendered": "typeOf"})

    group = enumerate_members(five_tuple=_COUNT_FIVE_TUPLE, slots=_CHAIN_SLOTS, graph=reader)

    assert {member.stat_var for member in group.members} == {
        "Count_Person_Gendered",
        "Count_Person_Aged",
    }
    assert group.inconsistent_candidates == ()


_BUDGET_ARCS: dict[str, dict[str, tuple[str, ...]]] = {
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
"""40 candidates, each with its own distinct `bucket` value: no subset
relation between any two, so all 40 are consistent, apparent-`{}` winners
once their `constraintProperties` is dropped, and none dominates another
truly either -- an unconstrained-vs-unconstrained tie that must be
re-verified together before the fallback budget can apply."""


def test_reverification_rereads_all_provisional_winners_before_using_the_budget():
    """All 40 candidates look unconstrained in the first batch.
    The re-read restores `bucket`; the fallback budget covers suspicious single-dcid reads."""
    reader = FlakyMemberArcReader(
        arcs=_BUDGET_ARCS,
        drop_first_batch=dict.fromkeys(_BUDGET_ARCS, "constraintProperties"),
    )

    group = enumerate_members(
        five_tuple=_COUNT_FIVE_TUPLE,
        slots={"bucket": _UNBOUND},
        graph=reader,
        member_cap=40,
    )

    assert [member.stat_var for member in group.members] == list(_BUDGET_ARCS)
    assert group.inconsistent_candidates == ()
    # One initial batch plus one uncapped batched re-verify round covering
    # all 40 provisional winners; no single-dcid fallback is needed.
    assert reader.calls["node_arcs_many"] == 2
