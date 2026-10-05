"""Pins absence reconciliation for dropped constraint and unresolved-denominator arcs."""

from __future__ import annotations

import pytest

from statvar_members.enumerate import enumerate_members
from statvar_members.types import UNRESOLVED_NODE, FiveTuple

from .conftest import FakeMemberArcReader, FlakyMemberArcReader

_UNRESOLVED_DENOMINATOR_ASK = FiveTuple(measured_property="amount", denominator_unresolved=True)
_NO_DENOMINATOR_ASK = FiveTuple(measured_property="amount")
_UNBOUND = ("unbound",)


def _arcs(**overrides: object) -> dict[str, tuple[str, ...]]:
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


@pytest.mark.parametrize(
    ("arcs", "dropped_arc", "five_tuple", "slots", "expected_members"),
    [
        pytest.param(
            {
                "Amount_Scheme": _arcs(constraintProperties=("scheme",), scheme=("ODAGrants",)),
                "Amount_OtherScheme": _arcs(constraintProperties=("region",), region=("Kenya",)),
            },
            {"Amount_Scheme": "constraintProperties"},
            _NO_DENOMINATOR_ASK,
            {"scheme": ("set", ("ODAGrants",))},
            ["Amount_Scheme"],
            id="constraint-properties",
        ),
        pytest.param(
            {
                "Amount_Gendered": _arcs(constraintProperties=("gender",), gender=("Male",)),
                "Amount_SchemeOnly": _arcs(constraintProperties=("scheme",), scheme=("ODAGrants",)),
            },
            {"Amount_Gendered": "gender"},
            _NO_DENOMINATOR_ASK,
            {"gender": _UNBOUND},
            ["Amount_Gendered"],
            id="constraint-value",
        ),
        pytest.param(
            {
                "Amount_WithDenom": _arcs(measurementDenominator=(UNRESOLVED_NODE,)),
                "Amount_OtherDenom": _arcs(measurementDenominator=("Count_Person",)),
            },
            {"Amount_WithDenom": "measurementDenominator"},
            _UNRESOLVED_DENOMINATOR_ASK,
            {},
            ["Amount_WithDenom"],
            id="unresolved-denominator",
        ),
    ],
)
def test_non_filter_covered_dropped_arc_is_reconciled_and_the_candidate_is_admitted(
    arcs, dropped_arc, five_tuple, slots, expected_members
):
    reader = FlakyMemberArcReader(arcs=arcs, drop_first_batch=dropped_arc)

    group = enumerate_members(five_tuple=five_tuple, slots=slots, graph=reader)

    assert [member.stat_var for member in group.members] == expected_members
    assert group.inconsistent_candidates == ()


def test_dropped_denominator_arc_on_a_concrete_ask_is_reconciled_and_the_candidate_is_admitted():
    """A concrete denominator is filter-covered, so its dropped arc is re-probed and restored.
    The sibling forces a multi-dcid batch; its unrelated constraint is excluded."""
    arcs = {
        "Amount_WithDenom": _arcs(measurementDenominator=("Count_Person",)),
        "Amount_WithDenom_Scheme": _arcs(
            measurementDenominator=("Count_Person",),
            constraintProperties=("scheme",),
            scheme=("ODAGrants",),
        ),
    }
    reader = FlakyMemberArcReader(
        arcs=arcs, drop_first_batch={"Amount_WithDenom": "measurementDenominator"}
    )

    group = enumerate_members(
        five_tuple=FiveTuple(measured_property="amount", measurement_denominator="Count_Person"),
        slots={},
        graph=reader,
    )

    assert [member.stat_var for member in group.members] == ["Amount_WithDenom"]
    assert group.inconsistent_candidates == ()


def test_a_role_rejection_on_an_empty_role_arc_is_re_read_not_trusted():
    """A role-bound identity rejects a candidate whose `observationProperties` read empty,
    but a batch read can drop that arc, so the rejection is re-probed. The re-read restores
    the recipient role and admits the candidate. A subject-only sibling that declares no
    role reads empty again and stays an ordinary non-match, never an inconsistent one."""
    arcs = {
        "Amount_ToRecipient": _arcs(observationProperties=("financingRecipient",)),
        "Amount_SubjectOnly": _arcs(),
    }
    reader = FlakyMemberArcReader(
        arcs=arcs, drop_first_batch={"Amount_ToRecipient": "observationProperties"}
    )

    group = enumerate_members(
        five_tuple=_NO_DENOMINATOR_ASK,
        slots={},
        graph=reader,
        role_keys=("financingRecipient",),
    )

    assert [member.stat_var for member in group.members] == ["Amount_ToRecipient"]
    assert group.members[0].declared_roles == ("financingRecipient",)
    assert group.inconsistent_candidates == ()


def test_a_candidate_genuinely_lacking_the_bound_constraint_is_still_rejected():
    """The server-side filter excludes a candidate with no `scheme` arc before any forward read."""
    arcs = {"Amount_NoScheme": _arcs()}
    reader = FakeMemberArcReader(arcs=arcs)

    group = enumerate_members(
        five_tuple=_NO_DENOMINATOR_ASK,
        slots={"scheme": ("value", "ODAGrants")},
        graph=reader,
    )

    assert group.candidates_considered == 0
    assert group.members == ()
    assert group.inconsistent_candidates == ()
    # One call, over an empty candidate list: the filter excludes
    # `Amount_NoScheme` before any forward-arc fetch, so no absence
    # reconciliation round ever runs.
    assert reader.calls["node_arcs_many"] == 1


def test_determinism_regardless_of_which_candidate_the_drop_lands_on():
    """Dropping the same arc from either candidate preserves the reference member set."""
    arcs = {
        "Amount_Gendered": _arcs(constraintProperties=("gender",), gender=("Male",)),
        "Amount_Gendered_Two": _arcs(constraintProperties=("gender",), gender=("Female",)),
    }
    reference = enumerate_members(
        five_tuple=_NO_DENOMINATOR_ASK,
        slots={"gender": _UNBOUND},
        graph=FakeMemberArcReader(arcs=arcs),
    )
    reference_stat_vars = {member.stat_var for member in reference.members}

    for drop_first_batch in (
        {"Amount_Gendered": "constraintProperties"},
        {"Amount_Gendered_Two": "gender"},
    ):
        reader = FlakyMemberArcReader(arcs=arcs, drop_first_batch=drop_first_batch)
        group = enumerate_members(
            five_tuple=_NO_DENOMINATOR_ASK, slots={"gender": _UNBOUND}, graph=reader
        )
        assert {member.stat_var for member in group.members} == reference_stat_vars
        assert group.inconsistent_candidates == ()


def test_reprobe_budget_is_respected():
    """40 candidates exceed the 32-candidate re-probe budget.
    The first 32 recover; the remaining 8 are inconsistent."""
    bucket_refs = tuple(f"Bucket{i:02d}" for i in range(40))
    arcs = {
        f"Amount_Bucket{i:02d}": _arcs(constraintProperties=("bucket",), bucket=(f"Bucket{i:02d}",))
        for i in range(40)
    }
    reader = FlakyMemberArcReader(
        arcs=arcs, drop_first_batch=dict.fromkeys(arcs, "constraintProperties")
    )

    group = enumerate_members(
        five_tuple=_NO_DENOMINATOR_ASK,
        slots={"bucket": ("set", bucket_refs)},
        graph=reader,
        member_cap=40,
    )

    assert len(group.members) == 32
    assert len(group.inconsistent_candidates) == 8


def test_concrete_denominator_ask_with_more_than_reprobe_budget_denominator_less_siblings():
    """The server-side filter excludes 40 denominator-less siblings before they
    count against the 32-candidate re-probe budget."""
    arcs = {
        "Amount_WithDenom": _arcs(measurementDenominator=("Count_Person",)),
        **{f"Amount_NoDenom{i:02d}": _arcs() for i in range(40)},
    }
    reader = FakeMemberArcReader(arcs=arcs)

    group = enumerate_members(
        five_tuple=FiveTuple(measured_property="amount", measurement_denominator="Count_Person"),
        slots={},
        graph=reader,
    )

    assert group.candidates_considered == 1
    assert [member.stat_var for member in group.members] == ["Amount_WithDenom"]
    assert group.inconsistent_candidates == ()
