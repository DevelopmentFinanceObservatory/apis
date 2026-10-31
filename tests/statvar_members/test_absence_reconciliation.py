"""Pins absence reconciliation for dropped constraint and unresolved-denominator arcs."""

from __future__ import annotations

import pytest

from statvar_members.enumerate import SlotBinding, enumerate_members
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


class _PartlyCompleteReader(FlakyMemberArcReader):
    """Reads the dcids in `complete` in full, as the origin catalogue's tables answer them,
    and the rest as a live graph whose batch reads drop arcs. Records every arcs read."""

    def __init__(self, *, complete: frozenset[str], **kwargs) -> None:
        super().__init__(**kwargs)
        self._complete = complete
        self.reads: list[tuple[str, ...]] = []

    def arcs_complete(self, dcid: str) -> bool:
        return dcid in self._complete

    def node_arcs_many(self, dcids, *, properties, individual_probe_budget=None, deadline=None):
        self.reads.append(tuple(dcids))
        return super().node_arcs_many(
            dcids, properties=properties, individual_probe_budget=individual_probe_budget
        )


def _occupation_arcs(occupation: str) -> dict[str, tuple[str, ...]]:
    return _arcs(constraintProperties=("occupation",), occupation=(occupation,))


def _age_arcs(age: str) -> dict[str, tuple[str, ...]]:
    return _arcs(constraintProperties=("age",), age=(age,))


@pytest.mark.parametrize(
    ("live_ages", "nurse_live", "undismissed", "rereadable"),
    [(0, True, 0, False), (33, False, 1, True)],
    ids=["live_member_recovered", "live_siblings_past_the_budget"],
)
def test_only_complete_reads_skip_the_absence_re_read(
    live_ages: int, nurse_live: bool, undismissed: int, rereadable: bool
):
    """40 siblings read in full lack the set-bound `occupation`, more than the 32-candidate
    re-read budget, and are dismissed without a re-read. A live member whose first batch
    drops its constraint is re-read and recovered. Live siblings past the budget stay
    undismissed, and a later walk could settle them."""
    complete_ages = {f"Amount_Age{i:02d}": _age_arcs(f"Age{i:02d}") for i in range(40)}
    live = {f"Amount_LiveAge{i:02d}": _age_arcs(f"LiveAge{i:02d}") for i in range(live_ages)}
    arcs = {
        "Amount_Teacher": _occupation_arcs("Teacher"),
        "Amount_Nurse": _occupation_arcs("Nurse"),
        **complete_ages,
        **live,
    }
    complete = frozenset(complete_ages) | {"Amount_Teacher"}
    if not nurse_live:
        complete |= {"Amount_Nurse"}
    reader = _PartlyCompleteReader(
        complete=complete,
        arcs=arcs,
        drop_first_batch={"Amount_Nurse": "constraintProperties"} if nurse_live else None,
    )
    slots: dict[str, SlotBinding] = {
        "occupation": ("set", ("Teacher", "Nurse")),
        "age": _UNBOUND,
    }

    group = enumerate_members(five_tuple=_NO_DENOMINATOR_ASK, slots=slots, graph=reader)

    assert [member.stat_var for member in group.members] == ["Amount_Nurse", "Amount_Teacher"]
    assert len(group.undismissed_candidates) == undismissed
    assert group.undismissed_rereadable is rereadable
    first, *rereads = reader.reads
    assert set(first) == set(arcs)
    assert not {dcid for read in rereads for dcid in read} & frozenset(complete_ages)


def _gender_arcs(gender: str) -> dict[str, tuple[str, ...]]:
    return _arcs(constraintProperties=("gender",), gender=(gender,))


class _StubReader(_PartlyCompleteReader):
    """Answers forward reads of the dcids in `bodies` from those bodies, while the reverse
    query still finds them by `arcs`, as our mixer's in-arcs and out-arcs disagree for the
    `dc/base/Mongolia_Demographics` StatVars. A `None` body reads absent."""

    def __init__(self, *, bodies: dict[str, dict[str, tuple[str, ...]] | None], **kwargs) -> None:
        super().__init__(**kwargs)
        self._bodies = bodies

    def node_arcs_many(self, dcids, *, properties, individual_probe_budget=None, deadline=None):
        read = super().node_arcs_many(dcids, properties=properties)
        for dcid in set(read) & set(self._bodies):
            body = self._bodies[dcid]
            read[dcid] = None if body is None else {prop: body.get(prop, ()) for prop in properties}
        return read


_MEMBER_OF_ONLY = {"memberOf": ("dc/g/Amount_Stub",)}


@pytest.mark.parametrize(
    ("with_root", "slots", "body", "stub_complete", "members", "stubs", "undismissed"),
    [
        pytest.param(
            True,
            {"gender": _UNBOUND},
            _MEMBER_OF_ONLY,
            True,
            ["Amount"],
            ("Amount_Stub",),
            (),
            id="member_of_only_settled_under_a_confirmed_root",
        ),
        pytest.param(
            True,
            {"gender": _UNBOUND},
            None,
            True,
            ["Amount"],
            ("Amount_Stub",),
            (),
            id="absent_settled_under_a_confirmed_root",
        ),
        pytest.param(
            True,
            {"gender": _UNBOUND},
            _MEMBER_OF_ONLY,
            False,
            ["Amount"],
            (),
            ("Amount_Stub",),
            id="refused_when_the_read_is_not_complete",
        ),
        pytest.param(
            True,
            {"gender": _UNBOUND},
            {**_MEMBER_OF_ONLY, "typeOf": ("StatisticalVariable",)},
            True,
            ["Amount"],
            (),
            ("Amount_Stub",),
            id="refused_when_the_body_carries_a_definition_arc",
        ),
        pytest.param(
            False,
            {"gender": _UNBOUND},
            _MEMBER_OF_ONLY,
            True,
            ["Amount_Female", "Amount_Male"],
            (),
            ("Amount_Stub",),
            id="refused_without_a_root",
        ),
        pytest.param(
            True,
            {"gender": ("set", ("Male", "Female"))},
            _MEMBER_OF_ONLY,
            True,
            ["Amount_Female", "Amount_Male"],
            (),
            ("Amount_Stub",),
            id="refused_when_a_set_slot_rules_the_root_out",
        ),
    ],
)
def test_a_complete_stub_settles_only_under_a_confirmed_root(
    with_root: bool,
    slots: dict[str, SlotBinding],
    body: dict[str, tuple[str, ...]] | None,
    stub_complete: bool,
    members: list[str],
    stubs: tuple[str, ...],
    undismissed: tuple[str, ...],
):
    """The reverse query names `Amount_Stub` under the five-tuple, but its forward body is
    one `memberOf` arc, as dev's `Count_Person_15OrMoreYears_Female_Unmarried` reads. Under a
    confirmed root a complete read carrying no definition arc settles it out and the group
    names it. Without a confirmed root it could stand for the root, a read the reader does not
    hold complete could have dropped its arcs, and a body with a definition arc contradicts
    the reverse read, so each of those still refuses."""
    arcs = {
        "Amount_Male": _gender_arcs("Male"),
        "Amount_Female": _gender_arcs("Female"),
        "Amount_Stub": _arcs(),
        **({"Amount": _arcs()} if with_root else {}),
    }
    complete = frozenset(arcs) - (set() if stub_complete else {"Amount_Stub"})
    reader = _StubReader(bodies={"Amount_Stub": body}, complete=complete, arcs=arcs)

    group = enumerate_members(five_tuple=_NO_DENOMINATOR_ASK, slots=slots, graph=reader)

    assert [member.stat_var for member in group.members] == members
    assert group.stub_candidates == stubs
    assert group.undismissed_candidates == undismissed
    assert group.undismissed_rereadable is not stub_complete
    assert "Amount_Stub" not in group.family
