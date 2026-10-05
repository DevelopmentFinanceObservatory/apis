"""Pins `enumerate_members` against a fake member-arc reader."""

from __future__ import annotations

import inspect

import pytest

from statvar_members.enumerate import (
    _EXTRA_CANDIDATE_PROBE_BUDGET,
    CANDIDATE_CAP,
    Member,
    MemberEnumerationTooLarge,
    MemberQueryUnanchored,
    MemberRoleDisagreement,
    MultiValuedConstraint,
    NotAStatisticalVariable,
    enumerate_members,
)
from statvar_members.types import UNRESOLVED_NODE, FiveTuple

from .conftest import (
    FakeMemberArcReader,
    FakeSubjectPage,
    FlakyMemberArcReader,
    kenya_reader,
    male_female_only_reader,
)

_KENYA_FIVE_TUPLE = FiveTuple(
    population_type="Person", measured_property="lifeExpectancy", stat_type="measuredValue"
)
_COUNT_FIVE_TUPLE = FiveTuple(
    population_type="Person", measured_property="count", stat_type="measuredValue"
)
_UNBOUND = ("unbound",)


def test_unbound_gender_reduces_to_the_unconstrained_kenya_member():
    reader = kenya_reader()

    group = enumerate_members(
        five_tuple=_KENYA_FIVE_TUPLE, slots={"gender": _UNBOUND}, graph=reader
    )

    assert group.members == (
        Member(
            stat_var="LifeExpectancy_Person", constraints={}, declared_roles=("observationAbout",)
        ),
    )
    assert group.truncated is False
    assert group.five_tuple == _KENYA_FIVE_TUPLE


def _incomparable_member_case(case: str):
    if case == "unbound":
        return (
            male_female_only_reader(),
            _COUNT_FIVE_TUPLE,
            {"gender": _UNBOUND},
            None,
        )
    if case == "set-bound":
        return (
            kenya_reader(),
            _KENYA_FIVE_TUPLE,
            {"gender": ("set", ("Male", "Female"))},
            None,
        )

    arcs = {
        f"Count_Person_Bucket{i}": {
            "typeOf": ("StatisticalVariable",),
            "populationType": ("Person",),
            "measuredProperty": ("count",),
            "statType": ("measuredValue",),
            "measurementQualifier": (),
            "measurementDenominator": (),
            "constraintProperties": ("bucket",),
            "observationProperties": (),
            "bucket": (f"Value{i}",),
        }
        for i in range(30)
    }
    return (
        FakeMemberArcReader(arcs=arcs),
        _COUNT_FIVE_TUPLE,
        {"bucket": ("set", ("Value0", "Value1"))},
        25,
    )


@pytest.mark.parametrize(
    ("case", "expected_stat_vars"),
    [
        pytest.param(
            "unbound",
            {"Count_Person_Male", "Count_Person_Female"},
            id="unbound-incomparable",
        ),
        pytest.param(
            "set-bound",
            {"LifeExpectancy_Person_Male", "LifeExpectancy_Person_Female"},
            id="set-bound",
        ),
        pytest.param(
            "set-bound-under-cap",
            {"Count_Person_Bucket0", "Count_Person_Bucket1"},
            id="set-bound-under-cap",
        ),
    ],
)
def test_incomparable_members_are_retained(case: str, expected_stat_vars: set[str]):
    reader, five_tuple, slots, member_cap = _incomparable_member_case(case)

    group = enumerate_members(
        five_tuple=five_tuple,
        slots=slots,
        graph=reader,
        **({"member_cap": member_cap} if member_cap is not None else {}),
    )

    assert {member.stat_var for member in group.members} == expected_stat_vars


def test_value_bound_slot_selects_one_member():
    reader = kenya_reader()

    group = enumerate_members(
        five_tuple=_KENYA_FIVE_TUPLE, slots={"gender": ("value", "Male")}, graph=reader
    )

    assert [member.stat_var for member in group.members] == ["LifeExpectancy_Person_Male"]


def test_value_bound_slot_matching_nothing_returns_an_empty_group():
    reader = kenya_reader()

    group = enumerate_members(
        five_tuple=_KENYA_FIVE_TUPLE,
        slots={"gender": ("value", "Nonbinary")},
        graph=reader,
    )

    assert group.members == ()
    assert group.truncated is False


def test_a_multi_valued_constraint_matching_the_ask_raises():
    # The bound value matches the first of two gender values. A Member cannot
    # represent the second, so the candidate must raise instead of losing it.
    reader = FakeMemberArcReader(
        arcs={
            "LifeExpectancy_Person_Male": {
                "typeOf": ("StatisticalVariable",),
                "populationType": ("Person",),
                "measuredProperty": ("lifeExpectancy",),
                "statType": ("measuredValue",),
                "measurementQualifier": (),
                "measurementDenominator": (),
                "constraintProperties": ("gender",),
                "observationProperties": (),
                "gender": ("Male", "Female"),
            },
        }
    )

    with pytest.raises(MultiValuedConstraint) as excinfo:
        enumerate_members(
            five_tuple=_KENYA_FIVE_TUPLE,
            slots={"gender": ("value", "Male")},
            graph=reader,
        )

    assert excinfo.value.stat_var == "LifeExpectancy_Person_Male"
    assert excinfo.value.property == "gender"
    assert excinfo.value.values == ("Male", "Female")


def test_a_multi_valued_constraint_not_matching_the_ask_is_excluded_not_raised():
    # The binding matches neither value, so the ordinary slot check excludes
    # the candidate before the multi-value guard runs.
    reader = FakeMemberArcReader(
        arcs={
            "LifeExpectancy_Person_Male": {
                "typeOf": ("StatisticalVariable",),
                "populationType": ("Person",),
                "measuredProperty": ("lifeExpectancy",),
                "statType": ("measuredValue",),
                "measurementQualifier": (),
                "measurementDenominator": (),
                "constraintProperties": ("gender",),
                "observationProperties": (),
                "gender": ("Female", "Male"),
            },
        }
    )

    group = enumerate_members(
        five_tuple=_KENYA_FIVE_TUPLE,
        slots={"gender": ("value", "Nonbinary")},
        graph=reader,
    )

    assert group.members == ()


def test_five_tuple_with_no_statvars_at_all_returns_an_empty_group_not_an_exception():
    reader = FakeMemberArcReader(arcs={})

    group = enumerate_members(five_tuple=_COUNT_FIVE_TUPLE, slots={}, graph=reader)

    assert group.members == ()
    assert group.candidates_considered == 0
    assert group.truncated is False


def test_measured_property_absent_raises_member_query_unanchored():
    reader = FakeMemberArcReader(arcs={})

    with pytest.raises(MemberQueryUnanchored) as excinfo:
        enumerate_members(five_tuple=FiveTuple(population_type="Person"), slots={}, graph=reader)

    assert excinfo.value.five_tuple == FiveTuple(population_type="Person")


def test_one_member_arm_with_only_measured_property_matches_a_bare_statvar():
    """A `FiveTuple` naming only `measured_property` -- every other field
    absent -- matches a StatVar carrying no other five-tuple arc at all,
    exercising the sparsest arm the entry contract admits."""
    reader = FakeMemberArcReader(
        arcs={
            "Count": {
                "typeOf": ("StatisticalVariable",),
                "populationType": (),
                "measuredProperty": ("count",),
                "statType": (),
                "measurementQualifier": (),
                "measurementDenominator": (),
                "constraintProperties": (),
                "observationProperties": (),
            },
        }
    )

    group = enumerate_members(
        five_tuple=FiveTuple(measured_property="count"), slots={}, graph=reader
    )

    assert [member.stat_var for member in group.members] == ["Count"]


@pytest.mark.parametrize(
    ("five_tuple", "arcs", "expected_candidates"),
    [
        pytest.param(
            FiveTuple(measured_property="count", stat_type="measuredValue"),
            {
                "Count_Person": {
                    "typeOf": ("StatisticalVariable",),
                    "populationType": ("Person",),
                    "measuredProperty": ("count",),
                    "statType": ("measuredValue",),
                    "measurementQualifier": (),
                    "measurementDenominator": (),
                    "constraintProperties": (),
                    "observationProperties": (),
                },
            },
            1,
            id="absent_population_type_uses_forward_confirmation",
        ),
        pytest.param(
            FiveTuple(measured_property="amount", measurement_denominator="Count_Person"),
            {
                "Amount_OtherDenom": {
                    "typeOf": ("StatisticalVariable",),
                    "populationType": (),
                    "measuredProperty": ("amount",),
                    "statType": (),
                    "measurementQualifier": (),
                    "measurementDenominator": ("Count_Household",),
                    "constraintProperties": (),
                    "observationProperties": (),
                },
            },
            0,
            id="concrete_denominator_filters_other_dcid_server_side",
        ),
        pytest.param(
            FiveTuple(measured_property="amount", denominator_unresolved=True),
            {
                "Amount_ResolvedDenom": {
                    "typeOf": ("StatisticalVariable",),
                    "populationType": (),
                    "measuredProperty": ("amount",),
                    "statType": (),
                    "measurementQualifier": (),
                    "measurementDenominator": ("Count_Person",),
                    "constraintProperties": (),
                    "observationProperties": (),
                },
            },
            1,
            id="unresolved_denominator_uses_forward_confirmation",
        ),
        pytest.param(
            FiveTuple(measured_property="amount"),
            {
                "Amount_ResolvedDenom": {
                    "typeOf": ("StatisticalVariable",),
                    "populationType": (),
                    "measuredProperty": ("amount",),
                    "statType": (),
                    "measurementQualifier": (),
                    "measurementDenominator": ("Count_Person",),
                    "constraintProperties": (),
                    "observationProperties": (),
                },
            },
            1,
            id="absent_denominator_uses_forward_confirmation",
        ),
        pytest.param(
            FiveTuple(measured_property="amount", measurement_denominator="Count_Person"),
            {
                "Amount_WithDenom": {
                    "typeOf": ("StatisticalVariable",),
                    "populationType": (),
                    "measuredProperty": ("amount",),
                    "statType": (),
                    "measurementQualifier": (),
                    "measurementDenominator": (UNRESOLVED_NODE,),
                    "constraintProperties": (),
                    "observationProperties": (),
                },
            },
            0,
            id="concrete_denominator_filters_marker_server_side",
        ),
    ],
)
def test_optional_five_tuple_fields_preserve_filter_and_confirmation_boundaries(
    five_tuple, arcs, expected_candidates
):
    """Concrete values filter server-side; absent and unresolved values require
    forward confirmation, which the candidate count distinguishes."""
    group = enumerate_members(
        five_tuple=five_tuple,
        slots={},
        graph=FakeMemberArcReader(arcs=arcs),
    )

    assert group.candidates_considered == expected_candidates
    assert group.members == ()


def test_truncated_page_still_confirms_a_dominating_member_inside_the_window():
    # `force_truncated` reports every `node_subjects` page truncated
    # regardless of how few candidates it actually holds, so the reverse
    # query's own page here is the same six-dcid Kenya family as the
    # non-truncated case above -- exercising confirmation over a truncated
    # page that happens to already hold the dominating member.
    reader = kenya_reader(force_truncated=True)

    group = enumerate_members(
        five_tuple=_KENYA_FIVE_TUPLE, slots={"gender": _UNBOUND}, graph=reader
    )

    assert group.truncated is True
    assert group.members == (
        Member(
            stat_var="LifeExpectancy_Person", constraints={}, declared_roles=("observationAbout",)
        ),
    )
    # The confirmed member's own constraint keys equal the identity's bound
    # keys exactly (empty: `gender` is unbound), and the only property
    # outside that set (`gender` itself) reads "aggregate" by default -- the
    # antichain reduction proves itself complete despite the page having
    # truncated, so no candidate the page never read could still be hiding
    # a dominating member.
    assert group.proven_complete is True
    # A truncated page still runs the ordinary forward-arc confirmation
    # step over the candidates its own page holds.
    assert reader.calls["node_arcs_many"] > 0
    # The group carries the reverse query's own `max_results` as
    # `candidate_cap`, so a caller reporting the truncated refusal states the
    # window it crossed without reading the module's constant back.
    assert group.candidate_cap == CANDIDATE_CAP


class _FewCandidatesTruncatedReader(FakeMemberArcReader):
    """A `FakeMemberArcReader` whose `node_subjects` reports
    `truncated=True` while returning exactly the candidates it was given --
    the shape `dc_graph.GraphClient.node_subjects` returns
    when a continuation token survives at or under the page cap, so the
    page holds fewer candidates than the cap. `candidates_considered` is
    then the page's own size, strictly below `candidate_cap`.
    """

    def __init__(
        self, *, arcs: dict[str, dict[str, tuple[str, ...]]], page: tuple[str, ...]
    ) -> None:
        super().__init__(arcs=arcs)
        self._page = page

    def node_subjects(self, dcid, *, property, filters=None, max_results):
        self.calls["node_subjects"] += 1
        return FakeSubjectPage(dcids=self._page, truncated=True)


def test_a_truncated_page_holding_fewer_candidates_than_the_cap_reports_what_it_read():
    """The truncated page returns 37 of `CANDIDATE_CAP` candidates; the set-bound result
    remains an unproven sample."""
    page = tuple(f"Count_Person_Detail{i:02d}" for i in range(37))
    arcs: dict[str, dict[str, tuple[str, ...]]] = {
        dcid: {
            "typeOf": ("StatisticalVariable",),
            "populationType": ("Person",),
            "measuredProperty": ("count",),
            "statType": ("measuredValue",),
            "measurementQualifier": (),
            "measurementDenominator": (),
            "constraintProperties": ("gender",),
            "observationProperties": (),
            "gender": ("Male",),
        }
        for dcid in page
    }
    reader = _FewCandidatesTruncatedReader(arcs=arcs, page=page)

    group = enumerate_members(
        five_tuple=_COUNT_FIVE_TUPLE, slots={"gender": ("set", ("Male", "Female"))}, graph=reader
    )

    assert group.truncated is True
    assert group.proven_complete is False
    assert group.candidates_considered == 37
    assert group.candidate_cap == CANDIDATE_CAP


def test_a_truncated_unproven_group_is_a_sample_of_the_window_not_the_population():
    """The withheld unconstrained root would dominate both gendered survivors.
    Because the page misses that root, its truncated result remains a sample."""
    arcs = {
        "Count_Person": {
            "typeOf": ("StatisticalVariable",),
            "populationType": ("Person",),
            "measuredProperty": ("count",),
            "statType": ("measuredValue",),
            "measurementQualifier": (),
            "measurementDenominator": (),
            "constraintProperties": (),
            "observationProperties": (),
        },
        **_male_female_arcs(),
    }
    reader = _WithheldFromWalkReader(
        arcs=arcs, withheld=frozenset({"Count_Person"}), force_truncated=True
    )

    group = enumerate_members(
        five_tuple=_COUNT_FIVE_TUPLE, slots={"gender": _UNBOUND}, graph=reader
    )

    assert group.truncated is True
    assert group.proven_complete is False
    assert {member.stat_var for member in group.members} == {
        "Count_Person_Male",
        "Count_Person_Female",
    }
    assert group.candidate_cap == CANDIDATE_CAP


def test_truncated_page_with_two_incomparable_members_is_not_proven_complete():
    # Neither `Count_Person_Male` nor `Count_Person_Female` has constraint
    # keys equal to the identity's own bound keys (empty: `gender` is
    # unbound here too), so no confirmed member can stand in for the whole
    # family -- the truncated page's own antichain does not prove itself
    # complete, and a dominating, less-constrained member could still sit
    # past the window.
    reader = male_female_only_reader(force_truncated=True)

    group = enumerate_members(
        five_tuple=_COUNT_FIVE_TUPLE, slots={"gender": _UNBOUND}, graph=reader
    )

    assert group.truncated is True
    assert {member.stat_var for member in group.members} == {
        "Count_Person_Male",
        "Count_Person_Female",
    }
    assert group.proven_complete is False


def test_truncated_page_with_a_not_declared_unbound_property_is_not_proven_complete():
    # The confirmed member's own constraint keys still equal the identity's
    # bound keys exactly (empty), but `gender` -- the one property outside
    # that set -- reads "not_declared" rather than the default "aggregate":
    # `dominates_constraint_keys` itself refuses to let a member's silence
    # on a "not_declared" property stand in for a genuine value of it, so an
    # unseen `gender`-bound sibling past the truncated page's own window
    # cannot be ruled out.
    reader = kenya_reader(force_truncated=True)

    group = enumerate_members(
        five_tuple=_KENYA_FIVE_TUPLE,
        slots={"gender": _UNBOUND},
        graph=reader,
        readings={"gender": "not_declared"},
    )

    assert group.truncated is True
    # `gender` reading "not_declared" also blocks the antichain reduction
    # itself, from the page's own candidates: `LifeExpectancy_Person`'s
    # empty constraint set no longer dominates its `gender`-bearing
    # siblings, so all three survive as an antichain, not just the one this
    # module's other truncated-page test confirms alone.
    assert {member.stat_var for member in group.members} == {
        "LifeExpectancy_Person",
        "LifeExpectancy_Person_Male",
        "LifeExpectancy_Person_Female",
    }
    assert group.proven_complete is False


def test_truncated_page_with_a_set_bound_slot_is_not_proven_complete():
    # A `"set"`-bound slot accepts more than one ref, so more than one true
    # member can legitimately share the same constraint keys including that
    # property -- one per ref the graph holds data for. A confirmed member
    # already carrying that property proves nothing about whether every
    # such ref was read before the page truncated, so the proof never fires
    # when a slot is `"set"`-bound, even though both confirmed members here
    # already have constraint keys equal to the identity's own bound keys.
    reader = male_female_only_reader(force_truncated=True)

    group = enumerate_members(
        five_tuple=_COUNT_FIVE_TUPLE,
        slots={"gender": ("set", ("Male", "Female"))},
        graph=reader,
    )

    assert group.truncated is True
    assert {member.stat_var for member in group.members} == {
        "Count_Person_Male",
        "Count_Person_Female",
    }
    assert group.proven_complete is False


def test_truncated_page_with_no_matching_candidate_confirms_an_empty_group():
    # The page finds no match, but truncation leaves the rest of the population unknown.
    reader = kenya_reader(force_truncated=True)

    group = enumerate_members(
        five_tuple=_KENYA_FIVE_TUPLE, slots={"gender": ("value", "Nonbinary")}, graph=reader
    )

    assert group.truncated is True
    assert group.members == ()
    assert group.proven_complete is False
    assert reader.calls["node_arcs_many"] > 0


def test_a_raw_count_between_1001_and_the_cap_confirms_every_candidate():
    # 1,001 candidates: past a 1,000-candidate count and well under the
    # measured cost bound, so a raw candidate count this size is confirmed
    # in full.
    arcs = {
        f"Count_Person_Bucket{i}": {
            "typeOf": ("StatisticalVariable",),
            "populationType": ("Person",),
            "measuredProperty": ("count",),
            "statType": ("measuredValue",),
            "measurementQualifier": (),
            "measurementDenominator": (),
            "constraintProperties": (),
            "observationProperties": (),
        }
        for i in range(1_001)
    }
    reader = FakeMemberArcReader(arcs=arcs)

    group = enumerate_members(
        five_tuple=_COUNT_FIVE_TUPLE, slots={}, graph=reader, member_cap=1_001
    )

    assert group.truncated is False
    assert group.candidates_considered == 1_001
    # Every candidate shares the same empty constraint set, so the
    # antichain reduction can never dominate one away: all 1,001 survive.
    assert len(group.members) == 1_001


def test_candidate_cap_still_confirms_the_truncated_page_and_can_exceed_member_cap():
    # Every candidate shares the same empty constraint set, so none
    # dominates any other and all `CANDIDATE_CAP` candidates the truncated
    # page holds survive the antichain reduction -- confirmed the same way
    # a complete page's candidates would be, past the default
    # `member_cap=700`. `CANDIDATE_CAP` bounds the raw read cost; it is
    # `member_cap`, not truncation, that refuses a fan-out this wide.
    arcs = {
        f"Count_Person_Bucket{i}": {
            "typeOf": ("StatisticalVariable",),
            "populationType": ("Person",),
            "measuredProperty": ("count",),
            "statType": ("measuredValue",),
            "measurementQualifier": (),
            "measurementDenominator": (),
            "constraintProperties": (),
            "observationProperties": (),
        }
        for i in range(CANDIDATE_CAP + 1)
    }
    reader = FakeMemberArcReader(arcs=arcs)

    with pytest.raises(MemberEnumerationTooLarge) as excinfo:
        enumerate_members(five_tuple=_COUNT_FIVE_TUPLE, slots={}, graph=reader)

    assert excinfo.value.count == CANDIDATE_CAP
    assert reader.calls["node_arcs_many"] > 0


def test_a_value_bound_slot_narrows_a_raw_population_over_the_cap_to_enumerate():
    # `CANDIDATE_CAP` candidates share the five-tuple but bind `bucket` to
    # a value the ask doesn't name, plus one further candidate bound to the
    # value the ask does name -- over `CANDIDATE_CAP` total, so the raw
    # five-tuple population alone would always truncate. Pushing the
    # `value`-bound `bucket` filter server-side narrows the reverse query
    # to the one matching candidate, well under the cap.
    arcs = {
        f"Count_Person_Other{i}": {
            "typeOf": ("StatisticalVariable",),
            "populationType": ("Person",),
            "measuredProperty": ("count",),
            "statType": ("measuredValue",),
            "measurementQualifier": (),
            "measurementDenominator": (),
            "constraintProperties": ("bucket",),
            "observationProperties": (),
            "bucket": ("Other",),
        }
        for i in range(CANDIDATE_CAP)
    }
    arcs["Count_Person_Target"] = {
        "typeOf": ("StatisticalVariable",),
        "populationType": ("Person",),
        "measuredProperty": ("count",),
        "statType": ("measuredValue",),
        "measurementQualifier": (),
        "measurementDenominator": (),
        "constraintProperties": ("bucket",),
        "observationProperties": (),
        "bucket": ("Target",),
    }
    reader = FakeMemberArcReader(arcs=arcs)

    group = enumerate_members(
        five_tuple=_COUNT_FIVE_TUPLE, slots={"bucket": ("value", "Target")}, graph=reader
    )

    assert group.truncated is False
    assert [member.stat_var for member in group.members] == ["Count_Person_Target"]


def test_truncation_still_applies_after_narrowing_by_a_value_bound_slot():
    # All candidates now share the value-bound slot too, so pushing it
    # server-side narrows nothing: the filtered population itself still
    # exceeds `CANDIDATE_CAP`, and the page truncates the same as an
    # unfiltered population would. Every candidate shares the same
    # constraint set (`{bucket}`, all bound to "Target"), so none dominates
    # any other and the truncated page's own candidates all survive
    # confirmation, past the default `member_cap=700`.
    arcs = {
        f"Count_Person_Target{i}": {
            "typeOf": ("StatisticalVariable",),
            "populationType": ("Person",),
            "measuredProperty": ("count",),
            "statType": ("measuredValue",),
            "measurementQualifier": (),
            "measurementDenominator": (),
            "constraintProperties": ("bucket",),
            "observationProperties": (),
            "bucket": ("Target",),
        }
        for i in range(CANDIDATE_CAP + 1)
    }
    reader = FakeMemberArcReader(arcs=arcs)

    with pytest.raises(MemberEnumerationTooLarge) as excinfo:
        enumerate_members(
            five_tuple=_COUNT_FIVE_TUPLE, slots={"bucket": ("value", "Target")}, graph=reader
        )

    assert excinfo.value.count == CANDIDATE_CAP
    assert reader.calls["node_arcs_many"] > 0


class _FilterRecordingReader(FakeMemberArcReader):
    """Records the last `filters` dict `node_subjects` received, for
    verifying which slot bindings `_reverse_query` pushes server-side
    without depending on `enumerate.py`'s own private helpers."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.last_filters: dict[str, str] = {}

    def node_subjects(self, dcid, *, property, filters=None, max_results):
        self.last_filters = dict(filters or {})
        return super().node_subjects(
            dcid, property=property, filters=filters, max_results=max_results
        )


def test_reverse_query_filters_carry_only_the_value_bound_slots():
    reader = _FilterRecordingReader(arcs={})

    enumerate_members(
        five_tuple=_KENYA_FIVE_TUPLE,
        slots={
            "gender": ("value", "Male"),
            "age": ("set", ("Years0", "Years1")),
            "region": _UNBOUND,
        },
        graph=reader,
    )

    assert reader.last_filters == {
        "populationType": "Person",
        "statType": "measuredValue",
        "gender": "Male",
    }


def test_member_cap_raises_after_the_antichain_reduction_when_nothing_dominates():
    # Every candidate shares the same empty constraint set, so none is a
    # proper subset of another and the antichain reduction keeps all 30 --
    # the reduced count equals the raw count here, and the cap still
    # refuses, naming the reduced (post-reduction) count.
    arcs = {
        f"Count_Person_Bucket{i}": {
            "typeOf": ("StatisticalVariable",),
            "populationType": ("Person",),
            "measuredProperty": ("count",),
            "statType": ("measuredValue",),
            "measurementQualifier": (),
            "measurementDenominator": (),
            "constraintProperties": (),
            "observationProperties": (),
        }
        for i in range(30)
    }
    reader = FakeMemberArcReader(arcs=arcs)

    with pytest.raises(MemberEnumerationTooLarge) as exc_info:
        enumerate_members(five_tuple=_COUNT_FIVE_TUPLE, slots={}, graph=reader, member_cap=25)

    assert exc_info.value.count == 30
    assert exc_info.value.member_cap == 25


def test_stable_antichain_larger_than_reprobe_budget_raises_at_its_full_size():
    """Every stable winner counts toward the post-verification cap, even
    when the batch contains more members than `_REPROBE_BUDGET`."""
    arcs = {
        f"Count_Person_Bucket{i:03d}": {
            "typeOf": ("StatisticalVariable",),
            "populationType": ("Person",),
            "measuredProperty": ("count",),
            "statType": ("measuredValue",),
            "measurementQualifier": (),
            "measurementDenominator": (),
            "constraintProperties": ("bucket",),
            "observationProperties": (),
            "bucket": (f"Value{i:03d}",),
        }
        for i in range(101)
    }
    reader = FakeMemberArcReader(arcs=arcs)

    with pytest.raises(MemberEnumerationTooLarge) as exc_info:
        enumerate_members(
            five_tuple=_COUNT_FIVE_TUPLE,
            slots={"bucket": _UNBOUND},
            graph=reader,
            member_cap=100,
        )

    assert exc_info.value.count == 101
    assert exc_info.value.member_cap == 100
    # Initial confirmation plus one complete stable-winner re-read: none of
    # the 101 members needs a single-dcid fallback or is discarded.
    assert reader.calls["node_arcs_many"] == 2


def test_bare_representative_dominates_more_than_the_cap_of_specific_siblings():
    # An all-unbound ask over a bare representative plus more than
    # `member_cap` more specific siblings: the raw consistent set (101)
    # exceeds `member_cap` (25), but the antichain reduction collapses
    # every sibling into the one dominating, least-constrained
    # representative before the cap is checked, so this enumerates cleanly
    # to that single member.
    arcs = {
        "Count_Person": {
            "typeOf": ("StatisticalVariable",),
            "populationType": ("Person",),
            "measuredProperty": ("count",),
            "statType": ("measuredValue",),
            "measurementQualifier": (),
            "measurementDenominator": (),
            "constraintProperties": (),
            "observationProperties": (),
        },
        **{
            f"Count_Person_Bucket{i}": {
                "typeOf": ("StatisticalVariable",),
                "populationType": ("Person",),
                "measuredProperty": ("count",),
                "statType": ("measuredValue",),
                "measurementQualifier": (),
                "measurementDenominator": (),
                "constraintProperties": ("bucket",),
                "observationProperties": (),
                "bucket": (f"Value{i}",),
            }
            for i in range(100)
        },
    }
    reader = FakeMemberArcReader(arcs=arcs)

    group = enumerate_members(
        five_tuple=_COUNT_FIVE_TUPLE,
        slots={"bucket": _UNBOUND},
        graph=reader,
        member_cap=25,
    )

    assert group.members == (
        Member(stat_var="Count_Person", constraints={}, declared_roles=("observationAbout",)),
    )


def test_surviving_members_with_different_roles_raise_naming_both_signatures():
    arcs = {
        "Count_Person_Male": {
            "typeOf": ("StatisticalVariable",),
            "populationType": ("Person",),
            "measuredProperty": ("count",),
            "statType": ("measuredValue",),
            "measurementQualifier": (),
            "measurementDenominator": (),
            "constraintProperties": ("gender",),
            "observationProperties": ("unitOfMeasure",),
            "gender": ("Male",),
        },
        "Count_Person_Female": {
            "typeOf": ("StatisticalVariable",),
            "populationType": ("Person",),
            "measuredProperty": ("count",),
            "statType": ("measuredValue",),
            "measurementQualifier": (),
            "measurementDenominator": (),
            "constraintProperties": ("gender",),
            "observationProperties": ("financingRecipient",),
            "gender": ("Female",),
        },
    }
    reader = FakeMemberArcReader(arcs=arcs)

    with pytest.raises(MemberRoleDisagreement) as exc_info:
        enumerate_members(five_tuple=_COUNT_FIVE_TUPLE, slots={"gender": _UNBOUND}, graph=reader)

    assert exc_info.value.roles_by_stat_var == {
        "Count_Person_Male": ("unitOfMeasure",),
        "Count_Person_Female": ("financingRecipient",),
    }
    message = str(exc_info.value)
    assert "Count_Person_Male" in message
    assert "Count_Person_Female" in message
    assert "unitOfMeasure" in message
    assert "financingRecipient" in message


def test_surviving_members_with_same_roles_in_different_wire_order_do_not_raise():
    arcs = {
        "Count_Person_Male": {
            "typeOf": ("StatisticalVariable",),
            "populationType": ("Person",),
            "measuredProperty": ("count",),
            "statType": ("measuredValue",),
            "measurementQualifier": (),
            "measurementDenominator": (),
            "constraintProperties": ("gender",),
            "observationProperties": ("unitOfMeasure", "measurementMethod"),
            "gender": ("Male",),
        },
        "Count_Person_Female": {
            "typeOf": ("StatisticalVariable",),
            "populationType": ("Person",),
            "measuredProperty": ("count",),
            "statType": ("measuredValue",),
            "measurementQualifier": (),
            "measurementDenominator": (),
            "constraintProperties": ("gender",),
            "observationProperties": ("measurementMethod", "unitOfMeasure"),
            "gender": ("Female",),
        },
    }
    reader = FakeMemberArcReader(arcs=arcs)

    group = enumerate_members(
        five_tuple=_COUNT_FIVE_TUPLE, slots={"gender": _UNBOUND}, graph=reader
    )

    assert {member.stat_var for member in group.members} == {
        "Count_Person_Male",
        "Count_Person_Female",
    }


def test_antichain_eliminated_role_divergence_does_not_raise():
    """`Count_Person_Male` disagrees with `Count_Person` on `declared_roles`,
    but the antichain reduction drops it (`{} ⊂ {gender}`) before the group
    is returned, so the disagreement is never real."""
    arcs = {
        "Count_Person": {
            "typeOf": ("StatisticalVariable",),
            "populationType": ("Person",),
            "measuredProperty": ("count",),
            "statType": ("measuredValue",),
            "measurementQualifier": (),
            "measurementDenominator": (),
            "constraintProperties": (),
            "observationProperties": (),
        },
        "Count_Person_Male": {
            "typeOf": ("StatisticalVariable",),
            "populationType": ("Person",),
            "measuredProperty": ("count",),
            "statType": ("measuredValue",),
            "measurementQualifier": (),
            "measurementDenominator": (),
            "constraintProperties": ("gender",),
            "observationProperties": ("unitOfMeasure",),
            "gender": ("Male",),
        },
    }
    reader = FakeMemberArcReader(arcs=arcs)

    group = enumerate_members(
        five_tuple=_COUNT_FIVE_TUPLE, slots={"gender": _UNBOUND}, graph=reader
    )

    assert group.members == (
        Member(stat_var="Count_Person", constraints={}, declared_roles=("observationAbout",)),
    )


def test_reconciliation_inconsistent_candidate_gates_role_check():
    """Ghost passes the reverse filter but has empty forward reads. It is reported inconsistent,
    so disagreement among the confirmed members does not raise."""
    arcs = {
        "Count_Person_Male": {
            "typeOf": ("StatisticalVariable",),
            "populationType": ("Person",),
            "measuredProperty": ("count",),
            "statType": ("measuredValue",),
            "measurementQualifier": (),
            "measurementDenominator": (),
            "constraintProperties": ("gender",),
            "observationProperties": ("unitOfMeasure",),
            "gender": ("Male",),
        },
        "Count_Person_Female": {
            "typeOf": ("StatisticalVariable",),
            "populationType": ("Person",),
            "measuredProperty": ("count",),
            "statType": ("measuredValue",),
            "measurementQualifier": (),
            "measurementDenominator": (),
            "constraintProperties": ("gender",),
            "observationProperties": ("financingRecipient",),
            "gender": ("Female",),
        },
        "Count_Person_Ghost": {
            "typeOf": ("StatisticalVariable",),
            "populationType": ("Person",),
            "measuredProperty": ("count",),
            "statType": ("measuredValue",),
            "measurementQualifier": (),
            "measurementDenominator": (),
            "constraintProperties": (),
            "observationProperties": (),
        },
    }
    reader = FlakyMemberArcReader(arcs=arcs, empty_always=["Count_Person_Ghost"])

    group = enumerate_members(
        five_tuple=_COUNT_FIVE_TUPLE, slots={"gender": _UNBOUND}, graph=reader
    )

    assert group.inconsistent_candidates == ("Count_Person_Ghost",)
    assert {member.stat_var for member in group.members} == {
        "Count_Person_Male",
        "Count_Person_Female",
    }
    assert {frozenset(member.declared_roles) for member in group.members} == {
        frozenset(("unitOfMeasure",)),
        frozenset(("financingRecipient",)),
    }


def test_reverification_keeps_stable_winners_beyond_the_fallback_budget():
    """33 stable winners are all accepted from the batch; the fallback
    budget applies only to winners whose batch re-read looks suspicious."""
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
        for i in range(33)
    }
    reader = FakeMemberArcReader(arcs=arcs)

    group = enumerate_members(
        five_tuple=_COUNT_FIVE_TUPLE,
        slots={"bucket": _UNBOUND},
        graph=reader,
        member_cap=40,
    )

    assert len(group.members) == 33
    assert group.inconsistent_candidates == ()
    assert reader.calls["node_arcs_many"] == 2


def test_reverification_reports_only_suspicious_winners_beyond_fallback_budget():
    """A batch degradation makes all 33 winners suspicious. The first 32
    recover through their single-dcid fallbacks; the unprobed remainder is
    still reported inconsistent."""
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
        for i in range(33)
    }
    reader = FlakyMemberArcReader(
        arcs=arcs,
        drop_second_batch=dict.fromkeys(arcs, "typeOf"),
    )

    group = enumerate_members(
        five_tuple=_COUNT_FIVE_TUPLE,
        slots={"bucket": _UNBOUND},
        graph=reader,
        member_cap=40,
    )

    assert len(group.members) == 32
    assert group.inconsistent_candidates == ("Count_Person_Bucket32",)
    assert reader.calls["node_arcs_many"] == 34


def test_candidate_missing_statistical_variable_raises_not_a_statistical_variable():
    """The batch matches the five-tuple and slot but finds no `StatisticalVariable` type.
    It raises without a second per-member graph read."""
    reader = FakeMemberArcReader(
        arcs={
            "Count_Person_Male": {
                "typeOf": (),
                "populationType": ("Person",),
                "measuredProperty": ("count",),
                "statType": ("measuredValue",),
                "measurementQualifier": (),
                "measurementDenominator": (),
                "constraintProperties": ("gender",),
                "observationProperties": (),
                "gender": ("Male",),
            },
        }
    )

    with pytest.raises(NotAStatisticalVariable) as exc_info:
        enumerate_members(
            five_tuple=_COUNT_FIVE_TUPLE,
            slots={"gender": ("value", "Male")},
            graph=reader,
        )

    assert exc_info.value.stat_var == "Count_Person_Male"
    assert exc_info.value.type_of == ()


def test_non_statistical_variable_failing_a_bound_slot_does_not_block_valid_members():
    """The `Female` slot mismatch excludes this non-StatVar before the type check; the valid
    `Male` member still resolves."""
    reader = FakeMemberArcReader(
        arcs={
            "Count_Person_Male": {
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
            "Count_Person_NotAVar": {
                "typeOf": (),
                "populationType": ("Person",),
                "measuredProperty": ("count",),
                "statType": ("measuredValue",),
                "measurementQualifier": (),
                "measurementDenominator": (),
                "constraintProperties": ("gender",),
                "observationProperties": (),
                "gender": ("Female",),
            },
        }
    )

    group = enumerate_members(
        five_tuple=_COUNT_FIVE_TUPLE, slots={"gender": ("value", "Male")}, graph=reader
    )

    assert [member.stat_var for member in group.members] == ["Count_Person_Male"]
    assert group.inconsistent_candidates == ()


def test_non_statistical_variable_dominated_in_the_antichain_does_not_block_valid_members():
    """The unconstrained member dominates the gendered candidate before type reverification.
    The non-StatVar loser is dropped; only `Count_Person` resolves."""
    reader = FakeMemberArcReader(
        arcs={
            "Count_Person": {
                "typeOf": ("StatisticalVariable",),
                "populationType": ("Person",),
                "measuredProperty": ("count",),
                "statType": ("measuredValue",),
                "measurementQualifier": (),
                "measurementDenominator": (),
                "constraintProperties": (),
                "observationProperties": (),
            },
            "Count_Person_Male": {
                "typeOf": (),
                "populationType": ("Person",),
                "measuredProperty": ("count",),
                "statType": ("measuredValue",),
                "measurementQualifier": (),
                "measurementDenominator": (),
                "constraintProperties": ("gender",),
                "observationProperties": (),
                "gender": ("Male",),
            },
        }
    )

    group = enumerate_members(
        five_tuple=_COUNT_FIVE_TUPLE, slots={"gender": _UNBOUND}, graph=reader
    )

    assert [member.stat_var for member in group.members] == ["Count_Person"]
    assert group.inconsistent_candidates == ()


def test_non_statistical_variable_on_the_reverse_arc_is_excluded_before_confirmation():
    """A reverse query also finds a `MeasurementDenominatorEnum` node (#876).
    Its non-StatVar `typeOf` excludes it before reduction, preserving the real count member."""
    reader = FakeMemberArcReader(
        arcs={
            "Count_Person": {
                "typeOf": ("StatisticalVariable",),
                "populationType": ("Person",),
                "measuredProperty": ("count",),
                "statType": ("measuredValue",),
                "measurementQualifier": (),
                "measurementDenominator": (),
                "constraintProperties": (),
                "observationProperties": (),
            },
            "Count_Person_IncomeOfUSDollar15000To24999": {
                "typeOf": ("MeasurementDenominatorEnum",),
                "populationType": ("Person",),
                "measuredProperty": ("count",),
                "statType": ("measuredValue",),
                "measurementQualifier": (),
                "measurementDenominator": (),
                "constraintProperties": (),
                "observationProperties": (),
            },
        }
    )

    group = enumerate_members(
        five_tuple=_COUNT_FIVE_TUPLE, slots={"gender": _UNBOUND}, graph=reader
    )

    assert [member.stat_var for member in group.members] == ["Count_Person"]
    assert group.inconsistent_candidates == ()


def test_candidate_with_absent_type_of_arc_still_appears_as_a_member():
    """A dropped `typeOf` arc says nothing about the node's type.
    The winner's single-dcid re-read restores it before the member is returned."""
    arcs = {
        "Count_Person": {
            "typeOf": ("StatisticalVariable",),
            "populationType": ("Person",),
            "measuredProperty": ("count",),
            "statType": ("measuredValue",),
            "measurementQualifier": (),
            "measurementDenominator": (),
            "constraintProperties": (),
            "observationProperties": (),
        },
        "Count_Person_Male": {
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
    }
    reader = FlakyMemberArcReader(arcs=arcs, drop_first_batch={"Count_Person": "typeOf"})

    group = enumerate_members(
        five_tuple=_COUNT_FIVE_TUPLE, slots={"gender": _UNBOUND}, graph=reader
    )

    assert [member.stat_var for member in group.members] == ["Count_Person"]


def test_candidate_with_statistical_variable_plus_another_type_stays_a_member():
    """A candidate whose `typeOf` includes `StatisticalVariable` alongside
    another type is still a StatVar: the exclusion only fires when the arc is
    present and excludes `StatisticalVariable`."""
    reader = FakeMemberArcReader(
        arcs={
            "Count_Person": {
                "typeOf": ("StatisticalVariable", "MeasurementDenominatorEnum"),
                "populationType": ("Person",),
                "measuredProperty": ("count",),
                "statType": ("measuredValue",),
                "measurementQualifier": (),
                "measurementDenominator": (),
                "constraintProperties": (),
                "observationProperties": (),
            },
        }
    )

    group = enumerate_members(
        five_tuple=_COUNT_FIVE_TUPLE, slots={"gender": _UNBOUND}, graph=reader
    )

    assert [member.stat_var for member in group.members] == ["Count_Person"]


def test_member_cap_default_is_700():
    # Health elects 241 members (benchmark 5); 700 holds a broad read near
    # 128 MiB and 13 s, per-member cost measured at 0.18 MiB/0.019 s.
    assert inspect.signature(enumerate_members).parameters["member_cap"].default == 700


# `extra_candidates`: a caller-posted StatVar dcid list unioned into the
# walked page before confirmation, standing in for a candidate the walk
# itself did not read (e.g. a root member past a truncated page's own
# window).


class _WithheldFromWalkReader(FakeMemberArcReader):
    """A `FakeMemberArcReader` whose `node_subjects` walk never returns a
    named dcid, even though `node_arcs_many` still holds it -- simulating a
    member sitting past a truncated page's own window, reachable only as a
    posted `extra_candidates` entry rather than something the walk itself
    read.
    """

    def __init__(
        self,
        *,
        arcs: dict[str, dict[str, tuple[str, ...]]],
        withheld: frozenset[str],
        force_truncated: bool = False,
    ) -> None:
        super().__init__(arcs=arcs, force_truncated=force_truncated)
        self._withheld = withheld

    def node_subjects(self, dcid, *, property, filters=None, max_results):
        page = super().node_subjects(
            dcid, property=property, filters=filters, max_results=max_results
        )
        return type(page)(
            dcids=tuple(d for d in page.dcids if d not in self._withheld), truncated=page.truncated
        )


_FOREIGN_ARCS: dict[str, dict[str, tuple[str, ...]]] = {
    "Amount_EconomicActivity": {
        "typeOf": ("StatisticalVariable",),
        "populationType": ("EconomicActivity",),
        "measuredProperty": ("amount",),
        "statType": ("measuredValue",),
        "measurementQualifier": (),
        "measurementDenominator": (),
        "constraintProperties": (),
        "observationProperties": (),
    },
}


def _male_female_arcs() -> dict[str, dict[str, tuple[str, ...]]]:
    return {
        "Count_Person_Male": {
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
        "Count_Person_Female": {
            "typeOf": ("StatisticalVariable",),
            "populationType": ("Person",),
            "measuredProperty": ("count",),
            "statType": ("measuredValue",),
            "measurementQualifier": (),
            "measurementDenominator": (),
            "constraintProperties": ("gender",),
            "observationProperties": (),
            "gender": ("Female",),
        },
    }


def test_a_root_extra_candidate_makes_a_truncated_page_prove_complete():
    # `Count_Person` sits past the truncated page's own window -- the walk
    # never returns it -- but the caller posts it as an extra candidate.
    # Its constraint keys equal the identity's own bound keys (empty:
    # `gender` is unbound), so once confirmed it dominates the two
    # gender-bound survivors the window did hold, and the antichain
    # reduction proves itself complete, exactly as it would if the walk
    # had read it directly (see
    # `test_truncated_page_still_confirms_a_dominating_member_inside_the_window`).
    arcs = {
        "Count_Person": {
            "typeOf": ("StatisticalVariable",),
            "populationType": ("Person",),
            "measuredProperty": ("count",),
            "statType": ("measuredValue",),
            "measurementQualifier": (),
            "measurementDenominator": (),
            "constraintProperties": (),
            "observationProperties": (),
        },
        **_male_female_arcs(),
    }
    reader = _WithheldFromWalkReader(
        arcs=arcs, withheld=frozenset({"Count_Person"}), force_truncated=True
    )

    group = enumerate_members(
        five_tuple=_COUNT_FIVE_TUPLE,
        slots={"gender": _UNBOUND},
        graph=reader,
        extra_candidates=("Count_Person",),
    )

    assert group.truncated is True
    assert [member.stat_var for member in group.members] == ["Count_Person"]
    assert group.proven_complete is True


def test_a_non_root_extra_candidate_does_not_prove_completeness():
    # `Count_Person_Nonbinary`, posted as an extra candidate past the
    # truncated page's own window, confirms as an ordinary member, but its
    # constraint keys are `{gender}`, not the identity's own empty bound
    # key set -- it dominates nothing and cannot stand in for the whole
    # family, so posting it does not make the reduction provably complete.
    arcs = {
        **_male_female_arcs(),
        "Count_Person_Nonbinary": {
            "typeOf": ("StatisticalVariable",),
            "populationType": ("Person",),
            "measuredProperty": ("count",),
            "statType": ("measuredValue",),
            "measurementQualifier": (),
            "measurementDenominator": (),
            "constraintProperties": ("gender",),
            "observationProperties": (),
            "gender": ("Nonbinary",),
        },
    }
    reader = _WithheldFromWalkReader(
        arcs=arcs, withheld=frozenset({"Count_Person_Nonbinary"}), force_truncated=True
    )

    group = enumerate_members(
        five_tuple=_COUNT_FIVE_TUPLE,
        slots={"gender": _UNBOUND},
        graph=reader,
        extra_candidates=("Count_Person_Nonbinary",),
    )

    assert group.truncated is True
    assert {member.stat_var for member in group.members} == {
        "Count_Person_Male",
        "Count_Person_Female",
        "Count_Person_Nonbinary",
    }
    assert group.proven_complete is False


def test_a_foreign_extra_candidate_is_excluded_and_changes_nothing():
    # `Amount_EconomicActivity` matches neither `_COUNT_FIVE_TUPLE`'s
    # populationType/measuredProperty nor anything in `slots`:
    # `_confirm_member` excludes it the ordinary way, the same as any
    # non-matching walked candidate would be. No refusal, no special case
    # -- the result is identical to the same ask with no extra candidates
    # at all.
    reader = FakeMemberArcReader(
        arcs={**_male_female_arcs(), **_FOREIGN_ARCS}, force_truncated=True
    )

    without_extra = enumerate_members(
        five_tuple=_COUNT_FIVE_TUPLE, slots={"gender": _UNBOUND}, graph=reader
    )
    with_extra = enumerate_members(
        five_tuple=_COUNT_FIVE_TUPLE,
        slots={"gender": _UNBOUND},
        graph=reader,
        extra_candidates=("Amount_EconomicActivity",),
    )

    assert with_extra == without_extra


def test_extra_candidates_already_in_the_walked_page_are_not_duplicated():
    # A posted dcid the walk already read is deduplicated against
    # `candidate_dcids` rather than confirmed twice.
    reader = male_female_only_reader()

    group = enumerate_members(
        five_tuple=_COUNT_FIVE_TUPLE,
        slots={"gender": _UNBOUND},
        graph=reader,
        extra_candidates=("Count_Person_Male",),
    )

    assert group.candidates_considered == 2


class _ProbeBudgetTrackingReader(FakeMemberArcReader):
    """Tracks the `individual_probe_budget` each `node_arcs_many` call
    carries, and how many dcids absent from `arcs` that call resolves --
    capped at the call's own budget, the same way
    `dc_graph.GraphClient.node_arcs_many`'s own individual-probe
    phase stops probing a still-empty suspect once its budget is spent and
    lets the group retry's own empty result stand for the rest (see that
    method's own docstring). A dcid present in `arcs` resolves the ordinary
    batched way, uncounted and unbounded, exactly as `FakeMemberArcReader`
    already does.
    """

    def __init__(self, *, arcs: dict[str, dict[str, tuple[str, ...]]]) -> None:
        super().__init__(arcs=arcs)
        self.probe_budgets: list[int | None] = []
        self.individual_probes = 0

    def node_arcs_many(
        self,
        dcids,
        *,
        properties,
        individual_probe_budget=None,
        deadline: float | None = None,
    ) -> dict[str, dict[str, tuple[str, ...]] | None]:
        self.probe_budgets.append(individual_probe_budget)
        self.calls["node_arcs_many"] += 1
        result: dict[str, dict[str, tuple[str, ...]] | None] = {}
        probed = 0
        for dcid in dcids:
            stat_var_arcs = self._arcs.get(dcid)
            if stat_var_arcs is None:
                if individual_probe_budget is None or probed < individual_probe_budget:
                    probed += 1
                    self.individual_probes += 1
                result[dcid] = None
                continue
            result[dcid] = {
                prop: tuple(dict.fromkeys(stat_var_arcs.get(prop, ()))) for prop in properties
            }
        return result


def test_extra_candidates_beyond_the_probe_budget_are_not_individually_probed():
    # 40 nonexistent extras, 8 more than `_EXTRA_CANDIDATE_PROBE_BUDGET`
    # (32). The walked-candidate call carries no budget at all (`None`), and
    # the extras call's own probing stops at the budget rather than paying
    # for one sequential single-dcid read per nonexistent dcid -- the real
    # members the walk finds are unaffected either way, since a nonexistent
    # extra was never going to match `slots`.
    reader = _ProbeBudgetTrackingReader(arcs=_male_female_arcs())
    fake_extras = tuple(f"NoSuchStatVar{i}" for i in range(_EXTRA_CANDIDATE_PROBE_BUDGET + 8))

    group = enumerate_members(
        five_tuple=_COUNT_FIVE_TUPLE,
        slots={"gender": _UNBOUND},
        graph=reader,
        extra_candidates=fake_extras,
    )

    # The first two calls are `_confirm_group`'s own split: walked
    # candidates unbudgeted, then extras budgeted. Later calls (e.g.
    # `_reverify_antichain`'s own re-verification round) never carry this
    # budget, so only the first two are pinned by position.
    assert reader.probe_budgets[:2] == [None, _EXTRA_CANDIDATE_PROBE_BUDGET]
    assert reader.probe_budgets.count(_EXTRA_CANDIDATE_PROBE_BUDGET) == 1
    assert reader.individual_probes == _EXTRA_CANDIDATE_PROBE_BUDGET
    assert {member.stat_var for member in group.members} == {
        "Count_Person_Male",
        "Count_Person_Female",
    }
