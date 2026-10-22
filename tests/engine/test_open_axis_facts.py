"""Tests open-constraint facts from served members and their publication on `ResolvedSpec`."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from code_index.shared import get_index
from search.engine.retrieval.bound_values import OpenAxisFact, open_constraint_axes
from search.engine.retrieval.member_facts import ArcsMemberFacts, IndexMemberFacts
from search.engine.spec_build import _open_axis_facts
from search.models import GraphRef, OpenConstraintAxisFact
from search.render.plan import render_sentence
from tests.engine._builders import conjoin_spec, shape_five_tuple
from tests.render._helpers import _spec, _what_slot

_SHARED_PURPOSE_A = (
    "Amount_FinancialFlow_Gross_Disbursement_CRSTypeOfFlow10_CRSTypeOfFinance110_"
    "CRSCooperationModalityA01_CRSPurposeCode11330_Nominal"
)
_SHARED_PURPOSE_B = (
    "Amount_FinancialFlow_Gross_Disbursement_CRSTypeOfFlow10_CRSTypeOfFinance110_"
    "CRSCooperationModalityA02_CRSPurposeCode11330_Nominal"
)
_NO_MODALITY = (
    "Amount_FinancialFlow_Gross_Disbursement_CRSTypeOfFlow10_CRSTypeOfFinance110_"
    "CRSPurposeCode11110_Nominal"
)


def _index_members(*dcids: str) -> dict[str, IndexMemberFacts]:
    by_dcid = {row.dcid: row for row in get_index().statvars}
    return {dcid: IndexMemberFacts(row=by_dcid[dcid]) for dcid in dcids}


def _arcs_from_real_row(dcid: str) -> dict:
    """A node_arcs-shaped dict asserting the same constraints as the real
    fixture row `dcid`, proving open_constraint_axes reads ArcsMemberFacts
    (recall-sourced) identically to IndexMemberFacts (index-sourced), the
    one property the MemberFacts protocol exists to guarantee."""
    row = next(r for r in get_index().statvars if r.dcid == dcid)
    return {
        "constraintProperties": {"nodes": [{"dcid": p} for p in row.constraint_properties]},
        **{p: {"nodes": [{"dcid": v}]} for p, v in row.constraint_properties.items()},
    }


def _fact(axes: tuple[OpenAxisFact, ...], prop: str) -> OpenAxisFact:
    return next(a for a in axes if a.property == prop)


# ---------------------------------------------------------------------------
# open_constraint_axes
# ---------------------------------------------------------------------------


def test_members_sharing_one_value_publish_it_as_not_varying():
    members = _index_members(_SHARED_PURPOSE_A, _SHARED_PURPOSE_B)
    axes = open_constraint_axes({}, members)
    purpose = _fact(axes, "crsPurpose")
    assert purpose.values == frozenset({"CRSPurposeCode11330"})
    assert purpose.undeclared_count == 0


def test_members_that_differ_publish_more_than_one_value():
    members = _index_members(_SHARED_PURPOSE_A, _SHARED_PURPOSE_B)
    axes = open_constraint_axes({}, members)
    modality = _fact(axes, "crsCooperationModality")
    assert modality.values == frozenset({"CRSCooperationModalityA01", "CRSCooperationModalityA02"})


def test_undeclared_count_equals_members_missing_the_key():
    members = _index_members(_SHARED_PURPOSE_A, _SHARED_PURPOSE_B, _NO_MODALITY)
    axes = open_constraint_axes({}, members)
    modality = _fact(axes, "crsCooperationModality")
    assert modality.member_values[_NO_MODALITY] is None
    assert modality.undeclared_count == 1
    assert None in modality.values


def test_a_bound_property_never_reaches_the_output():
    members = _index_members(_SHARED_PURPOSE_A, _SHARED_PURPOSE_B)
    axes = open_constraint_axes({"crsPurpose": ["CRSPurposeCode11330"]}, members)
    assert all(a.property != "crsPurpose" for a in axes)


def test_index_sourced_and_recall_sourced_members_agree():
    index_members = _index_members(_SHARED_PURPOSE_A, _SHARED_PURPOSE_B, _NO_MODALITY)
    arcs_members = {
        dcid: ArcsMemberFacts(arcs=_arcs_from_real_row(dcid))
        for dcid in (_SHARED_PURPOSE_A, _SHARED_PURPOSE_B, _NO_MODALITY)
    }

    from_index = {
        a.property: (a.values, a.undeclared_count) for a in open_constraint_axes({}, index_members)
    }
    from_arcs = {
        a.property: (a.values, a.undeclared_count) for a in open_constraint_axes({}, arcs_members)
    }
    assert from_index == from_arcs


def test_no_members_publishes_nothing_to_report():
    assert open_constraint_axes({}, {}) == ()


def test_a_member_with_no_constraint_properties_at_all_publishes_nothing():
    members = {"sv": ArcsMemberFacts(arcs={})}
    assert open_constraint_axes({}, members) == ()


# ---------------------------------------------------------------------------
# _open_axis_facts: the wire-facing adapter
# ---------------------------------------------------------------------------


def test_no_open_axis_publishes_none_not_an_empty_list():
    assert _open_axis_facts(()) is None


def test_an_axis_every_member_declares_nothing_for_still_publishes_its_own_entry():
    # Distinct from the no-axis case above: this axis is present in the
    # OpenAxisFact tuple _open_axis_facts is handed (some member's arc named
    # it, per open_constraint_axes' own contract), it is simply undeclared
    # on every one of these particular members.
    facts = (OpenAxisFact(property="crsCooperationModality", member_values={"a": None, "b": None}),)

    published = _open_axis_facts(facts)

    assert published is not None
    assert published == [
        OpenConstraintAxisFact(
            property="crsCooperationModality",
            varies=False,
            value=None,
            distinct_count=0,
            undeclared_count=2,
        )
    ]


def test_varying_axis_publishes_varies_true():
    members = _index_members(_SHARED_PURPOSE_A, _SHARED_PURPOSE_B)
    facts = open_constraint_axes({}, members)

    published = _open_axis_facts(facts)

    assert published is not None
    modality = next(f for f in published if f.property == "crsCooperationModality")
    assert modality.varies is True
    assert modality.value is None
    assert modality.distinct_count == 2
    purpose = next(f for f in published if f.property == "crsPurpose")
    assert purpose.varies is False
    assert purpose.distinct_count == 1
    assert purpose.value == GraphRef(ref="CRSPurposeCode11330", label="vocational training")


def test_agreed_value_absent_when_every_member_declares_nothing():
    # Here, `varies=False` means every member omits the axis.
    # The published value is None.
    facts = (
        OpenAxisFact(
            property="crsCooperationModality", member_values={"a": None, "b": None, "c": None}
        ),
    )

    published = _open_axis_facts(facts)

    assert published is not None
    assert published[0].value is None
    assert published[0].distinct_count == 0


def test_distinct_count_counts_every_distinct_declared_value_on_a_varying_axis():
    facts = (
        OpenAxisFact(
            property="crsCooperationModality",
            member_values={
                "a": "CRSCooperationModalityA01",
                "b": "CRSCooperationModalityA02",
                "c": None,
            },
        ),
    )

    published = _open_axis_facts(facts)

    assert published is not None
    assert published[0].varies is True
    assert published[0].value is None
    assert published[0].distinct_count == 2


def test_each_declared_value_publishes_its_member_count():
    members = _index_members(_SHARED_PURPOSE_A, _SHARED_PURPOSE_B, _NO_MODALITY)

    published = _open_axis_facts(open_constraint_axes({}, members))

    assert published is not None
    by_property = {f.property: f for f in published}
    assert [(v.value.ref, v.count) for v in by_property["crsPurpose"].values] == [
        ("CRSPurposeCode11330", 2),
        ("CRSPurposeCode11110", 1),
    ]
    assert by_property["crsPurpose"].values[0].value.label == "vocational training"
    assert [(v.value.ref, v.count) for v in by_property["crsCooperationModality"].values] == [
        ("CRSCooperationModalityA01", 1),
        ("CRSCooperationModalityA02", 1),
    ]
    for fact in published:
        assert sum(v.count for v in fact.values) + fact.undeclared_count == len(members)
        assert len(fact.values) == fact.distinct_count


def test_values_tied_on_member_count_order_by_label():
    """The composer's `ranked_values` rule: ties order by label, whatever the dcids' order."""
    facts = (
        OpenAxisFact(
            property="undataSex",
            member_values={"sv1": "undata/SEX-A", "sv2": "undata/SEX-B"},
            value_names={"undata/SEX-A": "Zeta", "undata/SEX-B": "Alpha"},
        ),
    )

    published = _open_axis_facts(facts)

    assert published is not None
    assert [v.value.ref for v in published[0].values] == ["undata/SEX-B", "undata/SEX-A"]


def _un_member_arcs(sex_name: str | None) -> dict:
    """A UN System member asserting Sex=Male, with or without the value's graph name."""
    node = {"dcid": "undata/SEX-M"} | ({"name": sex_name} if sex_name is not None else {})
    return {
        "constraintProperties": {"nodes": [{"dcid": "undataSex"}]},
        "undataSex": {"nodes": [node]},
    }


def test_a_value_the_code_index_lacks_publishes_its_graph_name():
    facts = open_constraint_axes({}, {"sv": ArcsMemberFacts(arcs=_un_member_arcs("Male"))})

    published = _open_axis_facts(facts)

    assert published is not None
    assert published[0].value == GraphRef(ref="undata/SEX-M", label="Male")


def test_a_value_with_no_graph_name_publishes_its_dcid():
    facts = open_constraint_axes({}, {"sv": ArcsMemberFacts(arcs=_un_member_arcs(None))})

    published = _open_axis_facts(facts)

    assert published is not None
    assert published[0].value == GraphRef(ref="undata/SEX-M", label="undata/SEX-M")


@pytest.mark.parametrize(
    ("prop", "value", "arc_name", "expected"),
    [
        pytest.param(
            "undataAge",
            "undata/AGE-M24T59",
            "24 to 59 months old",
            "ODA Grants for 24 to 59 months old.",
            id="un-value-renders-its-graph-name",
        ),
        pytest.param(
            "activitySource",
            "GrossDomesticProduction",
            "GrossDomesticProduction",
            "ODA Grants for gross domestic production.",
            id="base-value-named-by-its-dcid-renders-humanised",
        ),
    ],
)
def test_a_constant_value_the_code_index_lacks_renders_from_its_arc_name(
    prop: str, value: str, arc_name: str, expected: str
):
    """The live graph names `undata/AGE-M24T59` "24 to 59 months old" and names
    `GrossDomesticProduction` by its own dcid, which the renderer humanises."""
    arcs = {
        "constraintProperties": {"nodes": [{"dcid": prop}]},
        prop: {"nodes": [{"dcid": value, "name": arc_name}]},
    }
    published = _open_axis_facts(open_constraint_axes({}, {"sv": ArcsMemberFacts(arcs=arcs)}))

    spec = _spec(slots=[_what_slot("ODA Grants")], open_axis_facts=published)

    assert render_sentence(spec) == expected


# ---------------------------------------------------------------------------
# The field is absent from identity.
# ---------------------------------------------------------------------------


def test_open_axis_facts_never_affects_spec_id():
    shape = shape_five_tuple("Person", "count")
    spec = conjoin_spec(shape, "Val")

    fabricated = [
        OpenConstraintAxisFact(
            property="crsCooperationModality", varies=True, distinct_count=2, undeclared_count=1
        )
    ]
    with_facts = spec.model_copy(
        update={"resolved": spec.resolved.model_copy(update={"open_axis_facts": fabricated})}
    )

    assert with_facts.spec_id == spec.spec_id
    assert with_facts.definition == spec.definition


def test_open_axis_facts_forbids_extra_fields():
    # `extra="forbid"` makes an unexpected field a validation error.
    with pytest.raises(ValidationError):
        OpenConstraintAxisFact(
            property="p", varies=False, distinct_count=0, undeclared_count=0, extra="nope"
        )
