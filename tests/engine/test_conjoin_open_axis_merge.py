"""Tests open-axis facts across same-shape merges.
Merged facts cover all served members, including silent members.
"""

from __future__ import annotations

import pytest

from search.engine.retrieval.bound_values import OpenAxisFact
from search.engine.same_shape import collapse_same_shape
from search.models import GraphRef
from tests.engine._conjoin_fixtures import _region
from tests.engine.test_conjoin_measure_merge import _recipient_only_shape, _where_bound_spec


def test_a_merged_answer_derives_facts_from_the_merged_member_set_not_the_base():
    """Two members each declare a different value for the same open axis;
    the merge must union both, publishing varies=True, a read the base
    member alone (varies=False, one value) could never produce."""
    shape = _recipient_only_shape()
    spec_kenya = _where_bound_spec(shape, "country/KEN", "sv_kenya")
    spec_ethiopia = _where_bound_spec(shape, "country/ETH", "sv_ethiopia")
    r_a = _region(
        spec_kenya,
        "aid to Kenya",
        earliest_index=0,
        open_axis_facts=(OpenAxisFact(property="crsPurpose", member_values={"sv_kenya": "A01"}),),
    )
    r_b = _region(
        spec_ethiopia,
        "aid to Ethiopia",
        earliest_index=1,
        open_axis_facts=(
            OpenAxisFact(property="crsPurpose", member_values={"sv_ethiopia": "A02"}),
        ),
    )

    effective, residual = collapse_same_shape([r_a, r_b])

    assert len(effective) == 1
    assert residual == []
    merged = effective[0]
    assert merged.open_axis_facts == (
        OpenAxisFact(
            property="crsPurpose", member_values={"sv_kenya": "A01", "sv_ethiopia": "A02"}
        ),
    )
    published = merged.spec.resolved.open_axis_facts
    assert published is not None
    assert len(published) == 1
    assert published[0].property == "crsPurpose"
    assert published[0].varies is True
    assert published[0].undeclared_count == 0
    # Two distinct declared values across the merged group, disagreeing:
    # no single agreed value to publish, but the served breadth still counts.
    assert published[0].value is None
    assert published[0].distinct_count == 2


def test_every_member_declaring_nothing_agrees_and_counts_every_undeclared_member():
    """Neither region's own served members declare a value for the axis at
    all: declaring nothing is itself an agreeing value (varies=False),
    and the merged undeclared_count must count every region's undeclared
    members rather than reuse either region's own count alone."""
    shape = _recipient_only_shape()
    spec_kenya = _where_bound_spec(shape, "country/KEN", "sv_kenya")
    spec_ethiopia = _where_bound_spec(shape, "country/ETH", "sv_ethiopia")
    r_a = _region(
        spec_kenya,
        "aid to Kenya",
        earliest_index=0,
        open_axis_facts=(OpenAxisFact(property="crsPurpose", member_values={"sv_kenya": None}),),
    )
    r_b = _region(
        spec_ethiopia,
        "aid to Ethiopia",
        earliest_index=1,
        open_axis_facts=(
            OpenAxisFact(
                property="crsPurpose", member_values={"sv_ethiopia": None, "sv_ethiopia_b": None}
            ),
        ),
    )

    effective, _residual = collapse_same_shape([r_a, r_b])

    merged = effective[0]
    assert merged.open_axis_facts == (
        OpenAxisFact(
            property="crsPurpose",
            member_values={"sv_kenya": None, "sv_ethiopia": None, "sv_ethiopia_b": None},
        ),
    )
    published = merged.spec.resolved.open_axis_facts
    assert published is not None
    assert published[0].varies is False
    assert published[0].undeclared_count == 3
    # Every member agrees on declaring nothing: no value to publish.
    assert published[0].value is None
    assert published[0].distinct_count == 0


def test_regions_sharing_a_statvar_count_it_once():
    """A where merge of two regions served by the same StatVar, which leaves the axis
    undeclared, publishes one undeclared member: the served member count."""
    shape = _recipient_only_shape()
    spec_france = _where_bound_spec(shape, "country/FRA", "sv_shared")
    spec_germany = _where_bound_spec(shape, "country/DEU", "sv_shared")
    regions = [
        _region(
            spec,
            query,
            earliest_index=index,
            open_axis_facts=(
                OpenAxisFact(property="crsPurpose", member_values={"sv_shared": None}),
            ),
        )
        for index, (spec, query) in enumerate(
            [(spec_france, "aid to France"), (spec_germany, "aid to Germany")]
        )
    ]

    effective, _residual = collapse_same_shape(regions)

    (merged,) = effective
    published = merged.spec.resolved.open_axis_facts
    assert published is not None
    assert published[0].undeclared_count == 1


def test_one_member_declaring_and_another_silent_still_varies():
    """One region's own member declares a value, the other's declares
    nothing at all: the merged read must still register as differing
    (varies=True) rather than dropping the silent member's own signal, and
    still carries its undeclared_count forward."""
    shape = _recipient_only_shape()
    spec_kenya = _where_bound_spec(shape, "country/KEN", "sv_kenya")
    spec_ethiopia = _where_bound_spec(shape, "country/ETH", "sv_ethiopia")
    r_a = _region(
        spec_kenya,
        "aid to Kenya",
        earliest_index=0,
        open_axis_facts=(OpenAxisFact(property="crsPurpose", member_values={"sv_kenya": "A01"}),),
    )
    r_b = _region(
        spec_ethiopia,
        "aid to Ethiopia",
        earliest_index=1,
        open_axis_facts=(OpenAxisFact(property="crsPurpose", member_values={"sv_ethiopia": None}),),
    )

    effective, _residual = collapse_same_shape([r_a, r_b])

    merged = effective[0]
    assert merged.open_axis_facts == (
        OpenAxisFact(property="crsPurpose", member_values={"sv_kenya": "A01", "sv_ethiopia": None}),
    )
    published = merged.spec.resolved.open_axis_facts
    assert published is not None
    assert published[0].varies is True
    assert published[0].undeclared_count == 1
    # Varying because the silent member disagrees with the declaring one:
    # exactly one distinct declared value across the group, but no single
    # agreed value since not every member reached it.
    assert published[0].value is None
    assert published[0].distinct_count == 1


def test_both_members_agreeing_on_one_real_value_publishes_it():
    """Both regions' own served members declare the same single value: the
    merged read still agrees (varies=False), and the agreed value survives
    the merge as a GraphRef (dcid-as-label, since no label is fetched for
    it here)."""
    shape = _recipient_only_shape()
    spec_kenya = _where_bound_spec(shape, "country/KEN", "sv_kenya")
    spec_ethiopia = _where_bound_spec(shape, "country/ETH", "sv_ethiopia")
    r_a = _region(
        spec_kenya,
        "aid to Kenya",
        earliest_index=0,
        open_axis_facts=(OpenAxisFact(property="crsPurpose", member_values={"sv_kenya": "A01"}),),
    )
    r_b = _region(
        spec_ethiopia,
        "aid to Ethiopia",
        earliest_index=1,
        open_axis_facts=(
            OpenAxisFact(property="crsPurpose", member_values={"sv_ethiopia": "A01"}),
        ),
    )

    effective, _residual = collapse_same_shape([r_a, r_b])

    merged = effective[0]
    published = merged.spec.resolved.open_axis_facts
    assert published is not None
    assert published[0].varies is False
    assert published[0].value == GraphRef(ref="A01", label="A01")
    assert published[0].distinct_count == 1


def test_a_merged_agreed_value_keeps_the_graph_name_either_region_carried():
    shape = _recipient_only_shape()
    spec_kenya = _where_bound_spec(shape, "country/KEN", "sv_kenya")
    spec_ethiopia = _where_bound_spec(shape, "country/ETH", "sv_ethiopia")
    r_a = _region(
        spec_kenya,
        "aid to Kenya",
        earliest_index=0,
        open_axis_facts=(
            OpenAxisFact(property="undataSex", member_values={"sv_kenya": "undata/SEX-M"}),
        ),
    )
    r_b = _region(
        spec_ethiopia,
        "aid to Ethiopia",
        earliest_index=1,
        open_axis_facts=(
            OpenAxisFact(
                property="undataSex",
                member_values={"sv_ethiopia": "undata/SEX-M"},
                value_names={"undata/SEX-M": "Male"},
            ),
        ),
    )

    effective, _residual = collapse_same_shape([r_a, r_b])

    published = effective[0].spec.resolved.open_axis_facts
    assert published is not None
    assert published[0].value == GraphRef(ref="undata/SEX-M", label="Male")


@pytest.mark.parametrize(
    ("ethiopia_serves", "ethiopia_facts", "counts", "undeclared"),
    [
        pytest.param(
            ("sv_shared", "sv_ethiopia"),
            (
                OpenAxisFact(
                    property="crsPurpose", member_values={"sv_shared": "A01", "sv_ethiopia": "A02"}
                ),
            ),
            [("A01", 1), ("A02", 1)],
            1,
            id="both-parts-have-the-fact",
        ),
        pytest.param(
            ("sv_ethiopia",),
            (),
            [("A01", 1)],
            2,
            id="a-part-without-the-fact-counts-as-undeclared",
        ),
    ],
)
def test_merged_value_counts_and_undeclared_count_sum_to_the_served_statvars(
    ethiopia_serves, ethiopia_facts, counts, undeclared
):
    """A where-axis merge can serve one StatVar at both places, and a part whose served
    StatVars never assert the property has no fact for it. Each value's count and
    undeclared_count still sum to the distinct served StatVars."""
    shape = _recipient_only_shape()
    spec_kenya = _where_bound_spec(shape, "country/KEN", "sv_shared", also_serves=("sv_kenya",))
    spec_ethiopia = _where_bound_spec(
        shape, "country/ETH", ethiopia_serves[0], also_serves=ethiopia_serves[1:]
    )
    r_a = _region(
        spec_kenya,
        "aid to Kenya",
        earliest_index=0,
        open_axis_facts=(
            OpenAxisFact(
                property="crsPurpose", member_values={"sv_shared": "A01", "sv_kenya": None}
            ),
        ),
    )
    r_b = _region(
        spec_ethiopia, "aid to Ethiopia", earliest_index=1, open_axis_facts=ethiopia_facts
    )

    effective, _residual = collapse_same_shape([r_a, r_b])

    merged = effective[0].spec.resolved
    (published,) = merged.open_axis_facts or []
    assert [(v.value.ref, v.count) for v in published.values] == counts
    assert published.undeclared_count == undeclared
    assert (
        sum(v.count for v in published.values) + published.undeclared_count
        == merged.stat_vars.member_count
    )


def test_a_group_with_no_open_axis_facts_at_all_publishes_none():
    """Neither member declares any open constraint property: the merge
    must not fabricate one, matching build_spec's own default."""
    shape = _recipient_only_shape()
    spec_kenya = _where_bound_spec(shape, "country/KEN", "sv_kenya")
    spec_ethiopia = _where_bound_spec(shape, "country/ETH", "sv_ethiopia")
    r_a = _region(spec_kenya, "aid to Kenya", earliest_index=0)
    r_b = _region(spec_ethiopia, "aid to Ethiopia", earliest_index=1)

    effective, _residual = collapse_same_shape([r_a, r_b])

    merged = effective[0]
    assert merged.open_axis_facts == ()
    assert merged.spec.resolved.open_axis_facts is None
