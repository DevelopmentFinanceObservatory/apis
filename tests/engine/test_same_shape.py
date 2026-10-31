"""Tests how `collapse_same_shape` merges role populations and StatVar counts.

Each recipient has its own donor population. The merged answer unions known donors and reports
unknown when any member's population is unknown.
"""

from __future__ import annotations

import dataclasses

import pytest

from search.engine.region.result import RegionResult
from search.engine.same_shape import collapse_same_shape
from search.engine.spec_build import NamedGroup, RoleAxisFactInput, build_spec
from search.heading.departure_fact import DepartureFact
from search.models import (
    BindingUnbound,
    BindingValue,
    CoverageBare,
    Entity,
    EntityGroup,
    EntityOptions,
    EntityRoleDirectional,
    GraphRef,
    RoleAxis,
    RoleAxisFact,
    Shape,
    Slot,
    SlotKey,
    SlotOption,
    SlotValue,
    StatVar,
)
from tests.engine._conjoin_fixtures import _region
from tests.engine.test_conjoin_measure_merge import _RECIPIENT_PROP, _gr, _recipient_only_shape


def _entity_options(role: RoleAxis, dcids: frozenset[str] | None) -> list[EntityOptions]:
    """Mirrors region/option_coverage.py's _entity_options_from_facts for a
    single role axis, at test-fixture scale: real dcid-as-label options when
    the population is known, the explicit-unknown placeholder otherwise."""
    if dcids is None:
        return [
            EntityOptions(entity_ref=None, role=role, options=[], truncated=True, source="observed")
        ]
    options = [
        SlotOption(value=SlotValue(ref=GraphRef(ref=dcid, label=dcid), value_kind="entity"))
        for dcid in sorted(dcids)
    ]
    return [
        EntityOptions(
            entity_ref=None, role=role, options=options, truncated=False, source="observed"
        )
    ]


def _role_bound_region(
    recipient_dcid: str,
    *,
    origin_dcids: frozenset[str] | None,
    earliest_index: int,
    options_listed: bool = True,
) -> RegionResult:
    """One recipient's answer with its own donor population. Without ``options_listed``
    the Spec withholds its donor options, as an answer read without option coverage
    does, and only the region's role facts hold the population."""
    shape = _recipient_only_shape()
    where_slot = Slot(
        key=SlotKey(axis="where", property=_gr(_RECIPIENT_PROP), label="recipient"),
        binding=BindingValue(value=SlotValue(ref=_gr(recipient_dcid), value_kind="entity")),
    )
    entity = Entity(
        ref=_gr(recipient_dcid),
        role=EntityRoleDirectional(role=_gr(_RECIPIENT_PROP), axis="destination"),
    )
    facts = (RoleAxisFactInput(role="origin", member_dcids=origin_dcids),)
    spec = build_spec(
        open_roles=(),
        measure_elected_properties=frozenset(),
        shape=shape,
        slots=[where_slot],
        stat_vars=[
            StatVar(
                ref=_gr(f"sv_{recipient_dcid}"),
                shape_id=shape.shape_id,
                slot_values=[],
                constraint_properties=[],
            )
        ],
        entities=[entity],
        coverage=CoverageBare(has_data=True),
        role_axis_facts=facts,
        entity_options=_entity_options("origin", origin_dcids if options_listed else None),
    )
    region = _region(spec, f"aid to {recipient_dcid}", earliest_index=earliest_index)
    return dataclasses.replace(region, role_axis_facts=facts)


@pytest.mark.parametrize("options_listed", [True, False], ids=["listed", "withheld"])
def test_a_merged_answer_unions_each_members_own_known_origin_population(options_listed: bool):
    """Two recipients, each with its own known and partly overlapping donor population:
    the merge counts the deduplicated union, whether or not the members' Specs list their
    donors. Listed donors merge into the union, and withheld ones stay withheld."""
    r_a = _role_bound_region(
        "country/KEN",
        origin_dcids=frozenset({"country/FRA", "country/DEU"}),
        earliest_index=0,
        options_listed=options_listed,
    )
    r_b = _role_bound_region(
        "country/UGA",
        origin_dcids=frozenset({"country/DEU", "country/JPN"}),
        earliest_index=1,
        options_listed=options_listed,
    )

    effective, residual = collapse_same_shape([r_a, r_b])

    assert len(effective) == 1
    assert residual == []
    merged = effective[0]

    facts = merged.spec.resolved.role_axis_facts
    assert facts == [
        RoleAxisFact(
            role="origin", member_count=3, members_available=True, holds_multilaterals=False
        )
    ]
    assert merged.role_axis_facts == (
        RoleAxisFactInput(
            role="origin", member_dcids=frozenset({"country/FRA", "country/DEU", "country/JPN"})
        ),
    )

    (options,) = merged.spec.resolved.entity_options
    assert options.role == "origin"
    dcids = {o.value.ref.ref for o in options.options if o.value.ref}
    if options_listed:
        assert (dcids, options.truncated) == ({"country/FRA", "country/DEU", "country/JPN"}, False)
    else:
        assert (dcids, options.truncated) == (set(), True)


def test_a_merged_answer_refuses_to_claim_a_population_when_a_member_is_unknown():
    """One recipient's own donor population is known, the other's could not
    be derived (a member_dcids=None corner on that member's own resolve):
    the merge cannot honestly de-duplicate a union from only one known
    side, so it must publish an explicit unknown on both fields rather
    than silently reporting only the known member's own population."""
    r_a = _role_bound_region(
        "country/KEN", origin_dcids=frozenset({"country/FRA"}), earliest_index=0
    )
    r_b = _role_bound_region("country/UGA", origin_dcids=None, earliest_index=1)

    effective, residual = collapse_same_shape([r_a, r_b])

    assert len(effective) == 1
    assert residual == []
    merged = effective[0]

    facts = merged.spec.resolved.role_axis_facts
    # The fixture's StatVars belong to no dataset the code index lists, so no served dataset
    # publishes a multilateral donor either.
    assert facts == [
        RoleAxisFact(
            role="origin", member_count=None, members_available=False, holds_multilaterals=False
        )
    ]

    entity_opts = merged.spec.resolved.entity_options
    assert len(entity_opts) == 1
    assert entity_opts[0].role == "origin"
    assert entity_opts[0].options == []
    assert entity_opts[0].truncated is True


# ---------------------------------------------------------------------------
# collapse_same_shape: resolved.stat_vars.member_count merge rule (#893)
# ---------------------------------------------------------------------------
#
# A where-axis merge groups specs that differ only on place (e.g. "population
# of Kenya and Uganda"), and the same StatVar is typically confirmed at every
# member's own place -- summing member_count would double-count the
# intersection, so same_shape.py publishes max() instead, always marked
# inexact. A what/how merge groups members that differ from each other on
# the diff key's own bound value, but that is not a disjointness guarantee:
# a StatVar whose own constraints are silent on the diff axis's property
# survives every bound value on that axis (role_resolve.py's
# graph_confirm_resolve keeps a member when `constraints.get(prop_dcid)` is
# None) and can land in more than one member's own .items, so summing would
# double-count it. The merge instead reads all_stat_vars, the ref-deduped
# union of every member's own .items already built above: len(all_stat_vars)
# is the true union size when every member's own .items is its complete
# population (member_count_exact true and truncated false on every member),
# and otherwise the merge publishes max(len(all_stat_vars), the largest
# member count), the tightest lower bound available when some member's own
# .items is itself only a sample. A zero-diff roster merge (two mentions
# that resolve to the same identity, differing only in which unbound slots
# each one's own extraction happened to enumerate) is neither: every merge
# gate already pins one resolved identity across the group, so each
# member's own member_count measures the same population rather than a
# disjoint or overlapping one, and the merge publishes max() of the member
# counts (the tightest available lower bound of that one population), exact
# whenever any member's own count is.


def test_a_merged_answer_keeps_every_members_departure_facts():
    """A fact survives with its warning, so the later member's fact reaches the merge."""
    first = _role_bound_region(
        "country/KEN", origin_dcids=frozenset({"country/FRA"}), earliest_index=0
    )
    second = _role_bound_region(
        "country/UGA", origin_dcids=frozenset({"country/DEU"}), earliest_index=1
    )
    first_fact = DepartureFact(code="direction_discarded", binding_ref="where:", asked="Kenya")
    second_fact = DepartureFact(code="direction_discarded", binding_ref="where:", asked="Uganda")
    first = dataclasses.replace(first, departure_facts={("direction_discarded", "a"): first_fact})
    second = dataclasses.replace(
        second, departure_facts={("direction_discarded", "b"): second_fact}
    )

    (merged,), _ = collapse_same_shape([first, second])

    assert merged.departure_facts == {
        ("direction_discarded", "a"): first_fact,
        ("direction_discarded", "b"): second_fact,
    }


def _where_bound_spec_with_member_count(
    recipient_dcid: str, sv_dcid: str, *, member_count: int, member_count_exact: bool = True
):
    shape = _recipient_only_shape()
    where_slot = Slot(
        key=SlotKey(axis="where", property=_gr(_RECIPIENT_PROP), label="recipient"),
        binding=BindingValue(value=SlotValue(ref=_gr(recipient_dcid), value_kind="entity")),
    )
    entity = Entity(
        ref=_gr(recipient_dcid),
        role=EntityRoleDirectional(role=_gr(_RECIPIENT_PROP), axis="destination"),
    )
    return build_spec(
        open_roles=(),
        measure_elected_properties=frozenset(),
        shape=shape,
        slots=[where_slot],
        stat_vars=[
            StatVar(
                ref=_gr(sv_dcid), shape_id=shape.shape_id, slot_values=[], constraint_properties=[]
            )
        ],
        entities=[entity],
        coverage=CoverageBare(has_data=True),
        stat_vars_member_count=member_count,
        stat_vars_member_count_exact=member_count_exact,
    )


def test_where_axis_merge_publishes_max_of_member_counts_marked_inexact():
    """Where-axis counts 5 and 7 overlap, so the merged lower bound is 7 and inexact."""
    spec_kenya = _where_bound_spec_with_member_count("country/KEN", "sv_kenya", member_count=5)
    spec_uganda = _where_bound_spec_with_member_count("country/UGA", "sv_uganda", member_count=7)
    r_a = _region(spec_kenya, "population of Kenya", earliest_index=0)
    r_b = _region(spec_uganda, "population of Uganda", earliest_index=1)

    effective, residual = collapse_same_shape([r_a, r_b])

    assert len(effective) == 1
    assert residual == []
    merged = effective[0].spec.resolved.stat_vars

    assert merged.member_count == 7
    assert merged.member_count_exact is False


def _recipient_and_open_what_shape() -> Shape:
    return Shape(
        shape_id="recipient_and_open_what_test_shape",
        label="Test recipient-plus-open-what shape",
        population_type=_gr("FinancialFlow"),
        measured_property=_gr("amount"),
        stat_type=_gr("measuredValue"),
        measurement_qualifier=None,
        measurement_denominator=None,
        slot_keys=[
            SlotKey(axis="where", property=_gr(_RECIPIENT_PROP), label="recipient"),
            SlotKey(axis="what", property=_gr("prop/B"), label="prop B"),
        ],
        member_count=1,
    )


def _roster_variant_spec(
    recipient_dcid: str,
    sv_dcid: str,
    *,
    member_count: int,
    member_count_exact: bool = True,
    carry_open_what_slot: bool,
):
    """Same identity as _where_bound_spec_with_member_count's shape family --
    same recipient, same StatVar -- but built from
    _recipient_and_open_what_shape, whose second slot key ("what"/prop B) is
    never bound. One variant's own resolved.slots roster carries an explicit
    BindingUnbound Slot for that key, the other's roster omits the key
    entirely; _slot_signature reads both as the same ("unbound",) state, so
    the two compare as a zero-differing-key, asymmetric-roster merge rather
    than a real conflict.
    """
    shape = _recipient_and_open_what_shape()
    where_slot = Slot(
        key=SlotKey(axis="where", property=_gr(_RECIPIENT_PROP), label="recipient"),
        binding=BindingValue(value=SlotValue(ref=_gr(recipient_dcid), value_kind="entity")),
    )
    slots = [where_slot]
    if carry_open_what_slot:
        slots.append(
            Slot(
                key=SlotKey(axis="what", property=_gr("prop/B"), label="prop B"),
                binding=BindingUnbound(),
            )
        )
    entity = Entity(
        ref=_gr(recipient_dcid),
        role=EntityRoleDirectional(role=_gr(_RECIPIENT_PROP), axis="destination"),
    )
    return build_spec(
        open_roles=(),
        measure_elected_properties=frozenset(),
        shape=shape,
        slots=slots,
        stat_vars=[
            StatVar(
                ref=_gr(sv_dcid), shape_id=shape.shape_id, slot_values=[], constraint_properties=[]
            )
        ],
        entities=[entity],
        coverage=CoverageBare(has_data=True),
        stat_vars_member_count=member_count,
        stat_vars_member_count_exact=member_count_exact,
    )


def test_zero_diff_roster_merge_reports_the_shared_population_not_its_double():
    """Two mentions of one identity keep count 5 despite different unbound-slot rosters."""
    spec_a = _roster_variant_spec(
        "country/KEN", "sv_kenya", member_count=5, carry_open_what_slot=False
    )
    spec_b = _roster_variant_spec(
        "country/KEN", "sv_kenya", member_count=5, carry_open_what_slot=True
    )
    r_a = _region(spec_a, "population of Kenya", earliest_index=0)
    r_b = _region(spec_b, "population of Kenya", earliest_index=1)

    effective, residual = collapse_same_shape([r_a, r_b])

    assert len(effective) == 1
    assert residual == []
    merged = effective[0].spec.resolved.stat_vars

    assert merged.member_count == 5
    assert merged.member_count_exact is True


def test_zero_diff_roster_merge_takes_the_max_and_is_exact_if_any_member_is():
    """The same population has counts 5 and 3; the exact count 5 is the merged result."""
    spec_a = _roster_variant_spec(
        "country/KEN",
        "sv_kenya",
        member_count=5,
        member_count_exact=True,
        carry_open_what_slot=False,
    )
    spec_b = _roster_variant_spec(
        "country/KEN",
        "sv_kenya",
        member_count=3,
        member_count_exact=False,
        carry_open_what_slot=True,
    )
    r_a = _region(spec_a, "population of Kenya", earliest_index=0)
    r_b = _region(spec_b, "population of Kenya", earliest_index=1)

    effective, residual = collapse_same_shape([r_a, r_b])

    assert len(effective) == 1
    assert residual == []
    merged = effective[0].spec.resolved.stat_vars

    assert merged.member_count == 5
    assert merged.member_count_exact is True


def _what_axis_shape() -> Shape:
    return Shape(
        shape_id="what_axis_member_count_test_shape",
        label="Test what-axis shape",
        population_type=_gr("TestPop"),
        measured_property=_gr("testMeasure"),
        stat_type=_gr("measuredValue"),
        measurement_qualifier=None,
        measurement_denominator=None,
        slot_keys=[SlotKey(axis="what", property=_gr("prop/A"), label="prop A")],
        member_count=1,
    )


def _what_bound_spec_with_member_count(
    value_dcid: str, sv_dcid: str, *, member_count: int, member_count_exact: bool = True
):
    shape = _what_axis_shape()
    slot = Slot(
        key=SlotKey(axis="what", property=_gr("prop/A"), label="prop A"),
        binding=BindingValue(value=SlotValue(ref=_gr(value_dcid), value_kind="enum_value")),
    )
    return build_spec(
        open_roles=(),
        measure_elected_properties=frozenset(),
        shape=shape,
        slots=[slot],
        stat_vars=[
            StatVar(
                ref=_gr(sv_dcid), shape_id=shape.shape_id, slot_values=[], constraint_properties=[]
            )
        ],
        entities=[],
        coverage=CoverageBare(has_data=True),
        stat_vars_member_count=member_count,
        stat_vars_member_count_exact=member_count_exact,
    )


def test_what_axis_merge_does_not_double_count_a_shared_stat_var():
    """A shared StatVar counts once across what-axis values."""
    spec_a = _what_bound_spec_with_member_count("male", "sv/shared", member_count=1)
    spec_b = _what_bound_spec_with_member_count("female", "sv/shared", member_count=1)
    r_a = _region(spec_a, "test a", earliest_index=0)
    r_b = _region(spec_b, "test b", earliest_index=1)

    effective, residual = collapse_same_shape([r_a, r_b])

    assert len(effective) == 1
    assert residual == []
    merged = effective[0].spec.resolved.stat_vars

    assert merged.member_count == 1
    assert merged.member_count_exact is True


def test_what_axis_merge_publishes_the_floor_when_both_members_are_exact_but_truncated():
    """Truncated lists cannot prove the union of counts 4 and 6.

    The merged count is 6 and inexact.
    """
    spec_a = _what_bound_spec_with_member_count("c1", "sv/a", member_count=4)
    spec_b = _what_bound_spec_with_member_count("c2", "sv/b", member_count=6)
    r_a = _region(spec_a, "test a", earliest_index=0)
    r_b = _region(spec_b, "test b", earliest_index=1)

    effective, residual = collapse_same_shape([r_a, r_b])

    assert len(effective) == 1
    assert residual == []
    merged = effective[0].spec.resolved.stat_vars
    assert merged.member_count == 6
    assert merged.member_count_exact is False


def test_what_axis_merge_publishes_the_floor_when_a_member_is_truncated_or_inexact():
    """One inexact item list makes the deduplicated union a lower bound."""
    spec_a = _what_bound_spec_with_member_count("c1", "sv/a", member_count=4)
    spec_b = _what_bound_spec_with_member_count(
        "c2", "sv/b", member_count=6, member_count_exact=False
    )
    r_a = _region(spec_a, "test a", earliest_index=0)
    r_b = _region(spec_b, "test b", earliest_index=1)

    effective, _residual = collapse_same_shape([r_a, r_b])

    merged = effective[0].spec.resolved.stat_vars
    assert merged.member_count == 6
    assert merged.member_count_exact is False


_G7 = NamedGroup(
    label="G7",
    dcid=None,
    member_dcids=frozenset({"country/CAN", "country/FRA"}),
    total="GroupOfSeven",
)


def _g7_region(recipient_dcid: str, *, earliest_index: int) -> RegionResult:
    """Aid from the G7, read as its members, to one recipient."""
    shape = _recipient_only_shape()
    where_slot = Slot(
        key=SlotKey(axis="where", property=_gr(_RECIPIENT_PROP), label="recipient"),
        binding=BindingValue(value=SlotValue(ref=_gr(recipient_dcid), value_kind="entity")),
    )
    donor = EntityRoleDirectional(role=_gr("financingProvider"), axis="origin")
    entities = [
        Entity(
            ref=_gr(recipient_dcid),
            role=EntityRoleDirectional(role=_gr(_RECIPIENT_PROP), axis="destination"),
        ),
        *(Entity(ref=_gr(dcid), role=donor) for dcid in sorted(_G7.member_dcids)),
    ]
    spec = build_spec(
        open_roles=(),
        measure_elected_properties=frozenset(),
        shape=shape,
        slots=[where_slot],
        stat_vars=[
            StatVar(
                ref=_gr(f"sv_{recipient_dcid}"),
                shape_id=shape.shape_id,
                slot_values=[],
                constraint_properties=[],
            )
        ],
        entities=entities,
        coverage=CoverageBare(has_data=True),
        named_groups=(_G7,),
    )
    return _region(spec, f"aid from the G7 to {recipient_dcid}", earliest_index=earliest_index)


def test_a_merged_answer_keeps_the_total_its_members_reading_leaves_aside():
    """Each member reads the node-less G7 as its members and keeps its total aside
    (`ResolvedSpec.group_totals`). The merged answer, rebuilt from the members' groups,
    keeps it too, so its departure can still name what the members leave out."""
    first = _g7_region("country/KEN", earliest_index=0)
    second = _g7_region("country/UGA", earliest_index=1)
    assert first.spec.resolved.group_totals == {"origin": "GroupOfSeven"}

    (merged,), _ = collapse_same_shape([first, second])

    assert (merged.spec.resolved.entity_groups or {}).get("origin") == EntityGroup(label="G7")
    assert merged.spec.resolved.group_totals == {"origin": "GroupOfSeven"}
