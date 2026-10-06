"""Test measure-head availability pairing and its pruning rules.

A pair must share population type, constraint values, and rate wording. Pruning
uses this rule only for named places. Denominator-resolution pairing also covers
entity-less queries. The fixture recall checks under-five and under-14 cohorts.
"""

from __future__ import annotations

import json
from pathlib import Path

from search.engine.quantity import NO_REQUIREMENT, quantity_requirement
from search.engine.region.availability import (
    _are_availability_siblings,
    _prune_availability_siblings,
    _shapes_are_measure_head_siblings,
)
from search.engine.retrieval.derive import derive_shapes
from search.engine.shape import ShapeDraft
from tests.fixtures import FakeGraph

_FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"

# Members from the "child mortality rate" recall.
_UNDER_FIVE_RATE_SV = "MortalityRate_Person_Upto4Years_AsFractionOf_Count_BirthEvent_LiveBirth"
_UNDER_14_COUNT_SV = "Count_Death_Upto14Years_AsAFractionOf_Count_Person_Upto14Years"


def _shape(
    shape_id: str,
    *,
    label: str | None = None,
    pop_type: str = "Person",
    meas_prop: str = "count",
    meas_denom_dcid: str | None = None,
    meas_denom_unresolved: bool = False,
    place_available: bool | None = None,
    # `None` is a meaningful caller-supplied value here (an unstamped/
    # unestablished taxonomy, distinct from an established-but-empty `{}`),
    # so a `None`-sentinel default would collapse that distinction for a
    # caller that explicitly passes `slot_taxonomy=None`. The shared `{}`
    # Each call passes the shared empty mapping to ShapeDraft without editing it.
    slot_taxonomy: dict[str, list[str]] | None = {},  # noqa: B006
    members: tuple[str, ...] = (),
) -> ShapeDraft:
    return ShapeDraft(
        shape_id=shape_id,
        label=label if label is not None else shape_id,
        pop_type_dcid=pop_type,
        meas_prop_dcid=meas_prop,
        stat_type_dcid="measuredValue",
        meas_qual_dcid=None,
        meas_denom_dcid=meas_denom_dcid,
        slot_keys=(),
        meas_denom_unresolved=meas_denom_unresolved,
        place_available=place_available,
        slot_taxonomy=slot_taxonomy,
        sv_arc_facts={m: {} for m in members} if members else None,
    )


# _shapes_are_measure_head_siblings


def test_shared_rate_head_does_not_pair_across_a_pop_type_mismatch():
    """Different population types block pairing despite a shared rate head."""
    rate = _shape(
        "rate",
        label="Mortality Rate of Children Younger Than 4 Years",
        meas_denom_dcid="Count_BirthEvent_LiveBirth",
    )
    counts = _shape(
        "counts",
        label="Count of Mortality Event: Years Upto 14 (As fraction of Count Person Upto 14 Years)",
        pop_type="MortalityEvent",
        meas_denom_dcid="Count_Person_Upto14Years",
    )

    assert _shapes_are_measure_head_siblings(rate, counts) is False


def test_shared_rate_head_pairs_two_labels_that_both_name_the_rate():
    under_five = _shape("under_five", label="Under-five mortality rate")
    infant = _shape("infant", label="Infant mortality rate")

    assert _shapes_are_measure_head_siblings(under_five, infant) is True


def test_close_inflection_of_the_head_still_pairs():
    rate = _shape("rate", label="Literacy rate")
    counts = _shape("counts", label="Number of illiterate persons")

    assert _shapes_are_measure_head_siblings(rate, counts) is False
    assert _shapes_are_measure_head_siblings(rate, _shape("lit", label="Literate persons")) is True


def test_different_rate_heads_do_not_pair():
    mortality = _shape("mortality", label="Mortality Rate of Children Younger Than 4 Years")
    birth = _shape("birth", label="Crude Birth Rate")

    assert _shapes_are_measure_head_siblings(mortality, birth) is False


def test_pair_with_no_rate_wording_at_all_does_not_pair():
    a = _shape("a", label="Population: Years Upto 18")
    b = _shape("b", label="Number of maternal deaths")

    assert _shapes_are_measure_head_siblings(a, b) is False


# The structural gate also compares population type and constraint values.


def _same_cohort_count_and_rate_pair() -> tuple[ShapeDraft, ShapeDraft]:
    """A synthetic count/rate pair with matching population and constraints.

    Shared by the named and entity-less pruning tests.
    """
    rate = _shape(
        "rate",
        label="Under-five mortality rate",
        slot_taxonomy={"what:age": ["YearsUpto4"]},
        place_available=True,
    )
    counts = _shape(
        "counts",
        label="Under-five mortality count",
        slot_taxonomy={"what:age": ["YearsUpto4"]},
        place_available=False,
    )
    return rate, counts


def test_same_cohort_count_and_rate_still_pair_and_prune():
    """A matching count/rate pair prunes the empty side at a named place."""
    rate, counts = _same_cohort_count_and_rate_pair()

    assert _shapes_are_measure_head_siblings(rate, counts) is True

    result, _ = _prune_availability_siblings(
        [rate, counts], named_place=True, quantity=NO_REQUIREMENT, denominator_kinds={}
    )

    assert [s.shape_id for s in result] == ["rate"]


def test_population_narrowed_survivor_is_refused_and_both_sides_stay():
    """R3 blocks substituting a rural reading for a general query."""
    general = _shape(
        "general",
        label="Under-five mortality rate",
        place_available=False,
    )
    rural = _shape(
        "rural",
        label="Rural under-five mortality rate",
        place_available=True,
    )

    assert _shapes_are_measure_head_siblings(general, rural) is True

    result, substitutions = _prune_availability_siblings(
        [general, rural],
        named_place=True,
        variable="under-five mortality rate",
        quantity=quantity_requirement(["under-five mortality rate"]),
        denominator_kinds={},
    )

    assert {s.shape_id for s in result} == {"general", "rural"}
    assert substitutions == []


def test_registry_twin_survivor_is_substituted_despite_the_narrowing_r3_would_refuse():
    """A registered mortality twin bypasses the rural-narrowing refusal."""
    general = _shape(
        "general",
        label="Under-five mortality rate",
        place_available=False,
        members=("MortalityRate_Person_Upto4Years_AsFractionOf_Count_BirthEvent_LiveBirth",),
    )
    rural = _shape(
        "rural",
        label="Rural under-five mortality rate",
        place_available=True,
        members=("worldBank/SH_DYN_MORT",),
    )

    assert _shapes_are_measure_head_siblings(general, rural) is True

    result, substitutions = _prune_availability_siblings(
        [general, rural],
        named_place=True,
        variable="under-five mortality rate",
        quantity=quantity_requirement(["under-five mortality rate"]),
        denominator_kinds={},
    )

    assert [s.shape_id for s in result] == ["rural"]
    assert [(s.shape_id, o.shape_id) for s, o in substitutions] == [("general", "rural")]


def test_shared_rate_head_and_pop_type_do_not_pair_across_an_age_value_mismatch():
    """Different age cohorts stay separate despite matching population and rate wording."""
    under_five = _shape(
        "under_five",
        label="Mortality rate for children younger than 5 years",
        slot_taxonomy={"what:age": ["YearsUpto4"]},
    )
    infant = _shape(
        "infant",
        label="Mortality rate for children younger than 1 year",
        slot_taxonomy={"what:age": ["YearsUpto1"]},
    )

    assert _shapes_are_measure_head_siblings(under_five, infant) is False


def test_unstamped_slot_taxonomy_on_either_side_refuses_the_pair():
    """Unknown slot taxonomy prevents structural pairing."""
    rate = _shape("rate", label="Under-five mortality rate", slot_taxonomy=None)
    counts = _shape("counts", label="Under-five mortality count", slot_taxonomy={})

    assert _shapes_are_measure_head_siblings(rate, counts) is False
    assert _shapes_are_measure_head_siblings(counts, rate) is False


def test_shared_property_under_different_axes_with_the_same_values_still_pairs():
    """Matching property values pair across axes because comparison ignores axis."""
    rate = _shape(
        "rate", label="Under-five mortality rate", slot_taxonomy={"how:age": ["YearsUpto4"]}
    )
    counts = _shape(
        "counts", label="Under-five mortality count", slot_taxonomy={"what:age": ["YearsUpto4"]}
    )

    assert _shapes_are_measure_head_siblings(rate, counts) is True


# _prune_availability_siblings, measure-head rule


def test_measure_head_pop_type_mismatch_leaves_the_empty_side_unpruned():
    """A population-type mismatch preserves the empty count."""
    rate = _shape(
        "rate",
        label="Mortality Rate of Children Younger Than 4 Years",
        meas_denom_dcid="Count_BirthEvent_LiveBirth",
        place_available=True,
    )
    counts = _shape(
        "counts",
        label="Count of Mortality Event: Years Upto 14 (As fraction of Count Person Upto 14 Years)",
        pop_type="MortalityEvent",
        meas_denom_dcid="Count_Person_Upto14Years",
        place_available=False,
    )

    result, _ = _prune_availability_siblings(
        [rate, counts], named_place=True, quantity=NO_REQUIREMENT, denominator_kinds={}
    )

    assert {s.shape_id for s in result} == {"rate", "counts"}


def test_pair_sharing_no_rate_head_is_untouched():
    mortality = _shape(
        "mortality",
        label="Mortality Rate of Children Younger Than 4 Years",
        meas_denom_dcid="Count_BirthEvent_LiveBirth",
        place_available=True,
    )
    birth = _shape(
        "birth",
        label="Crude Birth Rate",
        meas_denom_dcid="Count_Person",
        place_available=False,
    )

    result, _ = _prune_availability_siblings(
        [mortality, birth], named_place=True, quantity=NO_REQUIREMENT, denominator_kinds={}
    )

    assert {s.shape_id for s in result} == {"mortality", "birth"}


def test_pair_with_both_sides_empty_is_untouched():
    rate = _shape(
        "rate",
        label="Mortality Rate of Children Younger Than 4 Years",
        meas_denom_dcid="Count_BirthEvent_LiveBirth",
        place_available=False,
    )
    counts = _shape(
        "counts",
        label="Count of Mortality Event: Years Upto 14 (As fraction of Count Person Upto 14 Years)",
        pop_type="MortalityEvent",
        meas_denom_dcid="Count_Person_Upto14Years",
        place_available=False,
    )

    result, _ = _prune_availability_siblings(
        [rate, counts], named_place=True, quantity=NO_REQUIREMENT, denominator_kinds={}
    )

    assert {s.shape_id for s in result} == {"rate", "counts"}


def test_denominator_resolution_pairing_still_drops_its_empty_side():
    unresolved = _shape("unresolved", meas_denom_unresolved=True, place_available=False)
    resolved = _shape("resolved", meas_denom_dcid="Count_Person", place_available=True)

    assert _are_availability_siblings(unresolved, resolved) is True

    result, _ = _prune_availability_siblings(
        [unresolved, resolved], named_place=True, quantity=NO_REQUIREMENT, denominator_kinds={}
    )

    assert [s.shape_id for s in result] == ["resolved"]


# The named-place restriction on the measure-head rule


def _entityless_sdg_rate_and_country_sibling() -> list[ShapeDraft]:
    """Build a world-only SDG rate and country sibling with different population types."""
    world_only_rate = _shape(
        "sdg_sh_dyn_mort",
        label="Under-five mortality rate [under 5 years old]",
        pop_type="SDG_SH_DYN_MORT",
        meas_prop="value",
        place_available=False,
    )
    country_sibling = _shape(
        "count_mortalityevent_infant",
        label="Infant mortality rate",
        pop_type="MortalityEvent",
        meas_denom_dcid="Count_BirthEvent_LiveBirth",
        place_available=True,
    )
    assert _shapes_are_measure_head_siblings(world_only_rate, country_sibling) is False
    return [world_only_rate, country_sibling]


def test_entityless_query_keeps_a_world_only_rate_and_its_country_sibling():
    """The entity-less sample keeps this structurally mismatched pair."""
    shapes = _entityless_sdg_rate_and_country_sibling()

    result, _ = _prune_availability_siblings(
        shapes, named_place=False, quantity=NO_REQUIREMENT, denominator_kinds={}
    )

    assert {s.shape_id for s in result} == {"sdg_sh_dyn_mort", "count_mortalityevent_infant"}


def test_pop_type_mismatched_sdg_pair_stays_unpruned_with_named_place_true():
    """The named-place rule also preserves the SDG/MortalityEvent mismatch."""
    shapes = _entityless_sdg_rate_and_country_sibling()

    result, _ = _prune_availability_siblings(
        shapes, named_place=True, quantity=NO_REQUIREMENT, denominator_kinds={}
    )

    assert {s.shape_id for s in result} == {"sdg_sh_dyn_mort", "count_mortalityevent_infant"}


def test_named_place_false_leaves_a_genuine_measure_head_pair_unpruned():
    """A genuine matching pair stays intact without a named place."""
    rate, counts = _same_cohort_count_and_rate_pair()

    result, _ = _prune_availability_siblings(
        [rate, counts], named_place=False, quantity=NO_REQUIREMENT, denominator_kinds={}
    )

    assert {s.shape_id for s in result} == {"rate", "counts"}


def test_denominator_resolution_pairing_prunes_an_entityless_query():
    """Denominator-resolution pairing also prunes without a named place."""
    unresolved = _shape("unresolved", meas_denom_unresolved=True, place_available=False)
    resolved = _shape("resolved", meas_denom_dcid="Count_Person", place_available=True)

    result, _ = _prune_availability_siblings(
        [unresolved, resolved], named_place=False, quantity=NO_REQUIREMENT, denominator_kinds={}
    )

    assert [s.shape_id for s in result] == ["resolved"]


# The live recall for "child mortality rate"


def _derive_child_mortality_shapes() -> list[ShapeDraft]:
    entry = json.loads((_FIXTURES / "graph_detect.json").read_text())["child mortality rate"][
        "ours"
    ]
    svs = entry["svs"]
    scores = (
        dict(zip(svs, entry["cosine_scores"], strict=True)) if entry.get("cosine_scores") else None
    )
    return derive_shapes(
        confirmed_svs=svs,
        graph=FakeGraph(),
        sv_scores=scores,
        sv_types=entry.get("types"),
        place_dcids=["country/ETH"],
    )


def test_recalled_under_five_rate_and_empty_under_14_count_are_not_measure_head_siblings():
    """The under-five rate and under-14 count differ in population and age cohort."""
    shapes = _derive_child_mortality_shapes()
    by_member = {
        next(iter(shape.sv_arc_facts or {})): shape for shape in shapes if shape.sv_arc_facts
    }
    under_five = by_member[_UNDER_FIVE_RATE_SV]
    under_14 = by_member[_UNDER_14_COUNT_SV]

    # Ethiopia has data for the under-five rate and none for the under-14 count.
    assert under_five.place_available is True
    assert under_14.place_available is False
    assert _shapes_are_measure_head_siblings(under_five, under_14) is False


def test_prune_keeps_the_empty_under_14_count_when_cohorts_differ():
    """Keep the empty under-14 count because its cohort differs from the rate."""
    shapes = _derive_child_mortality_shapes()
    survivors, _ = _prune_availability_siblings(
        shapes, named_place=True, quantity=NO_REQUIREMENT, denominator_kinds={}
    )
    surviving_members = {sv for shape in survivors for sv in (shape.sv_arc_facts or {})}

    assert _UNDER_FIVE_RATE_SV in surviving_members
    assert _UNDER_14_COUNT_SV in surviving_members
