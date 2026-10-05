"""Tests five-tuple grouping and single-slot merge behaviour, including option unions and caps."""

from __future__ import annotations

import pytest

from search.engine.same_shape import collapse_same_shape, five_tuple_key
from search.engine.spec_build import build_spec
from search.models import (
    BindingSet,
    CoverageBare,
    EntityOptions,
    PipelineStep,
    SlotKey,
    SlotOption,
    SlotOptions,
    SlotValue,
    StatVar,
)
from tests.engine._builders import conjoin_spec as _spec
from tests.engine._builders import constraint_value_slot as _cv_slot
from tests.engine._builders import gref as _gr
from tests.engine._builders import shape_five_tuple as _shape
from tests.engine._builders import unbound_slot as _unbound_slot
from tests.engine._conjoin_fixtures import _ECON_SHAPE, _PERSON_SHAPE, _region
from tests.engine._harness import ref_dcid


def _slot_options(prop: str, *dcids: str, truncated: bool = True) -> SlotOptions:
    """Build options for one constraint slot."""
    return SlotOptions(
        key=SlotKey(axis="what", property=_gr(prop), label=prop),
        options=[SlotOption(value=SlotValue(ref=_gr(d), value_kind="enum_value")) for d in dcids],
        truncated=truncated,
        source="observed",
    )


def _spec_with_extra_slots(shape, constraint_val: str, *extra_slots, sv_dcid: str = "test_sv"):
    """Build a spec with extra slots to test unbound-slot roster merging (#590)."""
    return build_spec(
        measure_elected_properties=frozenset(),
        shape=shape,
        slots=[_cv_slot("constraintProp", constraint_val), *extra_slots],
        stat_vars=[
            StatVar(
                ref=_gr(sv_dcid), shape_id=shape.shape_id, slot_values=[], constraint_properties=[]
            )
        ],
        entities=[],
        coverage=CoverageBare(has_data=True),
        pipeline_trace=[PipelineStep(step="extract", ran=True)],
    )


# ---------------------------------------------------------------------------
# five_tuple_key
# ---------------------------------------------------------------------------


def test_five_tuple_key_basic():
    spec = _spec(_PERSON_SHAPE, "any")
    key = five_tuple_key(spec)
    assert key == ("Person", "count", "measuredValue", None, None, False)


def test_five_tuple_key_with_qualifier():
    spec = _spec(_ECON_SHAPE, "any")
    key = five_tuple_key(spec)
    assert key == ("EconomicActivity", "amount", "measuredValue", "Nominal", None, False)


def test_five_tuple_key_distinct_shapes():
    spec_person = _spec(_PERSON_SHAPE, "any")
    spec_econ = _spec(_ECON_SHAPE, "any")
    assert five_tuple_key(spec_person) != five_tuple_key(spec_econ)


def test_five_tuple_key_denominator_unresolved_reads_dcid_and_flag_together():
    """A denominator arc naming a dangling ProvisionalNode target still
    carries that target's dcid (see Shape.measurement_denominator's own
    docstring). The key carries it as a separate element alongside
    measurement_denominator_unresolved."""
    shape = _shape("EconomicActivity", "amount", denom="PerCapita", denom_unresolved=True)
    spec = _spec(shape, "any")
    assert five_tuple_key(spec) == (
        "EconomicActivity",
        "amount",
        "measuredValue",
        None,
        "PerCapita",
        True,
    )


def test_five_tuple_key_distinguishes_absent_from_unresolved_denominator():
    """A denominator-absent shape and one whose denominator names a dangling
    reference must not collapse to the same key just because both surface a
    None-ish denominator to a caller that only reads the dcid."""
    absent = _spec(_shape("EconomicActivity", "amount"), "any")
    unresolved = _spec(
        _shape("EconomicActivity", "amount", denom="PerCapita", denom_unresolved=True), "any"
    )
    assert five_tuple_key(absent) != five_tuple_key(unresolved)


def test_five_tuple_key_distinguishes_absent_from_dcid_less_unresolved_denominator():
    """A denominator arc that is present but whose dangling target carries no
    dcid of its own reads measurement_denominator.dcid as None, the same as
    an absent denominator arc -- the two are indistinguishable by dcid alone
    on either side of the comparison. Only measurement_denominator_unresolved
    tells them apart."""
    absent = _spec(_shape("EconomicActivity", "amount"), "any")
    dcid_less_unresolved = _spec(_shape("EconomicActivity", "amount", denom_unresolved=True), "any")
    assert five_tuple_key(absent) != five_tuple_key(dcid_less_unresolved)


def test_five_tuple_key_distinguishes_unresolved_from_resolved_same_dcid():
    """Two shapes whose measurement_denominator names the same dcid must
    still produce different keys when one is a resolved StatVar and the
    other is a dangling ProvisionalNode target. The dcid alone cannot tell
    them apart. Only the flag can."""
    resolved = _spec(_shape("EconomicActivity", "amount", denom="PerCapita"), "any")
    unresolved = _spec(
        _shape("EconomicActivity", "amount", denom="PerCapita", denom_unresolved=True), "any"
    )
    assert five_tuple_key(resolved) != five_tuple_key(unresolved)


def test_collapse_same_shape_does_not_merge_across_denominator_resolution_states():
    """collapse_same_shape groups by five_tuple_key. Two regions that would
    otherwise merge (one differing "what"-axis slot) stay unmerged when they
    disagree on measurement_denominator_unresolved, because they land in
    different five_tuple_key groups and each group of size 1 passes through
    unchanged."""
    shape_resolved = _shape("EconomicActivity", "amount", denom="PerCapita")
    shape_unresolved = _shape(
        "EconomicActivity", "amount", denom="PerCapita", denom_unresolved=True
    )
    spec_a = _spec(shape_resolved, "ValA")
    spec_b = _spec(shape_unresolved, "ValB")
    r_a = _region(spec_a, "var A", earliest_index=0)
    r_b = _region(spec_b, "var B", earliest_index=1)

    effective, residual = collapse_same_shape([r_a, r_b])

    assert len(effective) == 2
    assert residual == []


# ---------------------------------------------------------------------------
# collapse_same_shape — single-slot merge
# ---------------------------------------------------------------------------


def test_collapse_same_shape_single_slot_merges():
    """Two definite regions with same five-tuple and one differing value slot → merged."""
    spec_a = _spec(_PERSON_SHAPE, "ValA")
    spec_b = _spec(_PERSON_SHAPE, "ValB")
    r_a = _region(spec_a, "var A", earliest_index=0)
    r_b = _region(spec_b, "var B", earliest_index=1)

    effective, residual = collapse_same_shape([r_a, r_b])

    assert len(effective) == 1
    assert residual == []
    merged = effective[0]
    assert merged.status == "definite"
    merged_binding = merged.spec.resolved.slots[0].binding
    assert isinstance(merged_binding, BindingSet)
    merged_dcids = {ref_dcid(v.ref) for v in merged_binding.values}
    assert merged_dcids == {"ValA", "ValB"}


def test_collapse_same_shape_unions_slot_options():
    """The merged options list must include bound values from both branches."""
    spec_a = _spec(
        _PERSON_SHAPE,
        "ValA",
        slot_options=[_slot_options("constraintProp", "ValA")],
    )
    spec_b = _spec(
        _PERSON_SHAPE,
        "ValB",
        slot_options=[_slot_options("constraintProp", "ValB")],
    )
    r_a = _region(spec_a, "var A", earliest_index=0)
    r_b = _region(spec_b, "var B", earliest_index=1)

    effective, residual = collapse_same_shape([r_a, r_b])

    assert len(effective) == 1
    assert residual == []
    merged = effective[0]
    merged_binding = merged.spec.resolved.slots[0].binding
    assert isinstance(merged_binding, BindingSet)
    bound_dcids = {ref_dcid(v.ref) for v in merged_binding.values}

    assert len(merged.spec.resolved.slot_options) == 1
    option_dcids = {ref_dcid(o.value.ref) for o in merged.spec.resolved.slot_options[0].options}
    # Every bound value must appear in its own slot's options.
    assert bound_dcids <= option_dcids
    assert option_dcids == {"ValA", "ValB"}


def test_collapse_same_shape_preserves_entity_options():
    """Donor entity_options is invariant across a same-shape merge group (the
    merge only ever differs on one slot, e.g. the recipient/where axis for
    dev-finance), so it must carry over from the group unchanged rather than
    being reset to [] by build_spec's own entity_options-omitted default."""
    donor_options = [
        EntityOptions(
            entity_ref="country/JPN",
            role="origin",
            options=[SlotOption(value=SlotValue(ref=_gr("country/JPN"), value_kind="entity"))],
            truncated=False,
            source="seed",
        )
    ]
    spec_a = _spec(
        _PERSON_SHAPE,
        "ValA",
        entity_options=donor_options,
    )
    spec_b = _spec(
        _PERSON_SHAPE,
        "ValB",
        entity_options=donor_options,
    )
    r_a = _region(spec_a, "var A", earliest_index=0)
    r_b = _region(spec_b, "var B", earliest_index=1)

    effective, _ = collapse_same_shape([r_a, r_b])

    assert len(effective) == 1
    assert effective[0].spec.resolved.entity_options == donor_options


def test_collapse_same_shape_unions_equivalent_readings():
    """A member's equivalent_readings names a twin for that member's own resolved
    StatVar; the merge must carry every member's entry forward, not just
    group[0]'s, since the merged spec's stat_vars span every member's."""
    twin_a = _spec(_PERSON_SHAPE, "TwinA", sv_dcid="twin_a_sv")
    twin_b = _spec(_PERSON_SHAPE, "TwinB", sv_dcid="twin_b_sv")
    spec_a = _spec(_PERSON_SHAPE, "ValA", equivalent_readings=[twin_a])
    spec_b = _spec(_PERSON_SHAPE, "ValB", equivalent_readings=[twin_b])
    r_a = _region(spec_a, "var A", earliest_index=0)
    r_b = _region(spec_b, "var B", earliest_index=1)

    effective, _ = collapse_same_shape([r_a, r_b])

    merged_twins = effective[0].spec.resolved.equivalent_readings
    assert merged_twins is not None
    assert {t.spec_id for t in merged_twins} == {twin_a.spec_id, twin_b.spec_id}


def test_collapse_same_shape_equivalent_readings_dedupes_by_spec_id():
    """The same twin named by both branches (e.g. both members answer with the
    StatVar the registry pairs with one common twin) contributes once."""
    twin = _spec(_PERSON_SHAPE, "Twin", sv_dcid="twin_sv")
    spec_a = _spec(_PERSON_SHAPE, "ValA", equivalent_readings=[twin])
    spec_b = _spec(_PERSON_SHAPE, "ValB", equivalent_readings=[twin])
    r_a = _region(spec_a, "var A", earliest_index=0)
    r_b = _region(spec_b, "var B", earliest_index=1)

    effective, _ = collapse_same_shape([r_a, r_b])

    merged_twins = effective[0].spec.resolved.equivalent_readings
    assert merged_twins is not None
    assert len(merged_twins) == 1


def test_collapse_same_shape_equivalent_readings_none_when_absent():
    spec_a = _spec(_PERSON_SHAPE, "ValA")
    spec_b = _spec(_PERSON_SHAPE, "ValB")
    r_a = _region(spec_a, "var A", earliest_index=0)
    r_b = _region(spec_b, "var B", earliest_index=1)

    effective, _ = collapse_same_shape([r_a, r_b])

    assert effective[0].spec.resolved.equivalent_readings is None


def test_collapse_same_shape_slot_options_dedupe_and_truncated_is_disjunctive():
    """A dcid appearing in both branches' options is not duplicated; truncated is
    true when any contributing branch reported truncated."""
    spec_a = _spec(
        _PERSON_SHAPE,
        "ValA",
        slot_options=[_slot_options("constraintProp", "ValA", "Shared", truncated=False)],
    )
    spec_b = _spec(
        _PERSON_SHAPE,
        "ValB",
        slot_options=[_slot_options("constraintProp", "Shared", "ValB", truncated=True)],
    )
    r_a = _region(spec_a, "var A", earliest_index=0)
    r_b = _region(spec_b, "var B", earliest_index=1)

    effective, _ = collapse_same_shape([r_a, r_b])

    merged_options = effective[0].spec.resolved.slot_options[0]
    option_dcids = [ref_dcid(o.value.ref) for o in merged_options.options]
    assert option_dcids == ["ValA", "Shared", "ValB"]  # first-seen order, no duplicate
    assert merged_options.truncated is True  # branch B was truncated


def test_union_slot_options_recaps_when_merge_exceeds_search_max_options(search_settings):
    """A merged options list over SEARCH_MAX_OPTIONS is capped and marked truncated."""
    search_settings(max_options=3)
    spec_a = _spec(
        _PERSON_SHAPE,
        "ValA",
        slot_options=[_slot_options("constraintProp", "OptA", "OptB", truncated=False)],
    )
    spec_b = _spec(
        _PERSON_SHAPE,
        "ValB",
        slot_options=[_slot_options("constraintProp", "OptC", "OptD", truncated=False)],
    )
    r_a = _region(spec_a, "var A", earliest_index=0)
    r_b = _region(spec_b, "var B", earliest_index=1)

    effective, residual = collapse_same_shape([r_a, r_b])

    assert len(effective) == 1
    assert residual == []
    merged_options = effective[0].spec.resolved.slot_options[0]
    assert [ref_dcid(o.value.ref) for o in merged_options.options] == ["OptA", "OptB", "OptC"]
    assert merged_options.truncated is True


def test_union_slot_options_under_cap_truncated_is_member_or(search_settings):
    """Below the cap, `truncated` reflects the member flags."""
    search_settings(max_options=10)
    spec_a = _spec(
        _PERSON_SHAPE,
        "ValA",
        slot_options=[_slot_options("constraintProp", "ValA", truncated=False)],
    )
    spec_b = _spec(
        _PERSON_SHAPE,
        "ValB",
        slot_options=[_slot_options("constraintProp", "ValB", truncated=True)],
    )
    r_a = _region(spec_a, "var A", earliest_index=0)
    r_b = _region(spec_b, "var B", earliest_index=1)

    effective, _residual = collapse_same_shape([r_a, r_b])

    merged_options = effective[0].spec.resolved.slot_options[0]
    assert [ref_dcid(o.value.ref) for o in merged_options.options] == ["ValA", "ValB"]
    assert merged_options.truncated is True  # member B's own flag, not the cap


@pytest.mark.parametrize(
    "asked", [("Alpha", "Beta"), ("Beta", "Alpha")], ids=["identity-order", "reversed"]
)
def test_collapse_same_shape_states_its_set_in_identity_order(asked: tuple[str, str]):
    """The merged BindingSet holds its values in the order the merged identity holds its
    refs, whatever order the parts asked them in, since a resubmit rebuilds the set from
    those refs and must state it alike."""
    first, second = asked
    r_a = _region(_spec(_PERSON_SHAPE, first), "var A", earliest_index=0)
    r_b = _region(_spec(_PERSON_SHAPE, second), "var B", earliest_index=1)

    effective, _ = collapse_same_shape([r_a, r_b])

    spec = effective[0].spec
    binding = spec.resolved.slots[0].binding
    assert isinstance(binding, BindingSet)
    assert [ref_dcid(v.ref) for v in binding.values] == ["Alpha", "Beta"]
    (identity_slot,) = [s for s in spec.definition.identity.slots if s.binding.state == "set"]
    assert identity_slot.binding.refs == ["Alpha", "Beta"]


def test_collapse_same_shape_dedupe_same_value():
    """Identical slot value across regions: < 2 distinct refs, so the multislot rule applies."""
    spec_a = _spec(_PERSON_SHAPE, "SameVal")
    spec_b = _spec(_PERSON_SHAPE, "SameVal")
    r_a = _region(spec_a, "var A", earliest_index=0)
    r_b = _region(spec_b, "var B", earliest_index=1)

    # < 2 distinct refs → SAME_SHAPE_MULTISLOT rule kicks in
    effective, residual = collapse_same_shape([r_a, r_b])

    assert len(effective) == 1
    assert residual == [("var B", 1)]


def test_collapse_same_shape_folds_absent_vs_unbound_slot_roster():
    """ "population and total population in Kenya" (#590): two regions share
    every bound slot, and differ only in which unbound slots each identity's
    own roster enumerates -- one names incomeStatus and maritalStatus as
    explicitly unbound, the other's roster never mentions them at all. An
    absent slot key and an explicit BindingUnbound one both mean "no
    constraint on this axis", so this must fold into one definite region
    rather than surface as two identical answers, and the fold must carry
    the union roster forward so the wider identity's own editing affordance
    (incomeStatus, maritalStatus) is not dropped."""
    spec_a = _spec(_PERSON_SHAPE, "SameVal")
    spec_b = _spec_with_extra_slots(
        _PERSON_SHAPE,
        "SameVal",
        _unbound_slot("what", "incomeStatus", "incomeStatus"),
        _unbound_slot("what", "maritalStatus", "maritalStatus"),
    )
    r_a = _region(spec_a, "var A", earliest_index=0)
    r_b = _region(spec_b, "var B", earliest_index=1)

    effective, residual = collapse_same_shape([r_a, r_b])

    assert len(effective) == 1
    assert residual == []
    merged_slots = effective[0].spec.resolved.slots
    # No slot became a BindingSet -- nothing actually differed.
    assert all(not isinstance(sl.binding, BindingSet) for sl in merged_slots)
    merged_keys = {(sl.key.axis, sl.key.property.dcid) for sl in merged_slots}
    assert ("what", "incomeStatus") in merged_keys
    assert ("what", "maritalStatus") in merged_keys


def test_collapse_same_shape_does_not_fold_unbound_against_a_real_bound_value():
    """The direction #590 warns against: one region leaves a slot explicitly
    unbound (open to the whole taxonomy) while the other binds that same
    slot to one specific value. These are not the same "no constraint"
    state -- a bound value narrows the answer -- so they must NOT fold, even
    though both look like a harmless roster difference at a glance. Folding
    here would silently drop the narrower part's own constraint."""
    spec_a = _spec_with_extra_slots(
        _PERSON_SHAPE,
        "SameVal",
        _unbound_slot("what", "incomeStatus", "incomeStatus"),
    )
    spec_b = _spec_with_extra_slots(
        _PERSON_SHAPE,
        "SameVal",
        _cv_slot("incomeStatus", "InPoverty"),
    )
    r_a = _region(spec_a, "var A", earliest_index=0)
    r_b = _region(spec_b, "var B", earliest_index=1)

    effective, residual = collapse_same_shape([r_a, r_b])

    # Refused merge: group[0] stands alone in `effective` and "var B" is
    # named in residual -- conjoin.py's _group_definite_regions reads this
    # residual to keep both members as their own separate parts.
    assert len(effective) == 1
    assert residual == [("var B", 1)]


def test_collapse_same_shape_unions_unresolved_slots():
    """One member's own unresolved_slots (an axis its resolved StatVar
    asserted but the graph could not read) survives into the merged spec
    even though the other member reports none."""
    spec_a = _spec(_PERSON_SHAPE, "ValA", unresolved_slots=["otherProp"])
    spec_b = _spec(_PERSON_SHAPE, "ValB")
    r_a = _region(spec_a, "var A", earliest_index=0)
    r_b = _region(spec_b, "var B", earliest_index=1)

    effective, _ = collapse_same_shape([r_a, r_b])

    assert effective[0].spec.resolved.unresolved_slots == ["otherProp"]


def test_collapse_same_shape_unresolved_slots_empty_when_none_present():
    """No group member reports an unresolved axis: the merged spec reports none."""
    spec_a = _spec(_PERSON_SHAPE, "ValA")
    spec_b = _spec(_PERSON_SHAPE, "ValB")
    r_a = _region(spec_a, "var A", earliest_index=0)
    r_b = _region(spec_b, "var B", earliest_index=1)

    effective, _ = collapse_same_shape([r_a, r_b])

    assert effective[0].spec.resolved.unresolved_slots == []


def test_collapse_same_shape_unresolved_slots_excludes_axis_bound_on_merged_spec():
    """A property landing in unresolved_slots is excluded once the merge
    itself binds that same axis, even if a member's own spec named it."""
    spec_a = _spec(_PERSON_SHAPE, "ValA", unresolved_slots=["constraintProp"])
    spec_b = _spec(_PERSON_SHAPE, "ValB")
    r_a = _region(spec_a, "var A", earliest_index=0)
    r_b = _region(spec_b, "var B", earliest_index=1)

    effective, _ = collapse_same_shape([r_a, r_b])

    merged = effective[0]
    assert isinstance(merged.spec.resolved.slots[0].binding, BindingSet)
    assert merged.spec.resolved.unresolved_slots == []


def test_collapse_same_shape_spec_id_recomputed():
    """Merged spec has a new spec_id distinct from either input spec."""
    spec_a = _spec(_PERSON_SHAPE, "ValA")
    spec_b = _spec(_PERSON_SHAPE, "ValB")
    r_a = _region(spec_a, "var A", earliest_index=0)
    r_b = _region(spec_b, "var B", earliest_index=1)

    effective, _ = collapse_same_shape([r_a, r_b])

    merged_id = effective[0].spec.spec_id
    assert merged_id != spec_a.spec_id
    assert merged_id != spec_b.spec_id


def test_collapse_same_shape_earliest_index_is_min():
    """Merged region's earliest_index is the minimum of the group."""
    spec_a = _spec(_PERSON_SHAPE, "ValA")
    spec_b = _spec(_PERSON_SHAPE, "ValB")
    # r_b has earliest_index=0, r_a has earliest_index=2
    r_a = _region(spec_a, "var A", earliest_index=2)
    r_b = _region(spec_b, "var B", earliest_index=0)

    effective, _ = collapse_same_shape([r_a, r_b])

    assert effective[0].earliest_index == 0


def test_collapse_same_shape_passthrough_single_region():
    """A single definite region passes through unchanged (no merge)."""
    spec_a = _spec(_PERSON_SHAPE, "ValA")
    r_a = _region(spec_a, "var A")

    effective, residual = collapse_same_shape([r_a])

    assert len(effective) == 1
    assert effective[0] is r_a
    assert residual == []
