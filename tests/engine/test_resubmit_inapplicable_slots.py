"""Applicability is recomputed from confirmed StatVar arcs because the wire's
unbound binding cannot distinguish a widened slot from an inapplicable one.
Tests cover direct arc classification, both resubmit branches, and hash stability.

Only TestMechanismDirect produces a non-empty `inapplicable_slots`. The resubmit fixtures
never declare an axis their members lack, so they prove the wiring and an empty result.
"""

from __future__ import annotations

from search.engine.region import resolve_spec_resubmit
from search.engine.region.arcs import inapplicable_slots_from_arcs
from search.engine.spec_id import compute_spec_id
from search.models import (
    BindingUnbound,
    Entity,
    EntityRoleSubject,
    GraphRef,
    ResubmitIdentityPayload,
    Slot,
    SlotKey,
)
from tests.engine._base_identity import (
    GDP_SHAPE,
    PERSON_COUNT_SHAPE,
    make_base,
    make_identity,
    subject_entity,
)
from tests.engine._builders import unbound_slot as _unbound_slot
from tests.engine._builders import value_slot as _value_slot
from tests.engine._fake_coverage import empty_coverage_snapshot
from tests.engine._fake_sdmx import FakeSdmxClient
from tests.engine._harness import walk_options
from tests.engine.test_resubmit_standard_widen import (
    _ACTIVITY_SOURCE_UNBOUND,
    _GDP_ENTITY_DCID,
)
from tests.fixtures import FakeGraph

# ---------------------------------------------------------------------------
# TestMechanismDirect: inapplicable_slots_from_arcs against real fixture arcs
# ---------------------------------------------------------------------------


class TestMechanismDirect:
    """Classifies unbound what/how slots against confirmed StatVar arcs."""

    def test_three_genuinely_undeclared_properties_are_reported(self):
        """Count_Person has no constraintProperties. Its three unbound slots
        are inapplicable; the bound place and when/source axes are excluded."""
        graph = FakeGraph()
        count_person_arcs = graph.node_arcs("Count_Person")
        assert count_person_arcs is not None
        assert "constraintProperties" not in count_person_arcs, (
            "fixture drift: Count_Person is expected to carry no constraint arcs "
            "at all, which is what makes it the reported regression's own case"
        )

        slots = [
            _unbound_slot("how", "gender", "gender"),
            _unbound_slot("how", "placeOfResidenceClassification", "place of residence"),
            _unbound_slot("how", "age", "age"),
            _value_slot("where", "someProp", "place", "country/IND"),
            Slot(
                key=SlotKey(axis="when", property=None, label="time window"),
                binding=BindingUnbound(),
            ),
        ]
        result = inapplicable_slots_from_arcs(
            slots=slots,
            sv_dcids=["Count_Person"],
            arcs_by_sv={"Count_Person": count_person_arcs},
        )
        assert set(result) == {"gender", "placeOfResidenceClassification", "age"}
        assert len(result) == 3

    def test_property_asserted_by_any_resolved_sv_is_not_reported(self):
        """Count_Person_Female asserts gender, so the property is applicable
        even though Count_Person does not."""
        graph = FakeGraph()
        arcs_by_sv = {
            "Count_Person": graph.node_arcs("Count_Person"),
            "Count_Person_Female": graph.node_arcs("Count_Person_Female"),
        }
        slots = [_unbound_slot("how", "gender", "gender")]
        result = inapplicable_slots_from_arcs(
            slots=slots, sv_dcids=list(arcs_by_sv), arcs_by_sv=arcs_by_sv
        )
        assert result == []

    def test_bound_slot_never_reported_even_if_arcs_lack_it(self):
        """Bound slots stay out of inapplicable_slots, even when the member
        has no constraint arcs for them."""
        slots = [_value_slot("how", "gender", "gender", "Female")]
        result = inapplicable_slots_from_arcs(
            slots=slots,
            sv_dcids=["Count_Person"],
            arcs_by_sv={"Count_Person": {}},  # no constraint arcs at all
        )
        assert result == []

    def test_where_when_source_axes_never_candidates(self):
        """Only what/how slots are ever candidates -- mirrors
        spec_build.inapplicable_property_dcid's own scope."""
        slots = [
            _value_slot("where", "someProp", "place", "country/IND"),
            Slot(
                key=SlotKey(axis="when", property=None, label="time window"),
                binding=BindingUnbound(),
            ),
            Slot(
                key=SlotKey(axis="source", property=None, label="data source"),
                binding=BindingUnbound(),
            ),
        ]
        result = inapplicable_slots_from_arcs(
            slots=slots, sv_dcids=["Count_Person"], arcs_by_sv={"Count_Person": {}}
        )
        assert result == []

    def test_sv_missing_from_arcs_map_contributes_nothing(self):
        """A missing arc-map entry or absent-node sentinel does not assert the
        property."""
        slots = [_unbound_slot("how", "gender", "gender")]
        result = inapplicable_slots_from_arcs(
            slots=slots, sv_dcids=["SomeUnconfirmedSV"], arcs_by_sv={"SomeUnconfirmedSV": None}
        )
        assert result == ["gender"]


def _assert_never_bound_and_inapplicable(spec) -> None:
    """Assert that no inapplicable property has a bound slot."""
    bound_props = {
        s.key.property.ref
        for s in spec.resolved.slots
        if s.key.property is not None and s.binding.kind != "unbound"
    }
    assert not (set(spec.resolved.inapplicable_slots) & bound_props)


# ---------------------------------------------------------------------------
# TestEndToEndResubmit: both resubmit call sites, through resolve_spec_resubmit
# ---------------------------------------------------------------------------


class TestEndToEndResubmit:
    def test_standard_promote_unchanged_count_person_stays_coherent(self):
        """An unchanged Count_Person resubmit stays definite. Its base declares
        no extra axes, so no inapplicable slot is reported."""
        entities = [subject_entity("country/IND")]
        base_identity = make_identity(PERSON_COUNT_SHAPE, entities=entities)
        inp = ResubmitIdentityPayload(
            identity=base_identity,
        )
        region = resolve_spec_resubmit(
            inp=inp,
            graph=FakeGraph(),
            sdmx=FakeSdmxClient(observations={}, availability={}),
            coverage=empty_coverage_snapshot(),
            options=walk_options(FakeGraph()),
        )
        assert region.status == "definite"
        spec = region.specs[0]
        assert spec.resolved.inapplicable_slots == []
        _assert_never_bound_and_inapplicable(spec)

    def test_standard_promote_widen_is_not_reported_inapplicable(self):
        """Widened activitySource remains unbound but is not inapplicable
        because the GDP member still asserts it."""
        entities = [subject_entity(_GDP_ENTITY_DCID)]
        inp = ResubmitIdentityPayload(
            identity=make_base(GDP_SHAPE, slots=[_ACTIVITY_SOURCE_UNBOUND], entities=entities)[1],
        )
        region = resolve_spec_resubmit(
            inp=inp,
            graph=FakeGraph(),
            sdmx=FakeSdmxClient(observations={}, availability={}),
            coverage=empty_coverage_snapshot(),
            options=walk_options(FakeGraph()),
        )
        assert region.status == "definite"
        spec = region.specs[0]
        activity_slot = next(
            s
            for s in spec.resolved.slots
            if s.key.property is not None and s.key.property.ref == "activitySource"
        )
        assert activity_slot.binding.kind == "unbound"
        assert "activitySource" not in spec.resolved.inapplicable_slots
        _assert_never_bound_and_inapplicable(spec)


# ---------------------------------------------------------------------------
# TestHashNeutrality
# ---------------------------------------------------------------------------


class TestHashNeutrality:
    def test_unchanged_standard_resubmit_spec_id_is_stable(self):
        """Repeated inapplicable-slot recomputation leaves spec_id unchanged."""
        entities = [subject_entity("country/IND")]
        base_identity = make_identity(PERSON_COUNT_SHAPE, entities=entities)
        inp = ResubmitIdentityPayload(
            identity=base_identity,
        )
        region_a = resolve_spec_resubmit(
            inp=inp,
            graph=FakeGraph(),
            sdmx=FakeSdmxClient(observations={}, availability={}),
            coverage=empty_coverage_snapshot(),
            options=walk_options(FakeGraph()),
        )
        region_b = resolve_spec_resubmit(
            inp=inp,
            graph=FakeGraph(),
            sdmx=FakeSdmxClient(observations={}, availability={}),
            coverage=empty_coverage_snapshot(),
            options=walk_options(FakeGraph()),
        )
        assert region_a.specs[0].spec_id == region_b.specs[0].spec_id
        assert region_a.specs[0].spec_id == "spec_f28a3b0c6d46c2ab8bb99edb1c14799f"

    def test_widened_resubmit_spec_id_unaffected_by_inapplicable_recompute(self):
        """The widening repro from TestEndToEndResubmit above: inapplicable_slots
        comes back empty (a pure widen, not an inapplicable property -- see
        that test), and the spec_id matches
        TestWideningDropsBaseConstraint.test_unbound_over_bound_base_widens_not_carries_forward's
        own spec (same input) in test_resubmit_standard_widen.py --
        the inapplicable-slots recomputation cannot perturb identity for this
        widen scenario, since inapplicable_slots is excluded from the
        identity projection."""
        entities = [subject_entity(_GDP_ENTITY_DCID)]
        inp = ResubmitIdentityPayload(
            identity=make_base(GDP_SHAPE, slots=[_ACTIVITY_SOURCE_UNBOUND], entities=entities)[1],
        )
        region = resolve_spec_resubmit(
            inp=inp,
            graph=FakeGraph(),
            sdmx=FakeSdmxClient(observations={}, availability={}),
            coverage=empty_coverage_snapshot(),
            options=walk_options(FakeGraph()),
        )
        spec = region.specs[0]
        # activitySource ends up unbound (the widen); shape declares no other
        # what/how slot for the GDP five-tuple, so the ONLY hashed slot is
        # activitySource=unbound -- identical to what a plain unbound slot
        # (no inapplicable concept involved at all) would hash to.
        reference_slots = [
            Slot(
                key=SlotKey(
                    axis="how",
                    property=GraphRef(ref="activitySource", label="activity source"),
                    label="activity source",
                ),
                binding=BindingUnbound(),
            )
        ]
        reference_entities = [
            Entity(
                ref=GraphRef(ref=_GDP_ENTITY_DCID, label=_GDP_ENTITY_DCID),
                entity_type=None,
                role=EntityRoleSubject(),
            )
        ]
        reference_id = compute_spec_id(
            GDP_SHAPE, reference_slots, entities=reference_entities, open_roles=()
        )
        assert spec.spec_id == reference_id
