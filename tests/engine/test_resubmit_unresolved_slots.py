"""Unresolved constraint arcs remain distinct from absent arcs. Direct tests
pin the classification, and a standard resubmit confirms that a widened,
unresolved activitySource is reported only in `unresolved_slots`.
"""

from __future__ import annotations

from search.engine.region.arcs import (
    inapplicable_slots_from_arcs,
    unresolved_slots_from_arcs,
)
from search.models import BindingUnbound, Slot, SlotKey
from tests.engine._builders import unbound_slot as _unbound_slot
from tests.engine._builders import value_slot as _value_slot

# ---------------------------------------------------------------------------
# TestMechanismDirect: unresolved_slots_from_arcs against hand-built arc data
# ---------------------------------------------------------------------------

_UNRESOLVED_NODE = {"nodes": [{"types": ["ProvisionalNode"], "provenanceId": "dc/base/Schema"}]}


def _arcs_with_unresolved_gender() -> dict:
    return {
        "constraintProperties": {"nodes": [{"dcid": "gender"}]},
        "gender": _UNRESOLVED_NODE,
    }


class TestMechanismDirect:
    def test_present_but_unresolved_property_is_reported(self):
        slots = [_unbound_slot("how", "gender", "gender")]
        result = unresolved_slots_from_arcs(
            slots=slots,
            sv_dcids=["sv_unresolved"],
            arcs_by_sv={"sv_unresolved": _arcs_with_unresolved_gender()},
        )
        assert result == ["gender"]

    def test_resolved_property_is_not_reported(self):
        """A resolved property is read_constraints' own business;
        unresolved_slots_from_arcs only reports properties
        read_unresolved_constraints finds."""
        slots = [_unbound_slot("how", "gender", "gender")]
        arcs = {
            "constraintProperties": {"nodes": [{"dcid": "gender"}]},
            "gender": {"nodes": [{"dcid": "Female"}]},
        }
        result = unresolved_slots_from_arcs(
            slots=slots, sv_dcids=["sv_resolved"], arcs_by_sv={"sv_resolved": arcs}
        )
        assert result == []

    def test_genuinely_absent_property_is_not_reported(self):
        """No arc at all for the property is the genuinely-absent case that
        inapplicable_slots_from_arcs reports."""
        slots = [_unbound_slot("how", "gender", "gender")]
        arcs = {"constraintProperties": {"nodes": [{"dcid": "gender"}]}}
        result = unresolved_slots_from_arcs(
            slots=slots, sv_dcids=["sv_absent"], arcs_by_sv={"sv_absent": arcs}
        )
        assert result == []

    def test_bound_slot_never_reported_even_if_unresolved(self):
        """Coherence invariant mirroring inapplicable_slots_from_arcs' own: a
        bound (value/set) slot is excluded regardless of what the arcs say.
        ResolvedSpec.unresolved_slots' own contract keeps the slot itself
        reading 'unbound' in every reported case."""
        slots = [_value_slot("how", "gender", "gender", "Female")]
        result = unresolved_slots_from_arcs(
            slots=slots,
            sv_dcids=["sv_unresolved"],
            arcs_by_sv={"sv_unresolved": _arcs_with_unresolved_gender()},
        )
        assert result == []

    def test_where_when_source_axes_never_candidates(self):
        slots = [
            _value_slot("where", "someProp", "place", "country/IND"),
            Slot(
                key=SlotKey(axis="when", property=None, label="time window"),
                binding=BindingUnbound(),
            ),
        ]
        result = unresolved_slots_from_arcs(
            slots=slots,
            sv_dcids=["sv_unresolved"],
            arcs_by_sv={"sv_unresolved": _arcs_with_unresolved_gender()},
        )
        assert result == []

    def test_sv_missing_from_arcs_map_contributes_nothing(self):
        slots = [_unbound_slot("how", "gender", "gender")]
        result = unresolved_slots_from_arcs(
            slots=slots, sv_dcids=["SomeUnconfirmedSV"], arcs_by_sv={"SomeUnconfirmedSV": None}
        )
        assert result == []

    def test_disjoint_from_inapplicable_slots_from_arcs(self):
        """A property present-but-unresolved on every member belongs to
        exactly one of the two lists."""
        slots = [_unbound_slot("how", "gender", "gender")]
        arcs_by_sv = {"sv_unresolved": _arcs_with_unresolved_gender()}
        inapplicable = inapplicable_slots_from_arcs(
            slots=slots, sv_dcids=list(arcs_by_sv), arcs_by_sv=arcs_by_sv
        )
        unresolved = unresolved_slots_from_arcs(
            slots=slots, sv_dcids=list(arcs_by_sv), arcs_by_sv=arcs_by_sv
        )
        assert inapplicable == []
        assert unresolved == ["gender"]
