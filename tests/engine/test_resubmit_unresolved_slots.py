"""Unresolved constraint arcs remain distinct from absent arcs. Direct tests
pin the classification, and a standard resubmit confirms that a widened,
unresolved activitySource is reported only in `unresolved_slots`.
"""

from __future__ import annotations

from search.engine.region import resolve_spec_resubmit
from search.engine.region.arcs import (
    inapplicable_slots_from_arcs,
    unresolved_slots_from_arcs,
)
from search.models import BindingUnbound, ResubmitIdentityPayload, Slot, SlotKey
from tests.engine._base_identity import GDP_SHAPE, make_base, make_identity, subject_entity
from tests.engine._builders import unbound_slot as _unbound_slot
from tests.engine._builders import value_slot as _value_slot
from tests.engine._fake_coverage import empty_coverage_snapshot
from tests.engine._fake_sdmx import FakeSdmxClient
from tests.engine.test_resubmit_standard_widen_and_seed import (
    _ACTIVITY_SOURCE_BOUND,
    _ACTIVITY_SOURCE_UNBOUND,
    _GDP_ENTITY_DCID,
)
from tests.fixtures import FakeGraph, _load_json

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


# ---------------------------------------------------------------------------
# TestEndToEndResubmit: the standard-promote call site, through
# resolve_spec_resubmit
# ---------------------------------------------------------------------------

_UNRESOLVED_SOURCE_SV_DCID = "Amount_EconomicActivity_GrossDomesticProduction_UnresolvedSource"

_UNRESOLVED_SOURCE_ARCS = {
    "populationType": {"nodes": [{"dcid": "EconomicActivity"}]},
    "measuredProperty": {"nodes": [{"dcid": "amount"}]},
    "statType": {"nodes": [{"dcid": "measuredValue"}]},
    "measurementQualifier": {"nodes": [{"dcid": "Nominal"}]},
    "constraintProperties": {"nodes": [{"dcid": "activitySource"}]},
    "activitySource": _UNRESOLVED_NODE,
}


def _graph_with_unresolved_source_sv() -> FakeGraph:
    nodes = {
        **_load_json("graph_nodes.json"),
        _UNRESOLVED_SOURCE_SV_DCID: {
            "label": "unresolved-activitySource GDP",
            "arcs": _UNRESOLVED_SOURCE_ARCS,
        },
    }
    # Reuse a real GDP entry's own observation facets so the synthetic SV has
    # coverage at _GDP_ENTITY_DCID and the promote path reaches "definite"
    # rather than no_observations.
    real_gdp_key = f"Amount_EconomicActivity_GrossDomesticProduction_Nominal|{_GDP_ENTITY_DCID}"
    synthetic_key = f"{_UNRESOLVED_SOURCE_SV_DCID}|{_GDP_ENTITY_DCID}"
    default_obs = _load_json("graph_obs.json")
    obs = {**default_obs, synthetic_key: default_obs[real_gdp_key]}
    return FakeGraph(nodes=nodes, obs=obs, detect={}, resolve={})


class TestEndToEndResubmit:
    def test_widened_axis_with_unresolved_arc_is_reported_unresolved_not_inapplicable(self):
        """base_identity binds activitySource=GrossDomesticProduction; this
        request posts activitySource unbound (a genuine widen, mirroring
        TestWideningDropsBaseConstraint's resolved-value sibling test), and
        the seed's own activitySource arc is present but unresolved. The slot
        must read unbound (the widen) and must be reported in unresolved_slots
        (the graph could not read this axis's own value), and is excluded
        from inapplicable_slots: the arc asserts something, it just doesn't
        resolve."""
        entities = [subject_entity(_GDP_ENTITY_DCID)]
        base_spec_id, base_identity = make_base(
            GDP_SHAPE,
            slots=[_ACTIVITY_SOURCE_BOUND],
            entities=entities,
        )
        inp = ResubmitIdentityPayload(
            identity=make_identity(GDP_SHAPE, [_ACTIVITY_SOURCE_UNBOUND], entities=entities),
            base_spec_id=base_spec_id,
            base_identity=base_identity,
            resolved_stat_var_dcids=[_UNRESOLVED_SOURCE_SV_DCID],
        )
        region = resolve_spec_resubmit(
            inp=inp,
            graph=_graph_with_unresolved_source_sv(),
            sdmx=FakeSdmxClient(observations={}, availability={}),
            coverage=empty_coverage_snapshot(),
        )
        assert region.status == "definite"
        spec = region.specs[0]
        activity_slot = next(
            s
            for s in spec.resolved.slots
            if s.key.property is not None and s.key.property.dcid == "activitySource"
        )
        assert activity_slot.binding.kind == "unbound"
        assert "activitySource" in spec.resolved.unresolved_slots
        assert "activitySource" not in spec.resolved.inapplicable_slots
