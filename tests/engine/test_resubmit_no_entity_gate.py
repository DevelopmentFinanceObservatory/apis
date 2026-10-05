"""A no-entity resubmit with split ownership of two bound properties returns
no_data under NO_MEMBER_ASSERTS_BOUND_PROPERTIES. Neither sibling asserts both.
"""

from __future__ import annotations

from search.engine.region import resolve_spec_resubmit
from search.engine.region.codes import NO_MEMBER_ASSERTS_BOUND_PROPERTIES
from search.models import ResubmitIdentityPayload
from tests.engine._base_identity import make_base, make_identity
from tests.engine._builders import value_slot as _value_slot
from tests.engine._fake_coverage import empty_coverage_snapshot
from tests.engine._fake_sdmx import FakeSdmxClient
from tests.fixtures import FakeGraph

_POP_TYPE = "TestSplitOwnershipEvent"
_MEAS_PROP = "testSplitOwnershipIncidence"
_STAT_TYPE = "measuredValue"
# Mirrors _sv_arcs' own populationType/measuredProperty/statType below --
# resolve_spec_resubmit's standard path re-derives the shape from the posted
# resolved_stat_var_dcids' real arcs and requires it to match the posted
# shape structurally.
_SHAPE: dict = {
    "kind": "statvar_schema",
    "population_type": _POP_TYPE,
    "measured_property": _MEAS_PROP,
    "stat_type": _STAT_TYPE,
}

_PROP_GENDER = "gender"
_VAL_FEMALE = "Female"
_PROP_RESIDENCE = "placeOfResidenceClassification"
_VAL_URBAN = "Urban"

# Split ownership: SV_A asserts gender=Female only, SV_B asserts
# residence=Urban only -- neither asserts both bound properties, so the
# no-entity branch's assert-every-bound-property eligibility rule (the same
# rule the entity-probe applies) admits neither as a full asserter.
_SV_A = "TEST/ResubmitSplitOwnership_GenderOnly"
_SV_B = "TEST/ResubmitSplitOwnership_ResidenceOnly"


def _sv_arcs(*, label: str, prop: str, value: str) -> dict:
    return {
        "typeOf": {"nodes": [{"dcid": "StatisticalVariable"}]},
        "name": {"nodes": [{"value": label}]},
        "populationType": {"nodes": [{"dcid": _POP_TYPE}]},
        "measuredProperty": {"nodes": [{"dcid": _MEAS_PROP}]},
        "statType": {"nodes": [{"dcid": _STAT_TYPE}]},
        "constraintProperties": {"nodes": [{"dcid": prop}]},
        prop: {"nodes": [{"dcid": value}]},
    }


_NODES = {
    _SV_A: {
        "label": "Test split-ownership incidence, female (any residence)",
        "type": "StatisticalVariable",
        "arcs": _sv_arcs(
            label="Test split-ownership incidence, female (any residence)",
            prop=_PROP_GENDER,
            value=_VAL_FEMALE,
        ),
    },
    _SV_B: {
        "label": "Test split-ownership incidence, urban (any gender)",
        "type": "StatisticalVariable",
        "arcs": _sv_arcs(
            label="Test split-ownership incidence, urban (any gender)",
            prop=_PROP_RESIDENCE,
            value=_VAL_URBAN,
        ),
    },
}


def test_no_entity_split_ownership_neither_asserter_wins_flags_gate():
    """A plain resubmit (no edit, no place entity) of two StatVars that each
    assert only one of the two bound properties must degrade to honest
    no_data(no_matching_slice), never a "definite" spec claiming both bound
    axes on a member that asserts only one. flagged_match must carry the
    representative under gate=no_member_asserts_bound_properties, mirroring
    the fresh path's equivalent no-entity outcome."""
    gender_slot = _value_slot("how", _PROP_GENDER, "gender", _VAL_FEMALE, "Female")
    residence_slot = _value_slot("how", _PROP_RESIDENCE, "place of residence", _VAL_URBAN, "Urban")

    graph = FakeGraph(nodes=_NODES, resolve={}, obs={})

    base_spec_id, base_identity = make_base(
        _SHAPE,
        slots=[gender_slot, residence_slot],
    )
    inp = ResubmitIdentityPayload(
        identity=make_identity(_SHAPE, [gender_slot, residence_slot]),
        base_spec_id=base_spec_id,
        base_identity=base_identity,
        resolved_stat_var_dcids=[_SV_A, _SV_B],
    )

    region = resolve_spec_resubmit(
        inp=inp,
        graph=graph,
        sdmx=FakeSdmxClient(observations={}, availability={}),
        coverage=empty_coverage_snapshot(),
    )

    assert region.status == "no_data"
    assert region.no_data_reason == "no_matching_slice"
    assert region.flagged_match is not None
    flagged_spec = region.flagged_match[0]
    # No place entity here to be unavailable AT -- distinct from the
    # entity-probe's own flagged-unavailable carrier; see
    # test_fresh_path_emission_guard.py's matching no-entity assertion.
    assert flagged_spec.resolved.available_at_place is None
    assert flagged_spec.resolved.confidence is None
    assert region.gate == NO_MEMBER_ASSERTS_BOUND_PROPERTIES
