"""Check that arcs-dict and MemberFacts readers produce identical grounding results."""

from __future__ import annotations

import pytest

from search.engine.region.arcs import (
    inapplicable_slots_from_arcs,
    inapplicable_slots_from_member_facts,
    unresolved_slots_from_arcs,
    unresolved_slots_from_member_facts,
)
from search.engine.retrieval.member_facts import ArcsMemberFacts
from search.engine.spec_build import build_stat_vars
from search.models import GraphRef, Slot, SlotKey

_PROP_RESOLVED = "testResolvedProp"
_PROP_UNRESOLVED = "testUnresolvedProp"
_PROP_ABSENT_EVERYWHERE = "testAbsentProp"
_VAL_RESOLVED = "TestValue1"

_SV_FULL = "TEST/FullMember"
_SV_PARTIAL = "TEST/PartialMember"

# SV_FULL has resolved and unresolved properties. SV_PARTIAL has neither.
_ARCS_BY_SV: dict[str, dict] = {
    _SV_FULL: {
        "constraintProperties": {"nodes": [{"dcid": _PROP_RESOLVED}, {"dcid": _PROP_UNRESOLVED}]},
        _PROP_RESOLVED: {"nodes": [{"dcid": _VAL_RESOLVED}]},
        _PROP_UNRESOLVED: {"nodes": [{}]},
    },
    _SV_PARTIAL: {
        "constraintProperties": {"nodes": []},
    },
}

_MEMBER_FACTS = {sv: ArcsMemberFacts(arcs=arcs) for sv, arcs in _ARCS_BY_SV.items()}


def _slot(prop_dcid: str, *, axis: str = "how") -> Slot:
    from search.models import BindingUnbound

    return Slot(
        key=SlotKey(axis=axis, property=GraphRef(dcid=prop_dcid, label=prop_dcid), label=prop_dcid),
        binding=BindingUnbound(),
        expansion=None,
    )


_SLOTS = [
    _slot(_PROP_RESOLVED),
    _slot(_PROP_UNRESOLVED),
    _slot(_PROP_ABSENT_EVERYWHERE),
]


@pytest.mark.parametrize("sv_dcids", [[_SV_FULL], [_SV_PARTIAL], [_SV_FULL, _SV_PARTIAL], []])
def test_inapplicable_slots_agree_between_arcs_and_member_facts_forms(sv_dcids):
    from_arcs = inapplicable_slots_from_arcs(
        slots=_SLOTS, sv_dcids=sv_dcids, arcs_by_sv=_ARCS_BY_SV
    )
    from_member_facts = inapplicable_slots_from_member_facts(
        slots=_SLOTS, sv_dcids=sv_dcids, member_facts=_MEMBER_FACTS
    )
    assert from_arcs == from_member_facts


@pytest.mark.parametrize("sv_dcids", [[_SV_FULL], [_SV_PARTIAL], [_SV_FULL, _SV_PARTIAL], []])
def test_unresolved_slots_agree_between_arcs_and_member_facts_forms(sv_dcids):
    from_arcs = unresolved_slots_from_arcs(slots=_SLOTS, sv_dcids=sv_dcids, arcs_by_sv=_ARCS_BY_SV)
    from_member_facts = unresolved_slots_from_member_facts(
        slots=_SLOTS, sv_dcids=sv_dcids, member_facts=_MEMBER_FACTS
    )
    assert from_arcs == from_member_facts


def test_the_three_state_partition_is_exactly_as_documented():
    """Resolved, unresolved, and absent properties land in distinct states."""
    inapplicable = inapplicable_slots_from_member_facts(
        slots=_SLOTS, sv_dcids=[_SV_FULL], member_facts=_MEMBER_FACTS
    )
    unresolved = unresolved_slots_from_member_facts(
        slots=_SLOTS, sv_dcids=[_SV_FULL], member_facts=_MEMBER_FACTS
    )
    assert inapplicable == [_PROP_ABSENT_EVERYWHERE]
    assert unresolved == [_PROP_UNRESOLVED]
    assert _PROP_RESOLVED not in inapplicable
    assert _PROP_RESOLVED not in unresolved


def test_build_stat_vars_agrees_between_arcs_and_member_facts_forms():
    """A bound slot yields identical StatVar values from either source."""
    from search.models import BindingValue, SlotValue

    bound_slot = Slot(
        key=SlotKey(
            axis="how", property=GraphRef(dcid=_PROP_RESOLVED, label=_PROP_RESOLVED), label="p"
        ),
        binding=BindingValue(
            value=SlotValue(
                ref=GraphRef(dcid=_VAL_RESOLVED, label=_VAL_RESOLVED), value_kind="enum_value"
            )
        ),
        expansion=None,
    )
    sv_refs = [GraphRef(dcid=_SV_FULL, label=_SV_FULL)]

    from_arcs = build_stat_vars(sv_refs, "test_shape", [bound_slot], arcs_by_sv=_ARCS_BY_SV)
    from_member_facts = build_stat_vars(
        sv_refs, "test_shape", [bound_slot], member_facts=_MEMBER_FACTS
    )

    assert from_arcs == from_member_facts
    assert len(from_arcs[0].slot_values) == 1
    assert from_arcs[0].slot_values[0].value.ref.dcid == _VAL_RESOLVED
    assert from_arcs[0].constraint_properties == [_PROP_RESOLVED, _PROP_UNRESOLVED]


def test_build_stat_vars_requires_exactly_one_source():
    sv_refs = [GraphRef(dcid=_SV_FULL, label=_SV_FULL)]
    with pytest.raises(ValueError, match="exactly one"):
        build_stat_vars(sv_refs, "test_shape", [])
    with pytest.raises(ValueError, match="exactly one"):
        build_stat_vars(
            sv_refs, "test_shape", [], arcs_by_sv=_ARCS_BY_SV, member_facts=_MEMBER_FACTS
        )
