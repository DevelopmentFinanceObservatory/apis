"""Test time-window binding and slot value kinds by axis."""

from __future__ import annotations

from search.engine.bind import SlotBindingDraft
from search.engine.shape import SlotKeyDraft
from search.engine.spec_build import bind_when_slot, build_slot
from search.models import (
    Axis,
    BindingKind,
    BindingSet,
    BindingUnbound,
    BindingValue,
    GraphRef,
    Slot,
    SlotKey,
    TimeWindow,
)

# bind_when_slot


def _bare_slot(axis: Axis) -> Slot:
    """Build an unbound slot for the given axis."""
    return Slot(
        key=SlotKey(axis=axis, property=None, label=axis),
        binding=BindingUnbound(),
    )


def test_bind_when_slot_no_op_when_window_none():
    slots = [_bare_slot("what"), _bare_slot("when"), _bare_slot("source")]
    result = bind_when_slot(slots, window=None)
    assert result is slots


def test_bind_when_slot_replaces_when_slot_with_time_window():
    window = TimeWindow(start_year=2015, end_year=2020)
    slots = [_bare_slot("what"), _bare_slot("when"), _bare_slot("source")]
    result = bind_when_slot(slots, window=window)

    when_slot = next(s for s in result if s.key.axis == "when")
    assert isinstance(when_slot.binding, BindingValue)
    assert when_slot.binding.value.value_kind == "time_window"
    assert when_slot.binding.value.time_window == window
    assert when_slot.binding.value.ref is None


def test_bind_when_slot_source_stays_unbound():
    window = TimeWindow(start_year=2015)
    slots = [_bare_slot("when"), _bare_slot("source")]
    result = bind_when_slot(slots, window=window)

    source_slot = next(s for s in result if s.key.axis == "source")
    assert isinstance(source_slot.binding, BindingUnbound)


def test_bind_when_slot_preserves_order_and_other_bindings():
    window = TimeWindow(start_year=2020)
    slots = [
        _bare_slot("what"),
        _bare_slot("how"),
        _bare_slot("where"),
        _bare_slot("when"),
        _bare_slot("source"),
    ]
    result = bind_when_slot(slots, window=window)

    assert [s.key.axis for s in result] == ["what", "how", "where", "when", "source"]
    for s in result:
        if s.key.axis != "when":
            assert isinstance(s.binding, BindingUnbound)


# build_slot value_kind dispatch by axis


def _sdk(axis: Axis, prop: str) -> SlotKeyDraft:
    return SlotKeyDraft(axis=axis, property_dcid=prop, label=prop)


def _bd(kind: BindingKind, dcids: list[str]) -> SlotBindingDraft:
    return SlotBindingDraft(axis="what", property_dcid="prop", kind=kind, value_dcids=dcids)


def _gr(dcid: str) -> GraphRef:
    return GraphRef(dcid=dcid, label=dcid)


def test_build_slot_value_kind_where_axis_is_entity():
    slot = build_slot(
        _sdk("where", "DevelopmentFinanceRecipient"),
        _bd("value", ["country/ETH"]),
        [_gr("country/ETH")],
        property_ref=_gr("DevelopmentFinanceRecipient"),
    )
    assert isinstance(slot.binding, BindingValue)
    assert slot.binding.value.value_kind == "entity"


def test_build_slot_value_kind_what_axis_is_enum_value():
    slot = build_slot(
        _sdk("what", "DevelopmentFinanceScheme"),
        _bd("value", ["ODAGrants"]),
        [_gr("ODAGrants")],
        property_ref=_gr("DevelopmentFinanceScheme"),
    )
    assert isinstance(slot.binding, BindingValue)
    assert slot.binding.value.value_kind == "enum_value"


def test_build_slot_value_kind_how_axis_is_enum_value():
    slot = build_slot(
        _sdk("how", "DevelopmentFinancePurpose"),
        _bd("value", ["DAC/Health"]),
        [_gr("DAC/Health")],
        property_ref=_gr("DevelopmentFinancePurpose"),
    )
    assert isinstance(slot.binding, BindingValue)
    assert slot.binding.value.value_kind == "enum_value"


def test_build_slot_set_what_axis_value_kind_is_enum_value():
    slot = build_slot(
        _sdk("what", "DevelopmentFinanceScheme"),
        _bd("set", ["ODAGrants", "ODALoans"]),
        [_gr("ODAGrants"), _gr("ODALoans")],
        property_ref=_gr("DevelopmentFinanceScheme"),
    )
    assert isinstance(slot.binding, BindingSet)
    for sv in slot.binding.values:
        assert sv.value_kind == "enum_value"


def test_build_slot_where_partial_set_one_value_is_entity():
    slot = build_slot(
        _sdk("where", "DevelopmentFinanceRecipient"),
        _bd("set", ["country/ETH", "country/KEN"]),
        [_gr("country/ETH")],  # only one grounded
        property_ref=_gr("DevelopmentFinanceRecipient"),
    )
    assert isinstance(slot.binding, BindingValue)
    assert slot.binding.value.value_kind == "entity"
