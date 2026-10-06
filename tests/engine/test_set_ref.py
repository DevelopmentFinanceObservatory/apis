"""Test set-reference detection, propagation, and geographic-slot exclusion."""

from __future__ import annotations

from collections.abc import Callable

import pytest

from search.engine.bind import SlotBindingDraft
from search.engine.errors import GraphInfraError
from search.engine.region import RegionResult, detect_set_ref
from search.engine.same_shape import collapse_same_shape
from search.engine.shape import SlotKeyDraft
from search.engine.spec_build import build_slot, build_spec
from search.models import (
    Axis,
    BindingSet,
    BindingValue,
    CoverageBare,
    GraphRef,
    Shape,
    Slot,
    SlotKey,
    SlotValue,
    StatVar,
)
from tests.engine._builders import gref as _gr
from tests.fixtures import FakeGraph

# Agg has 3 children. AggPartial has 4, and partial-set tests use 2.


def _agg_nodes() -> dict:
    """Minimal _nodes dict with a parent Agg and its three full children."""
    return {
        "Agg": {
            "label": "Aggregate Label",
            "type": "TestEnum",
            "arcs": {},
        },
        "c1": {
            "label": "Child One",
            "type": "TestEnum",
            "arcs": {
                "isPartOf": {"nodes": [{"dcid": "Agg"}]},
            },
        },
        "c2": {
            "label": "Child Two",
            "type": "TestEnum",
            "arcs": {
                "isPartOf": {"nodes": [{"dcid": "Agg"}]},
            },
        },
        "c3": {
            "label": "Child Three",
            "type": "TestEnum",
            "arcs": {
                "isPartOf": {"nodes": [{"dcid": "Agg"}]},
            },
        },
        # Tests use two of four children to exercise the partial-set guard.
        "AggPartial": {
            "label": "Partial Aggregate",
            "type": "TestEnum",
            "arcs": {},
        },
        "p1": {
            "label": "Partial Child One",
            "type": "TestEnum",
            "arcs": {"isPartOf": {"nodes": [{"dcid": "AggPartial"}]}},
        },
        "p2": {
            "label": "Partial Child Two",
            "type": "TestEnum",
            "arcs": {"isPartOf": {"nodes": [{"dcid": "AggPartial"}]}},
        },
        "p3": {
            "label": "Partial Child Three",
            "type": "TestEnum",
            "arcs": {"isPartOf": {"nodes": [{"dcid": "AggPartial"}]}},
        },
        "p4": {
            "label": "Partial Child Four",
            "type": "TestEnum",
            "arcs": {"isPartOf": {"nodes": [{"dcid": "AggPartial"}]}},
        },
        # AggNoLabel intentionally has no label.
        "AggNoLabel": {
            "type": "TestEnum",
            "arcs": {},
        },
        "nl1": {
            "label": "No-label Child One",
            "type": "TestEnum",
            "arcs": {"isPartOf": {"nodes": [{"dcid": "AggNoLabel"}]}},
        },
        "nl2": {
            "label": "No-label Child Two",
            "type": "TestEnum",
            "arcs": {"isPartOf": {"nodes": [{"dcid": "AggNoLabel"}]}},
        },
    }


def _fake_graph() -> FakeGraph:
    return FakeGraph(nodes=_agg_nodes(), obs={}, detect={}, resolve={})


def _multiple_parent_graph() -> FakeGraph:
    nodes = _agg_nodes()
    nodes["multi_parent"] = {
        "label": "Multi Parent Child",
        "type": "TestEnum",
        "arcs": {
            "isPartOf": {"nodes": [{"dcid": "Agg"}, {"dcid": "AggPartial"}]},
        },
    }
    return FakeGraph(nodes=nodes, obs={}, detect={}, resolve={})


# ---------------------------------------------------------------------------
# FakeGraph.child_dcids
# ---------------------------------------------------------------------------


def test_child_dcids_returns_full_children() -> None:
    g = _fake_graph()
    children = set(g.child_dcids("Agg"))
    assert children == {"c1", "c2", "c3"}


@pytest.mark.parametrize(
    "parent_dcid",
    [
        pytest.param("NonExistentParent", id="absent-parent"),
        pytest.param("c1", id="leaf-node"),
    ],
)
def test_child_dcids_returns_empty_when_parent_has_no_children(parent_dcid: str) -> None:
    assert _fake_graph().child_dcids(parent_dcid) == []


def test_child_dcids_raise_on_call() -> None:
    g = FakeGraph(raise_on_call=True)
    with pytest.raises(GraphInfraError):
        g.child_dcids("Agg")


# ---------------------------------------------------------------------------
# detect_set_ref
# ---------------------------------------------------------------------------


def test_detect_set_ref_full_children_returns_graphref() -> None:
    """All 3 Agg children resolve to its labelled GraphRef."""
    g = _fake_graph()
    result = detect_set_ref(value_dcids=["c1", "c2", "c3"], graph=g)
    assert result is not None
    assert result.dcid == "Agg"
    assert result.label == "Aggregate Label"


@pytest.mark.parametrize(
    ("value_dcids", "graph_factory"),
    [
        pytest.param(["p1", "p2"], _fake_graph, id="partial-subset"),
        pytest.param(["c1", "p1"], _fake_graph, id="members-span-two-parents"),
        pytest.param(["Agg", "c1"], _fake_graph, id="member-has-no-ispartof-arc"),
        pytest.param(["nl1", "nl2"], _fake_graph, id="parent-has-no-label"),
        pytest.param(["c1"], _fake_graph, id="single-member"),
        pytest.param([], _fake_graph, id="empty-members"),
        pytest.param(["multi_parent", "c1"], _multiple_parent_graph, id="member-has-two-parents"),
    ],
)
def test_detect_set_ref_returns_none_for_incomplete_or_ambiguous_sets(
    value_dcids: list[str], graph_factory: Callable[[], FakeGraph]
) -> None:
    assert detect_set_ref(value_dcids=value_dcids, graph=graph_factory()) is None


# ---------------------------------------------------------------------------
# build_slot passes set_ref into BindingSet.
# ---------------------------------------------------------------------------


def _key_draft(axis: Axis = "what") -> SlotKeyDraft:
    return SlotKeyDraft(axis=axis, property_dcid="prop/A", label="prop A")


def test_build_slot_set_ref_none_by_default() -> None:
    """Omitting set_ref leaves BindingSet.set_ref=None (backward compat)."""
    draft = SlotBindingDraft(
        axis="what", property_dcid="prop/A", kind="set", value_dcids=["c1", "c2"]
    )
    slot = build_slot(
        _key_draft("what"),
        draft,
        [_gr("c1"), _gr("c2")],
        property_ref=_gr("prop/A"),
    )
    assert isinstance(slot.binding, BindingSet)
    assert slot.binding.set_ref is None


def test_build_slot_set_ref_propagates_to_binding_set() -> None:
    """A supplied set_ref appears in the returned BindingSet."""
    draft = SlotBindingDraft(
        axis="what", property_dcid="prop/A", kind="set", value_dcids=["c1", "c2"]
    )
    ref = _gr("Agg", "Aggregate Label")
    slot = build_slot(
        _key_draft("what"),
        draft,
        [_gr("c1"), _gr("c2")],
        property_ref=_gr("prop/A"),
        set_ref=ref,
    )
    assert isinstance(slot.binding, BindingSet)
    assert slot.binding.set_ref == ref


# ---------------------------------------------------------------------------
# collapse_same_shape passes set_ref_for to merged slots.
# ---------------------------------------------------------------------------


def _shape(pop: str = "Pop", mp: str = "MP", axis: Axis = "what") -> Shape:
    return Shape(
        shape_id=f"{pop}/{mp}",
        label="Test",
        population_type=_gr(pop),
        measured_property=_gr(mp),
        stat_type=_gr("measuredValue"),
        measurement_qualifier=None,
        measurement_denominator=None,
        slot_keys=[SlotKey(axis=axis, property=_gr("prop/A"), label="prop A")],
        member_count=1,
    )


def _value_slot(axis: Axis, val: str) -> Slot:
    return Slot(
        key=SlotKey(axis=axis, property=_gr("prop/A"), label="prop A"),
        binding=BindingValue(value=SlotValue(ref=_gr(val), value_kind="enum_value")),
    )


def _entity_slot(axis: Axis, val: str) -> Slot:
    return Slot(
        key=SlotKey(axis=axis, property=_gr("prop/A"), label="prop A"),
        binding=BindingValue(value=SlotValue(ref=_gr(val), value_kind="entity")),
    )


def _region(shape: Shape, slot: Slot, sv: str = "sv/X", index: int = 0) -> RegionResult:
    spec = build_spec(
        open_roles=(),
        measure_elected_properties=frozenset(),
        shape=shape,
        slots=[slot],
        stat_vars=[
            StatVar(ref=_gr(sv), shape_id=shape.shape_id, slot_values=[], constraint_properties=[])
        ],
        entities=[],
        coverage=CoverageBare(has_data=True),
    )
    return RegionResult(
        variable_text="test",
        status="definite",
        specs=(spec,),
        no_data_reason=None,
        warnings=(),
        timing_by_step={},
        earliest_index=index,
    )


def test_collapse_same_shape_what_axis_set_ref_for_called() -> None:
    """A what-axis merge stores the set_ref_for result in BindingSet."""
    sh = _shape(axis="what")
    r1 = _region(sh, _value_slot("what", "c1"), sv="sv/1", index=0)
    r2 = _region(sh, _value_slot("what", "c2"), sv="sv/2", index=1)

    parent_ref = GraphRef(dcid="Agg", label="Aggregate Label")
    captured: list[list[str]] = []

    def fake_set_ref_for(dcids: list[str]) -> GraphRef | None:
        captured.append(list(dcids))
        return parent_ref

    effective, residual = collapse_same_shape([r1, r2], set_ref_for=fake_set_ref_for)
    assert residual == []
    assert len(effective) == 1
    merged = effective[0].spec
    diff_slot = next(s for s in merged.resolved.slots if s.binding.kind == "set")
    assert isinstance(diff_slot.binding, BindingSet)
    assert diff_slot.binding.set_ref == parent_ref
    # Verify the correct dcids were passed to set_ref_for.
    assert len(captured) == 1
    assert set(captured[0]) == {"c1", "c2"}


def test_collapse_same_shape_set_ref_for_none_leaves_set_ref_none() -> None:
    """With set_ref_for=None (default), merged BindingSet.set_ref stays None."""
    sh = _shape(axis="what")
    r1 = _region(sh, _value_slot("what", "c1"), sv="sv/1", index=0)
    r2 = _region(sh, _value_slot("what", "c2"), sv="sv/2", index=1)

    effective, residual = collapse_same_shape([r1, r2])
    assert residual == []
    diff_slot = next(s for s in effective[0].spec.resolved.slots if s.binding.kind == "set")
    assert isinstance(diff_slot.binding, BindingSet)
    assert diff_slot.binding.set_ref is None


def test_collapse_same_shape_where_axis_not_labelled() -> None:
    """Where-axis merges skip set_ref_for entirely (geographic containment guard)."""
    sh = _shape(axis="where")
    r1 = _region(sh, _entity_slot("where", "c1"), sv="sv/1", index=0)
    r2 = _region(sh, _entity_slot("where", "c2"), sv="sv/2", index=1)

    called: list[bool] = []

    def should_not_call(_dcids: list[str]) -> GraphRef | None:
        called.append(True)
        return GraphRef(dcid="WRONG", label="WRONG")

    effective, _ = collapse_same_shape([r1, r2], set_ref_for=should_not_call)
    assert called == [], "set_ref_for must not be called for where-axis slots"
    diff_slot = next(s for s in effective[0].spec.resolved.slots if s.binding.kind == "set")
    assert isinstance(diff_slot.binding, BindingSet)
    assert diff_slot.binding.set_ref is None


def test_collapse_same_shape_how_axis_set_ref_for_called() -> None:
    """The how axis is a taxonomy axis, so set_ref_for runs."""
    sh = _shape(axis="how")
    slot1 = Slot(
        key=SlotKey(axis="how", property=_gr("prop/A"), label="prop A"),
        binding=BindingValue(value=SlotValue(ref=_gr("h1"), value_kind="enum_value")),
    )
    slot2 = Slot(
        key=SlotKey(axis="how", property=_gr("prop/A"), label="prop A"),
        binding=BindingValue(value=SlotValue(ref=_gr("h2"), value_kind="enum_value")),
    )
    r1 = _region(sh, slot1, sv="sv/1", index=0)
    r2 = _region(sh, slot2, sv="sv/2", index=1)

    how_ref = GraphRef(dcid="HowParent", label="How Parent")
    effective, _ = collapse_same_shape([r1, r2], set_ref_for=lambda _: how_ref)
    diff_slot = next(s for s in effective[0].spec.resolved.slots if s.binding.kind == "set")
    assert isinstance(diff_slot.binding, BindingSet)
    assert diff_slot.binding.set_ref == how_ref


def test_collapse_same_shape_set_ref_for_returns_none_propagates() -> None:
    """When set_ref_for returns None (no full-children match), set_ref stays None."""
    sh = _shape(axis="what")
    r1 = _region(sh, _value_slot("what", "c1"), sv="sv/1", index=0)
    r2 = _region(sh, _value_slot("what", "c2"), sv="sv/2", index=1)

    effective, _ = collapse_same_shape([r1, r2], set_ref_for=lambda _: None)
    diff_slot = next(s for s in effective[0].spec.resolved.slots if s.binding.kind == "set")
    assert isinstance(diff_slot.binding, BindingSet)
    assert diff_slot.binding.set_ref is None
