"""Index-sourced slot options show the bound code's children and parent, or two levels unbound.

The fixture checks CRSPurposeCode120 and its three children, parent, and member counts against
 the packaged index. It also checks all nine flat `crsTypeOfFlow` codes, the unbound two-level
 crsPurpose menu, and the pre-bind skeleton's roots-only and capped lists.
"""

from __future__ import annotations

import asyncio

from code_index import CodeIndex
from code_index.shared import get_index
from search.engine.bind import SlotBindingDraft
from search.engine.region.grounding import _ground_answer
from search.engine.region.sentence_events import _build_shape_event
from search.engine.shape import ShapeDraft, SlotKeyDraft
from tests.fixtures import FakeGraph

_PURPOSE = "crsPurpose"
_HEALTH = "CRSPurposeCode120"
_HEALTH_CHILDREN = ("CRSPurposeCode121", "CRSPurposeCode122", "CRSPurposeCode123")
_HEALTH_PARENT = "CRSPurposeCode100"

_FLOW = "crsTypeOfFlow"

_NODES = {
    _HEALTH: {"label": "health"},
    "CRSPurposeCode121": {"label": "general health"},
    "CRSPurposeCode122": {"label": "basic health"},
    "CRSPurposeCode123": {"label": "noncommunicable disease"},
    _HEALTH_PARENT: {"label": "social infrastructure and services"},
}


def _row_parent(index: CodeIndex, dcid: str) -> str | None:
    row = index.code_by_dcid(dcid)
    assert row is not None
    return row.parent


def _row_label(index: CodeIndex, dcid: str) -> str:
    row = index.code_by_dcid(dcid)
    assert row is not None
    return row.label


def test_fixture_matches_the_real_packaged_index():
    """Independent cross-check of this file's own fixture constants against
    the real packaged index, so a corpus refresh that moves the health
    family fails loudly here rather than the tests below silently testing
    stale assumptions."""
    index = get_index()
    declaration = index.declarations[_PURPOSE]
    assert declaration.hierarchical is True
    assert _row_parent(index, _HEALTH) == _HEALTH_PARENT
    for child in _HEALTH_CHILDREN:
        assert _row_parent(index, child) == _HEALTH
    counts = index.value_counts(_PURPOSE)
    for dcid in (*_HEALTH_CHILDREN, _HEALTH_PARENT):
        assert counts.get(dcid, 0) > 0


def _shape_draft(*, property_dcid: str) -> ShapeDraft:
    return ShapeDraft(
        shape_id="test_shape",
        label="test shape",
        pop_type_dcid="FinancialFlow",
        meas_prop_dcid="amount",
        stat_type_dcid="measuredValue",
        meas_qual_dcid=None,
        meas_denom_dcid=None,
        slot_keys=(SlotKeyDraft(axis="how", property_dcid=property_dcid, label="purpose"),),
        slot_taxonomy_index_sourced=True,
    )


def _slot_options(*, property_dcid: str, bound_dcid: str | None, nodes: dict):
    shape_draft = _shape_draft(property_dcid=property_dcid)
    bindings = (
        [
            SlotBindingDraft(
                axis="how", property_dcid=property_dcid, kind="value", value_dcids=[bound_dcid]
            )
        ]
        if bound_dcid is not None
        else []
    )
    result = _ground_answer(
        shape_draft,
        bindings=bindings,
        sv_dcids=[],
        roles={},
        graph=FakeGraph(nodes=nodes),
    )
    slot_options = result[7]
    assert len(slot_options) == 1
    return slot_options[0]


def test_bound_hierarchical_slot_publishes_children_and_parent_not_the_rest():
    so = _slot_options(property_dcid=_PURPOSE, bound_dcid=_HEALTH, nodes=_NODES)
    assert so.truncated is False
    assert so.source == "observed"
    by_dcid = {o.value.ref.dcid: o for o in so.options}
    assert set(by_dcid) == {*_HEALTH_CHILDREN, _HEALTH_PARENT}
    counts = get_index().value_counts(_PURPOSE)
    for dcid in _HEALTH_CHILDREN:
        opt = by_dcid[dcid]
        assert opt.relation == "child"
        assert opt.member_count == counts[dcid]
        assert opt.parent == _HEALTH
    parent_opt = by_dcid[_HEALTH_PARENT]
    assert parent_opt.relation == "parent"
    assert parent_opt.member_count == counts[_HEALTH_PARENT]
    assert parent_opt.parent is None  # CRSPurposeCode100 is a root


def test_unbound_slot_publishes_each_root_then_its_children():
    index = get_index()
    counts = index.value_counts(_PURPOSE)

    def ordered(dcids):
        return sorted(dcids, key=lambda d: (-counts.get(d, 0), d))

    roots = ordered(r.dcid for r in index.codes if r.code_system == _PURPOSE and r.parent is None)
    expected = []
    for root in roots:
        expected.append((root, "root", None))
        children = ordered(r.dcid for r in index.codes if r.parent == root)
        expected.extend((child, "child", root) for child in children)

    so = _slot_options(property_dcid=_PURPOSE, bound_dcid=None, nodes={})
    assert so.truncated is False
    assert [(o.value.ref.dcid, o.relation, o.parent) for o in so.options] == expected
    # Health sits under social infrastructure without a pick of its root first.
    assert (_HEALTH, "child", _HEALTH_PARENT) in expected
    for opt in so.options:
        assert opt.member_count == counts.get(opt.value.ref.dcid, 0)


def test_flat_system_slot_publishes_all_its_values_with_no_invented_hierarchy():
    """crsTypeOfFlow is A4c's real flat system -- hierarchical=False, nine
    codes, no specializationOf on any of them. Bound or not, its options are
    every value it carries, each honestly a root, never a child or parent."""
    index = get_index()
    declaration = index.declarations[_FLOW]
    assert declaration.hierarchical is False
    flow_dcids = [r.dcid for r in index.codes if r.code_system == _FLOW]
    labels = {d: _row_label(index, d) for d in flow_dcids}

    bound_value = flow_dcids[0]
    so = _slot_options(
        property_dcid=_FLOW,
        bound_dcid=bound_value,
        nodes={d: {"label": labels[d]} for d in flow_dcids},
    )
    dcids = {o.value.ref.dcid for o in so.options}
    assert dcids == set(flow_dcids)
    assert all(o.relation == "root" for o in so.options)
    assert all(o.parent is None for o in so.options)


def _shape_event_options(shape_draft: ShapeDraft, nodes: dict) -> list:
    event = asyncio.run(
        _build_shape_event(shape_draft, variable_index=0, graph=FakeGraph(nodes=nodes))
    )
    assert len(event.slot_options) == 1
    return event.slot_options[0].options


def test_shape_event_sends_roots_only_for_an_index_sourced_hierarchical_slot():
    """The pre-bind skeleton holds the roots even when every child has a label."""
    index = get_index()
    rows = [r for r in index.codes if r.code_system == _PURPOSE]
    roots = {r.dcid for r in rows if r.parent is None}
    options = _shape_event_options(
        _shape_draft(property_dcid=_PURPOSE), {r.dcid: {"label": r.label} for r in rows}
    )
    assert {o.value.ref.dcid for o in options} == roots
    assert all(o.relation == "root" and o.parent is None for o in options)


def test_shape_event_caps_a_recall_sourced_list_at_fifteen():
    values = [f"CRSPurposeCode{n}" for n in range(1, 21)]
    shape_draft = ShapeDraft(
        shape_id="test_shape",
        label="test shape",
        pop_type_dcid="FinancialFlow",
        meas_prop_dcid="amount",
        stat_type_dcid="measuredValue",
        meas_qual_dcid=None,
        meas_denom_dcid=None,
        slot_keys=(SlotKeyDraft(axis="how", property_dcid=_PURPOSE, label="purpose"),),
        slot_taxonomy={f"how:{_PURPOSE}": values},
    )
    options = _shape_event_options(shape_draft, {d: {"label": d} for d in values})
    assert [o.value.ref.dcid for o in options] == values[:15]
    assert all(o.relation is None for o in options)
