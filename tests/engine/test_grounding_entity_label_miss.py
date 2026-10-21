"""Checks that missing labels preserve confirmed entity and where-slot dcids."""

from __future__ import annotations

import dataclasses

import pytest

from search.engine.bind import SlotBindingDraft
from search.engine.region import grounding as grounding_module
from search.engine.region.codes import ENTITY_LABEL_UNRESOLVED
from search.engine.region.grounding import _build_entity, _ground_answer
from search.engine.retrieval.member_facts import ArcsMemberFacts
from search.engine.shape import ShapeDraft, SlotKeyDraft
from search.engine.spec_id import compute_spec_id
from search.models import BindingSet, BindingUnbound, BindingValue
from tests.engine._base_identity import PERSON_COUNT_SHAPE
from tests.fixtures import FakeGraph

SHAPE_DRAFT = ShapeDraft(
    shape_id="person_count_measuredvalue",
    label="Total population",
    pop_type_dcid="Person",
    meas_prop_dcid="count",
    stat_type_dcid="measuredValue",
    meas_qual_dcid=None,
    meas_denom_dcid=None,
    slot_keys=(),
)

WHERE_KEY = SlotKeyDraft(
    axis="where", property_dcid="DevelopmentFinanceRecipient", label="Recipient"
)


def _shape_with_where_slot() -> ShapeDraft:
    return dataclasses.replace(SHAPE_DRAFT, slot_keys=(WHERE_KEY,))


def test_label_miss_keeps_entity_with_dcid_echo_label():
    """A missing label falls back to the dcid while retaining both entities."""
    nodes = {
        "country/USA": {"label": "United States", "type": "Country"},
        "country/ZZZ": {"type": "Country"},  # confirmed node, no ->name arc
    }
    graph = FakeGraph(nodes=nodes)
    roles = {"country/USA": "observationAbout", "country/ZZZ": "observationAbout"}

    result = _ground_answer(
        SHAPE_DRAFT, bindings=[], sv_dcids=[], roles=roles, graph=graph, open_axes=frozenset()
    )
    entity_objects_data = result[5]

    assert len(entity_objects_data) == 2
    by_dcid = {er.ref: er for _rd, er, _etr in entity_objects_data}
    assert by_dcid["country/USA"].label == "United States"
    assert by_dcid["country/ZZZ"].label == "country/ZZZ"


def test_label_miss_keeps_every_served_stat_var_with_a_dcid_echo_label():
    """A served member with no name stays among the answer's members. Dropping it left a
    definite answer whose one member had no name with no member (`assemble` refuses that)."""
    nodes = {
        "Count_Person": {"label": "Total population", "type": "StatisticalVariable"},
        "dc/unnamed": {"type": "StatisticalVariable"},  # served, no ->name arc
    }

    result = _ground_answer(
        SHAPE_DRAFT,
        bindings=[],
        sv_dcids=["dc/unnamed", "Count_Person"],
        roles={},
        open_axes=frozenset(),
        graph=FakeGraph(nodes=nodes),
    )

    assert [(ref.ref, ref.label) for ref in result[4]] == [
        ("dc/unnamed", "dc/unnamed"),
        ("Count_Person", "Total population"),
    ]


def test_label_miss_emits_exactly_one_warning_naming_the_dcid():
    nodes = {
        "country/USA": {"label": "United States", "type": "Country"},
        "country/ZZZ": {"type": "Country"},
    }
    graph = FakeGraph(nodes=nodes)
    roles = {"country/USA": "observationAbout", "country/ZZZ": "observationAbout"}

    result = _ground_answer(
        SHAPE_DRAFT, bindings=[], sv_dcids=[], roles=roles, graph=graph, open_axes=frozenset()
    )
    warnings = result[9]

    assert len(warnings) == 1
    assert warnings[0].code == ENTITY_LABEL_UNRESOLVED
    assert "country/ZZZ" in warnings[0].message


def test_no_warning_when_all_labels_resolve():
    nodes = {
        "country/USA": {"label": "United States", "type": "Country"},
        "country/ETH": {"label": "Ethiopia", "type": "Country"},
    }
    graph = FakeGraph(nodes=nodes)
    roles = {"country/USA": "observationAbout", "country/ETH": "observationAbout"}

    result = _ground_answer(
        SHAPE_DRAFT, bindings=[], sv_dcids=[], roles=roles, graph=graph, open_axes=frozenset()
    )
    entity_objects_data = result[5]
    warnings = result[9]

    assert len(entity_objects_data) == 2
    assert warnings == []


def test_spec_id_distinguishes_two_different_missed_entities():
    """The dcid fallback preserves distinct entity ids in the minted spec id."""
    graph = FakeGraph(nodes={})  # every dcid below is an unresolved label miss

    def _spec_id_for(dcid: str) -> str:
        roles = {dcid: "observationAbout"}
        result = _ground_answer(
            SHAPE_DRAFT, bindings=[], sv_dcids=[], roles=roles, graph=graph, open_axes=frozenset()
        )
        entity_objects_data, role_refs = result[5], result[1]
        entities = [_build_entity(rd, er, etr, role_refs) for rd, er, etr in entity_objects_data]
        return compute_spec_id(PERSON_COUNT_SHAPE, slots=[], entities=entities, open_roles=())

    assert _spec_id_for("country/AAA") != _spec_id_for("country/BBB")


def test_where_slot_value_label_miss_keeps_dcid_echoed_binding():
    """A missing label preserves the recipient binding with its dcid as label."""
    nodes = {
        "DevelopmentFinanceRecipient": {"label": "Recipient"},
        "country/ZZZ": {"type": "Country"},  # confirmed node, no ->name arc
    }
    graph = FakeGraph(nodes=nodes)
    roles = {"country/ZZZ": "observationAbout"}
    binding = SlotBindingDraft(
        axis="where",
        property_dcid="DevelopmentFinanceRecipient",
        kind="value",
        value_dcids=["country/ZZZ"],
    )

    result = _ground_answer(
        _shape_with_where_slot(),
        bindings=[binding],
        sv_dcids=[],
        roles=roles,
        graph=graph,
        open_axes=frozenset(),
    )
    slot = result[2][0]
    entity_objects_data = result[5]
    warnings = result[9]

    assert isinstance(slot.binding, BindingValue)
    assert slot.binding.value.ref.ref == "country/ZZZ"
    assert slot.binding.value.ref.label == "country/ZZZ"
    assert [er.ref for _rd, er, _etr in entity_objects_data] == ["country/ZZZ"]
    assert len(warnings) == 1
    assert warnings[0].code == ENTITY_LABEL_UNRESOLVED


def test_where_slot_set_label_miss_retains_both_values():
    """A missing label preserves both entities and where-slot values."""
    nodes = {
        "DevelopmentFinanceRecipient": {"label": "Recipient"},
        "country/USA": {"label": "United States", "type": "Country"},
        "country/ZZZ": {"type": "Country"},
    }
    graph = FakeGraph(nodes=nodes)
    roles = {"country/USA": "observationAbout", "country/ZZZ": "observationAbout"}
    binding = SlotBindingDraft(
        axis="where",
        property_dcid="DevelopmentFinanceRecipient",
        kind="set",
        value_dcids=["country/USA", "country/ZZZ"],
    )

    result = _ground_answer(
        _shape_with_where_slot(),
        bindings=[binding],
        sv_dcids=[],
        roles=roles,
        graph=graph,
        open_axes=frozenset(),
    )
    slot = result[2][0]
    entity_objects_data = result[5]
    warnings = result[9]

    assert isinstance(slot.binding, BindingSet)
    by_dcid = {v.ref.ref: v.ref.label for v in slot.binding.values}
    assert by_dcid == {"country/USA": "United States", "country/ZZZ": "country/ZZZ"}
    assert {er.ref for _rd, er, _etr in entity_objects_data} == {"country/USA", "country/ZZZ"}
    assert len(warnings) == 1
    assert warnings[0].code == ENTITY_LABEL_UNRESOLVED
    assert "country/ZZZ" in warnings[0].message


def test_where_value_not_in_roles_with_label_miss_stays_omitted():
    """A label fallback requires a resolve-confirmed dcid in `roles`."""
    nodes = {
        "DevelopmentFinanceRecipient": {"label": "Recipient"},
        "country/ZZZ": {"type": "Country"},  # confirmed node, no ->name arc
    }
    graph = FakeGraph(nodes=nodes)
    binding = SlotBindingDraft(
        axis="where",
        property_dcid="DevelopmentFinanceRecipient",
        kind="value",
        value_dcids=["country/ZZZ"],
    )

    result = _ground_answer(
        _shape_with_where_slot(),
        bindings=[binding],
        sv_dcids=[],
        roles={},
        graph=graph,
        open_axes=frozenset(),
    )
    slot = result[2][0]
    warnings = result[9]

    assert isinstance(slot.binding, BindingUnbound)
    assert warnings == []


_CONSTRAINT_PROPERTY = "DevelopmentFinanceScheme"
_PROVISIONAL_SCHEME = "ProvisionalScheme"


def _shape_with_constraint_slot() -> ShapeDraft:
    """A shape whose elected SV explicitly constrains the provisional dcid."""
    constraint_key = SlotKeyDraft(axis="what", property_dcid=_CONSTRAINT_PROPERTY, label="Scheme")
    return dataclasses.replace(
        SHAPE_DRAFT,
        slot_keys=(constraint_key,),
        sv_arc_facts={
            "ElectedStatVar": {
                "constraintProperties": {"nodes": [{"dcid": _CONSTRAINT_PROPERTY}]},
                _CONSTRAINT_PROPERTY: {"nodes": [{"dcid": _PROVISIONAL_SCHEME}]},
            }
        },
    )


def test_constraint_value_keeps_evidenced_unlabelled_provisional_dcid():
    """An exact elected-SV constraint arc confirms the unlabeled filter value."""
    shape = _shape_with_constraint_slot()
    graph = FakeGraph(nodes={_CONSTRAINT_PROPERTY: {"label": "Scheme"}})
    binding = SlotBindingDraft(
        axis="what",
        property_dcid=_CONSTRAINT_PROPERTY,
        kind="value",
        value_dcids=[_PROVISIONAL_SCHEME],
    )

    result = _ground_answer(
        shape, bindings=[binding], sv_dcids=[], roles={}, graph=graph, open_axes=frozenset()
    )
    slot = result[2][0]

    assert isinstance(slot.binding, BindingValue)
    assert slot.binding.value.ref.ref == _PROVISIONAL_SCHEME
    assert slot.binding.value.ref.label == _PROVISIONAL_SCHEME


def _scheme_arcs(scheme: str) -> dict:
    return {
        "constraintProperties": {"nodes": [{"dcid": _CONSTRAINT_PROPERTY}]},
        _CONSTRAINT_PROPERTY: {"nodes": [{"dcid": scheme}]},
    }


@pytest.mark.parametrize(
    ("sv_dcids", "kept"),
    [
        pytest.param(["ServedStatVar"], True, id="a-served-member-asserts-it"),
        pytest.param([], False, id="only-an-unserved-member-asserts-it"),
    ],
)
def test_constraint_value_absent_from_recalled_arcs_stays_when_a_served_member_asserts_it(
    monkeypatch, sv_dcids, kept
):
    """#1059: a bound value no label names and no recalled arc carries stays bound under
    its dcid when a member the answer serves asserts it, since materialise narrowed to
    those members. A member materialise read but does not serve is no evidence."""
    monkeypatch.setattr(grounding_module, "get_index", lambda: _FakeCodeIndex({}))
    graph = FakeGraph(nodes={_CONSTRAINT_PROPERTY: {"label": "Scheme"}})
    binding = SlotBindingDraft(
        axis="what",
        property_dcid=_CONSTRAINT_PROPERTY,
        kind="value",
        value_dcids=["UnrecalledScheme"],
    )

    result = _ground_answer(
        _shape_with_constraint_slot(),
        bindings=[binding],
        sv_dcids=sv_dcids,
        roles={},
        graph=graph,
        member_facts={"ServedStatVar": ArcsMemberFacts(arcs=_scheme_arcs("UnrecalledScheme"))},
        open_axes=frozenset(),
    )
    slot = result[2][0]

    if kept:
        assert isinstance(slot.binding, BindingValue)
        assert (slot.binding.value.ref.ref, slot.binding.value.ref.label) == (
            "UnrecalledScheme",
            "UnrecalledScheme",
        )
    else:
        assert isinstance(slot.binding, BindingUnbound)


class _FakeCodeRow:
    def __init__(self, label: str) -> None:
        self.label = label


class _FakeCodeIndex:
    """Minimal code_index.query.CodeIndex stand-in: only code_by_dcid, the
    one method _ground_answer's node_labels_batch-miss fallback reads."""

    def __init__(self, rows: dict[str, str]) -> None:
        self._rows = rows

    def code_by_dcid(self, dcid: str) -> _FakeCodeRow | None:
        label = self._rows.get(dcid)
        return _FakeCodeRow(label) if label is not None else None


def test_constraint_value_uses_index_label_when_graph_omits_it(monkeypatch):
    """node_labels_batch omits _PROVISIONAL_SCHEME (absent from the graph
    fixture's own nodes); the fixture code index carries it, so _ground_answer's
    label fallback renders the index's own label instead of the dcid echo
    test_constraint_value_keeps_evidenced_unlabelled_provisional_dcid pins for
    an index-less dcid."""
    monkeypatch.setattr(
        grounding_module, "get_index", lambda: _FakeCodeIndex({_PROVISIONAL_SCHEME: "Grant"})
    )
    shape = _shape_with_constraint_slot()
    graph = FakeGraph(nodes={_CONSTRAINT_PROPERTY: {"label": "Scheme"}})
    binding = SlotBindingDraft(
        axis="what",
        property_dcid=_CONSTRAINT_PROPERTY,
        kind="value",
        value_dcids=[_PROVISIONAL_SCHEME],
    )

    result = _ground_answer(
        shape, bindings=[binding], sv_dcids=[], roles={}, graph=graph, open_axes=frozenset()
    )
    slot = result[2][0]

    assert isinstance(slot.binding, BindingValue)
    assert slot.binding.value.ref.ref == _PROVISIONAL_SCHEME
    assert slot.binding.value.ref.label == "Grant"


def test_constraint_value_index_miss_still_echoes_dcid(monkeypatch):
    """The index fallback is scoped to dcids the index actually carries: a
    dcid it doesn't know still falls back to the dcid echo, unchanged from
    test_constraint_value_keeps_evidenced_unlabelled_provisional_dcid."""
    monkeypatch.setattr(grounding_module, "get_index", lambda: _FakeCodeIndex({}))
    shape = _shape_with_constraint_slot()
    graph = FakeGraph(nodes={_CONSTRAINT_PROPERTY: {"label": "Scheme"}})
    binding = SlotBindingDraft(
        axis="what",
        property_dcid=_CONSTRAINT_PROPERTY,
        kind="value",
        value_dcids=[_PROVISIONAL_SCHEME],
    )

    result = _ground_answer(
        shape, bindings=[binding], sv_dcids=[], roles={}, graph=graph, open_axes=frozenset()
    )
    slot = result[2][0]

    assert isinstance(slot.binding, BindingValue)
    assert slot.binding.value.ref.label == _PROVISIONAL_SCHEME


def test_constraint_set_keeps_labelled_and_evidenced_values_in_input_order():
    """A set drops invented unlabeled dcids while preserving its valid input order."""
    graph = FakeGraph(
        nodes={
            _CONSTRAINT_PROPERTY: {"label": "Scheme"},
            "Grant": {"label": "Grant"},
        }
    )
    binding = SlotBindingDraft(
        axis="what",
        property_dcid=_CONSTRAINT_PROPERTY,
        kind="set",
        value_dcids=["Grant", "InventedScheme", _PROVISIONAL_SCHEME],
    )

    result = _ground_answer(
        _shape_with_constraint_slot(),
        bindings=[binding],
        sv_dcids=[],
        roles={},
        graph=graph,
        open_axes=frozenset(),
    )
    slot = result[2][0]

    assert isinstance(slot.binding, BindingSet)
    assert [(value.ref.ref, value.ref.label) for value in slot.binding.values] == [
        ("Grant", "Grant"),
        (_PROVISIONAL_SCHEME, _PROVISIONAL_SCHEME),
    ]
