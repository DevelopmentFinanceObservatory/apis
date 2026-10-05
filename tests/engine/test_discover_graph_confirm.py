"""Tests graph confirmation specificity, slot labels, and unresolved bound values."""

from __future__ import annotations

from series_coverage.schema import ABSENT_ENTITY, RoleNames
from tests.engine._fake_coverage import (
    FakeCoverageService,
    build_coverage_index,
)
from tests.engine._fake_sdmx import FakeSdmxClient
from tests.fixtures import FakeGraph


def test_constraint_property_label_misses_humanize_only_at_fresh_wire_slots() -> None:
    """A label miss leaves binder bytes intact and humanizes the response Slots."""
    from search.engine.bind import _build_universal_bind_prompt
    from search.engine.region.grounding import _ground_answer
    from search.engine.shape import shape_draft_from

    properties = [
        "placeOfResidenceClassification",
        "armedForcesStatus",
        "educationalAttainment",
        "incomeStatus",
        "residenceType",
        "literacyStatus",
        "maritalStatus",
    ]
    expected_labels = [
        "place of residence classification",
        "armed forces status",
        "educational attainment",
        "income status",
        "residence type",
        "literacy status",
        "marital status",
    ]
    prop_labels = {property_dcid: property_dcid for property_dcid in properties}
    shape = shape_draft_from(
        shape_id="test_shape",
        label="Test",
        pop_type_dcid="",
        meas_prop_dcid="",
        stat_type_dcid="",
        meas_qual_dcid=None,
        meas_denom_dcid=None,
        constraint_props=properties,
        # derive_shapes produces this exact map on a graph name lookup miss.
        prop_labels=prop_labels,
        prop_observed_values={property_dcid: [] for property_dcid in properties},
        recalled=False,
    )

    assert [slot_key.property_dcid for slot_key in shape.slot_keys[: len(properties)]] == properties
    assert [slot_key.label for slot_key in shape.slot_keys[: len(properties)]] == properties
    assert list(prop_labels) == properties
    assert set(prop_labels.values()) == set(properties)
    prompt = _build_universal_bind_prompt("test", [shape])
    for property_dcid in properties:
        assert f"property_dcid={property_dcid} ({property_dcid})" in prompt

    grounded = _ground_answer(shape, bindings=[], sv_dcids=[], roles={}, graph=FakeGraph(nodes={}))
    slots = grounded[2]
    slot_keys = grounded[3]
    assert [slot.key.label for slot in slots[: len(properties)]] == expected_labels
    assert [slot_key.label for slot_key in slot_keys[: len(properties)]] == expected_labels


def test_shape_event_humanizes_only_dcid_echo_slot_labels() -> None:
    """The direct pre-bind SlotKey construction follows the wire-label rule."""
    import asyncio

    from search.engine.region.sentence_events import _build_shape_event
    from search.engine.shape import shape_draft_from

    raw_property = "placeOfResidenceClassification"
    shape = shape_draft_from(
        shape_id="test_shape",
        label="Test",
        pop_type_dcid="",
        meas_prop_dcid="",
        stat_type_dcid="",
        meas_qual_dcid=None,
        meas_denom_dcid=None,
        constraint_props=[raw_property, "residenceType"],
        prop_labels={raw_property: raw_property, "residenceType": "Residence type"},
        prop_observed_values={raw_property: [], "residenceType": []},
        recalled=False,
    )

    assert [slot.label for slot in shape.slot_keys[:2]] == [raw_property, "Residence type"]

    event = asyncio.run(_build_shape_event(shape, variable_index=0, graph=FakeGraph(nodes={})))

    assert [slot_key.label for slot_key in event.slot_keys[:2]] == [
        "place of residence classification",
        "Residence type",
    ]


def test_graph_confirm_resolve_specificity_sort() -> None:
    """The StatVar matching both bound constraints sorts before its sibling, on a debtor
    flow confirmed against coverage."""
    from search.engine.retrieval.role_resolve import graph_confirm_resolve
    from search.engine.shape import shape_draft_from

    PROP_A = "prop:A"
    PROP_B = "prop:B"
    SV_SPECIFIC = "sv/Specific"
    SV_GENERAL = "sv/General"
    RECIPIENT = "country/KEN"

    # The specific StatVar matches both bindings. The general one matches only A.
    arcs_specific = {
        "observationProperties": {"nodes": [{"dcid": "debtor"}]},
        "populationType": {"nodes": [{"dcid": "Pop"}]},
        "measuredProperty": {"nodes": [{"dcid": "Prop"}]},
        "statType": {"nodes": [{"dcid": "measuredValue"}]},
        "constraintProperties": {"nodes": [{"dcid": PROP_A}, {"dcid": PROP_B}]},
        PROP_A: {"nodes": [{"dcid": "A1"}]},
        PROP_B: {"nodes": [{"dcid": "B1"}]},
    }
    arcs_general = {
        "observationProperties": {"nodes": [{"dcid": "debtor"}]},
        "populationType": {"nodes": [{"dcid": "Pop"}]},
        "measuredProperty": {"nodes": [{"dcid": "Prop"}]},
        "statType": {"nodes": [{"dcid": "measuredValue"}]},
        "constraintProperties": {"nodes": [{"dcid": PROP_A}, {"dcid": PROP_B}]},
        PROP_A: {"nodes": [{"dcid": "A1"}]},
        PROP_B: {"nodes": [{"dcid": "B_OTHER"}]},  # Differs from bound value B1.
    }

    obs = {
        f"{SV_SPECIFIC}|{RECIPIENT}": [
            {"earliestDate": "2010", "latestDate": "2022", "obsCount": 5}
        ],
        f"{SV_GENERAL}|{RECIPIENT}": [
            {"earliestDate": "2010", "latestDate": "2022", "obsCount": 3}
        ],
    }

    shape = shape_draft_from(
        shape_id="test_shape",
        label="Test",
        pop_type_dcid="Pop",
        meas_prop_dcid="Prop",
        stat_type_dcid="measuredValue",
        meas_qual_dcid=None,
        meas_denom_dcid=None,
        constraint_props=[PROP_A, PROP_B],
        prop_labels={PROP_A: "A", PROP_B: "B"},
        prop_observed_values={PROP_A: ["A1"], PROP_B: ["B1", "B_OTHER"]},
        sv_arc_facts={SV_SPECIFIC: arcs_specific, SV_GENERAL: arcs_general},
        representative_score=0.9,
        recalled=False,
        declared_roles=("debtor",),
    )

    # Bind A to A1 and B to B1.
    from search.engine.bind import SlotBindingDraft

    bindings = [
        SlotBindingDraft(axis="what", property_dcid=PROP_A, kind="value", value_dcids=["A1"]),
        SlotBindingDraft(axis="what", property_dcid=PROP_B, kind="value", value_dcids=["B1"]),
    ]

    g = FakeGraph(obs=obs, nodes={}, detect={}, resolve={})
    from search.engine.retrieval.retrieve import Materialised

    result = graph_confirm_resolve(
        shape=shape,
        bindings=bindings,
        subject_dcids=[RECIPIENT],
        origin_dcids=None,
        graph=g,
        sdmx=FakeSdmxClient(observations={}, availability={}),
        coverage_snapshot=FakeCoverageService(
            build_coverage_index(
                [
                    (sv, "f1", ABSENT_ENTITY, RECIPIENT, list(range(2010, 2023)))
                    for sv in (SV_SPECIFIC, SV_GENERAL)
                ],
                role_names={
                    sv: RoleNames(donor=None, recipient="debtor")
                    for sv in (SV_SPECIFIC, SV_GENERAL)
                },
            )
        ).snapshot(),
    )
    assert isinstance(result, Materialised), f"Expected Materialised but got {result!r}"
    # sv/Specific must appear before sv/General (more specific first)
    assert result.sv_dcids[0] == SV_SPECIFIC, (
        f"Most-specific SV must be first; got {result.sv_dcids}"
    )
