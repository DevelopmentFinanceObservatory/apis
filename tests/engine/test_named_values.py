"""A slot the bind leaves open binds the one value the variable phrase names by its label
(`phase_rank_bind._with_named_values`), whatever the binder drafted for it."""

from __future__ import annotations

from dataclasses import replace

import pytest

from search.engine.bind import SlotBindingDraft
from search.engine.region.phase_election import _settle_bindings
from search.engine.region.phase_rank_bind import _with_named_values
from search.engine.shape import ShapeDraft, SlotKeyDraft
from tests.engine.test_qualifier_guard import _state

_GAS = SlotKeyDraft(axis="how", property_dcid="emittedThing", label="emittedThing")
_SOURCE = SlotKeyDraft(axis="how", property_dcid="emissionSource", label="emissionSource")
_AGE_GROUP = SlotKeyDraft(
    axis="how", property_dcid="ageGroupClassification", label="ageGroupClassification"
)


def _nodes(*dcids: str) -> dict:
    return {"nodes": [{"dcid": dcid} for dcid in dcids]}


def _member(name: str, **constraints: str) -> dict:
    return {
        "name": {"nodes": [{"value": name}]},
        "constraintProperties": _nodes(*constraints),
        **{prop: _nodes(value) for prop, value in constraints.items()},
    }


# The emissions family "greenhouse gas emissions in Nigeria" recalls: one member per gas,
# one of them by source.
_EMISSIONS = ShapeDraft(
    shape_id="emissions",
    label="Annual Amount Of Emissions: Greenhouse Gas",
    pop_type_dcid="Emissions",
    meas_prop_dcid="amount",
    stat_type_dcid="measuredValue",
    meas_qual_dcid="Annual",
    meas_denom_dcid=None,
    slot_keys=(_GAS, _SOURCE),
    sv_arc_facts={
        "Annual_Emissions_GreenhouseGas": _member(
            "Annual Amount Of Emissions: Greenhouse Gas", emittedThing="GreenhouseGas"
        ),
        "Annual_Emissions_CarbonDioxide": _member(
            "Annual Amount Of Emissions: Carbon Dioxide", emittedThing="CarbonDioxide"
        ),
        "Annual_Emissions_CarbonDioxide_ForestryAndLandUse": _member(
            "Annual Amount Of Emissions: Carbon Dioxide, Forestry And Land Use",
            emittedThing="CarbonDioxide",
            emissionSource="ForestryAndLandUse",
        ),
    },
    slot_taxonomy={
        "how:emittedThing": ["GreenhouseGas", "CarbonDioxide"],
        "how:emissionSource": ["ForestryAndLandUse"],
    },
    value_labels={
        # The graph labels the gas with its dcid.
        "GreenhouseGas": "GreenhouseGas",
        "CarbonDioxide": "Carbon Dioxide",
        "ForestryAndLandUse": "Forestry And Land Use",
    },
)

# Infant mortality: the headline series is named "Infant mortality rate" and selects deaths
# under one year by age, while another member selects the Infant age group.
_INFANT_MORTALITY = ShapeDraft(
    shape_id="infant_mortality",
    label="Infant mortality rate",
    pop_type_dcid="MortalityEvent",
    meas_prop_dcid="count",
    stat_type_dcid="measuredValue",
    meas_qual_dcid=None,
    meas_denom_dcid="Count_BirthEvent_LiveBirth",
    slot_keys=(_AGE_GROUP,),
    sv_arc_facts={
        "Count_Death_0Years_AsFractionOf_Count_BirthEvent_LiveBirth": _member(
            "Infant mortality rate"
        ),
        "Count_Death_Infant_AsFractionOf_Count_BirthEvent_LiveBirth": _member(
            "Deaths of infants per live birth", ageGroupClassification="Infant"
        ),
    },
    slot_taxonomy={"how:ageGroupClassification": ["Infant"]},
    value_labels={"Infant": "Infant"},
)


# The emissions of carbon dioxide alone: every member holds the gas.
_CARBON_DIOXIDE = replace(
    _EMISSIONS,
    shape_id="carbon_dioxide",
    sv_arc_facts={
        dcid: arcs
        for dcid, arcs in (_EMISSIONS.sv_arc_facts or {}).items()
        if "CarbonDioxide" in dcid
    },
)


def _open(slot: SlotKeyDraft, **fields: object) -> SlotBindingDraft:
    return SlotBindingDraft(
        axis=slot.axis, property_dcid=slot.property_dcid, kind="unbound", **fields
    )


def _value(slot: SlotKeyDraft, value: str) -> SlotBindingDraft:
    return SlotBindingDraft(
        axis=slot.axis, property_dcid=slot.property_dcid, kind="value", value_dcids=[value]
    )


_EVERY_GAS = _open(_GAS, evidence="requested_all", requested_all_phrase="by gas")


@pytest.mark.parametrize(
    ("shape", "variable", "drafted", "bound"),
    [
        pytest.param(
            _EMISSIONS,
            "greenhouse gas emissions",
            [_open(_GAS), _open(_SOURCE)],
            {"emittedThing": ["GreenhouseGas"]},
            id="one-value-named",
        ),
        pytest.param(
            _EMISSIONS, "emissions", [_open(_GAS), _open(_SOURCE)], {}, id="no-value-named"
        ),
        pytest.param(
            _EMISSIONS,
            "greenhouse gas emissions by gas",
            [_EVERY_GAS, _open(_SOURCE)],
            {},
            id="asked-in-full",
        ),
        pytest.param(
            _EMISSIONS,
            "greenhouse gas and carbon dioxide emissions",
            [_open(_GAS), _open(_SOURCE)],
            {},
            id="two-values-named",
        ),
        pytest.param(
            _EMISSIONS,
            "emissions excluding greenhouse gas",
            [_open(_GAS), _open(_SOURCE)],
            {},
            id="value-excluded",
        ),
        pytest.param(
            _EMISSIONS,
            "greenhouse gas emissions",
            [_value(_GAS, "CarbonDioxide"), _open(_SOURCE)],
            {"emittedThing": ["CarbonDioxide"]},
            id="reply-bound-another-value",
        ),
        pytest.param(
            _EMISSIONS,
            "greenhouse gas emissions",
            [_open(_GAS), _value(_SOURCE, "ForestryAndLandUse")],
            {"emissionSource": ["ForestryAndLandUse"]},
            id="no-member-holds-both",
        ),
        pytest.param(
            _CARBON_DIOXIDE,
            "carbon dioxide emissions",
            [_open(_GAS), _open(_SOURCE)],
            {},
            id="every-member-holds-it",
        ),
        pytest.param(
            _INFANT_MORTALITY,
            "infant mortality rate",
            [_open(_AGE_GROUP)],
            {},
            id="label-names-the-measure",
        ),
    ],
)
def test_an_open_slot_binds_the_one_value_the_phrase_names(shape, variable, drafted, bound):
    """A slot the reply left open binds the value the phrase names by its label, and is
    reported as text-recognised, by its dcid in words where the graph labels it with its
    dcid. A slot asked for in full, a phrase naming two values or excluding the one it
    names, a value the reply bound, a value no member holds with the rest of the bindings,
    a value every member holds, and a word a member's name holds without the value, as
    "infant" in the infant mortality rate, leave the bindings as they were."""
    result, named = _with_named_values(drafted, shape, variable)

    assert {b.property_dcid: b.value_dcids for b in result if b.value_dcids} == bound
    assert named == {b.property_dcid for b in result if b.value_dcids and b not in drafted}
    assert [b.kind for b in result if b not in drafted] == ["value"] * len(named)


def test_settlement_binds_the_gas_a_reply_left_open():
    """The recorded Gemini reply to "greenhouse gas emissions" on the Nigeria menu left
    ``emittedThing`` unbound, so the answer read every gas with data there, the OECD's
    carbon dioxide, and never handed the part to the UN SDG series. Settlement binds the
    gas the phrase names, as text-recognised, so the part asks for greenhouse gas alone."""
    state = _state(
        variable="greenhouse gas emissions",
        shape_draft=_EMISSIONS,
        bindings=[_open(_GAS, evidence="mentioned"), _open(_SOURCE, evidence="mentioned")],
        unplaced=[],
        entities=("Nigeria",),
    )
    state.capped_shapes = [_EMISSIONS]

    _settle_bindings(state, _EMISSIONS)

    assert [(b.property_dcid, b.kind, b.value_dcids) for b in state.bind_result.bindings] == [
        ("emittedThing", "value", ["GreenhouseGas"]),
        ("emissionSource", "unbound", []),
    ]
    assert state.text_recognized_properties == {"emittedThing"}
