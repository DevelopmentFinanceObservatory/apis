"""Each StatVar's wire `slot_values` must reflect its own constraint arcs.

A synthetic two-member set binding exercises values that the recorded query fixtures lack.
"""

from __future__ import annotations

from materialise.graph import SubjectPage
from search.engine.bind import SlotBindingDraft, _UniversalBindOutput
from search.models import ParsedEntity, ParsedInput, ParsedPart, ResolveRequest
from tests.engine._fake_sdmx import FakeSdmxClient
from tests.engine._harness import materialise_double_from_obs, offline_resolve
from tests.fixtures import FakeGraph

_PROP = "testSlotValuesSource"
_VAL_A = "SlotValuesSourceA"
_VAL_B = "SlotValuesSourceB"
_QUERY = "arc derived slot values test variable in Kenya"

_SV_A = "Test_SlotValues_SourceA"
_SV_B = "Test_SlotValues_SourceB"

_FIVE_TUPLE_ARCS = {
    "typeOf": {"nodes": [{"dcid": "StatisticalVariable"}]},
    "populationType": {"nodes": [{"dcid": "TestSlotValuesActivity"}]},
    "measuredProperty": {"nodes": [{"dcid": "testSlotValuesExpenditure"}]},
    "statType": {"nodes": [{"dcid": "measuredValue"}]},
}


def _sv_arcs(label: str, value: str) -> dict:
    return {
        **_FIVE_TUPLE_ARCS,
        "name": {"nodes": [{"value": label}]},
        "constraintProperties": {"nodes": [{"dcid": _PROP}]},
        _PROP: {"nodes": [{"dcid": value}]},
    }


_NODES = {
    "country/KEN": {"label": "Kenya", "type": "Country"},
    _VAL_A: {"label": "Slot Values Source A", "type": "TestSlotValuesSourceEnum"},
    _VAL_B: {"label": "Slot Values Source B", "type": "TestSlotValuesSourceEnum"},
    _SV_A: {
        "label": "Slot values SourceA",
        "type": "StatisticalVariable",
        "arcs": _sv_arcs("Slot values SourceA", _VAL_A),
    },
    _SV_B: {
        "label": "Slot values SourceB",
        "type": "StatisticalVariable",
        "arcs": _sv_arcs("Slot values SourceB", _VAL_B),
    },
}


def _member_arcs(node_arcs: dict) -> dict[str, tuple[str, ...]]:
    """This fixture's `{"prop": {"nodes": [{"dcid"|"value": ...}, ...]}}`
    node-arc shape flattened into the `{property: (dcid_or_value, ...)}`
    tuple form the materialise-side FakeGraphReader's own `arcs=` expects."""
    return {
        prop: tuple(n.get("dcid") or n.get("value") for n in entry["nodes"])
        for prop, entry in node_arcs.items()
    }


class _SetBindLLM:
    async def generate_structured(self, *, prompt, system, schema):
        assert schema.__name__ == "_UniversalBindOutput"
        return _UniversalBindOutput(
            chosen_shape_index=0,
            bindings=[
                SlotBindingDraft(
                    axis="how",
                    property_dcid=_PROP,
                    kind="set",
                    value_dcids=[_VAL_A, _VAL_B],
                )
            ],
            also_plausible=None,
            ask=None,
        ), None


def _obs_facet(*, value: float, provenance_id: str) -> dict:
    return {
        "earliestDate": "2020",
        "latestDate": "2020",
        "obsCount": 1,
        "provenanceId": provenance_id,
        "dates": ["2020"],
        "values": [value],
    }


def test_two_stat_vars_carry_only_their_own_set_binding_member():
    """A how-axis set binding across two StatVars sharing one five-tuple:
    each StatVar's own arcs assert one member, so each StatVar's slot_values
    on the wire carries only its own member, never its sibling's."""
    obs = {
        f"{_SV_A}|country/KEN": [_obs_facet(value=10.0, provenance_id="dc/base/SlotValuesA")],
        f"{_SV_B}|country/KEN": [_obs_facet(value=20.0, provenance_id="dc/base/SlotValuesB")],
    }
    graph = FakeGraph(
        nodes=_NODES,
        detect={_QUERY: {"svs": [_SV_A, _SV_B], "cosine_scores": [0.95, 0.9]}},
        resolve={"Kenya": "country/KEN"},
        obs=obs,
    )
    materialise_graph, materialise_sdmx = materialise_double_from_obs(obs)
    materialise_graph._arcs = {
        _SV_A: _member_arcs(_NODES[_SV_A]["arcs"]),
        _SV_B: _member_arcs(_NODES[_SV_B]["arcs"]),
    }
    materialise_graph._subjects = {
        ("testSlotValuesExpenditure", "measuredProperty"): SubjectPage(
            dcids=(_SV_A, _SV_B), truncated=False
        ),
    }

    request = ResolveRequest(
        input=ParsedInput(
            parts=[ParsedPart(variable_text=_QUERY, entities=[ParsedEntity(text="Kenya")])]
        )
    )
    result = offline_resolve(
        request,
        graph=graph,
        llm=_SetBindLLM(),
        sdmx=FakeSdmxClient(observations={}, availability={}),
        materialise_graph=materialise_graph,
        materialise_sdmx=materialise_sdmx,
        pac=True,
    )

    assert result.parts[0].status == "definite", f"expected definite, got {result.parts[0]}"
    spec = result.parts[0].spec
    stat_vars_by_ref = {sv.ref: sv for sv in spec.resolved.stat_vars.items}
    assert set(stat_vars_by_ref) == {_SV_A, _SV_B}

    a_values = {v.value.ref.dcid for v in stat_vars_by_ref[_SV_A].slot_values}
    b_values = {v.value.ref.dcid for v in stat_vars_by_ref[_SV_B].slot_values}
    assert a_values == {_VAL_A}, f"{_SV_A} slot_values should carry only {_VAL_A}, got {a_values}"
    assert b_values == {_VAL_B}, f"{_SV_B} slot_values should carry only {_VAL_B}, got {b_values}"
