"""Test shape selection, bindings, and prompt fields in universal bind."""

from __future__ import annotations

import asyncio
import dataclasses

import pytest

from code_index.shared import get_index
from search.engine.bind import (
    SlotBindingDraft,
    _build_universal_bind_prompt,
    _UniversalBindOutput,
    bind_universal,
)
from search.engine.quantity_read import QuantityOutput
from search.engine.shape import ShapeDraft, SlotKeyDraft
from tests.engine._fake_coverage import empty_coverage_snapshot
from tests.engine._fake_sdmx import FakeSdmxClient
from tests.engine._quantity import NO_QUANTITY

# Helpers


def _run(coro):
    return asyncio.run(coro)


# Minimal recall data for resolve_variable's failed-bind cases.
_RESOLVE_SV = "Count_Person"
_RESOLVE_DETECT_QUERY = "population"

_RESOLVE_NODES: dict = {
    "Person": {"label": "Person"},
    "count": {"label": "count"},
    "measuredValue": {"label": "Measured Value"},
    "observationAbout": {"label": "observation about"},
    "country/KEN": {"label": "Kenya", "type": "Country"},
    _RESOLVE_SV: {
        "label": "Population",
        "type": "StatisticalVariable",
        "arcs": {
            "typeOf": {"nodes": [{"dcid": "StatisticalVariable"}]},
            "populationType": {"nodes": [{"dcid": "Person"}]},
            "measuredProperty": {"nodes": [{"dcid": "count"}]},
            "statType": {"nodes": [{"dcid": "measuredValue"}]},
        },
    },
}
_RESOLVE_DETECT: dict = {_RESOLVE_DETECT_QUERY: {"svs": [_RESOLVE_SV], "entities": []}}
_RESOLVE_RESOLVE: dict = {"Kenya": "country/KEN"}


def _shape(
    shape_id: str,
    *,
    slot_keys: tuple[SlotKeyDraft, ...] = (),
    slot_taxonomy: dict[str, list[str]] | None = None,
    score: float = 0.8,
) -> ShapeDraft:
    return ShapeDraft(
        shape_id=shape_id,
        label=shape_id,
        pop_type_dcid="Person",
        meas_prop_dcid="count",
        stat_type_dcid="measuredValue",
        meas_qual_dcid=None,
        meas_denom_dcid=None,
        slot_keys=slot_keys,
        slot_taxonomy=slot_taxonomy or {},
        representative_score=score,
    )


_GENDER_SLOT = SlotKeyDraft(axis="how", property_dcid="gender", label="gender")


class _StubLLM:
    """Return a fixed universal-bind result."""

    def __init__(self, output: _UniversalBindOutput):
        self._output = output

    async def generate_structured(self, *, prompt, system, schema):
        if schema is QuantityOutput:
            return NO_QUANTITY
        assert schema.__name__ == "_UniversalBindOutput"
        return self._output, {"input_tokens": 0, "output_tokens": 0, "cached_tokens": 0}


# chosen_shape_index / bindings


def test_bind_universal_chooses_shape_and_binds_slots():
    shapes = [
        _shape(
            "count_person_female",
            slot_keys=(_GENDER_SLOT,),
            slot_taxonomy={"how:gender": ["Male", "Female"]},
        ),
    ]
    output = _UniversalBindOutput(
        chosen_shape_index=0,
        bindings=[
            SlotBindingDraft(
                axis="how", property_dcid="gender", kind="value", value_dcids=["Female"]
            ),
        ],
        also_plausible=None,
        ask=None,
    )
    result, usage = _run(bind_universal("female population", shapes, llm=_StubLLM(output)))
    assert result.chosen_shape_index == 0
    assert result.ask is None
    assert len(result.bindings) == 1
    assert result.bindings[0].value_dcids == ["Female"]
    assert usage is not None


def test_bind_universal_unbinds_a_source_draft_the_model_writes():
    """Only the engine names a dataset, so a source-axis value from the model comes back
    unbound and every other draft comes back as written."""
    shapes = [_shape("count_person_gender", score=0.9)]
    gender = SlotBindingDraft(
        axis="how", property_dcid="gender", kind="value", value_dcids=["Female"]
    )
    output = _UniversalBindOutput(
        chosen_shape_index=0,
        bindings=[
            gender,
            SlotBindingDraft(
                axis="source", property_dcid=None, kind="value", value_dcids=["provenance/Made_Up"]
            ),
        ],
        also_plausible=None,
        ask=None,
    )

    result, _ = _run(bind_universal("female population", shapes, llm=_StubLLM(output)))

    assert result.bindings == [
        gender,
        SlotBindingDraft(axis="source", property_dcid=None, kind="unbound"),
    ]


_AGE_SLOT = SlotKeyDraft(axis="how", property_dcid="undataAge", label="age")


@pytest.mark.parametrize(
    ("written", "taxonomy", "expected"),
    [
        pytest.param(
            ["undata/AGE-Y0T4"],
            ["undata/AGE-Y0T4", "undata/AGE-Y5T9"],
            SlotBindingDraft(
                axis="how", property_dcid="undataAge", kind="value", value_dcids=["undata/AGE-Y0T4"]
            ),
            id="exact-id-binds",
        ),
        pytest.param(
            ["undataAGE-Y0T4"],
            ["undata/AGE-Y0T4", "undata/AGE-Y5T9"],
            SlotBindingDraft(
                axis="how", property_dcid="undataAge", kind="value", value_dcids=["undata/AGE-Y0T4"]
            ),
            id="separator-dropped-binds-the-taxonomy-value",
        ),
        pytest.param(
            ["CRSTypeOfFlow10"],
            ["undata/AGE-Y0T4"],
            SlotBindingDraft(
                axis="how", property_dcid="undataAge", kind="value", value_dcids=["CRSTypeOfFlow10"]
            ),
            id="index-code-outside-taxonomy-binds",
        ),
        pytest.param(
            ["undataAGE-Y0T4"],
            ["undata/AGE-Y0T4", "undata/AGE_Y0T4"],
            SlotBindingDraft(axis="how", property_dcid="undataAge", kind="unbound"),
            id="ambiguous-normalised-match-unbinds",
        ),
        pytest.param(
            ["undataAGE-Y0T4", "Female"],
            ["undata/AGE-Y0T4", "undata/AGE-Y5T9"],
            SlotBindingDraft(
                axis="how",
                property_dcid="undataAge",
                kind="set",
                value_dcids=["undata/AGE-Y0T4", "Female"],
            ),
            id="unmatched-value-binds-as-written",
        ),
    ],
)
def test_bind_universal_grounds_values_in_the_chosen_shapes_taxonomy(written, taxonomy, expected):
    """under5-mortality-ethiopia-2015-2020: the model wrote ``undataAGE-Y0T4`` for
    ``undata/AGE-Y0T4``. A value grounds by exact or unique normalised match, an ambiguous
    one opens its slot while the elected shape stays chosen, and one matching nothing binds
    as written for materialise to refuse as a value no member holds."""
    shapes = [
        _shape("other_shape", slot_keys=(_AGE_SLOT,), slot_taxonomy={"how:undataAge": ["x"]}),
        _shape("under5", slot_keys=(_AGE_SLOT,), slot_taxonomy={"how:undataAge": taxonomy}),
    ]
    output = _UniversalBindOutput(
        chosen_shape_index=1,
        bindings=[
            SlotBindingDraft(
                axis="how",
                property_dcid="undataAge",
                kind="value" if len(written) == 1 else "set",
                value_dcids=written,
            )
        ],
        also_plausible=None,
        ask=None,
    )

    result, _ = _run(bind_universal("under-five mortality", shapes, llm=_StubLLM(output)))

    assert result.chosen_shape_index == 1
    assert result.bindings == [expected]


def test_bind_universal_picks_non_first_shape():
    shapes = [
        _shape("wrong_shape", score=0.9),
        _shape("right_shape", score=0.6),
    ]
    output = _UniversalBindOutput(chosen_shape_index=1, bindings=[], also_plausible=None, ask=None)
    result, _ = _run(bind_universal("some variable", shapes, llm=_StubLLM(output)))
    assert result.chosen_shape_index == 1


def test_bind_universal_also_plausible_ranked():
    shapes = [_shape("a"), _shape("b"), _shape("c")]
    output = _UniversalBindOutput(
        chosen_shape_index=0,
        bindings=[],
        also_plausible=[2, 1],
        ask=None,
    )
    result, _ = _run(bind_universal("ambiguous variable", shapes, llm=_StubLLM(output)))
    assert result.also_plausible == [2, 1]


# CANNOT-BIND (ask) path


def test_bind_universal_off_topic_sets_ask():
    shapes = [_shape("unrelated_shape")]
    output = _UniversalBindOutput(
        chosen_shape_index=0,
        bindings=[],
        also_plausible=None,
        ask="Variable 'weather in London' does not match any candidate shape.",
    )
    result, _ = _run(bind_universal("weather in London", shapes, llm=_StubLLM(output)))
    assert result.ask is not None
    assert result.bindings == []


def test_bind_ask_triggers_no_data_in_resolve_variable():
    """When bind_universal returns ask, resolve_variable returns no_data(variable_not_resolved)."""
    import asyncio

    from search.engine.region import resolve_variable
    from tests.fixtures import FakeGraph, make_resolver

    class _AskLLM:
        async def generate_structured(self, *, prompt, system, schema):
            if schema.__name__ == "_UniversalBindOutput":
                return _UniversalBindOutput(
                    chosen_shape_index=0,
                    bindings=[],
                    also_plausible=None,
                    ask="Variable is completely off-topic.",
                ), None
            if schema is QuantityOutput:
                return NO_QUANTITY
            raise AssertionError(f"unexpected schema {schema.__name__!r}")

    graph = FakeGraph(
        nodes=_RESOLVE_NODES, detect=_RESOLVE_DETECT, resolve=_RESOLVE_RESOLVE, obs={}
    )

    result = asyncio.run(
        resolve_variable(
            "population",
            entities=["Kenya"],
            date_request=None,
            detect_query="population",
            role_query="population of Kenya",
            pac=True,
            graph=graph,
            resolver=make_resolver(graph),
            sdmx=FakeSdmxClient(observations={}, availability={}),
            llm=_AskLLM(),
            base_timing={},
            coverage=empty_coverage_snapshot(),
        )
    )
    assert result.status == "no_data"
    assert result.no_data_reason == "variable_not_resolved"


def test_bind_out_of_range_index_triggers_no_data():
    """An out-of-range chosen_shape_index returns no_data."""
    import asyncio

    from search.engine.region import resolve_variable
    from tests.fixtures import FakeGraph, make_resolver

    class _OutOfRangeLLM:
        async def generate_structured(self, *, prompt, system, schema):
            if schema.__name__ == "_UniversalBindOutput":
                return _UniversalBindOutput(
                    chosen_shape_index=99,
                    bindings=[],
                    also_plausible=None,
                    ask=None,
                ), None
            if schema is QuantityOutput:
                return NO_QUANTITY
            raise AssertionError(f"unexpected schema {schema.__name__!r}")

    graph = FakeGraph(
        nodes=_RESOLVE_NODES, detect=_RESOLVE_DETECT, resolve=_RESOLVE_RESOLVE, obs={}
    )

    result = asyncio.run(
        resolve_variable(
            "population",
            entities=["Kenya"],
            date_request=None,
            detect_query="population",
            role_query="population of Kenya",
            pac=True,
            graph=graph,
            resolver=make_resolver(graph),
            sdmx=FakeSdmxClient(observations={}, availability={}),
            llm=_OutOfRangeLLM(),
            base_timing={},
            coverage=empty_coverage_snapshot(),
        )
    )
    assert result.status == "no_data"
    assert result.no_data_reason == "variable_not_resolved"


# The prompt's member sample uses recall members, even when the index has more.


def test_bind_prompt_sample_stays_on_the_recall_pool_for_an_index_carried_shape():
    """The prompt sample contains one recalled member despite hundreds in the index."""
    index = get_index()
    target = ("FinancialFlow", "grantEquivalent", "measuredValue", "Nominal", None)
    population = [
        row.dcid
        for row in index.statvars
        if (
            row.population_type or "",
            row.measured_property,
            row.stat_type or "",
            row.measurement_qualifier,
            row.measurement_denominator,
        )
        == target
    ]
    assert len(population) > 100
    recalled_sv = population[0]

    shape = ShapeDraft(
        shape_id="test_bind_prompt_sample_shape",
        label="test shape",
        pop_type_dcid=target[0],
        meas_prop_dcid=target[1],
        stat_type_dcid=target[2],
        meas_qual_dcid=target[3],
        meas_denom_dcid=target[4],
        slot_keys=(),
        sv_arc_facts={recalled_sv: {}},
        sv_arc_facts_complete=False,
    )

    prompt = _build_universal_bind_prompt("test variable", [shape])

    assert f"sample_member_dcids: ['{recalled_sv}']" in prompt
    # Index-only members stay out of the prompt.
    for other_sv in population[1:20]:
        assert other_sv not in prompt


# Prompt controls for withheld axes and dropped qualifiers.

_MATERIAL_SLOT = SlotKeyDraft(axis="what", property_dcid="material", label="material")
_FINISH_SLOT = SlotKeyDraft(axis="how", property_dcid="finish", label="finish")


_HEAD_WIDGET_PROMPT = (
    "Variable phrase: widget variable\n\n"
    "CANDIDATE SHAPES (1 total):\n\n"
    "Shape 0:\n"
    "  label: widget\n"
    "  family: standard Data Commons\n"
    "  five_tuple: populationType=Person, measuredProperty=count, statType=measuredValue\n"
    "  retrieval_score: 0.800\n"
    "  constraint_slots:\n"
    "    axis=what, property_dcid=material (material)\n"
    "      taxonomy: ['Plastic', 'Metal']\n"
    "    axis=how, property_dcid=finish (finish)\n"
    "      taxonomy: ['Matte', 'Glossy']\n"
)


def test_no_dropped_phrases_keeps_a_recall_sourced_prompt_byte_identical_to_head():
    """An empty dropped-phrase list preserves the prompt bytes."""
    shape = _shape(
        "widget",
        slot_keys=(_MATERIAL_SLOT, _FINISH_SLOT),
        slot_taxonomy={"what:material": ["Plastic", "Metal"], "how:finish": ["Matte", "Glossy"]},
    )
    assert _build_universal_bind_prompt("widget variable", [shape]) == _HEAD_WIDGET_PROMPT
    # Explicit empty dropped_qualifier_phrases is the exact same call.
    assert (
        _build_universal_bind_prompt("widget variable", [shape], dropped_qualifier_phrases=())
        == _HEAD_WIDGET_PROMPT
    )


def test_a_recall_sourced_shapes_slot_shows_its_value_labels():
    """A walked or held-family shape labels its values from `value_labels`, as an
    index-sourced shape does from its code rows; a label repeating its dcid is left out."""
    shape = dataclasses.replace(
        _shape(
            "widget",
            slot_keys=(_MATERIAL_SLOT,),
            slot_taxonomy={"what:material": ["undata/MAT-P", "undata/MAT-M"]},
        ),
        value_labels={"undata/MAT-P": "Plastic", "undata/MAT-M": "undata/MAT-M"},
    )

    prompt = _build_universal_bind_prompt("widget variable", [shape])

    assert (
        "      taxonomy: ['undata/MAT-P', 'undata/MAT-M']\n      labels: undata/MAT-P = Plastic\n"
    ) in prompt


def test_dropped_qualifiers_block_renders_only_when_phrases_exist():
    shape = _shape("widget")
    without = _build_universal_bind_prompt("widget variable", [shape])
    assert "DROPPED QUALIFIERS" not in without

    with_phrases = _build_universal_bind_prompt(
        "widget variable", [shape], dropped_qualifier_phrases=("across all flows",)
    )
    assert "DROPPED QUALIFIERS" in with_phrases
    assert "across all flows" in with_phrases
    assert "requested_all" in with_phrases
    assert "requested_all_phrase" in with_phrases


def test_withheld_measure_axes_drop_an_index_sourced_shapes_slots_and_keep_the_default():
    shape = dataclasses.replace(
        _shape(
            "widget",
            slot_keys=(_MATERIAL_SLOT, _FINISH_SLOT),
            slot_taxonomy={
                "what:material": ["Plastic", "Metal"],
                "how:finish": ["Matte", "Glossy"],
            },
        ),
        slot_taxonomy_index_sourced=True,
    )

    withheld = _build_universal_bind_prompt(
        "widget variable", [shape], withheld_measure_axes=frozenset({"material"})
    )
    assert "property_dcid=material" not in withheld
    assert "property_dcid=finish" in withheld

    with_default = _build_universal_bind_prompt("widget variable", [shape])
    with_explicit_empty = _build_universal_bind_prompt(
        "widget variable", [shape], withheld_measure_axes=frozenset()
    )
    assert with_default == with_explicit_empty
    assert "property_dcid=material" in with_default
    assert "property_dcid=finish" in with_default


def test_withheld_measure_axes_leave_a_recall_sourced_shapes_slots_for_the_binder():
    """No election reads a recall-sourced shape's members, so the binder binds the axis."""
    shape = _shape(
        "widget",
        slot_keys=(_MATERIAL_SLOT, _FINISH_SLOT),
        slot_taxonomy={"what:material": ["Plastic", "Metal"], "how:finish": ["Matte", "Glossy"]},
    )

    withheld = _build_universal_bind_prompt(
        "widget variable", [shape], withheld_measure_axes=frozenset({"material"})
    )

    assert withheld == _HEAD_WIDGET_PROMPT
