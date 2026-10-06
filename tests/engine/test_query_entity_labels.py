"""Accepted query place text remains the presentation label after grounding."""

from __future__ import annotations

import asyncio

import pytest

from search.engine.bind import SlotBindingDraft, _UniversalBindOutput
from search.engine.clients.graph import Facet
from search.engine.region.codes import ENTITY_LABEL_UNRESOLVED
from search.engine.region.grounding import _build_entity, _ground_answer
from search.engine.region.pipeline import resolve_variable
from search.engine.retrieval.read_plan import plan_standard
from search.engine.shape import ShapeDraft, SlotKeyDraft
from search.engine.spec_build import build_shape_model
from search.render.heading import heading_from_parts
from search.render.plan import render_sentence
from search.render.tokens import plan_tokens, sentence_from_plan
from tests.engine._fake_coverage import empty_coverage_snapshot
from tests.engine._fake_sdmx import FakeSdmxClient
from tests.fixtures import FakeGraph, make_resolver

_WHERE_SLOT = SlotKeyDraft(axis="where", property_dcid=None, label="place")
_SHAPE = ShapeDraft(
    shape_id="person_count",
    label="Total population",
    pop_type_dcid="",
    meas_prop_dcid="",
    stat_type_dcid="",
    meas_qual_dcid=None,
    meas_denom_dcid=None,
    slot_keys=(_WHERE_SLOT,),
)


def _grounded_place(*, dcid: str, graph_label: str, query_text: str | None):
    graph = FakeGraph(nodes={dcid: {"label": graph_label, "type": "Country"}})
    grounded = _ground_answer(
        _SHAPE,
        [SlotBindingDraft(axis="where", property_dcid=None, kind="value", value_dcids=[dcid])],
        [],
        [(dcid, "observationAbout")],
        graph,
        query_entity_text_by_dcid={dcid: query_text} if query_text else None,
        open_axes=frozenset(),
    )
    entities = [_build_entity(*entity_data, grounded[1]) for entity_data in grounded[5]]
    shape = build_shape_model(_SHAPE, grounded[3], grounded[0], member_count=1, members=None)
    sentence = sentence_from_plan(
        plan_tokens(
            heading_from_parts(grounded[2], entities, shape, None, pinned_properties=frozenset())
        )
    )
    return grounded, sentence


@pytest.mark.parametrize(
    ("dcid", "graph_label", "query_text"),
    [
        ("country/SWZ", "Swaziland", "Eswatini"),
        ("country/TUR", "Turkey", "Turkiye"),
    ],
)
def test_accepted_place_alias_is_used_for_entity_and_sentence(
    dcid: str, graph_label: str, query_text: str
) -> None:
    grounded, sentence = _grounded_place(dcid=dcid, graph_label=graph_label, query_text=query_text)

    assert grounded[5][0][1].dcid == dcid
    assert grounded[5][0][1].label == query_text
    assert sentence == f"{query_text}'s total population."


@pytest.mark.parametrize("query_text", [None, "Swaziland"])
def test_omitted_or_stored_place_name_keeps_graph_label(query_text: str | None) -> None:
    grounded, sentence = _grounded_place(
        dcid="country/SWZ", graph_label="Swaziland", query_text=query_text
    )

    assert grounded[5][0][1].label == "Swaziland"
    assert sentence == "Swaziland's total population."


def test_query_place_text_is_used_when_the_graph_label_is_missing() -> None:
    dcid = "country/SWZ"
    graph = FakeGraph(nodes={dcid: {"type": "Country"}})
    grounded = _ground_answer(
        _SHAPE,
        [SlotBindingDraft(axis="where", property_dcid=None, kind="value", value_dcids=[dcid])],
        [],
        [(dcid, "observationAbout")],
        graph,
        query_entity_text_by_dcid={dcid: "Eswatini"},
        open_axes=frozenset(),
    )
    entities = [_build_entity(*entity_data, grounded[1]) for entity_data in grounded[5]]
    shape = build_shape_model(_SHAPE, grounded[3], grounded[0], member_count=1, members=None)
    sentence = sentence_from_plan(
        plan_tokens(
            heading_from_parts(grounded[2], entities, shape, None, pinned_properties=frozenset())
        )
    )

    assert grounded[5][0][1].label == "Eswatini"
    assert sentence == "Eswatini's total population."
    assert [warning.code for warning in grounded[9]] == [ENTITY_LABEL_UNRESOLVED]


def test_query_entity_text_never_relabels_stat_vars_or_sources() -> None:
    graph = FakeGraph(
        nodes={
            "country/SWZ": {"label": "Swaziland", "type": "Country"},
            "Count_Person_Swaziland": {"label": "Swaziland population"},
            "dc/base/SwazilandData": {"label": "Swaziland statistical office"},
        }
    )

    grounded = _ground_answer(
        _SHAPE,
        [
            SlotBindingDraft(
                axis="where", property_dcid=None, kind="value", value_dcids=["country/SWZ"]
            )
        ],
        ["Count_Person_Swaziland"],
        [("country/SWZ", "observationAbout")],
        graph,
        plan=plan_standard(
            {
                ("Count_Person_Swaziland", "country/SWZ"): [
                    Facet(
                        earliest_date="2020",
                        latest_date="2020",
                        obs_count=1,
                        facet_id="f1",
                        provenance_id="dc/base/SwazilandData",
                    )
                ]
            },
            members=["Count_Person_Swaziland"],
            years=None,
        ),
        query_entity_text_by_dcid={
            "country/SWZ": "Eswatini",
            "Count_Person_Swaziland": "not an entity label",
            "dc/base/SwazilandData": "not a source label",
        },
        open_axes=frozenset(),
    )

    assert grounded[4][0].label == "Swaziland population"
    assert grounded[6][0].label == "Swaziland statistical office"


class _FixedBindLLM:
    async def generate_structured(self, *, prompt: str, system: str, schema: type):
        assert schema is _UniversalBindOutput
        return _UniversalBindOutput(chosen_shape_index=0, bindings=[]), None


def test_pipeline_threads_accepted_place_text_to_resolved_entity_and_sentence() -> None:
    stat_var = "TestAliasPopulation"
    graph = FakeGraph(
        nodes={
            "country/SWZ": {"label": "Swaziland", "type": "Country"},
            stat_var: {
                "label": "Total population",
                "type": "StatisticalVariable",
                "arcs": {
                    "typeOf": {"nodes": [{"dcid": "StatisticalVariable"}]},
                    "name": {"nodes": [{"value": "Total population"}]},
                    "populationType": {"nodes": [{"dcid": "Person"}]},
                    "measuredProperty": {"nodes": [{"dcid": "count"}]},
                    "statType": {"nodes": [{"dcid": "measuredValue"}]},
                },
            },
        },
        obs={
            f"{stat_var}|country/SWZ": [
                {"earliestDate": "2020", "latestDate": "2020", "obsCount": 1}
            ]
        },
        detect={"population": {"svs": [stat_var], "cosine_scores": [0.9]}},
        resolve={"Eswatini": "country/SWZ"},
    )

    events = []

    async def emit(event) -> None:
        events.append(event)

    result = asyncio.run(
        resolve_variable(
            "population",
            entities=["Eswatini"],
            date_request=None,
            detect_query="population",
            role_query="population in Eswatini",
            pac=True,
            graph=graph,
            resolver=make_resolver(graph),
            llm=_FixedBindLLM(),
            sdmx=FakeSdmxClient(observations={}, availability={}),
            base_timing={},
            emit=emit,
            coverage=empty_coverage_snapshot(),
        )
    )

    spec = result.specs[0]
    assert spec.resolved.entities[0].label == "Eswatini"
    assert render_sentence(spec).startswith("Eswatini's ")
    sentence_event = next(event for event in events if event.kind == "sentence")
    assert sentence_from_plan(sentence_event.sentence_plan).startswith("Eswatini's ")
