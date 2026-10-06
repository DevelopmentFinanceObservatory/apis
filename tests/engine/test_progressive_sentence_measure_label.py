"""Check bind-stage sentence labels against the final measure label.

Measure-family slots fold into the `what` phrase. Shapes without a measure
family emit only the post-materialise sentence event.
"""

from __future__ import annotations

import asyncio

import pytest

from search.engine.events import Event, SentenceEvent
from search.engine.extract import DateRequest
from search.engine.region import phase_place_materialise, resolve_variable, sentence_events
from search.engine.spec_build import NamedGroup
from search.models import (
    BindingValue,
    GraphRef,
    IdentityEntityGroup,
    Shape,
    Slot,
    SlotKey,
    SlotValue,
    TimeWindow,
)
from search.render.heading import declared_axis_names, heading_from_parts
from search.render.plan import render_sentence_plan
from search.render.tokens import plan_tokens, sentence_from_plan
from tests.engine._base_identity import destination_entity
from tests.engine._fake_coverage import FakeCoverageService, patch_coverage
from tests.engine._fake_sdmx import FakeSdmxClient
from tests.engine._harness import door_roles_for_test_query
from tests.engine.test_measure_election_inapplicable_slots import (
    _COVERAGE_ROWS,
    _DETECT,
    _EXTRA_SV,
    _KEN_DCID,
    _NODES,
    _OBS,
    _PROP_FLOW_TYPE,
    _PROP_PURPOSE,
    _QUERY,
    _ROLE_NAMES,
    _BindPurposeToExtraLLM,
)
from tests.engine.test_standard_multi_place import _SV as _NO_FAMILY_SV
from tests.engine.test_standard_multi_place import _run as _run_no_family
from tests.fixtures import FakeGraph, make_resolver


def _run_with_events(
    monkeypatch, date_request: DateRequest | None = None
) -> tuple[object, list[Event]]:
    events: list[Event] = []

    async def _emit(event: Event) -> None:
        events.append(event)

    _auto_coverage_index_1 = patch_coverage(
        monkeypatch, _COVERAGE_ROWS, role_names={_EXTRA_SV: _ROLE_NAMES}
    )
    graph = FakeGraph(nodes=_NODES, detect=_DETECT, resolve={"Kenya": _KEN_DCID}, obs={})
    result = asyncio.run(
        resolve_variable(
            _QUERY,
            entities=["Kenya"],
            date_request=date_request,
            detect_query=_QUERY,
            role_query=_QUERY,
            entity_roles=door_roles_for_test_query(_QUERY, ["Kenya"]),
            pac=True,
            graph=graph,
            resolver=make_resolver(graph),
            sdmx=FakeSdmxClient(observations=_OBS, availability={}),
            llm=_BindPurposeToExtraLLM(),
            base_timing={},
            emit=_emit,
            coverage=FakeCoverageService(_auto_coverage_index_1).snapshot(),
        )
    )
    return result, events


def test_bind_stage_what_token_matches_the_final_measure_label(monkeypatch):
    """The bind-stage `what` token uses the elected label "grant equivalents"."""
    result, events = _run_with_events(monkeypatch)
    assert result.status == "definite"
    spec = result.specs[0]
    assert spec.resolved.measure_label is not None

    sentence_events = [e for e in events if isinstance(e, SentenceEvent)]
    assert len(sentence_events) == 2  # bind-stage, then the post-materialise convergence
    bind_stage_plan = sentence_events[0].sentence_plan
    what_token = next(tok for tok in bind_stage_plan if tok.kind == "slot" and tok.axis == "what")
    # what_token.text is sentence-capitalised (it opens the plan); measure_label
    # is stored lower-case, the same relationship the final sentence's own
    # what token has to it.
    assert what_token.text.lower() == spec.resolved.measure_label


@pytest.mark.parametrize("named", [True, False], ids=["group-named", "no-group"])
def test_bind_stage_sentence_carries_the_group_the_query_named(monkeypatch, named):
    """A role whose entities are exactly a group the query named reaches the bind-stage
    heading as that group; with no group named, none."""
    group = NamedGroup(label="the test group", dcid=None, member_dcids=frozenset({_KEN_DCID}))
    monkeypatch.setattr(
        phase_place_materialise, "request_named_groups", lambda _state: (group,) if named else ()
    )
    groups_rendered: list[object] = []

    def _spy(slots, entities, *args, **kwargs):
        # The pre-bind shape skeleton names no entity; the bind-stage sentence does.
        if entities:
            groups_rendered.append(kwargs.get("entity_groups"))
        return heading_from_parts(slots, entities, *args, **kwargs)

    monkeypatch.setattr(sentence_events, "heading_from_parts", _spy)
    _run_with_events(monkeypatch)

    expected = (
        {"destination": IdentityEntityGroup(dcid=None, label="the test group")} if named else {}
    )
    assert groups_rendered == [expected]


def test_bind_stage_measure_axis_how_slot_is_folded_not_listed(monkeypatch):
    """The elected `financialFlowType` stays folded into `what`, without a second clause."""
    _, events = _run_with_events(monkeypatch)
    sentence_events = [e for e in events if isinstance(e, SentenceEvent)]
    bind_stage_plan = sentence_events[0].sentence_plan
    flow_type_token = next(
        tok
        for tok in bind_stage_plan
        if tok.kind == "slot" and tok.axis == "how" and tok.property_dcid == _PROP_FLOW_TYPE
    )
    assert flow_type_token.text == ""
    assert "disbursement" not in sentence_from_plan(bind_stage_plan).lower()


def test_bind_stage_slots_carry_the_summable_flag_of_the_final_spec(monkeypatch):
    """The bind-stage event's `crsPurpose` slot reads summable, as the final spec's does."""
    result, events = _run_with_events(monkeypatch)
    bind_stage = next(e for e in events if isinstance(e, SentenceEvent))
    bind_purpose = next(
        s for s in bind_stage.slots if s.key.property and s.key.property.dcid == _PROP_PURPOSE
    )
    final_purpose = next(
        s
        for s in result.specs[0].resolved.slots
        if s.key.property and s.key.property.dcid == _PROP_PURPOSE
    )
    assert bind_purpose.summable is True
    assert final_purpose.summable is True


def test_bind_stage_text_is_a_prefix_of_the_final_text_when_only_materialise_fields_are_added():
    """Adding a time window extends the bind-stage sentence without changing its text."""
    how_slot = Slot(
        key=SlotKey(
            axis="how",
            property=GraphRef(dcid=_PROP_FLOW_TYPE, label=_PROP_FLOW_TYPE),
            label=_PROP_FLOW_TYPE,
        ),
        binding=BindingValue(
            value=SlotValue(
                ref=GraphRef(dcid="Disbursement", label="Disbursement"), value_kind="enum_value"
            )
        ),
    )
    shape = Shape(shape_id="widget", label="Widget", slot_keys=[how_slot.key], member_count=1)
    entities = [destination_entity("country/KEN", "Kenya")]
    measure_label = "grant equivalents"
    measure_label_axes = [_PROP_FLOW_TYPE]

    bind_stage_tokens = plan_tokens(
        heading_from_parts(
            [how_slot],
            entities,
            shape,
            window=None,
            measure_label=measure_label,
            measure_label_axes=measure_label_axes,
            pinned_properties=frozenset(),
        )
    )
    final_tokens = plan_tokens(
        heading_from_parts(
            [how_slot],
            entities,
            shape,
            window=TimeWindow(start_year=2020, end_year=2020),
            measure_label=measure_label,
            measure_label_axes=measure_label_axes,
            pinned_properties=frozenset(),
        )
    )
    bind_stage_text = sentence_from_plan(bind_stage_tokens)
    final_text = sentence_from_plan(final_tokens)

    assert "disbursement" not in bind_stage_text.lower()
    assert bind_stage_text.endswith(".")
    assert final_text.startswith(bind_stage_text[: -len(".")])
    assert final_text != bind_stage_text


def test_no_family_part_emits_only_the_final_sentence_event():
    """Without a measure family, the part emits only the final sentence event."""
    events: list = []

    async def _emit(event) -> None:
        events.append(event)

    obs = {
        f"{_NO_FAMILY_SV}|country/GTM": [
            {"earliestDate": "2020", "latestDate": "2020", "obsCount": 1}
        ]
    }
    result = _run_no_family(["Guatemala"], role_query="population Guatemala", obs=obs, emit=_emit)

    assert result.status == "definite"
    sentence_events = [e for e in events if isinstance(e, SentenceEvent)]
    assert len(sentence_events) == 1
    assert sentence_events[0].spec_id == result.specs[0].spec_id


def _last_sentence(events: list[Event]) -> str:
    """The sentence a client holds once the stream ends: the last one for the part."""
    return sentence_from_plan([e for e in events if isinstance(e, SentenceEvent)][-1].sentence_plan)


def test_the_bind_stage_sentence_states_no_window_and_the_final_one_the_years_served(
    monkeypatch,
):
    """Kenya holds 2018 to 2020. "2015 to 2020" streams a bind-stage sentence with no
    years, then a final one stating the years served, 2018 to 2020."""
    request = DateRequest(window=TimeWindow(start_year=2015, end_year=2020), latest=False)
    result, events = _run_with_events(monkeypatch, request)

    assert result.status == "definite"
    sentences = [e for e in events if isinstance(e, SentenceEvent)]
    assert len(sentences) == 2
    bind_stage = sentences[0]
    assert not any(
        tok.kind == "slot" and tok.axis == "when" and tok.text for tok in bind_stage.sentence_plan
    )
    assert "2015" not in sentence_from_plan(bind_stage.sentence_plan)
    assert "2018 to 2020" in _last_sentence(events)


def test_the_bind_stage_sentence_is_a_heading_and_the_final_one_adds_period_and_byline(
    monkeypatch,
):
    """C35: the bind-stage sentence holds heading tokens only, with no period, while the
    final sentence is the Spec's own plan: the heading with its period, then the byline."""
    request = DateRequest(window=TimeWindow(start_year=2015, end_year=2020), latest=False)
    result, events = _run_with_events(monkeypatch, request)

    bind_stage, final = [e for e in events if isinstance(e, SentenceEvent)]
    assert {tok.segment for tok in bind_stage.sentence_plan if tok.text} == {"heading"}
    assert not any(tok.axis == "when" and tok.text for tok in bind_stage.sentence_plan)

    assert final.sentence_plan == render_sentence_plan(
        result.specs[0], axis_declarations=declared_axis_names()
    )
    when = [tok for tok in final.sentence_plan if tok.axis == "when"]
    assert [(tok.text, tok.segment) for tok in when] == [("2018 to 2020", "heading")]
    byline = [tok for tok in final.sentence_plan if tok.segment == "byline"]
    # This world cites no provenance and serves no currency: the byline is the empty
    # source placeholder, after the heading's period.
    assert [(tok.kind, tok.axis, tok.text) for tok in byline] == [("slot", "source", "")]
    heading_end = final.sentence_plan.index(byline[0]) - 1
    assert final.sentence_plan[heading_end].text == "."
    assert {tok.segment for tok in final.sentence_plan[: heading_end + 1]} == {"heading"}


def test_an_answer_with_no_data_leaves_no_window_in_the_last_sentence(monkeypatch):
    """Nothing is recorded from 2022 to 2024, so the answer is `no_data` and sends no final
    sentence: the bind-stage sentence a client keeps states no years."""
    request = DateRequest(window=TimeWindow(start_year=2022, end_year=2024), latest=False)
    result, events = _run_with_events(monkeypatch, request)

    assert result.status == "no_data"
    assert [e for e in events if isinstance(e, SentenceEvent)]
    last = _last_sentence(events)
    assert not any(year in last for year in ("2022", "2024")), last


def test_the_bind_stage_heading_states_its_roles_as_the_final_one_does(monkeypatch):
    """C35: the bind-stage heading reads the flow family from the elected members' arcs,
    so its role tokens come in the final heading's order and wording ("... to Kenya", not
    "Kenya's ...")."""
    _result, events = _run_with_events(monkeypatch)

    bind_stage, final = [e for e in events if isinstance(e, SentenceEvent)]

    def roles(event: SentenceEvent) -> list[tuple[int, str, str | None, str | None]]:
        plan = event.sentence_plan
        return [
            (i, plan[i - 1].text if i else "", tok.role, tok.text)
            for i, tok in enumerate(plan)
            if tok.role is not None
        ]

    assert roles(bind_stage)
    assert [r[1:] for r in roles(bind_stage)] == [r[1:] for r in roles(final)]
