"""The frame each door builds for one part (`search.engine.frame`).

The two text doors ask the place-set world (`_place_set_world`) one selection each; the
event tests ask the fixture CRS index for "aid flows to Kenya".
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any
from unittest.mock import patch

import pytest

from code_index.shared import get_index
from search.engine.core import resolve_events
from search.engine.events import FrameEvent, InterpretationEvent, ShapeEvent, serialize_sse
from search.engine.extract import DateRequest, ExtractedEntity, Extraction
from search.engine.frame import Frame, door_frame, with_recall
from search.engine.quantity import quantity_requirement
from search.models import (
    ParsedEntity,
    RawTextInput,
    ResolveOptions,
    ResolveRequest,
    TimeWindow,
)
from tests.engine._fake_sdmx import FakeSdmxClient
from tests.engine._place_set_world import (
    AID,
    WHO,
    Doors,
    destination,
    members,
    origin,
)
from tests.engine.test_operations_e2e import _OpenFlowAndSectorBindLLM
from tests.engine.test_qualifier_suppression_and_provenance import _e2e_graph_for

_VARIABLE = "aid flows"
_EMPTY_SDMX = FakeSdmxClient(observations={}, availability={})


def _llm() -> _OpenFlowAndSectorBindLLM:
    return _OpenFlowAndSectorBindLLM(
        Extraction(
            kind="resolution",
            variables=[_VARIABLE],
            entities=[
                ExtractedEntity(text="Kenya", span="Kenya", flow_role="destination", role_span="to")
            ],
        )
    )


def _events(request: ResolveRequest) -> list[Any]:
    async def collect() -> list[Any]:
        return [
            event
            async for event in resolve_events(
                request, graph=_e2e_graph_for(_VARIABLE), llm=_llm(), sdmx=_EMPTY_SDMX
            )
        ]

    return asyncio.run(collect())


def _without_stated_by(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _without_stated_by(v) for k, v in value.items() if k != "stated_by"}
    if isinstance(value, list):
        return [_without_stated_by(item) for item in value]
    return value


@pytest.fixture
def doors(monkeypatch: pytest.MonkeyPatch) -> Doors:
    return Doors(monkeypatch)


@contextmanager
def _built_frames() -> Iterator[list[Frame]]:
    """Every frame the text doors complete after recall, in the order they are built."""
    frames: list[Frame] = []

    def recalled(*args: Any, **kwargs: Any) -> Frame:
        frames.append(with_recall(*args, **kwargs))
        return frames[-1]

    with patch("search.engine.region.phase_rank_bind.with_recall", recalled):
        yield frames


_FLOW = (
    "aid from France to Kenya",
    AID,
    (
        ExtractedEntity(text="France", span="France", flow_role="origin", role_span="from"),
        ExtractedEntity(text="Kenya", span="Kenya", flow_role="destination", role_span="to"),
    ),
    (origin("France"), destination("Kenya")),
)
_MEMBERS = (
    "how much do African countries spend on their own health care?",
    WHO,
    (ExtractedEntity(text="Africa", span="African", members=True),),
    (members("Africa"),),
)


@pytest.mark.parametrize(
    ("query", "stat_var", "extracted", "posted"), [_FLOW, _MEMBERS], ids=["flow", "members"]
)
def test_one_selection_gives_one_frame_through_both_text_doors(
    doors: Doors,
    query: str,
    stat_var: str,
    extracted: tuple[ExtractedEntity, ...],
    posted: tuple[ParsedEntity, ...],
):
    """A selection asked through raw_text and its parsed form builds one frame after
    recall. Extraction states the raw_text door's fields and the caller the parsed door's."""
    with _built_frames() as frames:
        doors.raw_text(query, stat_var, *extracted)
        doors.first_answer(stat_var, *posted)
    raw, parsed = frames

    assert _without_stated_by(raw.model_dump(mode="json")) == _without_stated_by(
        parsed.model_dump(mode="json")
    )
    assert all(place.dcids for place in raw.places)
    assert [frame.variable.stated_by for frame in frames] == ["extraction", "caller"]
    assert [{p.stated_by for p in frame.places} for frame in frames] == [
        {"extraction"},
        {"caller"},
    ]


def test_the_frame_event_follows_the_interpretation_and_precedes_the_shape():
    """The interpretation fires once for the request at entry, and each part's completed
    frame fires after its recall, before its shape."""
    events = _events(
        ResolveRequest(
            input=RawTextInput(query="aid flows to Kenya"), response_options=ResolveOptions()
        )
    )
    kinds = [
        event.kind
        for event in events
        if isinstance(event, InterpretationEvent | FrameEvent | ShapeEvent)
    ]
    assert kinds == ["interpretation", "frame", "shape"]
    (frame_event,) = [event for event in events if isinstance(event, FrameEvent)]
    sent = json.loads(serialize_sse(frame_event).split("data: ", 1)[1])
    assert sent["variable_index"] == 0
    assert sent["frame"]["variable"] == {"value": _VARIABLE, "stated_by": "extraction"}
    assert sent["frame"]["normalisation"]["form"] == {"value": ["level"], "stated_by": "default"}


def _completed(variable: str, *, date_request: DateRequest | None = None) -> Frame:
    frame = door_frame(
        door="parsed",
        variable=variable,
        entities=[],
        entity_roles={},
        member_entities=frozenset(),
        pins={},
        date_request=date_request,
        unit_text=None,
        scope=None,
        operations=(),
        open_axes=frozenset(),
    )
    return with_recall(
        frame,
        dropped_phrases=(),
        matches={},
        ties=(),
        readings=(),
        requirement=quantity_requirement([variable]),
        measure_class=None,
        place_sets={},
        role_bearing=False,
        slot_defaults=(),
    )


@pytest.mark.parametrize(
    ("variable", "date_request", "stat_type", "form", "window"),
    [
        ("GDP", None, ("measuredValue", "default"), (("level",), "default"), ("all", "default")),
        (
            "GDP growth",
            DateRequest(window=None, latest=True),
            ("growthRate", "reader"),
            (("level",), "default"),
            ("latest", "caller"),
        ),
        (
            "GDP per capita",
            DateRequest(window=TimeWindow(start_year=2010, end_year=2020), latest=False),
            ("measuredValue", "default"),
            (("per_person",), "reader"),
            (TimeWindow(start_year=2010, end_year=2020), "caller"),
        ),
    ],
    ids=["all-defaults", "growth-latest", "per-capita-window"],
)
def test_a_field_the_question_leaves_unstated_carries_its_default(
    variable, date_request, stat_type, form, window
):
    """The statType is measuredValue unless a measure word states another, the form a
    level unless the words name one, and the window every year unless one is named."""
    frame = _completed(variable, date_request=date_request)
    assert frame.measure is not None
    assert frame.normalisation is not None
    assert (frame.measure.stat_type.value, frame.measure.stat_type.stated_by) == stat_type
    assert (frame.normalisation.form.value, frame.normalisation.form.stated_by) == form
    assert (frame.window.value, frame.window.stated_by) == window
    assert frame.window_defaulted == (window[1] == "default")
    assert (frame.price_basis.value, frame.price_basis.stated_by) == (None, "default")


def test_the_stat_type_words_are_the_code_index_s_declaration(monkeypatch):
    """The frame reads the words that state a statType from the code index
    (`CodeIndex.stat_type_words`): an index declaring none leaves "GDP growth" at the
    default."""
    monkeypatch.setattr(get_index(), "stat_type_words", ())
    frame = _completed("GDP growth", date_request=None)
    assert frame.measure is not None
    assert (frame.measure.stat_type.value, frame.measure.stat_type.stated_by) == (
        "measuredValue",
        "default",
    )
