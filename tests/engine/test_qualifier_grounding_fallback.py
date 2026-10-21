"""A purpose the question names is pinned on every reading whose slots hold the purpose.

A reading none of whose members holds the named purpose is not admissible
(`admissibility.judge_pinned_parts`): with no other reading the part refuses
`qualifier_not_bound` before bind, naming the purpose and flagging nothing, and beside a
reading that holds it, that reading answers with the purpose bound. A measure sibling
answering for a data-less elected shape is judged by its own materialisation. The
`education` qualifier is the fixture code index's own.
"""

from __future__ import annotations

import pytest

from search.engine.bind import _UniversalBindOutput
from search.engine.events import Event, FrameEvent, ResponseEvent
from search.engine.extract import ExtractedEntity, Extraction
from search.engine.quantity_read import QuantityOutput
from search.models import DefinitePart, NoDataPart, Resolution, Spec
from tests.engine._fake_sdmx import FakeSdmxClient
from tests.engine._harness import make_request, offline_events
from tests.engine._quantity import NO_QUANTITY
from tests.fixtures import FakeGraph

_KEN = "country/KEN"
_VARIABLE = "education widgets"


@pytest.fixture(autouse=True)
def _log_coverage(search_settings):
    """The word "widgets" beside education is one no coverage reader covers, so the log mode
    keeps these grounding cases answering."""
    search_settings(qualifier_coverage="log")


_AMOUNT = "Amount_Widget"
_COUNT = "Count_Widget"
_HEALTH = "CRSPurposeCode12110"
_EDUCATION = "CRSPurposeCode11110"


def _arcs(sv: str, measured_property: str, purpose: str) -> dict:
    return {
        "typeOf": {"nodes": [{"dcid": "StatisticalVariable"}]},
        "name": {"nodes": [{"value": sv}]},
        "populationType": {"nodes": [{"dcid": "Widget"}]},
        "measuredProperty": {"nodes": [{"dcid": measured_property}]},
        "statType": {"nodes": [{"dcid": "measuredValue"}]},
        "constraintProperties": {"nodes": [{"dcid": "crsPurpose"}]},
        "crsPurpose": {"nodes": [{"dcid": purpose}]},
    }


class _LLM:
    """Extract the variable at Kenya and elect the first shape with nothing drafted, so the
    `education` qualifier alone binds the purpose. `also_plausible` offers the rest. With
    `nearest`, a bind over an empty menu asks and names the first NOT ELECTABLE reading."""

    def __init__(self, also_plausible: list[int] | None, *, nearest: bool = False) -> None:
        self._also_plausible = also_plausible
        self._nearest = nearest

    async def generate_structured(self, *, prompt, system, schema):
        if schema is QuantityOutput:
            return NO_QUANTITY
        if schema is Extraction:
            return Extraction(
                kind="resolution",
                variables=[_VARIABLE],
                entities=[
                    ExtractedEntity(
                        text="Kenya", span="Kenya", flow_role="unspecified", role_span=None
                    )
                ],
            ), None
        if self._nearest and "NOT ELECTABLE" in prompt:
            return _UniversalBindOutput(
                chosen_shape_index=0, ask="no reading holds the purpose", nearest_refused_index=0
            ), None
        return _UniversalBindOutput(
            chosen_shape_index=0, bindings=[], unplaced=[], also_plausible=self._also_plausible
        ), None


def _events(
    members: dict[str, tuple[str, str]],
    *,
    also_plausible=None,
    without_data=frozenset(),
    nearest: bool = False,
) -> list[Event]:
    """The events of resolving the variable over `members`, each a StatVar's measured
    property and purpose, recalled in order and each holding data at Kenya unless named in
    `without_data`."""
    graph = FakeGraph(
        nodes={
            _KEN: {"label": "Kenya", "type": "Country"},
            "crsPurpose": {"label": "purpose"},
            _HEALTH: {"label": "Health policy"},
            _EDUCATION: {"label": "Education policy"},
            **{
                sv: {"label": sv, "type": "StatisticalVariable", "arcs": _arcs(sv, *facts)}
                for sv, facts in members.items()
            },
        },
        detect={
            _VARIABLE: {
                "svs": list(members),
                "cosine_scores": [0.9 - 0.01 * rank for rank in range(len(members))],
            }
        },
        resolve={"Kenya": _KEN},
        obs={
            f"{sv}|{_KEN}": [
                {
                    "earliestDate": "2020",
                    "latestDate": "2021",
                    "obsCount": 2,
                    "provenanceId": "dc/base/WidgetFixture",
                }
            ]
            for sv in members
            if sv not in without_data
        },
    )
    return offline_events(
        make_request(f"{_VARIABLE} in Kenya"),
        graph=graph,
        llm=_LLM(also_plausible, nearest=nearest),
        sdmx=FakeSdmxClient(observations={}, availability={}),
    )


def _resolve(members: dict[str, tuple[str, str]], **kwargs) -> Resolution:
    """The response of resolving the variable over `members` (`_events`)."""
    (response,) = [e.response for e in _events(members, **kwargs) if isinstance(e, ResponseEvent)]
    assert isinstance(response, Resolution)
    return response


def _purpose(spec: Spec) -> tuple[list[tuple[str, str | None]], str, str]:
    """The trace's qualifier dispositions and slots, the purpose slot's binding kind and
    the identity's purpose binding state."""
    (slot,) = (s for s in spec.resolved.slots if s.key.property is not None)
    (identity_slot,) = spec.definition.identity.slots
    return (
        [(q.disposition, q.slot) for q in spec.resolved.trace.query_qualifiers],
        slot.binding.kind,
        identity_slot.binding.state,
    )


def test_a_reading_holding_no_member_of_the_named_purpose_refuses_before_bind():
    """The frame sent beside the refusal carries the code the refused reading's
    vocabulary matched in "education", the education sector."""
    events = _events({_AMOUNT: ("amount", _HEALTH)})
    (response,) = [e.response for e in events if isinstance(e, ResponseEvent)]
    (frame,) = [e.frame for e in events if isinstance(e, FrameEvent)]

    assert [(c.vocabulary, c.codes) for c in frame.measure.codes.value] == [
        (("crsPurpose",), ("CRSPurposeCode110",))
    ]
    assert isinstance(response, Resolution)
    (part,) = response.parts
    assert isinstance(part, NoDataPart)
    assert part.no_data.reason == "qualifier_not_bound"
    assert [(q.text, q.slot) for q in part.no_data.unbound_qualifiers] == [
        ("education", "crsPurpose")
    ]
    assert part.no_data.flagged_match is None
    assert part.departures == []
    assert "query_qualifier_dropped" not in [w.code for w in part.warnings]


def test_a_refused_reading_bind_names_as_nearest_is_flagged_and_refused_on_its_trace():
    """Bind declines the empty menu and names the amount, refused on the purpose it holds
    no member of, as nearest (`_declined_refusal`). The refusal names the terms the flagged
    amount's trace leaves unmet, the purpose on its slot alone under log coverage, and
    none of them departs beside it."""
    response = _resolve({_AMOUNT: ("amount", _HEALTH)}, nearest=True)

    (part,) = response.parts
    assert isinstance(part, NoDataPart)
    assert part.no_data.reason == "qualifier_not_bound"
    flagged = part.no_data.flagged_match
    assert flagged is not None
    refused = [(q.text, q.slot) for q in part.no_data.unbound_qualifiers]
    assert refused == [("education", "crsPurpose")]
    assert [(q.text, q.slot) for q in flagged.resolved.trace.query_qualifiers] == refused
    assert part.departures == []
    assert "query_qualifier_dropped" not in [w.code for w in part.warnings]


def test_a_definite_answer_whose_purpose_filter_held_keeps_the_bound_qualifier():
    response = _resolve({_AMOUNT: ("amount", _EDUCATION)})

    (part,) = response.parts
    assert isinstance(part, DefinitePart)
    assert _purpose(part.spec) == ([("bound", "crsPurpose")], "value", "value")


def test_the_reading_holding_the_named_purpose_answers_beside_one_that_holds_none():
    """The amount holds no education member, so it leaves the menu and the count, the one
    admissible reading, answers with the purpose bound."""
    response = _resolve({_AMOUNT: ("amount", _HEALTH), _COUNT: ("count", _EDUCATION)})

    (part,) = response.parts
    assert isinstance(part, DefinitePart)
    assert [sv.ref.ref for sv in part.spec.resolved.members] == [_COUNT]
    assert _purpose(part.spec) == ([("bound", "crsPurpose")], "value", "value")
    assert "query_qualifier_dropped" not in [w.code for w in part.warnings]


def test_a_measure_sibling_whose_purpose_filter_held_rides_on_the_refusal():
    """The elected amount shape holds no data. Its count sibling is no equivalent or
    price-basis sibling of it (status rule 1), so the part refuses, and the count, which
    bind nominated, rides on the refusal as its alternate match with the purpose filter
    held."""
    response = _resolve(
        {_AMOUNT: ("amount", _EDUCATION), _COUNT: ("count", _EDUCATION)},
        also_plausible=[1],
        without_data={_AMOUNT},
    )

    (part,) = response.parts
    assert isinstance(part, NoDataPart)
    assert part.no_data.gate == "only_alternate_has_data"
    alternate = part.no_data.alternate_match
    assert alternate is not None
    assert [sv.ref.ref for sv in alternate.resolved.members] == [_COUNT]
    assert _purpose(alternate)[1:] == ("value", "value")
