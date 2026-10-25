"""End-to-end tests for operations on the raw-text and parsed doors, and for the
options and slot token an entity_ambiguous part carries.

An operation rides on the part of the variable extract assigned it, and its phrase
reaches that variable's bind the way a dropped qualifier's does, with no
query_qualifier_dropped warning and no sentence note. The fixture is the fixture CRS
index with the query "standard grants", which leaves crsTypeOfFlow open for the binder.
"""

from __future__ import annotations

import asyncio

import pytest
from dfo_spec_kit.registry import load_registry

from search.engine.bind import SlotBindingDraft
from search.engine.core import resolve_events
from search.engine.events import ResponseEvent, SentenceEvent
from search.engine.extract import ExtractedDate, ExtractedEntity, ExtractedOperation, Extraction
from search.engine.quantity_read import QuantityOutput
from search.engine.region.codes import QUERY_QUALIFIER_DROPPED, SLOT_DEFAULTED
from search.models import (
    DatasetResubmitInput,
    DefinitePart,
    NoDataPart,
    Operation,
    ParsedEntity,
    ParsedInput,
    ParsedPart,
    RawTextInput,
    ResolveOptions,
    ResolveRequest,
    ResubmitIdentityEntry,
    ResubmitParsedEntry,
)
from tests.engine._binders import elected_reply
from tests.engine._fake_sdmx import FakeSdmxClient
from tests.engine._harness import make_request, offline_resolve
from tests.engine._quantity import NO_QUANTITY
from tests.engine.test_bind import (
    _RESOLVE_DETECT_QUERY,
    _RESOLVE_NODES,
    _RESOLVE_RESOLVE,
    _RESOLVE_SV,
)
from tests.engine.test_declared_default_e2e import _filter_for
from tests.engine.test_qualifier_suppression_and_provenance import (
    _E2E_KEN_DCID,
    _E2E_NODES,
    _E2E_OBS,
    _E2E_SV_DCID,
    _e2e_graph_for,
)
from tests.fixtures import FakeGraph

_QUERY = "standard grants"
_EMPTY_SDMX = FakeSdmxClient(observations={}, availability={})


class _ScriptedLLM:
    """Return a scripted extraction, record every bind prompt by variable, and claim
    `requested_all_phrase` on crsTypeOfFlow when one is given."""

    def __init__(self, extraction: Extraction, *, requested_all_phrase: str | None = None):
        self._extraction = extraction
        self._phrase = requested_all_phrase
        self.bind_prompts: dict[str, str] = {}

    async def generate_structured(self, *, prompt, system, schema):
        if schema.__name__ == "Extraction":
            return self._extraction, None
        if schema is QuantityOutput:
            return NO_QUANTITY
        assert schema.__name__ == "_UniversalBindOutput"
        variable = next(
            line.removeprefix("Variable phrase: ")
            for line in prompt.splitlines()
            if line.startswith("Variable phrase: ")
        )
        self.bind_prompts[variable] = prompt
        bindings = []
        if self._phrase is not None and variable == _QUERY:
            bindings.append(
                SlotBindingDraft(
                    axis="how",
                    property_dcid="crsTypeOfFlow",
                    kind="unbound",
                    value_dcids=[],
                    evidence="requested_all",
                    requested_all_phrase=self._phrase,
                )
            )
        return elected_reply(0, bindings=bindings, also_plausible=[]), None


def _extraction(**fields) -> Extraction:
    return Extraction(
        kind="resolution",
        **{"variables": [_QUERY], "entities": [ExtractedEntity(text="Kenya")], **fields},
    )


def _query_qualifier_warnings(response) -> list[str]:
    return [
        w.message
        for w in [*response.diagnostics.warnings, *(w for p in response.parts for w in p.warnings)]
        if w.code == QUERY_QUALIFIER_DROPPED
    ]


def _measures_not_applied(part) -> list[str | None]:
    """What each `measure_not_applied` departure on `part` says the request asked."""
    return [d.asked for d in part.departures if d.code == "measure_not_applied"]


_ODA_DEFAULT_NOTE = (
    "No flow was named, so the answer covers official development assistance, the default flow."
)


def _slot_defaulted_warnings(response) -> list[str]:
    return [
        w.message
        for w in [*response.diagnostics.warnings, *(w for p in response.parts for w in p.warnings)]
        if w.code == SLOT_DEFAULTED
    ]


def test_a_rank_operation_rides_on_its_part_with_the_donor_axis_open_and_no_warning():
    """'top 5 donors for standard grants' carries rank over origin, binds no donor, and
    neither warns nor notes that the phrase went unapplied."""
    llm = _ScriptedLLM(
        _extraction(
            operations=[
                ExtractedOperation(verb="rank", variable=0, phrase="top 5 donors", over="origin")
            ]
        )
    )
    response = offline_resolve(
        make_request("top 5 donors for standard grants in Kenya"),
        graph=_e2e_graph_for(_QUERY),
        llm=llm,
        sdmx=_EMPTY_SDMX,
    )

    rank = Operation(verb="rank", phrase="top 5 donors", over="origin")
    part = response.parts[0]
    assert part.status == "definite"
    assert part.operations == [rank]
    assert part.restatement is not None
    assert part.restatement.operations == [rank]
    assert not any(e.role == "origin" for e in part.spec.resolved.entities)
    assert _query_qualifier_warnings(response) == []
    assert "was not applied" not in (part.spec.resolved.rendered_sentence or "")
    assert "was not applied" not in (response.rendered_sentence or "")
    assert "  top 5 donors" in llm.bind_prompts[_QUERY]


def test_the_recorded_top_5_donors_reading_carries_its_rank_with_no_warning():
    """The recorded model output for the design's headline query reads "Top 5" as a rank
    over origin on "education aid"; the rank and the variable account for every word of
    it, so the answer carries the rank and no unapplied-qualifier warning."""
    response = offline_resolve(make_request("Top 5 donors for education aid"))

    assert [p.operations for p in response.parts] == [
        [Operation(verb="rank", phrase="Top 5", over="origin")]
    ]
    assert _query_qualifier_warnings(response) == []
    assert "was not applied" not in (response.rendered_sentence or "")


class _OpenFlowAndSectorBindLLM(_ScriptedLLM):
    """Bind leaves crsTypeOfFlow and crsPurpose open and claims no phrase for either."""

    async def generate_structured(self, *, prompt, system, schema):
        result, usage = await super().generate_structured(
            prompt=prompt, system=system, schema=schema
        )
        if schema.__name__ == "_UniversalBindOutput":
            result.bindings = [
                SlotBindingDraft(axis="how", property_dcid=prop, kind="unbound")
                for prop in ("crsTypeOfFlow", "crsPurpose")
            ]
        return result, usage


@pytest.mark.parametrize("phrase", ["Which sectors got the most", "Which got the most"])
def test_a_rank_over_sectors_of_aid_keeps_the_oda_default_and_the_sector_slot_open(phrase):
    """Both recorded readings of "Which sectors got the most aid in Kenya in 2022?": the
    phrase names the sector slot or no slot, never the flow slot, so "aid" still
    defaults the flow to ODA and the sector slot stays open for the rank."""
    response = offline_resolve(
        make_request("Which sectors got the most aid in Kenya in 2022?"),
        graph=_e2e_graph_for("aid"),
        llm=_OpenFlowAndSectorBindLLM(
            Extraction(
                kind="resolution",
                variables=["aid"],
                entities=[ExtractedEntity(text="Kenya")],
                operations=[
                    ExtractedOperation(verb="rank", variable=0, phrase=phrase, over="category")
                ],
            )
        ),
        sdmx=_EMPTY_SDMX,
    )

    part = response.parts[0]
    assert part.status == "definite"
    assert part.operations == [Operation(verb="rank", phrase=phrase, over="category")]
    flow_filter = _filter_for(part.spec, "crsTypeOfFlow")
    assert flow_filter.binding_kind == "value"
    assert [ref.ref for ref in flow_filter.refs] == ["CRSTypeOfFlow10"]
    assert flow_filter.provenance == "declared_default"
    sector_filter = _filter_for(part.spec, "crsPurpose")
    assert sector_filter.binding_kind == "unbound"
    assert sector_filter.refs == []
    assert _slot_defaulted_warnings(response) == []


def _graph_through_2022(variable: str) -> FakeGraph:
    """The fixture graph with the flow's points running to 2022, so a query naming 2022
    reads one there (`read_plan.has_points`)."""
    return FakeGraph(
        nodes=_E2E_NODES,
        detect={variable: {"svs": [_E2E_SV_DCID], "cosine_scores": [0.9]}},
        resolve={"Kenya": _E2E_KEN_DCID},
        obs={
            f"{_E2E_SV_DCID}|{_E2E_KEN_DCID}": [
                {"earliestDate": "2020", "latestDate": "2022", "obsCount": 3}
            ]
        },
    )


@pytest.mark.parametrize(
    ("query", "variable", "phrase", "dropped", "qualifier_warnings", "departs"),
    [
        # "most" is operation wording: the phrase keeps its warning and departs not at all.
        (
            "Which flows got the most aid in Kenya in 2022?",
            "aid",
            "Which got the most",
            "Which flows got the most aid in Kenya in 2022",
            ['The requested qualifier "flows got the most aid" was not applied.'],
            [],
        ),
        ("top 5 flows of aid to Kenya", "aid", "top 5", "top 5 flows", [], []),
        (
            "top 5 sectors by aid flows in Kenya",
            "aid",
            "top 5 sectors by aid flows",
            "top 5 sectors by aid flows",
            ['The requested qualifier "sectors by aid flows" was not applied.'],
            ["sectors by aid flows"],
        ),
        ("top 5 aid flows to Kenya", "aid flows", "top 5", "top 5 aid flows", [], []),
        ("top 5 flows of aid to Kenya", "flows of aid", "top 5", "top 5 flows of aid", [], []),
        (
            "Which aid flows to Kenya were largest in 2022?",
            "aid flows",
            "Which were largest",
            "Which aid flows to Kenya were largest in 2022",
            [],
            [],
        ),
    ],
)
def test_a_rank_whose_phrase_opens_no_flow_slot_states_the_oda_default(
    query, variable, phrase, dropped, qualifier_warnings, departs
):
    """A rank over flows whose phrase names no slot, or two, opens nothing, and bind
    claims nothing, so "aid" defaults the flow to ODA. The query names the flow slot, so
    the part warns slot_defaulted, and the sentence does not repeat it. A qualifier that asked for
    no more than the flow ("flows") loses its warning to that default; one that asked for
    more keeps it, and departs unless it is operation wording ("flows got the most aid").
    When the trim's variable exit takes the qualifier ("top 5 aid flows" beside "aid
    flows"), the default still warns."""
    response = offline_resolve(
        make_request(query),
        graph=_graph_through_2022(variable),
        llm=_OpenFlowAndSectorBindLLM(
            Extraction(
                kind="resolution",
                variables=[variable],
                entities=[ExtractedEntity(text="Kenya")],
                operations=[
                    ExtractedOperation(verb="rank", variable=0, phrase=phrase, over="category")
                ],
                dropped_qualifiers=[dropped],
                dates=[ExtractedDate(kind="point", start="2022")] if "2022" in query else [],
            )
        ),
        sdmx=_EMPTY_SDMX,
    )

    part = response.parts[0]
    flow_filter = _filter_for(part.spec, "crsTypeOfFlow")
    assert [ref.ref for ref in flow_filter.refs] == ["CRSTypeOfFlow10"]
    assert flow_filter.provenance == "declared_default"
    assert _filter_for(part.spec, "crsPurpose").binding_kind == "unbound"
    assert _slot_defaulted_warnings(response) == [_ODA_DEFAULT_NOTE]
    assert _query_qualifier_warnings(response) == qualifier_warnings
    sentence = part.spec.resolved.rendered_sentence or ""
    assert _ODA_DEFAULT_NOTE not in sentence
    assert "was not applied" not in sentence
    assert _measures_not_applied(part) == departs


_NORWAY_DCID = "country/NOR"


@pytest.mark.parametrize(
    ("query", "extraction", "slot_warnings"),
    [
        (
            "aid flows between Norway and Kenya",
            Extraction(
                kind="resolution",
                variables=["aid flows"],
                entities=[ExtractedEntity(text="Norway"), ExtractedEntity(text="Kenya")],
            ),
            [_ODA_DEFAULT_NOTE],
        ),
        (
            "aid flows to Kenya",
            Extraction(
                kind="resolution",
                variables=["aid"],
                entities=[ExtractedEntity(text="Kenya")],
                dropped_qualifiers=["flows"],
            ),
            [_ODA_DEFAULT_NOTE],
        ),
        (
            "aid flows to Kenya",
            Extraction(
                kind="resolution", variables=["aid"], entities=[ExtractedEntity(text="Kenya")]
            ),
            [_ODA_DEFAULT_NOTE],
        ),
        (
            "aid to Kenya",
            Extraction(
                kind="resolution", variables=["aid"], entities=[ExtractedEntity(text="Kenya")]
            ),
            [],
        ),
    ],
    ids=["recorded-aid-flows", "flows-dropped", "flows-in-query-only", "aid-to-kenya"],
)
def test_a_flow_default_warns_only_where_the_query_names_flows(query, extraction, slot_warnings):
    """With no operation and bind claiming nothing, "aid" defaults the flow to ODA. The
    part warns slot_defaulted when the variable, a dropped qualifier or the query names
    the flow: the extraction recorded for "aid flows between Norway and Kenya" carries
    "aid flows" as its variable. A dropped "flows" warns through slot_defaulted alone,
    with no unapplied-qualifier note. "aid to Kenya" names no flow and does not warn."""
    graph = FakeGraph(
        nodes={**_E2E_NODES, _NORWAY_DCID: {"label": "Norway", "type": "Country"}},
        detect={extraction.variables[0]: {"svs": [_E2E_SV_DCID], "cosine_scores": [0.9]}},
        resolve={"Kenya": _E2E_KEN_DCID, "Norway": _NORWAY_DCID},
        obs=_E2E_OBS,
    )
    response = offline_resolve(
        make_request(query),
        graph=graph,
        llm=_OpenFlowAndSectorBindLLM(extraction),
        sdmx=_EMPTY_SDMX,
    )

    part = response.parts[0]
    flow_filter = _filter_for(part.spec, "crsTypeOfFlow")
    assert [ref.ref for ref in flow_filter.refs] == ["CRSTypeOfFlow10"]
    assert flow_filter.provenance == "declared_default"
    assert _slot_defaulted_warnings(response) == slot_warnings
    assert _query_qualifier_warnings(response) == []
    sentence = part.spec.resolved.rendered_sentence or ""
    assert _ODA_DEFAULT_NOTE not in sentence
    assert "was not applied" not in sentence


# The fixture's measure noun is "Test grant equivalent flow", so "by flow" and "per flow"
# read as stated by the heading and write no departure; "across flows" does not.
@pytest.mark.parametrize(
    ("dropped", "departs"),
    [("by flow", []), ("per flow", []), ("across flows", ["across flows"])],
)
def test_a_flow_breakdown_keeps_its_warning_beside_the_default(dropped, departs):
    """A dropped breakdown phrase ("by flow") asked for every flow, and the answer gave
    ODA alone, so the part keeps the phrase's query_qualifier_dropped warning beside
    slot_defaulted, and its measure_not_applied departure unless the heading states
    every word of it."""
    response = offline_resolve(
        make_request(f"aid to Kenya {dropped}"),
        graph=_e2e_graph_for("aid"),
        llm=_OpenFlowAndSectorBindLLM(
            Extraction(
                kind="resolution",
                variables=["aid"],
                entities=[ExtractedEntity(text="Kenya")],
                dropped_qualifiers=[dropped],
            )
        ),
        sdmx=_EMPTY_SDMX,
    )

    part = response.parts[0]
    flow_filter = _filter_for(part.spec, "crsTypeOfFlow")
    assert [ref.ref for ref in flow_filter.refs] == ["CRSTypeOfFlow10"]
    assert flow_filter.provenance == "declared_default"
    note = f'The requested qualifier "{dropped}" was not applied.'
    assert _slot_defaulted_warnings(response) == [_ODA_DEFAULT_NOTE]
    assert _query_qualifier_warnings(response) == [note]
    assert note not in (part.spec.resolved.rendered_sentence or "")
    assert _measures_not_applied(part) == departs


def test_the_final_sentence_event_states_no_slot_default():
    """slot_defaulted is the election's warning: the part carries it, and neither the final
    sentence event nor the response's sentence, whose plans match, repeats it."""

    async def collect():
        return [
            event
            async for event in resolve_events(
                ResolveRequest(
                    input=RawTextInput(query="aid flows to Kenya"),
                    response_options=ResolveOptions(),
                ),
                graph=_e2e_graph_for("aid flows"),
                llm=_OpenFlowAndSectorBindLLM(
                    Extraction(
                        kind="resolution",
                        variables=["aid flows"],
                        entities=[ExtractedEntity(text="Kenya")],
                    )
                ),
                sdmx=_EMPTY_SDMX,
            )
        ]

    events = asyncio.run(collect())
    response = next(event.response for event in events if isinstance(event, ResponseEvent))
    part = response.parts[0]
    assert isinstance(part, DefinitePart)
    final_plan = [event for event in events if isinstance(event, SentenceEvent)][-1].sentence_plan
    assert _slot_defaulted_warnings(response) == [_ODA_DEFAULT_NOTE]
    assert f" {_ODA_DEFAULT_NOTE}" not in [token.text for token in final_plan]
    assert final_plan == part.spec.resolved.sentence_plan


def test_a_first_answer_warns_slot_defaulted_and_its_resubmit_does_not():
    """The first answer's election defaults the flow the query names and warns. The
    resubmit posts the Spec's identity and member, which carry the default, so it elects
    nothing and warns nothing."""
    graph = _e2e_graph_for("aid flows")
    first = offline_resolve(
        ResolveRequest(
            input=RawTextInput(query="aid flows to Kenya"),
            response_options=ResolveOptions(),
        ),
        graph=graph,
        llm=_OpenFlowAndSectorBindLLM(
            Extraction(
                kind="resolution",
                variables=["aid flows"],
                entities=[ExtractedEntity(text="Kenya")],
            )
        ),
        sdmx=_EMPTY_SDMX,
    )
    part = first.parts[0]
    assert isinstance(part, DefinitePart)
    assert _slot_defaulted_warnings(first) == [_ODA_DEFAULT_NOTE]

    spec = part.spec
    again = offline_resolve(
        ResolveRequest(
            input=DatasetResubmitInput(
                entries=[
                    ResubmitIdentityEntry(
                        variable_text="aid flows",
                        identity=spec.definition.identity,
                    )
                ]
            ),
            response_options=ResolveOptions(),
        ),
        graph=graph,
        llm=_OpenFlowAndSectorBindLLM(Extraction(kind="resolution", variables=[], entities=[])),
        sdmx=_EMPTY_SDMX,
    )
    assert isinstance(again.parts[0], DefinitePart)
    assert _slot_defaulted_warnings(again) == []


class _AskingBindLLM(_ScriptedLLM):
    """Bind asks instead of electing a shape."""

    async def generate_structured(self, *, prompt, system, schema):
        result, usage = await super().generate_structured(
            prompt=prompt, system=system, schema=schema
        )
        if schema.__name__ == "_UniversalBindOutput":
            result.ask = "no shape matches"
        return result, usage


@pytest.mark.parametrize(
    ("llm_type", "expected"),
    [
        (_ScriptedLLM, []),
        (_AskingBindLLM, ['The requested qualifier "standard grants" was not applied.']),
    ],
    ids=["elected", "bind-asks"],
)
def test_what_a_rank_leaves_of_a_qualifier_warns_only_when_the_answer_did_not_apply_it(
    llm_type, expected
):
    """The rank accounts for "top 5 donors" in "top 5 donors of standard grants"; the
    region then reports "standard grants" unless the elected shape applied it."""
    variable = "ODA standard grants"
    llm = llm_type(
        Extraction(
            kind="resolution",
            variables=[variable],
            entities=[ExtractedEntity(text="Kenya")],
            operations=[
                ExtractedOperation(verb="rank", variable=0, phrase="top 5 donors", over="origin")
            ],
            dropped_qualifiers=["top 5 donors of standard grants"],
        )
    )
    response = offline_resolve(
        make_request("top 5 donors of ODA standard grants in Kenya"),
        graph=_e2e_graph_for(variable),
        llm=llm,
        sdmx=_EMPTY_SDMX,
    )

    assert _query_qualifier_warnings(response) == expected


def test_a_requested_all_claim_on_an_operation_phrase_opens_its_slot():
    """Bind validates an operation phrase as it validates a dropped one, so a
    requested_all claim on it suppresses the declared crsTypeOfFlow default."""
    phrase = "top 5 flow types"
    response = offline_resolve(
        make_request(f"{phrase} for standard grants in Kenya"),
        graph=_e2e_graph_for(_QUERY),
        llm=_ScriptedLLM(
            _extraction(
                operations=[
                    ExtractedOperation(verb="rank", variable=0, phrase=phrase, over="category")
                ]
            ),
            requested_all_phrase=phrase,
        ),
        sdmx=_EMPTY_SDMX,
    )

    part = response.parts[0]
    assert part.status == "definite"
    flow_filter = _filter_for(part.spec, "crsTypeOfFlow")
    assert flow_filter.binding_kind == "unbound"
    assert flow_filter.provenance != "declared_default"
    assert part.operations == [Operation(verb="rank", phrase=phrase, over="category")]
    assert _query_qualifier_warnings(response) == []


class _UnclaimedOpenBindLLM(_ScriptedLLM):
    """Bind leaves crsTypeOfFlow open and claims no phrase for it."""

    async def generate_structured(self, *, prompt, system, schema):
        result, usage = await super().generate_structured(
            prompt=prompt, system=system, schema=schema
        )
        if schema.__name__ == "_UniversalBindOutput":
            result.bindings = [
                SlotBindingDraft(axis="how", property_dcid="crsTypeOfFlow", kind="unbound")
            ]
        return result, usage


def _rank_over_category(phrase: str, llm_type: type[_ScriptedLLM]):
    return offline_resolve(
        make_request(f"{phrase} for standard grants in Kenya"),
        graph=_e2e_graph_for(_QUERY),
        llm=llm_type(
            _extraction(
                operations=[
                    ExtractedOperation(verb="rank", variable=0, phrase=phrase, over="category")
                ]
            )
        ),
        sdmx=_EMPTY_SDMX,
    )


def test_an_operation_over_the_category_its_phrase_names_opens_that_slot_with_no_bind_claim():
    """'top 5 flow types' names the flow slot, so the slot stays open for the rank
    though bind leaves it open without a requested_all claim."""
    response = _rank_over_category("top 5 flow types", _UnclaimedOpenBindLLM)
    part = response.parts[0]

    assert part.status == "definite"
    flow_filter = _filter_for(part.spec, "crsTypeOfFlow")
    assert flow_filter.binding_kind == "unbound"
    assert flow_filter.provenance != "declared_default"
    assert _slot_defaulted_warnings(response) == []


def test_an_operation_over_another_category_leaves_the_flow_default_in_place():
    """'top 5 sectors' names the sector slot, so standard grants still default the
    flow slot to ODA. The reply leaves the cooperation modality out, so it reads open and
    the family serves the member stating none, which the fixture holds no data for at
    Kenya: the part flags that read."""
    part = _rank_over_category("top 5 sectors", _UnclaimedOpenBindLLM).parts[0]

    assert isinstance(part, NoDataPart)
    assert part.no_data.flagged_match is not None
    flow_filter = _filter_for(part.no_data.flagged_match, "crsTypeOfFlow")
    assert flow_filter.binding_kind == "value"
    assert flow_filter.provenance == "declared_default"


def test_an_operation_phrase_sends_bind_the_prompt_a_dropped_qualifier_did():
    """A phrase extract records as an operation instead of a dropped qualifier renders
    into the same bind prompt but for the frame, which states the operation and the axis it
    opens."""
    as_dropped = _ScriptedLLM(_extraction(dropped_qualifiers=["top 5 donors"]))
    as_operation = _ScriptedLLM(
        _extraction(
            operations=[
                ExtractedOperation(verb="rank", variable=0, phrase="top 5 donors", over="origin")
            ]
        )
    )
    for llm in (as_dropped, as_operation):
        offline_resolve(
            make_request("top 5 donors for standard grants in Kenya"),
            graph=_e2e_graph_for(_QUERY),
            llm=llm,
            sdmx=_EMPTY_SDMX,
        )

    operation_prompt = as_operation.bind_prompts[_QUERY]
    assert "DROPPED QUALIFIERS" in operation_prompt
    assert "  operations: rank over origin [extraction]\n" in operation_prompt
    assert "  open roles: origin [extraction]\n" in operation_prompt
    as_frame_without = operation_prompt.replace(
        "  operations: rank over origin [extraction]\n", "  operations: none [extraction]\n"
    ).replace("  open roles: origin [extraction]\n", "  open roles: none [extraction]\n")
    assert as_frame_without == as_dropped.bind_prompts[_QUERY]


def test_only_the_variable_an_operation_names_carries_it_or_sends_it_to_bind():
    """In a two-variable query the other leg's part and bind prompt stay untouched."""
    llm = _ScriptedLLM(
        _extraction(
            variables=[_QUERY, _RESOLVE_DETECT_QUERY],
            operations=[
                ExtractedOperation(verb="rank", variable=0, phrase="top 5 donors", over="origin")
            ],
        )
    )
    graph = FakeGraph(
        nodes={**_E2E_NODES, **_RESOLVE_NODES},
        detect={
            _QUERY: {"svs": [_E2E_SV_DCID], "cosine_scores": [0.9]},
            _RESOLVE_DETECT_QUERY: {"svs": [_RESOLVE_SV], "entities": []},
        },
        resolve=_RESOLVE_RESOLVE,
        obs=_E2E_OBS,
    )
    response = offline_resolve(
        make_request("top 5 donors for standard grants and population in Kenya"),
        graph=graph,
        llm=llm,
        sdmx=_EMPTY_SDMX,
    )

    by_variable = {part.variable_text: part for part in response.parts}
    rank = Operation(verb="rank", phrase="top 5 donors", over="origin")
    assert by_variable[_QUERY].operations == [rank]
    assert by_variable[_RESOLVE_DETECT_QUERY].operations == []
    assert "top 5 donors" in llm.bind_prompts[_QUERY]
    assert "DROPPED QUALIFIERS" not in llm.bind_prompts[_RESOLVE_DETECT_QUERY]
    assert _query_qualifier_warnings(response) == []


def test_a_forecast_resolves_over_the_years_the_data_holds():
    """'standard grants in Kenya in 2030' forecasts: the target year leaves the dates,
    the part carries the operation, and nothing warns."""
    response = offline_resolve(
        make_request("standard grants in Kenya in 2030"),
        graph=_e2e_graph_for(_QUERY),
        llm=_ScriptedLLM(
            _extraction(
                operations=[ExtractedOperation(verb="forecast", variable=0, phrase="in 2030")],
                dates=[ExtractedDate(kind="point", start="2030")],
            )
        ),
        sdmx=_EMPTY_SDMX,
    )

    part = response.parts[0]
    assert part.status == "definite"
    assert part.operations == [Operation(verb="forecast", phrase="in 2030", target_year=2030)]
    assert part.restatement is not None
    assert part.restatement.time_text is None
    assert part.restatement.operations == part.operations
    assert _query_qualifier_warnings(response) == []


def test_a_forecast_on_one_variable_leaves_the_other_variable_its_year():
    """'forecast standard grants and population in Kenya in 2050' forecasts grants only,
    so the grants part resolves over the data's years and the population part keeps 2050."""
    llm = _ScriptedLLM(
        _extraction(
            variables=[_QUERY, _RESOLVE_DETECT_QUERY],
            operations=[ExtractedOperation(verb="forecast", variable=0, phrase="forecast")],
            dates=[ExtractedDate(kind="point", start="2050")],
        )
    )
    graph = FakeGraph(
        nodes={**_E2E_NODES, **_RESOLVE_NODES},
        detect={
            _QUERY: {"svs": [_E2E_SV_DCID], "cosine_scores": [0.9]},
            _RESOLVE_DETECT_QUERY: {"svs": [_RESOLVE_SV], "entities": []},
        },
        resolve=_RESOLVE_RESOLVE,
        obs=_E2E_OBS,
    )
    response = offline_resolve(
        make_request("forecast standard grants and population in Kenya in 2050"),
        graph=graph,
        llm=llm,
        sdmx=_EMPTY_SDMX,
    )

    by_variable = {part.variable_text: part for part in response.parts}
    forecast, population = by_variable[_QUERY], by_variable[_RESOLVE_DETECT_QUERY]
    assert forecast.operations == [Operation(verb="forecast", phrase="forecast", target_year=2050)]
    assert forecast.restatement is not None
    assert forecast.restatement.time_text is None
    assert population.restatement is not None
    assert population.restatement.time_text == "2050"


@pytest.mark.parametrize("door", ["parsed", "dataset_resubmit"])
def test_a_reposted_forecast_part_keeps_its_target_year(door):
    """'forecast standard grants in Kenya to 2050' with the phrase "forecast": 2050 leaves
    the part's time_text and rides on the operation, so the part posted back on its own
    still forecasts to 2050."""
    response = offline_resolve(
        make_request("forecast standard grants in Kenya to 2050"),
        graph=_e2e_graph_for(_QUERY),
        llm=_ScriptedLLM(
            _extraction(
                operations=[ExtractedOperation(verb="forecast", variable=0, phrase="forecast")],
                dates=[ExtractedDate(kind="point", start="2050")],
            )
        ),
        sdmx=_EMPTY_SDMX,
    )
    forecast = Operation(verb="forecast", phrase="forecast", target_year=2050)
    restatement = response.parts[0].restatement
    assert restatement is not None
    assert restatement.time_text is None
    assert restatement.operations == [forecast]

    reposted = offline_resolve(
        _posted(door, restatement),
        graph=_e2e_graph_for(_QUERY),
        llm=_ScriptedLLM(_extraction()),
        sdmx=_EMPTY_SDMX,
    )

    part = reposted.parts[0]
    assert part.status == "definite"
    assert part.operations == [forecast]
    assert part.restatement is not None
    assert part.restatement.operations == [forecast]


def test_an_invalid_role_span_leaves_its_qualifier_unapplied_with_a_departure():
    """A role span validation rejects is cleared before the trim reads it, so "fragile
    states" stays an unapplied qualifier and the answer says so."""
    query = "top 5 donors for standard grants to Kenya in fragile states"
    response = offline_resolve(
        make_request(query),
        graph=_e2e_graph_for(_QUERY),
        llm=_ScriptedLLM(
            _extraction(
                entities=[
                    ExtractedEntity(
                        text="Kenya",
                        span="Kenya in fragile states",
                        flow_role="destination",
                        role_span="fragile states",
                    )
                ],
                operations=[
                    ExtractedOperation(
                        verb="rank", variable=0, phrase="top 5 donors", over="origin"
                    )
                ],
                dropped_qualifiers=["top 5 donors in fragile states"],
            )
        ),
        sdmx=_EMPTY_SDMX,
    )

    part = response.parts[0]
    assert part.status == "definite"
    assert _query_qualifier_warnings(response) == [
        'The requested qualifier "fragile states" was not applied.'
    ]
    assert "fragile states" not in (part.spec.resolved.rendered_sentence or "")
    assert _measures_not_applied(part) == ["fragile states"]


# ---------------------------------------------------------------------------
# entity_ambiguous: options and the ambiguous wording's slot token
# ---------------------------------------------------------------------------

_CONGO_NODES = {
    **_E2E_NODES,
    "country/COD": {"label": "Democratic Republic of the Congo", "type": "Country"},
    "country/COG": {"label": "Republic of the Congo", "type": "Country"},
}


def _congo_graph() -> FakeGraph:
    return FakeGraph(
        nodes=_CONGO_NODES,
        detect={_QUERY: {"svs": [_E2E_SV_DCID], "cosine_scores": [0.9]}},
        resolve={"Kenya": _E2E_KEN_DCID},
        obs={
            **_E2E_OBS,
            f"{_E2E_SV_DCID}|country/COD": [
                {"earliestDate": "2020", "latestDate": "2020", "obsCount": 3}
            ],
        },
    )


def _ambiguous_congo_llm() -> _ScriptedLLM:
    return _ScriptedLLM(
        _extraction(
            entities=[
                ExtractedEntity(text="Kenya"),
                ExtractedEntity(text="Congo", flow_role="destination", role_span="to"),
            ],
            operations=[
                ExtractedOperation(verb="rank", variable=0, phrase="top 5 donors", over="origin")
            ],
        )
    )


def _ambiguous_request() -> ResolveRequest:
    return ResolveRequest(
        input={"kind": "raw_text", "query": "top 5 donors for standard grants to Congo, Kenya"},
        response_options=ResolveOptions(),
    )


def test_an_ambiguous_part_carries_an_option_per_candidate_that_resolves_when_posted():
    """Each option pins Kenya, pins Congo to its candidate, and keeps the operation;
    posting the DRC option's part as a dataset_resubmit parsed entry resolves."""
    graph = _congo_graph()
    response = offline_resolve(
        _ambiguous_request(), graph=graph, llm=_ambiguous_congo_llm(), sdmx=_EMPTY_SDMX
    )

    part = response.parts[0]
    assert isinstance(part, NoDataPart)
    assert part.no_data.reason == "entity_ambiguous"
    candidates = part.no_data.ambiguous_entities[0].candidates
    options = part.no_data.options
    assert [(o.mentions, o.label) for o in options] == [(["Congo"], c.label) for c in candidates]
    rank = Operation(verb="rank", phrase="top 5 donors", over="origin")
    for option, candidate in zip(options, candidates, strict=True):
        (query_part,) = option.data_query.parts
        assert query_part.variable_text == _QUERY
        assert query_part.operations == [rank]
        assert [(e.text, e.role_text, e.dcid) for e in query_part.entities or ()] == [
            ("Kenya", None, _E2E_KEN_DCID),
            ("Congo", "to", candidate.ref),
        ]

    (picked,) = [o for o in options if o.data_query.parts[0].entities[1].dcid == "country/COD"]
    resubmitted = offline_resolve(
        ResolveRequest(
            input=DatasetResubmitInput(
                entries=[ResubmitParsedEntry(part=picked.data_query.parts[0])]
            )
        ),
        graph=graph,
        llm=_ambiguous_congo_llm(),
        sdmx=_EMPTY_SDMX,
    )

    (resolved,) = resubmitted.parts
    assert resolved.status == "definite"
    assert resolved.operations == [rank]
    assert {e.ref for e in resolved.spec.resolved.entities} >= {"country/COD", _E2E_KEN_DCID}


def test_a_group_named_beside_an_ambiguous_wording_is_echoed_unpinned():
    """Each option echoes a resolver group by its text alone: pinning it to one member's
    dcid would resubmit that member for the group."""
    g7 = ("CAN", "DEU", "FRA", "GBR", "ITA", "JPN", "USA")
    graph = FakeGraph(
        nodes={
            **_CONGO_NODES,
            **{f"country/{iso}": {"label": iso, "type": "Country"} for iso in g7},
        },
        detect={_QUERY: {"svs": [_E2E_SV_DCID], "cosine_scores": [0.9]}},
        resolve={},
        obs=_E2E_OBS,
    )
    llm = _ScriptedLLM(
        _extraction(
            entities=[
                ExtractedEntity(text="G7"),
                ExtractedEntity(text="Congo", flow_role="destination", role_span="to"),
            ]
        )
    )
    request = ResolveRequest(
        input={"kind": "raw_text", "query": "standard grants from the G7 to Congo"},
        response_options=ResolveOptions(),
    )

    part = offline_resolve(request, graph=graph, llm=llm, sdmx=_EMPTY_SDMX).parts[0]

    assert isinstance(part, NoDataPart)
    assert part.no_data.reason == "entity_ambiguous"
    options = part.no_data.options
    assert [o.data_query.parts[0].entities for o in options] == [
        [ParsedEntity(text="G7"), ParsedEntity(text="Congo", role_text="to", dcid=dcid)]
        for dcid in ("country/COD", "country/COG")
    ]


def test_an_ambiguous_wording_is_its_own_slot_token_in_the_sentence():
    """The envelope sentence marks "Congo" as a slot token with the part's index and the
    role the query gave it, and the text still reads as before."""
    response = offline_resolve(
        _ambiguous_request(), graph=_congo_graph(), llm=_ambiguous_congo_llm(), sdmx=_EMPTY_SDMX
    )

    part = response.parts[0]
    assert isinstance(part, NoDataPart)
    plan = response.sentence_plan or []
    (token,) = [t for t in plan if t.kind == "slot"]
    assert token.text == "Congo"
    assert token.part_index == 0
    assert token.role == "destination"
    assert token.entity_refs == []
    assert token.binding_ref is None
    assert response.rendered_sentence == part.rendered_sentence
    assert "".join(t.text for t in plan) == response.rendered_sentence
    assert response.rendered_sentence.startswith('"Congo" matches more than one entity')


def test_an_ambiguous_place_on_each_end_carries_an_option_per_end_and_candidate():
    """Congo named on both axes is a place on each end, so the part restates both and
    asks each end's choice. Each option pins one end to its candidate and leaves the
    other open, beside Kenya pinned, and keeps the operation."""
    llm = _ScriptedLLM(
        _extraction(
            entities=[
                ExtractedEntity(text="Kenya"),
                ExtractedEntity(text="Congo", flow_role="origin", role_span="from"),
                ExtractedEntity(text="Congo", flow_role="destination", role_span="to"),
            ],
            operations=[
                ExtractedOperation(verb="rank", variable=0, phrase="top 5 donors", over="origin")
            ],
        )
    )
    response = offline_resolve(
        ResolveRequest(
            input={
                "kind": "raw_text",
                "query": "top 5 donors for standard grants from Congo to Congo, Kenya",
            }
        ),
        graph=_congo_graph(),
        llm=llm,
        sdmx=_EMPTY_SDMX,
    )

    part = response.parts[0]
    assert isinstance(part, NoDataPart)
    assert part.no_data.reason == "entity_ambiguous"
    assert part.restatement is not None
    assert [(e.text, e.role_text) for e in part.restatement.entities or ()] == [
        ("Kenya", None),
        ("Congo", "from"),
        ("Congo", "to"),
    ]
    ends = [(dcid, None) for dcid in ("country/COD", "country/COG")] + [
        (None, dcid) for dcid in ("country/COD", "country/COG")
    ]
    assert len(part.no_data.options) == len(ends)
    for option, (origin, destination) in zip(part.no_data.options, ends, strict=True):
        (query_part,) = option.data_query.parts
        assert query_part.variable_text == _QUERY
        assert query_part.operations == [
            Operation(verb="rank", phrase="top 5 donors", over="origin")
        ]
        assert [(e.text, e.role_text, e.dcid) for e in query_part.entities or ()] == [
            ("Kenya", None, _E2E_KEN_DCID),
            ("Congo", "from", origin),
            ("Congo", "to", destination),
        ]


# ---------------------------------------------------------------------------
# Operations through the parsed doors' bind
# ---------------------------------------------------------------------------


def _raw_text_rank_bind_prompt() -> tuple[str, ParsedPart]:
    """Resolve the rank query on the raw-text door; return its bind prompt and the
    restatement a client would post back."""
    llm = _ScriptedLLM(
        _extraction(
            operations=[
                ExtractedOperation(verb="rank", variable=0, phrase="top 5 donors", over="origin")
            ]
        )
    )
    response = offline_resolve(
        make_request("top 5 donors for standard grants in Kenya"),
        graph=_e2e_graph_for(_QUERY),
        llm=llm,
        sdmx=_EMPTY_SDMX,
    )
    restatement = response.parts[0].restatement
    assert restatement is not None
    return llm.bind_prompts[_QUERY], restatement


def _posted(door: str, part: ParsedPart) -> ResolveRequest:
    if door == "parsed":
        return ResolveRequest(input=ParsedInput(parts=[part]))
    return ResolveRequest(input=DatasetResubmitInput(entries=[ResubmitParsedEntry(part=part)]))


@pytest.mark.parametrize("door", ["parsed", "dataset_resubmit"])
def test_a_posted_rank_operation_reaches_bind_as_the_raw_text_door_sends_it(door):
    """Posting the raw-text door's restatement back sends its operation phrase to bind
    beside the dropped phrases, so the donor axis stays open and the answer keeps the
    shape the raw-text door gave it."""
    raw_prompt, restatement = _raw_text_rank_bind_prompt()
    llm = _ScriptedLLM(_extraction())
    response = offline_resolve(
        _posted(door, restatement), graph=_e2e_graph_for(_QUERY), llm=llm, sdmx=_EMPTY_SDMX
    )

    part = response.parts[0]
    assert part.status == "definite"
    assert part.operations == restatement.operations
    assert not any(e.role == "origin" for e in part.spec.resolved.entities)
    assert _dropped_section(llm.bind_prompts[_QUERY]) == _dropped_section(raw_prompt)
    assert "  top 5 donors" in llm.bind_prompts[_QUERY]


@pytest.mark.parametrize("door", ["parsed", "dataset_resubmit"])
def test_a_posted_part_without_operations_sends_bind_no_dropped_phrases(door):
    """A part with no operations adds nothing to bind's prompt, so recorded bind
    responses for the parsed doors still replay."""
    llm = _ScriptedLLM(_extraction())
    part = ParsedPart(variable_text=_QUERY, entities=[ParsedEntity(text="Kenya")])
    offline_resolve(_posted(door, part), graph=_e2e_graph_for(_QUERY), llm=llm, sdmx=_EMPTY_SDMX)

    assert "DROPPED QUALIFIERS" not in llm.bind_prompts[_QUERY]


@pytest.mark.parametrize("door", ["parsed", "dataset_resubmit"])
def test_a_requested_all_claim_on_a_posted_operation_phrase_opens_its_slot(door):
    """A requested_all claim on a posted operation phrase suppresses the declared
    crsTypeOfFlow default, as it does on the raw-text door."""
    phrase = "top 5 flow types"
    part = ParsedPart(
        variable_text=_QUERY,
        entities=[ParsedEntity(text="Kenya")],
        operations=[Operation(verb="rank", phrase=phrase, over="category")],
    )
    response = offline_resolve(
        _posted(door, part),
        graph=_e2e_graph_for(_QUERY),
        llm=_ScriptedLLM(_extraction(), requested_all_phrase=phrase),
        sdmx=_EMPTY_SDMX,
    )

    resolved = response.parts[0]
    assert resolved.status == "definite"
    flow_filter = _filter_for(resolved.spec, "crsTypeOfFlow")
    assert flow_filter.binding_kind == "unbound"
    assert flow_filter.provenance != "declared_default"


@pytest.mark.parametrize("door", ["parsed", "dataset_resubmit"])
def test_a_posted_operation_over_a_named_category_opens_its_slot_with_no_bind_claim(door):
    """A posted rank over flow types keeps the flow slot open as the raw-text door does,
    with bind claiming nothing."""
    part = ParsedPart(
        variable_text=_QUERY,
        entities=[ParsedEntity(text="Kenya")],
        operations=[Operation(verb="rank", phrase="top 5 flow types", over="category")],
    )
    response = offline_resolve(
        _posted(door, part),
        graph=_e2e_graph_for(_QUERY),
        llm=_UnclaimedOpenBindLLM(_extraction()),
        sdmx=_EMPTY_SDMX,
    )

    flow_filter = _filter_for(response.parts[0].spec, "crsTypeOfFlow")
    assert flow_filter.binding_kind == "unbound"
    assert flow_filter.provenance != "declared_default"


@pytest.mark.parametrize("door", ["raw_text", "parsed", "dataset_resubmit"])
def test_every_door_the_registry_names_warns_slot_defaulted(door):
    """A part asking for "aid flows" to Kenya defaults the flow to ODA and warns
    slot_defaulted on each resolve door, and the registry row names that door."""
    part = ParsedPart(variable_text="aid flows", entities=[ParsedEntity(text="Kenya")])
    request = make_request("aid flows to Kenya") if door == "raw_text" else _posted(door, part)
    response = offline_resolve(
        request,
        graph=_e2e_graph_for("aid flows"),
        llm=_OpenFlowAndSectorBindLLM(
            Extraction(
                kind="resolution",
                variables=["aid flows"],
                entities=[ExtractedEntity(text="Kenya")],
            )
        ),
        sdmx=_EMPTY_SDMX,
    )

    flow_filter = _filter_for(response.parts[0].spec, "crsTypeOfFlow")
    assert flow_filter.provenance == "declared_default"
    assert _slot_defaulted_warnings(response) == [_ODA_DEFAULT_NOTE]
    doors = {
        door
        for mapping in load_registry()[SLOT_DEFAULTED].mappings
        if mapping.endpoint == "search.resolve"
        for door in mapping.doors or ()
    }
    assert door in doors


def _dropped_section(prompt: str) -> list[str]:
    """Return the bind prompt's dropped-qualifier block, header through its last item."""
    lines = prompt.splitlines()
    start = next(i for i, line in enumerate(lines) if "DROPPED QUALIFIERS" in line)
    end = next(
        (i for i in range(start + 1, len(lines)) if not lines[i].startswith("  ")), len(lines)
    )
    return lines[start:end]
