"""End-to-end tests for multi-entry `dataset_resubmit` requests and their shared envelope.
Requests use offline fixtures and fake model clients.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import re
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from search.engine.concurrency import create_owned_task
from search.engine.core import resolve_events
from search.engine.enrichment import _launch_trailing_tasks
from search.engine.entry import dataset_resubmit as _dataset_resubmit_mod
from search.engine.entry.dataset_entries import _build_candidates_region
from search.engine.errors import EngineInputError, GraphInfraError
from search.engine.events import StepEvent
from search.engine.region import pipeline as _pipeline_mod
from search.engine.region import relaunch as _relaunch_mod
from search.engine.region import resolve_spec_resubmit
from search.engine.region.facet_probe import facet_probe
from search.engine.region.observations import dispatch as _observations_dispatch_mod
from search.engine.region.result import RegionResult
from search.engine.spec_id import build_spec_identity
from search.models import (
    BaseIdentity,
    ChoiceOrigin,
    DatasetResubmitInput,
    EntityGroup,
    Operation,
    Option,
    OptionAvailability,
    ParsedEntity,
    ParsedInput,
    ParsedPart,
    ResolveOptions,
    ResolveRequest,
    ResubmitCandidate,
    ResubmitCandidatesEntry,
    ResubmitIdentityEntry,
    ResubmitIdentityPayload,
    ResubmitParsedEntry,
    ResubmitPlaceholderEntry,
    Slot,
    SlotBinding,
    TimeWindow,
    Unsupported,
)
from search.models import Entry as ScopeEntry
from search.render.plan import render_sentence
from search.scope import rules as scope_rules
from search.scope.tables import AxisTable
from tests.code_index.aid_identity import education_aid_identity
from tests.engine._axis_doubles import (
    TOTAL_POPULATION_INDIA_FAMILY,
    axis_confirming_reader,
)
from tests.engine._base_identity import (
    GDP_SHAPE,
    PERSON_COUNT_SHAPE,
    destination_entity,
    make_identity,
    origin_entity,
    subject_entity,
    where_slots_for,
)
from tests.engine._binders import BIRTH_RATE_CANDIDATES, KeyedBindReplyLLM
from tests.engine._builders import sv_arcs as _sv_arcs
from tests.engine._builders import value_slot as _value_slot
from tests.engine._fake_coverage import empty_coverage_snapshot
from tests.engine._fake_sdmx import FakeSdmxClient
from tests.engine._harness import (
    DEADLINE_TIMEOUT_S,
    frozen_monotonic_clock,
    make_request,
    offline_resolve,
    walk_options,
    with_member_arcs,
)
from tests.engine._refusal_detail import assert_names_plainly
from tests.engine._search_app import search_app
from tests.fixtures import FakeGraph, FakeLLM, _load_json, make_resolver

# Most entries here resolve several parts concurrently, so the parts' reads interleave and
# repeat in an order that varies between runs; the distinct reads are compared.
pytestmark = [pytest.mark.usefixtures("request_log"), pytest.mark.unordered_requests]

_PROBE = facet_probe(["sv"], {"observationAbout": ["country/KEN"]}, years=None)

_INDIA = "country/IND"
_USA = "country/USA"
_KEN = "country/KEN"

_ACTIVITY_SOURCE_SLOT = Slot.model_validate(
    {
        "key": {
            "axis": "how",
            "property": {"ref": "activitySource", "label": "activity source"},
            "label": "activity source",
        },
        "binding": {"kind": "unbound"},
    }
)


def _client(*, materialise_graph=None) -> TestClient:
    """A Search app over the shared fixture graph, whose resubmits walk the mint's member
    enumeration over that graph, through `materialise_graph` when given."""
    graph = FakeGraph()
    return TestClient(
        search_app(
            graph=graph,
            resolver=make_resolver(graph),
            llm=FakeLLM(),
            sdmx=FakeSdmxClient(),
            materialise_graph=with_member_arcs(materialise_graph, graph),
        ),
        raise_server_exceptions=True,
    )


def _parse_sse(body: str) -> list[tuple[str, dict]]:
    frames: list[tuple[str, dict]] = []
    event_type: str | None = None
    for line in body.splitlines():
        if line.startswith(":"):
            continue
        if line.startswith("event:"):
            event_type = line[len("event:") :].strip()
        elif line.startswith("data:") and event_type is not None:
            frames.append((event_type, json.loads(line[len("data:") :].strip())))
            event_type = None
    return frames


def _wire_request(entries: list, *, raw_query: str | None = None) -> dict:
    input_body: dict = {
        "kind": "dataset_resubmit",
        "entries": [e.model_dump(mode="json") for e in entries],
    }
    if raw_query is not None:
        input_body["raw_query"] = raw_query
    return {"input": input_body, "response_options": {}}


def _gdp_entry(entity_dcid: str, variable_text: str) -> ResubmitIdentityEntry:
    """Build a GDP identity entry; its StatVar declares `activitySource`."""
    entities = [subject_entity(entity_dcid)]
    identity = make_identity(GDP_SHAPE, [_ACTIVITY_SOURCE_SLOT], entities=entities)
    return ResubmitIdentityEntry(
        variable_text=variable_text,
        identity=identity,
    )


def _identity_entry_from_spec(
    spec, variable_text: str, *, separate: bool = False
) -> ResubmitIdentityEntry:
    """Build an identity entry from a resolved or candidate spec."""
    return ResubmitIdentityEntry(
        variable_text=variable_text,
        identity=spec.definition.identity,
        separate=separate,
    )


def _candidates_entry_from_part(
    part, *, candidates, variable_text: str, separate: bool = False
) -> ResubmitCandidatesEntry:
    return ResubmitCandidatesEntry(
        variable_text=variable_text,
        ordering=part.candidates.ordering,
        truncated=part.candidates.truncated,
        candidates=[
            ResubmitCandidate(
                identity=spec.definition.identity,
            )
            for spec in candidates
        ],
        separate=separate,
    )


def _spec_with_sv(part, ref: str):
    """Return the candidate spec containing the requested StatVar."""
    (spec,) = [
        spec
        for spec in part.candidates.specs
        if ref in {sv.ref.ref for sv in spec.resolved.members}
    ]
    return spec


_CRUDE_BIRTH_RATE = "Count_BirthEvent_LiveBirth_AsFractionOf_Count_Person"
_FERTILITY_RATE = "FertilityRate_Person_Female"


# ---------------------------------------------------------------------------
# 1. One input, N entries -> one envelope composed over all N parts.
# ---------------------------------------------------------------------------


def test_n_part_envelope_composes_sentence_over_every_part():
    """A three-entry request composes one sentence across identity, placeholder, and candidates."""
    ethiopia = offline_resolve(
        make_request("population and birth rate in Ethiopia"),
        llm=KeyedBindReplyLLM(BIRTH_RATE_CANDIDATES),
    )
    birth_rate_part = ethiopia.parts[1]
    assert birth_rate_part.status == "candidates"

    entries = [
        _gdp_entry(_INDIA, "GDP"),
        ResubmitPlaceholderEntry(
            variable_text="unemployment rate",
            status="no_data",
            reason="variable_not_resolved",
        ),
        _candidates_entry_from_part(
            birth_rate_part,
            candidates=birth_rate_part.candidates.specs,
            variable_text="birth rate candidates",
        ),
    ]
    result = offline_resolve(
        ResolveRequest(input=DatasetResubmitInput(entries=entries)),
    )

    assert result.status == "mixed"
    assert [p.status for p in result.parts] == ["definite", "no_data", "candidates"]
    assert result.rendered_sentence
    assert result.sentence_plan
    part_indices = {t.part_index for t in result.sentence_plan}
    assert {0, 1, 2} <= part_indices


def test_a_resubmit_drops_a_label_only_group_and_states_a_spec_sentence_without_departures():
    """C134: a posted group naming no graph node drops from the rebuilt part's Spec, since
    nothing ties its label to the role's entities, and the spec sentence holds heading and
    byline tokens only, the same sentence a fresh render of the spec states."""
    group = EntityGroup(ref=None, label="South Asia's largest economy")
    entities = [subject_entity(_INDIA)]
    _spec_id, identity = build_spec_identity(
        GDP_SHAPE,
        [_ACTIVITY_SOURCE_SLOT],
        entities=entities,
        open_roles=(),
    )
    entry = ResubmitIdentityEntry(
        variable_text="GDP",
        identity=identity,
        entity_groups={"observationAbout": group},
    )

    result = offline_resolve(ResolveRequest(input=DatasetResubmitInput(entries=[entry])))

    (part,) = result.parts
    assert part.status == "definite"
    assert part.spec.resolved.entity_groups is None
    plan = part.spec.resolved.sentence_plan
    assert plan
    assert {token.segment for token in plan} <= {"heading", "byline"}
    assert part.spec.resolved.rendered_sentence == render_sentence(part.spec)
    assert result.rendered_sentence is not None
    assert result.rendered_sentence.startswith(part.spec.resolved.rendered_sentence)


def test_every_entry_kind_echoes_its_posted_operations_onto_its_part():
    """Identity, placeholder, candidates, and parsed entries each carry their own
    operations back, a parsed entry through its `part`."""
    ethiopia = offline_resolve(
        make_request("population and birth rate in Ethiopia"),
        llm=KeyedBindReplyLLM(BIRTH_RATE_CANDIDATES),
    )
    birth_rate_part = ethiopia.parts[1]
    rank = Operation(verb="rank", phrase="top 5", over="place")
    growth = Operation(verb="growth", phrase="change in", over="year")
    explain = Operation(verb="explain", phrase="why did")
    forecast = Operation(verb="forecast", phrase="in 2030", over="year")

    identity = _gdp_entry(_INDIA, "GDP").model_copy(update={"operations": [rank]})
    placeholder = ResubmitPlaceholderEntry(
        variable_text="unemployment rate",
        status="no_data",
        reason="no_observations",
        operations=[growth],
    )
    candidates = _candidates_entry_from_part(
        birth_rate_part,
        candidates=birth_rate_part.candidates.specs,
        variable_text="birth rate candidates",
    ).model_copy(update={"operations": [explain]})
    parsed = ResubmitParsedEntry(
        part=ParsedPart(
            variable_text="fertility rate",
            entities=[ParsedEntity(text="Kenya")],
            operations=[forecast],
        )
    )
    entries = [identity, placeholder, candidates, parsed]

    result = offline_resolve(
        ResolveRequest(input=DatasetResubmitInput(entries=entries)),
    )

    assert [p.status for p in result.parts] == ["definite", "no_data", "candidates", "definite"]
    assert [p.operations for p in result.parts] == [[rank], [growth], [explain], [forecast]]


def test_entries_posted_without_operations_rebuild_parts_without_them():
    entry = ResubmitPlaceholderEntry(
        variable_text="unemployment rate", status="no_data", reason="no_observations"
    )
    result = offline_resolve(ResolveRequest(input=DatasetResubmitInput(entries=[entry])))
    assert result.parts[0].operations == []


# ---------------------------------------------------------------------------
# 2. The headline case: promoting a birth-rate candidate re-hoists Ethiopia
#    and source once both parts are definite and agree.
# ---------------------------------------------------------------------------


def test_promoting_a_candidate_hoists_shared_qualifiers_once_both_parts_agree():
    """Promoting Ethiopia's fertility-rate candidate makes both parts definite.
    The shared source token has no part index in buffered and SSE responses. The two
    windows differ (population runs to 2024, fertility rate to 2023), so each part keeps
    its own place and time tokens.
    """
    ethiopia = offline_resolve(
        make_request("population and birth rate in Ethiopia"),
        llm=KeyedBindReplyLLM(BIRTH_RATE_CANDIDATES),
    )
    population, birth_rate_candidates = ethiopia.parts
    assert population.status == "definite"
    assert birth_rate_candidates.status == "candidates"
    promoted_birth_rate = _spec_with_sv(birth_rate_candidates, _FERTILITY_RATE)

    request = _wire_request(
        [
            _identity_entry_from_spec(population.spec, "population"),
            _identity_entry_from_spec(promoted_birth_rate, "birth rate"),
        ]
    )
    client = _client(materialise_graph=axis_confirming_reader(*TOTAL_POPULATION_INDIA_FAMILY))
    buffered = client.post("/api/search/resolve", json=request)
    assert buffered.status_code == 200
    sse = client.post("/api/search/resolve", json=request, headers={"Accept": "text/event-stream"})
    assert sse.status_code == 200
    sse_response = next(d for ev, d in _parse_sse(sse.text) if ev == "response")["response"]

    for body in (buffered.json(), sse_response):
        assert body["status"] == "definite"
        assert [p["status"] for p in body["parts"]] == ["definite", "definite"]
        rendered = body["rendered_sentence"]
        assert "Ethiopia" in rendered
        # The unit phrase's price basis is the only parenthetical. A part never attaches one.
        unattached = rendered.replace(" (current prices)", "")
        assert "(" not in unattached and ")" not in unattached
        plan = body["sentence_plan"]
        where_tokens = [t for t in plan if t.get("axis") == "where"]
        when_tokens = [t for t in plan if t.get("axis") == "when"]
        source_tokens = [t for t in plan if t.get("axis") == "source"]
        assert len(source_tokens) == 1
        # Hoisted: the shared source carries no part_index at all (the wire drops a None
        # field entirely -- see conjoin/render's own SentenceToken serialization,
        # exercised the same way by test_envelope_composed_sentence.py's agree/disagree
        # goldens).
        assert "part_index" not in source_tokens[0]
        # The windows differ, so each part keeps its own possessive lead and time token.
        assert [(t["text"], t["part_index"]) for t in where_tokens] == [
            ("Ethiopia", 0),
            ("Ethiopia", 1),
        ]
        assert [(t["text"], t["part_index"]) for t in when_tokens] == [
            ("1960 to 2024", 0),
            ("1960 to 2023", 1),
        ]


# ---------------------------------------------------------------------------
# 3. Fold: two same-shape identities that fail the fold gate as posted
#    (a window disagreement) resolve UNFOLDED; editing one to agree resolves
#    FOLDED, with binding_folded on the later part. This is built from two
#    identities sharing one shape and differing only in gender (a single
#    "value" slot) plus, initially, their requested window.
# ---------------------------------------------------------------------------

_GENDER_ENTITY = "country/IND"
_FEMALE_SV = "Count_Person_Female"
_MALE_SV = "Count_Person_Male"


def _gender_fold_graph() -> FakeGraph:
    """Build female and synthetic male population nodes.
    Both carry India observations for 2001-2021.
    """
    fixtures_dir = Path(__file__).parent.parent / "fixtures"
    nodes = json.loads((fixtures_dir / "graph_nodes.json").read_text())
    obs = json.loads((fixtures_dir / "graph_obs.json").read_text())
    nodes[_MALE_SV] = {
        "label": "Male population",
        "type": "StatisticalVariable",
        "arcs": _sv_arcs(
            pop_type="Person",
            meas_prop="count",
            stat_type="measuredValue",
            label="Male population",
            gender="Male",
        ),
    }
    fixed_obs = [
        {
            "earliestDate": "2001",
            "latestDate": "2021",
            "obsCount": 3,
            "provenanceId": "dc/base/IndiaCensus",
        }
    ]
    obs[f"{_FEMALE_SV}|{_GENDER_ENTITY}"] = fixed_obs
    obs[f"{_MALE_SV}|{_GENDER_ENTITY}"] = fixed_obs
    return FakeGraph(nodes=nodes, obs=obs)


def _gender_entry(gender: str, sv_dcid: str, *, window: TimeWindow | None) -> ResubmitIdentityEntry:
    entities = [subject_entity(_GENDER_ENTITY)]
    gender_slot = _value_slot("how", "gender", "gender", gender, gender)
    _, identity = build_spec_identity(
        PERSON_COUNT_SHAPE,
        [gender_slot, *where_slots_for(entities)],
        entities=entities,
        requested_window=window,
        open_roles=(),
    )
    return ResubmitIdentityEntry(
        variable_text="count",
        identity=identity,
    )


def test_window_disagreement_blocks_fold_editing_to_agree_enables_it():
    graph = _gender_fold_graph()
    sdmx = FakeSdmxClient(observations={}, availability={})
    female_entry = _gender_entry("Female", _FEMALE_SV, window=None)

    misaligned_male = _gender_entry(
        "Male", _MALE_SV, window=TimeWindow(start_year=2001, end_year=2010)
    )
    unfolded = offline_resolve(
        ResolveRequest(input=DatasetResubmitInput(entries=[female_entry, misaligned_male])),
        graph=graph,
        sdmx=sdmx,
    )
    assert [p.status for p in unfolded.parts] == ["definite", "definite"]
    assert unfolded.parts[0].spec.spec_id != unfolded.parts[1].spec.spec_id
    assert not any(w.code == "binding_folded" for p in unfolded.parts for w in p.warnings)

    aligned_male = _gender_entry("Male", _MALE_SV, window=None)
    folded = offline_resolve(
        ResolveRequest(input=DatasetResubmitInput(entries=[female_entry, aligned_male])),
        graph=graph,
        sdmx=sdmx,
    )
    assert [p.status for p in folded.parts] == ["definite", "definite"]
    first, second = folded.parts
    assert first.spec.spec_id == second.spec.spec_id
    assert not any(w.code == "binding_folded" for w in first.warnings)
    assert any(w.code == "binding_folded" for w in second.warnings)
    gender_slot = next(
        s
        for s in first.spec.resolved.slots
        if s.key.property is not None and s.key.property.ref == "gender"
    )
    assert gender_slot.binding.kind == "set"
    assert {v.ref.ref for v in gender_slot.binding.values if v.ref is not None} == {
        "Female",
        "Male",
    }


# ---------------------------------------------------------------------------
# 4. A `candidates` entry comes back in posted order, always -- proven
#    non-vacuously by posting the reverse of the order a fresh broadest_first
#    re-sort would produce.
# ---------------------------------------------------------------------------


def test_candidates_entry_preserves_posted_order_even_reversed():
    """A reversed birth-rate/fertility-rate pair returns in posted order, not member-count
    order."""
    ethiopia = offline_resolve(
        make_request("population and birth rate in Ethiopia"),
        llm=KeyedBindReplyLLM(BIRTH_RATE_CANDIDATES),
    )
    birth_rate_part = ethiopia.parts[1]
    natural_order = list(birth_rate_part.candidates.specs)
    assert len(natural_order) == 2
    reversed_order = list(reversed(natural_order))

    entry = _candidates_entry_from_part(
        birth_rate_part, candidates=reversed_order, variable_text="birth rate"
    )
    result = offline_resolve(
        ResolveRequest(input=DatasetResubmitInput(entries=[entry])),
    )
    assert result.status == "candidates"
    part = result.parts[0]
    assert part.status == "candidates"
    resolved_svs = [[sv.ref.ref for sv in spec.resolved.members] for spec in part.candidates.specs]
    posted_svs = [[sv.ref.ref for sv in spec.resolved.members] for spec in reversed_order]
    # A fresh sort would reproduce natural_order; asserting against that
    # (rather than only against posted_svs) makes a regression to re-sorting
    # actually fail this test.
    assert resolved_svs == posted_svs
    assert resolved_svs != [[sv.ref.ref for sv in spec.resolved.members] for spec in natural_order]
    assert part.candidates.ordering == birth_rate_part.candidates.ordering
    assert part.candidates.max_candidates == birth_rate_part.candidates.max_candidates
    assert part.candidates.truncated == birth_rate_part.candidates.truncated


# ---------------------------------------------------------------------------
# 5. Each entry kind, end to end.
# ---------------------------------------------------------------------------


def test_identity_entry_resolves_definite():
    result = offline_resolve(
        ResolveRequest(input=DatasetResubmitInput(entries=[_gdp_entry(_INDIA, "GDP")]))
    )
    assert result.status == "definite"
    assert result.parts[0].status == "definite"


def test_candidates_entry_resolves_candidates():
    """The full birth-rate candidate set resubmits as candidates."""
    ethiopia = offline_resolve(
        make_request("population and birth rate in Ethiopia"),
        llm=KeyedBindReplyLLM(BIRTH_RATE_CANDIDATES),
    )
    birth_rate_part = ethiopia.parts[1]
    entry = _candidates_entry_from_part(
        birth_rate_part,
        candidates=birth_rate_part.candidates.specs,
        variable_text="birth rate",
        separate=True,
    )
    result = offline_resolve(
        ResolveRequest(input=DatasetResubmitInput(entries=[entry])),
    )
    assert result.status == "candidates"
    assert result.parts[0].status == "candidates"
    assert len(result.parts[0].candidates.specs) == 2
    assert result.parts[0].separate is True
    assert result.parts[0].model_dump(mode="json")["separate"] is True

    repeated_part = result.parts[0]
    repeated = offline_resolve(
        ResolveRequest(
            input=DatasetResubmitInput(
                entries=[
                    _candidates_entry_from_part(
                        repeated_part,
                        candidates=repeated_part.candidates.specs,
                        variable_text=repeated_part.variable_text,
                        separate=repeated_part.separate,
                    )
                ]
            )
        )
    )
    assert repeated.parts[0].separate is True


def test_candidates_entry_round_trips_a_raw_text_part_without_collapsing():
    """An unedited candidate set keeps its status and candidates after resubmission."""
    ethiopia = offline_resolve(
        make_request("population and birth rate in Ethiopia"),
        llm=KeyedBindReplyLLM(BIRTH_RATE_CANDIDATES),
    )
    birth_rate_part = ethiopia.parts[1]
    assert birth_rate_part.status == "candidates"
    entry = _candidates_entry_from_part(
        birth_rate_part,
        candidates=birth_rate_part.candidates.specs,
        variable_text="birth rate",
    )

    result = offline_resolve(
        ResolveRequest(input=DatasetResubmitInput(entries=[entry])),
    )

    assert result.status == "candidates"
    assert result.status == birth_rate_part.status
    part = result.parts[0]
    assert part.status == birth_rate_part.status
    assert len(part.candidates.specs) == len(birth_rate_part.candidates.specs)
    assert not any(
        w.code in ("candidates_collapsed_definite", "candidates_collapsed_no_data")
        for w in part.warnings
    )


def test_collapsed_candidates_entry_carries_the_warning_on_the_wire():
    """When only the fertility-rate candidate has data, the response is definite with
    `candidates_collapsed_definite` so clients can identify the collapse.
    """
    ethiopia = offline_resolve(
        make_request("population and birth rate in Ethiopia"),
        llm=KeyedBindReplyLLM(BIRTH_RATE_CANDIDATES),
    )
    birth_rate_part = ethiopia.parts[1]
    survivor = _spec_with_sv(birth_rate_part, _FERTILITY_RATE)
    no_data_entry = _gdp_entry(_USA, "GDP-USA")
    entry = ResubmitCandidatesEntry(
        variable_text="birth rate",
        ordering=birth_rate_part.candidates.ordering,
        truncated=birth_rate_part.candidates.truncated,
        candidates=[
            ResubmitCandidate(
                identity=survivor.definition.identity,
            ),
            ResubmitCandidate(
                identity=no_data_entry.identity,
            ),
        ],
    )

    result = offline_resolve(ResolveRequest(input=DatasetResubmitInput(entries=[entry])))

    assert result.status == "definite"
    part = result.parts[0]
    assert part.status == "definite"
    warning = next(w for w in part.warnings if w.code == "candidates_collapsed_definite")
    assert warning.message == (
        "2 candidates were posted; 1 still resolves to a definite answer, so the part "
        "is reported as that answer and not as the posted candidate set."
    )


def test_placeholder_entry_no_data_restates_stub_with_warning_and_reason():
    entry = ResubmitPlaceholderEntry(
        variable_text="unemployment rate",
        status="no_data",
        reason="no_observations",
        separate=True,
    )
    result = offline_resolve(ResolveRequest(input=DatasetResubmitInput(entries=[entry])))
    assert result.status == "no_data"
    part = result.parts[0]
    assert part.status == "no_data"
    assert part.no_data.reason == "no_observations"
    assert part.separate is True
    assert part.model_dump(mode="json")["separate"] is True
    warning = next(w for w in part.warnings if w.code == "part_restated")
    assert warning.severity == "info"
    assert "unemployment rate" in warning.message
    assert "no_data" in warning.message


def test_placeholder_entry_unsupported_restates_stub_with_warning_and_code():
    entry = ResubmitPlaceholderEntry(
        variable_text="GDP in PPP dollars",
        status="unsupported",
        code="unit_request_unsupported",
        separate=True,
    )
    result = offline_resolve(ResolveRequest(input=DatasetResubmitInput(entries=[entry])))
    assert result.status == "unsupported"
    part = result.parts[0]
    assert part.status == "unsupported"
    assert part.unsupported.code == "unit_request_unsupported"
    assert part.separate is True
    assert part.model_dump(mode="json")["separate"] is True
    warning = next(w for w in part.warnings if w.code == "part_restated")
    assert warning.severity == "info"
    assert "GDP in PPP dollars" in warning.message
    assert "unsupported" in warning.message


def test_placeholder_entry_unrecognised_code_names_the_code_on_the_wire():
    """A placeholder entry's posted `code` absent from this deployment's
    registry must fail the whole request with a `detail` naming the bogus
    code, distinct from the generic per-code catalogue default every other
    `request_invalid` raise site shares. Without it, a client engineer
    holding only a "Bad request" response cannot tell which field was
    wrong, what value it held, or that codes are even registry-checked."""
    entry = ResubmitPlaceholderEntry(
        variable_text="GDP in PPP dollars", status="unsupported", code="not_a_real_code"
    )
    client = _client()
    resp = client.post("/api/search/resolve", json=_wire_request([entry]))
    assert resp.status_code // 100 == 4
    body = resp.json()
    assert body["code"] == "request_invalid"
    assert body["pointer"] == "/input/entries/0/code"
    assert body["detail"] != "Bad request"
    assert "not_a_real_code" in body["detail"]


def test_parsed_entry_resolves_through_the_shared_per_part_pipeline():
    """Reuses "fertility rate"/"Kenya", the same variable+entity text
    test_parsed_pipeline.py's own single-part golden exercises, so this
    entry hits the exact same recall()/bind_universal FakeLLM fixture --
    no new prompt is minted."""
    entry = ResubmitParsedEntry(
        part=ParsedPart(variable_text="fertility rate", entities=[ParsedEntity(text="Kenya")])
    )
    result = offline_resolve(ResolveRequest(input=DatasetResubmitInput(entries=[entry])))
    assert result.status == "definite"
    part = result.parts[0]
    assert part.status == "definite"
    assert part.variable_text == "fertility rate"
    assert part.spec.definition.identity.entities == {"observationAbout": ["country/KEN"]}


@pytest.mark.parametrize(
    "entries, graph, expect_error, check_result",
    [
        pytest.param(
            [
                ResubmitParsedEntry(
                    part=ParsedPart(
                        variable_text="fertility rate",
                        entities=[ParsedEntity(text="Congo", dcid="country/COD")],
                    )
                ),
                ResubmitParsedEntry(
                    part=ParsedPart(
                        variable_text="fertility rate",
                        entities=[ParsedEntity(text="Congo", dcid="country/COG")],
                    )
                ),
            ],
            FakeGraph(
                obs={
                    **_load_json("graph_obs.json"),
                    "FertilityRate_Person_Female|country/COD": [
                        {"earliestDate": "1960", "latestDate": "2023", "obsCount": 64}
                    ],
                    "FertilityRate_Person_Female|country/COG": [],
                }
            ),
            None,
            lambda result: (
                result.parts[0].status == "definite"
                and result.parts[0].spec.definition.identity.entities
                == {"observationAbout": ["country/COD"]}
                and result.parts[1].status == "no_data"
                and result.parts[1].no_data.reason == "no_observations"
            ),
            id="per_part_pins_resolve_to_different_places",
        ),
        pytest.param(
            [
                ResubmitParsedEntry(
                    part=ParsedPart(
                        variable_text="fertility rate",
                        entities=[ParsedEntity(text="Kenya", dcid="country/DOES_NOT_EXIST")],
                    )
                )
            ],
            None,
            ("invalid_role_ref", 422, "/input/entries/0/part/entities/0/dcid"),
            None,
            id="unconfirmable_dcid_refuses_invalid_role_ref",
        ),
        pytest.param(
            [
                ResubmitParsedEntry(
                    part=ParsedPart(
                        variable_text="aid from Congo to Congo",
                        entities=[
                            ParsedEntity(text="Congo", role_text="from", members=True),
                            ParsedEntity(text="Congo", role_text="to"),
                        ],
                    )
                )
            ],
            None,
            ("request_invalid", None, "/input/entries/0/part/entities"),
            None,
            id="one_text_both_members_values_refuses_request_invalid",
        ),
    ],
)
def test_parsed_entry_dcid_pins(entries, graph, expect_error, check_result):
    """Two entries pin "Congo" to COD and COG independently. Only COD has observations.
    A text-keyed pins map would let the first pin win for both entries.
    Unconfirmable or conflicting pins, and one text posted with both `members` values,
    refuse the request.
    Errors use `/input/entries/{i}/part/...`, even after other entry kinds.
    """
    request = ResolveRequest(input=DatasetResubmitInput(entries=entries))
    if expect_error:
        code, status, pointer = expect_error
        with pytest.raises(EngineInputError) as exc_info:
            offline_resolve(request, graph=graph or FakeGraph())
        assert exc_info.value.code == code
        if status is not None:
            assert exc_info.value.status == status
        assert exc_info.value.pointer == pointer
    else:
        result = offline_resolve(request, graph=graph)
        assert check_result(result), result


def test_parsed_entry_pin_pointer_uses_entries_index_not_parsed_sublist_position():
    """The offending parsed entry sits at index 1 of `inp.entries` (a
    placeholder entry occupies index 0) -- confirm_part_pins's own
    pointer_prefixes are built from each entry's `inp.entries` position, not
    from its position among parsed entries alone, so the pointer must still
    read entries/1, not entries/0."""
    entries = [
        ResubmitPlaceholderEntry(
            variable_text="unemployment rate", status="no_data", reason="no_observations"
        ),
        ResubmitParsedEntry(
            part=ParsedPart(
                variable_text="fertility rate",
                entities=[ParsedEntity(text="Kenya", dcid="country/DOES_NOT_EXIST")],
            )
        ),
    ]
    with pytest.raises(EngineInputError) as exc_info:
        offline_resolve(ResolveRequest(input=DatasetResubmitInput(entries=entries)))
    assert exc_info.value.code == "invalid_role_ref"
    assert exc_info.value.pointer == "/input/entries/1/part/entities/0/dcid"


def test_parsed_entry_warns_when_variable_text_names_an_entity():
    """Parsed entries warn when variable text also names an entity."""
    entry = ResubmitParsedEntry(part=ParsedPart(variable_text="fertility rate in France"))
    result = offline_resolve(ResolveRequest(input=DatasetResubmitInput(entries=[entry])))
    warnings = {w.code: w for w in result.parts[0].warnings}
    assert "variable_text_entity_like" in warnings
    assert "France" in warnings["variable_text_entity_like"].message


def test_parsed_entry_structured_entities_suppress_the_entity_like_warning():
    # The same variable_text as the no-entities case above (so detect_svs
    # misses its fixture and degrades gracefully either way, never reaching a
    # bind call this test would need its own LLM fixture for) -- only
    # entities differs, isolating what suppresses the warning.
    entry = ResubmitParsedEntry(
        part=ParsedPart(
            variable_text="fertility rate in France", entities=[ParsedEntity(text="France")]
        )
    )
    result = offline_resolve(ResolveRequest(input=DatasetResubmitInput(entries=[entry])))
    assert not any(w.code == "variable_text_entity_like" for w in result.parts[0].warnings)


# ---------------------------------------------------------------------------
# 6. The three-class refusal contract, plus N=1.
# ---------------------------------------------------------------------------


def test_malformed_entry_fails_whole_request_with_repointed_pointer_and_missing_refs_detail():
    """Unresolvable posted entities on entry index 1 raise invalid_role_ref
    from resolve_spec_resubmit with pointer /input/identity/entities/...;
    this door re-points it onto /input/entries/1/... and fails the WHOLE
    request 4xx -- a malformed entry is a client bug, not a per-part outcome,
    so no partial `parts` list is ever returned alongside it. The wire
    detail names the missing dcids of the one rejected role."""
    missing_dcids = ["country/ZZZ_NOT_A_PLACE", "country/ZZZ_ALSO_NOT_A_PLACE"]
    entities = [subject_entity(dcid) for dcid in missing_dcids]
    invalid_entry = ResubmitIdentityEntry(
        variable_text="GDP2",
        identity=make_identity(GDP_SHAPE, [_ACTIVITY_SOURCE_SLOT], entities=entities),
    )
    entries = [_gdp_entry(_INDIA, "GDP1"), invalid_entry]
    client = _client()
    resp = client.post("/api/search/resolve", json=_wire_request(entries))
    assert resp.status_code // 100 == 4
    body = resp.json()
    assert body["code"] == "invalid_role_ref"
    assert body["pointer"] == "/input/entries/1/identity/entities/observationAbout"
    assert_names_plainly(body["detail"], sorted(missing_dcids))
    assert "parts" not in body


@pytest.mark.parametrize(
    ("counts", "pointer", "named"),
    [
        pytest.param(
            {"observationAbout": 300},
            "/input/entries/0/identity/entities/observationAbout",
            {300, 256},
            id="one-role-of-300",
        ),
        pytest.param(
            {"destination": 200, "origin": 100},
            "/input/entries/0/identity/entities",
            {300, 2, 256},
            id="200-and-100-across-two-roles",
        ),
    ],
)
def test_an_identity_over_the_place_bound_refuses_request_invalid_on_the_wire(
    counts, pointer, named
):
    """An identity binding more than 256 places answers 400 `request_invalid` at the role's
    path when one role exceeds alone, and at the entity map when only the sum of two roles does.
    The detail names the count and the bound."""
    build = {
        "observationAbout": subject_entity,
        "destination": destination_entity,
        "origin": origin_entity,
    }
    entities = [
        build[role](f"country/{role}{i}") for role, count in counts.items() for i in range(count)
    ]
    entry = ResubmitIdentityEntry(
        variable_text="GDP",
        identity=make_identity(GDP_SHAPE, [_ACTIVITY_SOURCE_SLOT], entities=entities),
    )
    resp = _client().post("/api/search/resolve", json=_wire_request([entry]))
    assert resp.status_code == 400
    body = resp.json()
    assert body["code"] == "request_invalid"
    assert body["pointer"] == pointer
    assert {int(number) for number in re.findall(r"\d+", body["detail"])} == named
    assert "parts" not in body


def test_a_composite_identity_its_components_cannot_serve_refuses_request_invalid_on_the_wire():
    """A composite whose imputed component's rows declare no co-operation modality cannot
    serve a bound modality, so a resubmit posting one refuses at the bound slot before any
    read."""
    document = education_aid_identity(source="provenance/ONEBilateralAndImputedMultilateral")
    document["slots"][0]["binding"] = {"state": "value", "ref": "CRSCooperationModalityA01"}
    identity = BaseIdentity.model_validate(document)
    entry = ResubmitIdentityEntry(variable_text="education aid", identity=identity)
    resp = _client().post("/api/search/resolve", json=_wire_request([entry]))
    assert resp.status_code == 400
    body = resp.json()
    assert body["code"] == "request_invalid"
    assert body["pointer"] == "/input/entries/0/identity/slots/0"
    assert "parts" not in body


def test_candidates_entry_beyond_search_max_candidates_refuses_request_invalid():
    """A candidates entry posting more candidates than this deployment's own
    SEARCH_MAX_CANDIDATES ceiling refuses request_invalid, naming the entry:
    the engine enforces its own real ceiling here rather than trusting a
    client-posted count checked only against itself."""
    ethiopia = offline_resolve(
        make_request("population and birth rate in Ethiopia"),
        llm=KeyedBindReplyLLM(BIRTH_RATE_CANDIDATES),
    )
    birth_rate_part = ethiopia.parts[1]
    oversized = list(birth_rate_part.candidates.specs) * 4  # 8 candidates
    entry = _candidates_entry_from_part(
        birth_rate_part, candidates=oversized, variable_text="birth rate"
    )
    client = _client()
    resp = client.post("/api/search/resolve", json=_wire_request([entry]))
    assert resp.status_code // 100 == 4
    body = resp.json()
    assert body["code"] == "request_invalid"
    assert body["pointer"] == "/input/entries/0/candidates"


def test_resolution_outcome_is_a_200_part_siblings_unaffected():
    """A promote that naturally resolves no_data (a valid entity with no
    GDP observations in this offline fixture bank) is a resolution outcome,
    not a malformed request: the whole response is 200, and the sibling
    entry's own definite outcome is untouched."""
    entries = [_gdp_entry(_INDIA, "GDP-India"), _gdp_entry(_USA, "GDP-USA")]
    result = offline_resolve(ResolveRequest(input=DatasetResubmitInput(entries=entries)))
    assert result.status == "mixed"
    india_part, usa_part = result.parts
    assert india_part.status == "definite"
    assert usa_part.status == "no_data"
    assert usa_part.no_data.reason == "no_observations"


def test_n_equals_1_resolves_like_a_single_resubmit():
    """A one-entry dataset_resubmit request is routed through the same
    `assemble_region` single-part shortcut a bare identity request hits, and
    must carry the identical posted identity."""
    entry = _gdp_entry(_INDIA, "GDP")
    dataset_result = offline_resolve(ResolveRequest(input=DatasetResubmitInput(entries=[entry])))
    assert dataset_result.status == "definite"
    assert len(dataset_result.parts) == 1

    plain_input = ResubmitIdentityPayload(
        identity=entry.identity,
    )
    plain_region = resolve_spec_resubmit(
        inp=plain_input,
        graph=FakeGraph(),
        sdmx=FakeSdmxClient(),
        coverage=empty_coverage_snapshot(),
        options=walk_options(FakeGraph()),
    )
    assert plain_region.status == "definite"
    assert dataset_result.parts[0].spec.spec_id == plain_region.specs[0].spec_id


def _record_parsed_boundary_raise(monkeypatch) -> list[str]:
    """Record `TimeoutError` raised by the parsed-entry fan-out."""
    real = _dataset_resubmit_mod.resolve_parsed_entries
    raised: list[str] = []

    async def _recording(*args, **kwargs):
        try:
            return await real(*args, **kwargs)
        except TimeoutError as exc:
            raised.append(str(exc))
            raise

    monkeypatch.setattr(_dataset_resubmit_mod, "resolve_parsed_entries", _recording)
    return raised


@pytest.mark.unrecorded_requests
def test_timeout_mid_request_fails_the_whole_request_with_no_partial_parts(
    monkeypatch, search_settings
):
    """A parsed leg advances the fake clock and blocks until cancelled.
    The GDP sibling must not leak as a partial response.
    """
    clock = frozen_monotonic_clock(monkeypatch)
    search_settings(request_timeout_s=DEADLINE_TIMEOUT_S)
    parsed_raises = _record_parsed_boundary_raise(monkeypatch)

    cancelled: list[str] = []

    async def _stall_past_deadline(*args, **kwargs):
        clock.now += DEADLINE_TIMEOUT_S * 10
        emit = kwargs.get("emit")
        if emit is not None:
            # Wake the fan-out loop, whose own wait is bounded by the same
            # clock, without waiting for anything real.
            await emit(StepEvent(step="materialise", ran=True))
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled.append("parsed leg")
            raise
        raise AssertionError("unreachable -- the deadline cancels this task")  # pragma: no cover

    monkeypatch.setattr("search.engine.fanout.resolve_variable", _stall_past_deadline)

    entries = [
        _gdp_entry(_INDIA, "GDP"),
        ResubmitParsedEntry(
            part=ParsedPart(variable_text="fertility rate", entities=[ParsedEntity(text="Kenya")])
        ),
    ]
    client = _client()
    resp = client.post(
        "/api/search/resolve",
        json=_wire_request(entries),
        headers={"Accept": "text/event-stream"},
    )
    assert resp.status_code == 200
    frames = _parse_sse(resp.text)
    event_types = [ev for ev, _ in frames]
    assert event_types == ["error"], f"Expected only an error event, got {event_types}"
    assert frames[0][1]["detail"] == "Request timed out"
    assert parsed_raises == ["dataset_resubmit: a parsed entry exceeded the pipeline deadline"], (
        "the parsed fan-out itself must be what raised"
    )
    assert cancelled == ["parsed leg"], "the still-pending sibling leg must be cancelled"


# ---------------------------------------------------------------------------
# 7. A parsed entry's trailing tasks: this door maps
#    enrichment._launch_trailing_tasks over every region, including a
#    parsed entry's own RegionResult (region/pipeline.py) -- which, unlike a
#    promote region (region/resubmit_promote.py), already carries a LIVE
#    slot_options_task/observations_task from its own eager streaming launch,
#    alongside the same relaunch closure a promote region uses to defer that
#    launch. _launch_trailing_tasks' own `is None` guard keeps a parsed
#    region's live task handle from being relaunched: without it, mapping the
#    helper over a parsed region would call the relaunch closure a second
#    time, launching a duplicate probe, overwriting the task handle, and
#    orphaning the original outside handle()'s own finally-block cleanup.
# ---------------------------------------------------------------------------


def test_launch_trailing_tasks_guards_against_relaunching_a_live_parsed_task():
    """A parsed region with live task handles keeps them and does not relaunch probes."""
    calls: list[str] = []
    spec = SimpleNamespace(spec_id="spec-id", resolved=SimpleNamespace(slot_options=["options"]))

    def relaunch_slot(*args):
        calls.append("slot")
        return "new-slot-task"

    def relaunch_observations(*args):
        calls.append("observations")
        return "new-observations-task"

    already_running_slot_task = object()
    already_running_observations_task = object()
    region = RegionResult(
        variable_text="fertility",
        status="definite",
        specs=(spec,),
        no_data_reason=None,
        warnings=(),
        timing_by_step={},
        facet_probe=_PROBE,
        slot_options_task=already_running_slot_task,
        observations_task=already_running_observations_task,
        slot_options_relaunch=relaunch_slot,
        observations_relaunch=relaunch_observations,
    )

    launched = _launch_trailing_tasks(region, launch_slot_options=True, launch_observations=True)

    assert calls == [], f"expected no relaunch calls for a region with a live task, got {calls}"
    assert launched.slot_options_task is already_running_slot_task
    assert launched.observations_task is already_running_observations_task


@pytest.mark.unrecorded_requests
def test_parsed_timeout_with_completed_identity_refusal_surfaces_4xx_not_504(
    monkeypatch, search_settings
):
    """A completed identity refusal takes precedence over a parsed-leg timeout.
    The fake clock advances only after the refusal completes; the response names entry 0.
    """
    clock = frozen_monotonic_clock(monkeypatch)
    search_settings(request_timeout_s=DEADLINE_TIMEOUT_S)

    async def _stall_until_promote_settles(*args, **kwargs):
        # The refusal has to be COMPLETED, not merely underway, for
        # `_first_completed_refusal` to prefer it: yield to the loop until no
        # promote task is left, then take the deadline.
        spins = 0
        while True:
            me = asyncio.current_task()
            pending = [
                t
                for t in asyncio.all_tasks()
                if t is not me and not t.done() and "._resolve_one" in t.get_coro().__qualname__
            ]
            if not pending:
                break
            spins += 1
            assert spins < 100_000, "the malformed entry's promote task never completed"
            await asyncio.sleep(0)
        clock.now += DEADLINE_TIMEOUT_S * 10
        emit = kwargs.get("emit")
        if emit is not None:
            await emit(StepEvent(step="materialise", ran=True))
        await asyncio.Event().wait()

    monkeypatch.setattr("search.engine.fanout.resolve_variable", _stall_until_promote_settles)

    entries = [
        _gdp_entry("country/ZZZ_NOT_A_PLACE", "GDP2"),
        ResubmitParsedEntry(
            part=ParsedPart(variable_text="fertility rate", entities=[ParsedEntity(text="Kenya")])
        ),
    ]
    client = _client()
    resp = client.post("/api/search/resolve", json=_wire_request(entries))
    assert resp.status_code // 100 == 4, f"expected a 4xx, got {resp.status_code}: {resp.text}"
    body = resp.json()
    assert body["code"] == "invalid_role_ref"
    assert body["pointer"] == "/input/entries/0/identity/entities/observationAbout"


@pytest.mark.unrecorded_requests
def test_parsed_timeout_without_completed_refusal_still_yields_504(monkeypatch, search_settings):
    """Without a completed refusal, the parsed-leg timeout returns 504."""
    clock = frozen_monotonic_clock(monkeypatch)
    search_settings(request_timeout_s=DEADLINE_TIMEOUT_S)
    parsed_raises = _record_parsed_boundary_raise(monkeypatch)

    async def _stall_past_deadline(*args, **kwargs):
        clock.now += DEADLINE_TIMEOUT_S * 10
        emit = kwargs.get("emit")
        if emit is not None:
            await emit(StepEvent(step="materialise", ran=True))
        await asyncio.Event().wait()

    monkeypatch.setattr("search.engine.fanout.resolve_variable", _stall_past_deadline)

    entries = [
        _gdp_entry(_INDIA, "GDP"),
        ResubmitParsedEntry(
            part=ParsedPart(variable_text="fertility rate", entities=[ParsedEntity(text="Kenya")])
        ),
    ]
    client = _client()
    resp = client.post("/api/search/resolve", json=_wire_request(entries))
    assert resp.status_code == 504
    assert resp.json()["code"] == "timeout"
    assert parsed_raises == ["dataset_resubmit: a parsed entry exceeded the pipeline deadline"], (
        "the parsed fan-out itself must be what raised, not a later stage"
    )


@pytest.mark.unrecorded_requests
def test_failing_request_cancels_and_awaits_live_parsed_trailing_tasks(
    monkeypatch, search_settings
):
    """Phase three (the promote gather) fails AFTER the
    parsed fan-out already returned success, so the failure surfaces from
    `_await_bounded(_gather_all())`, not from `resolve_parsed_entries`. The
    parsed entry's own RegionResult still carries LIVE slot_options_task/
    observations_task handles at that point (region/pipeline.py's eager
    streaming launch) -- `handle`'s own cleanup on a failing exit must
    cancel and await both rather than leaving them to outlive the request.

    The deadline is driven through `search.engine.deadline._await_bounded`,
    the one seam that bounds the promote gather: the door's own imported
    name for it raises `asyncio.TimeoutError` without the promote ever
    running out a real budget. That name is called exactly once per request
    -- the promote gather -- since the parsed fan-out's own pre-resolution
    pass binds its own copy.

    Asserted from inside the running loop: a `finally`/except cleanup that
    fires during this generator's own teardown is invisible once
    `asyncio.run` has already returned and torn the loop down (that would
    hide exactly the leak this test guards against)."""
    search_settings(request_timeout_s=DEADLINE_TIMEOUT_S)

    bounded_calls: list[float] = []

    async def _timeout_the_promote_gather(coro, *, deadline):
        bounded_calls.append(deadline)
        coro.close()
        raise TimeoutError("promote gather exceeded the pipeline deadline")

    monkeypatch.setattr(_dataset_resubmit_mod, "_await_bounded", _timeout_the_promote_gather)

    slot_tasks: list[asyncio.Task] = []
    obs_tasks: list[asyncio.Task] = []
    real_launch_slot_options = _relaunch_mod._launch_slot_options_task
    real_launch_observations = _observations_dispatch_mod._launch_observations_task

    async def _held_open(task: asyncio.Task):
        # The probe's handle stays live until the request's cleanup cancels it, however
        # soon the probe's own read finishes, so the test does not race the read.
        result = await task
        await asyncio.Event().wait()
        return result

    def _counting_launch_slot_options(*args, **kwargs):
        task = create_owned_task(_held_open(real_launch_slot_options(*args, **kwargs)))
        slot_tasks.append(task)
        return task

    def _counting_launch_observations(*args, **kwargs):
        task = create_owned_task(_held_open(real_launch_observations(*args, **kwargs)))
        obs_tasks.append(task)
        return task

    monkeypatch.setattr(_relaunch_mod, "_launch_slot_options_task", _counting_launch_slot_options)
    monkeypatch.setattr(
        _observations_dispatch_mod, "_launch_observations_task", _counting_launch_observations
    )

    entries = [
        _gdp_entry(_INDIA, "GDP"),
        ResubmitParsedEntry(
            part=ParsedPart(variable_text="fertility rate", entities=[ParsedEntity(text="Kenya")])
        ),
    ]
    request = ResolveRequest(
        input=DatasetResubmitInput(entries=entries),
        response_options=ResolveOptions(include_option_coverage=True, include_observations=True),
    )

    async def _drain() -> None:
        graph = FakeGraph()
        raised: BaseException | None = None
        try:
            async for _ev in resolve_events(
                request,
                graph=graph,
                resolver=make_resolver(graph),
                llm=FakeLLM(),
                stream_slot_options=True,
                sdmx=FakeSdmxClient(),
            ):
                pass
        except TimeoutError as exc:
            raised = exc
        assert raised is not None, "expected the stalled promote gather to time the request out"
        assert len(bounded_calls) == 1, (
            f"expected the promote gather to be the only bounded await, got {len(bounded_calls)}"
        )
        assert len(slot_tasks) == 1, (
            f"expected exactly one slot-options probe, got {len(slot_tasks)}"
        )
        assert len(obs_tasks) == 1, f"expected exactly one observations probe, got {len(obs_tasks)}"
        assert slot_tasks[0].done() and slot_tasks[0].cancelled(), (
            "the slot-options probe must be cancelled by the failing exit"
        )
        assert obs_tasks[0].done() and obs_tasks[0].cancelled(), (
            "the observations probe must be cancelled by the failing exit"
        )

    asyncio.run(_drain())


def test_successful_streaming_request_keeps_parsed_trailing_tasks_alive_after_response(
    monkeypatch,
):
    """The cleanup for a failing exit must
    NOT also sweep a successful request's own parsed trailing tasks -- those
    handles are load-bearing, consumed downstream by `_launch_trailing_tasks`
    and the streaming combiner. Matters as much as the leak test above: a
    blanket cancel-on-every-exit would pass that test while quietly breaking
    every request that streams option coverage or observations for a parsed
    entry."""
    slot_tasks: list[asyncio.Task] = []
    obs_tasks: list[asyncio.Task] = []
    real_launch_slot_options = _relaunch_mod._launch_slot_options_task
    real_launch_observations = _observations_dispatch_mod._launch_observations_task

    def _counting_launch_slot_options(*args, **kwargs):
        task = real_launch_slot_options(*args, **kwargs)
        slot_tasks.append(task)
        return task

    def _counting_launch_observations(*args, **kwargs):
        task = real_launch_observations(*args, **kwargs)
        obs_tasks.append(task)
        return task

    monkeypatch.setattr(_relaunch_mod, "_launch_slot_options_task", _counting_launch_slot_options)
    monkeypatch.setattr(
        _observations_dispatch_mod, "_launch_observations_task", _counting_launch_observations
    )

    # A single parsed entry, same isolation
    # test_parsed_entry_streaming_launches_exactly_one_slot_options_and_observations_probe
    # uses: a sibling identity/candidates entry also fires
    # `_launch_observations_task` through its own `_launch_trailing_tasks`
    # relaunch, which would conflate that (already load-bearing, unrelated)
    # call with the one this test is pinning.
    entries = [
        ResubmitParsedEntry(
            part=ParsedPart(variable_text="fertility rate", entities=[ParsedEntity(text="Kenya")])
        ),
    ]
    request = ResolveRequest(
        input=DatasetResubmitInput(entries=entries),
        response_options=ResolveOptions(include_option_coverage=True, include_observations=True),
    )

    async def _drain() -> list[str]:
        graph = FakeGraph()
        kinds = [
            ev.kind
            async for ev in resolve_events(
                request,
                graph=graph,
                resolver=make_resolver(graph),
                llm=FakeLLM(),
                stream_slot_options=True,
                sdmx=FakeSdmxClient(),
            )
        ]
        assert len(slot_tasks) == 1, (
            f"expected exactly one slot-options probe, got {len(slot_tasks)}"
        )
        assert len(obs_tasks) == 1, f"expected exactly one observations probe, got {len(obs_tasks)}"
        assert slot_tasks[0].done() and not slot_tasks[0].cancelled(), (
            "a successful request must not cancel the parsed entry's slot-options task"
        )
        assert obs_tasks[0].done() and not obs_tasks[0].cancelled(), (
            "a successful request must not cancel the parsed entry's observations task"
        )
        return kinds

    kinds = asyncio.run(_drain())
    assert "response" in kinds


def test_parsed_entry_streaming_launches_exactly_one_slot_options_and_observations_probe(
    monkeypatch,
):
    """End-to-end proof, through dataset_resubmit.handle itself (via
    resolve_events), that a streamed parsed entry launches exactly one
    slot-options probe and one observations probe, both fully awaited by the
    time the request completes. Wraps the actual probe launchers
    (region/option_coverage.py's _launch_slot_options_task, called through
    region/relaunch.py's _relaunch_slot_options;
    region/observations/dispatch.py's _launch_observations_task), exercising
    the real eager-launch-plus-relaunch-closure shape a parsed RegionResult
    carries (see _launch_trailing_tasks' own docstring)."""
    slot_tasks: list[asyncio.Task] = []
    obs_tasks: list[asyncio.Task] = []
    real_launch_slot_options = _relaunch_mod._launch_slot_options_task
    real_launch_observations = _observations_dispatch_mod._launch_observations_task

    def _counting_launch_slot_options(*args, **kwargs):
        task = real_launch_slot_options(*args, **kwargs)
        slot_tasks.append(task)
        return task

    def _counting_launch_observations(*args, **kwargs):
        task = real_launch_observations(*args, **kwargs)
        obs_tasks.append(task)
        return task

    monkeypatch.setattr(_relaunch_mod, "_launch_slot_options_task", _counting_launch_slot_options)
    monkeypatch.setattr(
        _observations_dispatch_mod, "_launch_observations_task", _counting_launch_observations
    )

    entry = ResubmitParsedEntry(
        part=ParsedPart(variable_text="fertility rate", entities=[ParsedEntity(text="Kenya")])
    )
    request = ResolveRequest(
        input=DatasetResubmitInput(entries=[entry]),
        response_options=ResolveOptions(include_option_coverage=True, include_observations=True),
    )

    async def _drain() -> list[str]:
        graph = FakeGraph()
        kinds = [
            ev.kind
            async for ev in resolve_events(
                request,
                graph=graph,
                resolver=make_resolver(graph),
                llm=FakeLLM(),
                stream_slot_options=True,
                sdmx=FakeSdmxClient(),
            )
        ]
        return kinds

    kinds = asyncio.run(_drain())

    assert "response" in kinds
    assert len(slot_tasks) == 1, f"expected exactly one slot-options probe, got {len(slot_tasks)}"
    assert len(obs_tasks) == 1, f"expected exactly one observations probe, got {len(obs_tasks)}"
    assert slot_tasks[0].done(), "slot-options task left un-awaited after the request completed"
    assert obs_tasks[0].done(), "observations task left un-awaited after the request completed"


# ---------------------------------------------------------------------------
# 8. Refusal precedence: static (no I/O) before dynamic (needs a live
#    resolve), lowest entry index within each class. The placeholder `code`
#    registry check runs ahead of every entry kind's own task creation, so a
#    bad code fails before paying for any real graph calls or losing a race
#    to a lower-index entry's own I/O-bound refusal.
# ---------------------------------------------------------------------------


def test_static_placeholder_code_check_outranks_a_lower_index_dynamic_refusal():
    """Entry 0 is the same malformed identity
    test_malformed_entry_fails_whole_request_with_repointed_pointer pins --
    an unresolvable posted entity, discoverable only after
    resolve_spec_resubmit's own graph I/O. Entry 1 is a placeholder whose
    code is absent from CODE_CATALOG, discoverable with a plain dictionary
    lookup and no I/O at all. Static beats dynamic as a class regardless of
    index: the response must name entry 1's code refusal, not entry 0's
    identity refusal, even though entry 0 sits at the lower index."""
    entries = [
        _gdp_entry("country/ZZZ_NOT_A_PLACE", "GDP"),
        ResubmitPlaceholderEntry(
            variable_text="GDP in PPP dollars", status="unsupported", code="not_a_real_code"
        ),
    ]
    client = _client()
    resp = client.post("/api/search/resolve", json=_wire_request(entries))
    assert resp.status_code // 100 == 4
    body = resp.json()
    assert body["code"] == "request_invalid"
    assert body["pointer"] == "/input/entries/1/code"
    assert "not_a_real_code" in body["detail"]


def test_static_placeholder_code_check_picks_the_lowest_index_within_its_own_class():
    """Two placeholder entries, both with a bad code: within the static
    class, the lowest entry index wins -- the plain `enumerate` loop ahead
    of task creation checks entries in posted order and raises on the
    first bad one it finds."""
    entries = [
        ResubmitPlaceholderEntry(variable_text="a", status="unsupported", code="not_a_real_code_0"),
        ResubmitPlaceholderEntry(variable_text="b", status="unsupported", code="not_a_real_code_1"),
    ]
    client = _client()
    resp = client.post("/api/search/resolve", json=_wire_request(entries))
    assert resp.status_code // 100 == 4
    body = resp.json()
    assert body["pointer"] == "/input/entries/0/code"
    assert "not_a_real_code_0" in body["detail"]


def test_static_candidates_ceiling_check_outranks_a_lower_index_dynamic_refusal(monkeypatch):
    """Entry 0 is the same malformed identity
    test_malformed_entry_fails_whole_request_with_repointed_pointer pins --
    an unresolvable posted entity, discoverable only after
    resolve_spec_resubmit's own graph I/O. Entry 1 is a candidates entry
    posting more candidates than SEARCH_MAX_CANDIDATES, discoverable with a
    plain length check and no I/O at all -- the same static class the
    placeholder code check above belongs to
    (dataset_entries.check_candidates_ceiling, called from handle's own
    static validation pass, ahead of every entry kind's own task creation).
    Static beats dynamic as a class regardless of index: the response must
    name entry 1's candidates refusal, not entry 0's identity refusal, even
    though entry 0 sits at the lower index.

    `asyncio.to_thread` is patched to run its wrapped call inline rather
    than handing it to a real thread-pool executor: unpatched, ANY refusal
    reached only via `resolve_spec_resubmit` needs at least one real
    executor round trip to surface, so entry 0's task can never finish
    within the same event-loop tick as entry 1's check, which would leave
    this test unable to prove the precedence rule it pins. Inlined, entry
    0's task raises synchronously, which is what puts the two checks on a
    genuine race: `asyncio.create_task` schedules entry 0's task before
    entry 1's (posted order), so entry 0's synchronous raise would complete
    first absent the static-check ordering. Entry 1's check instead runs
    in handle's own plain `for` loop before either task is ever created, so
    it wins regardless of scheduling."""

    async def _inline_to_thread(func, /, *args, **kwargs):
        return func(*args, **kwargs)

    monkeypatch.setattr("asyncio.to_thread", _inline_to_thread)

    ethiopia = offline_resolve(
        make_request("population and birth rate in Ethiopia"),
        llm=KeyedBindReplyLLM(BIRTH_RATE_CANDIDATES),
    )
    birth_rate_part = ethiopia.parts[1]
    oversized = list(birth_rate_part.candidates.specs) * 4  # 8 candidates
    entries = [
        _gdp_entry("country/ZZZ_NOT_A_PLACE", "GDP"),
        _candidates_entry_from_part(
            birth_rate_part, candidates=oversized, variable_text="birth rate"
        ),
    ]
    client = _client()
    resp = client.post("/api/search/resolve", json=_wire_request(entries))
    assert resp.status_code // 100 == 4
    body = resp.json()
    assert body["code"] == "request_invalid"
    assert body["pointer"] == "/input/entries/1/candidates"


def test_dynamic_refusal_picks_lowest_index_even_when_it_finishes_last(monkeypatch):
    """Both entries fail dynamic validation -- the same unresolvable-entity
    refusal test_malformed_entry_fails_whole_request_with_repointed_pointer
    pins, discoverable only after resolve_spec_resubmit's own graph I/O --
    but entry 1's own promote task is made to finish before entry 0's. The
    response must still name entry 0: among the promote tasks' own
    completed EngineInputErrors, the lowest entry index wins, regardless of
    which one's own I/O landed first.

    `asyncio.to_thread` is patched to control completion order directly,
    unlike the inline patch the static-vs-dynamic tests above use: the
    first call into it (entry 0's, since asyncio.create_task schedules the
    promote tasks in posted order) sleeps past the second call (entry 1's)
    before running the real resolve_spec_resubmit, so entry 1's own
    EngineInputError is the one asyncio.gather's default
    (return_exceptions=False) would propagate first -- the exact race this
    test pins.
    """
    call_order: list[int] = []
    completion_order: list[int] = []

    async def _to_thread_ordered(func, /, *args, **kwargs):
        call_index = len(call_order)
        call_order.append(call_index)
        if call_index == 0:
            await asyncio.sleep(0.02)
        try:
            return func(*args, **kwargs)
        finally:
            # Both entries refuse, so `func` leaves by raising: recording the
            # order in a `finally` is what makes a raising promote count as
            # finished here.
            completion_order.append(call_index)

    monkeypatch.setattr("asyncio.to_thread", _to_thread_ordered)

    entries = [
        _gdp_entry("country/ZZZ_NOT_A_PLACE_0", "GDP0"),
        _gdp_entry("country/ZZZ_NOT_A_PLACE_1", "GDP1"),
    ]
    client = _client()
    resp = client.post("/api/search/resolve", json=_wire_request(entries))
    # Fails closed: the assertion below only discriminates while entry 1's
    # promote finishes first, so a change to the order create_task schedules
    # them in must break this test rather than let it pass vacuously.
    assert completion_order == [1, 0], (
        f"entry 0's promote was meant to finish last, got {completion_order}"
    )
    assert resp.status_code // 100 == 4
    body = resp.json()
    assert body["pointer"] == "/input/entries/0/identity/entities/observationAbout"


# ---------------------------------------------------------------------------
# 9. Region-lifetime owner: a future exit path cannot escape trailing-task
#    cleanup either, proven at two seams where a leak is easy to reintroduce.
# ---------------------------------------------------------------------------


@pytest.mark.unrecorded_requests
def test_region_assembly_failure_still_cancels_live_parsed_trailing_tasks(monkeypatch):
    """`_placeholder_region` is faulted directly here, standing in for any
    future exception the region-assembly loop could raise, to prove the
    request's own concurrency scope (`search.engine.concurrency`, opened by
    `core.resolve_events`) -- not a patch keyed to one specific raise site --
    cancels a parsed entry's already-live trailing tasks on any failure in
    that loop, regardless of which local variable in `handle`'s own body was
    holding the region.

    A real trailing probe against this repo's fake clients could complete
    on its own fast enough to mask a leak, so the launched slot-options/
    observations tasks are substituted with a long-running stand-in and
    asserted `cancelled()` -- proof of an actual cancel, not of eventual
    completion. The stand-ins register with `create_owned_task` (rather than
    a bare `asyncio.create_task`) so they reach the same request scope the
    real launchers do."""

    def _boom(entry, index):
        raise RuntimeError("boom")

    monkeypatch.setattr("search.engine.entry.dataset_resubmit._placeholder_region", _boom)

    slot_tasks: list[asyncio.Task] = []
    obs_tasks: list[asyncio.Task] = []

    def _stalling_launch_slot_options(*args, **kwargs):
        task = create_owned_task(asyncio.sleep(5.0))
        slot_tasks.append(task)
        return task

    def _stalling_launch_observations(*args, **kwargs):
        task = create_owned_task(asyncio.sleep(5.0))
        obs_tasks.append(task)
        return task

    monkeypatch.setattr(_relaunch_mod, "_launch_slot_options_task", _stalling_launch_slot_options)
    monkeypatch.setattr(
        _observations_dispatch_mod, "_launch_observations_task", _stalling_launch_observations
    )

    entries = [
        ResubmitParsedEntry(
            part=ParsedPart(variable_text="fertility rate", entities=[ParsedEntity(text="Kenya")])
        ),
        ResubmitPlaceholderEntry(
            variable_text="unemployment rate", status="no_data", reason="no_observations"
        ),
    ]
    request = ResolveRequest(
        input=DatasetResubmitInput(entries=entries),
        response_options=ResolveOptions(include_option_coverage=True, include_observations=True),
    )

    async def _drain() -> None:
        graph = FakeGraph()
        raised: BaseException | None = None
        try:
            async for _ev in resolve_events(
                request,
                graph=graph,
                resolver=make_resolver(graph),
                llm=FakeLLM(),
                stream_slot_options=True,
                sdmx=FakeSdmxClient(),
            ):
                pass
        except RuntimeError as exc:
            raised = exc
        assert raised is not None and str(raised) == "boom"
        assert slot_tasks, "expected the parsed entry to launch a slot-options probe"
        assert obs_tasks, "expected the parsed entry to launch an observations probe"
        for task in (*slot_tasks, *obs_tasks):
            assert task.cancelled(), f"trailing task leaked past the failing request: {task}"

    asyncio.run(_drain())


@pytest.mark.unrecorded_requests
def test_parsed_fanout_timeout_cancels_a_completed_sibling_parsed_trailing_tasks(
    monkeypatch, search_settings
):
    """`resolve_parsed_entries` itself raises `asyncio.TimeoutError` after
    `out.regions` is already populated -- one parsed leg (fertility rate)
    completed, its sibling (GDP growth) still pending -- but before it
    builds the {index: RegionResult} dict `handle`'s own
    `parsed_regions_by_index` would otherwise be populated from:
    `parsed_regions_by_index` stays `{}` on this failing exit, so the
    completed sibling's own live trailing tasks are reachable only through
    `owned_regions`, populated inside `resolve_parsed_entries` itself rather
    than from this function's return value.

    A real trailing probe against this repo's fake clients could finish on
    its own fast enough to mask the leak (see the previous test's own
    docstring on why), so the launched slot-options/observations tasks are
    substituted with a long-running stand-in here too, asserted
    `cancelled()` from inside the running loop -- a `finally`/except cleanup
    that fires during this generator's own teardown is invisible once
    `asyncio.run` has already returned and torn the loop down.

    The deadline passes on the fake clock once the completed leg has launched
    both stand-ins, so the leg's own work never races it. The request runs on
    an empty process code index, as after `tests/code_index/test_shared.py`
    reset it, so the leg parses the index (about 0.2 s) before it launches."""
    clock = frozen_monotonic_clock(monkeypatch)
    search_settings(request_timeout_s=DEADLINE_TIMEOUT_S)

    real_resolve_variable = _pipeline_mod.resolve_variable

    async def _resolve_variable_stall_second_leg(variable, *args, **kwargs):
        if variable == "GDP growth":
            await asyncio.sleep(5.0)
            raise AssertionError(
                "unreachable -- the deadline cancels this task first"
            )  # pragma: no cover
        return await real_resolve_variable(variable, *args, **kwargs)

    monkeypatch.setattr("search.engine.fanout.resolve_variable", _resolve_variable_stall_second_leg)

    slot_tasks: list[asyncio.Task] = []
    obs_tasks: list[asyncio.Task] = []

    def _pass_the_deadline_once_both_launched() -> None:
        if slot_tasks and obs_tasks:
            clock.now += DEADLINE_TIMEOUT_S * 10

    def _stalling_launch_slot_options(*args, **kwargs):
        task = asyncio.create_task(asyncio.sleep(5.0))
        slot_tasks.append(task)
        _pass_the_deadline_once_both_launched()
        return task

    def _stalling_launch_observations(*args, **kwargs):
        task = asyncio.create_task(asyncio.sleep(5.0))
        obs_tasks.append(task)
        _pass_the_deadline_once_both_launched()
        return task

    monkeypatch.setattr(_relaunch_mod, "_launch_slot_options_task", _stalling_launch_slot_options)
    monkeypatch.setattr(
        _observations_dispatch_mod, "_launch_observations_task", _stalling_launch_observations
    )

    entries = [
        ResubmitParsedEntry(
            part=ParsedPart(variable_text="fertility rate", entities=[ParsedEntity(text="Kenya")])
        ),
        ResubmitParsedEntry(
            part=ParsedPart(variable_text="GDP growth", entities=[ParsedEntity(text="Kenya")])
        ),
    ]
    request = ResolveRequest(
        input=DatasetResubmitInput(entries=entries),
        response_options=ResolveOptions(include_option_coverage=True, include_observations=True),
    )

    async def _drain() -> None:
        graph = FakeGraph()
        raised: BaseException | None = None
        try:
            async for _ev in resolve_events(
                request,
                graph=graph,
                resolver=make_resolver(graph),
                llm=FakeLLM(),
                stream_slot_options=True,
                sdmx=FakeSdmxClient(),
            ):
                pass
        except TimeoutError as exc:
            raised = exc
        assert raised is not None, "expected the stalled parsed leg to time the request out"
        assert slot_tasks, "expected the completed parsed leg to launch a slot-options probe"
        assert obs_tasks, "expected the completed parsed leg to launch an observations probe"
        for task in (*slot_tasks, *obs_tasks):
            assert task.cancelled(), f"trailing task leaked past the failing request: {task}"

    asyncio.run(_drain())


@pytest.mark.unrecorded_requests
def test_parsed_fanout_infra_error_cancels_a_completed_sibling_parsed_trailing_tasks(
    monkeypatch,
):
    """`_run_variables_streaming` itself raises the `EngineInfraError` a
    failing leg's `resolve_variable` call raised (`fanout._stream_one_variable`
    routes ANY `EngineInfraError` straight to a `_TaskFailed` marker, never
    degrading it to a no_data placeholder even under `degrade_on_error=True`)
    -- after every leg has finished, one (GDP growth) via that error, its
    sibling (fertility rate) via a real `RegionResult`. `_run_variables_streaming`
    raises before ever reaching its own `out.regions` assignment, so the
    completed sibling's already-live trailing tasks are reachable to no
    caller at all: the fan-out cancels them itself on that exit -- see its
    own docstring.

    A real trailing probe against this repo's fake clients could finish on
    its own fast enough to mask the leak (see the two tests above), so the
    launched slot-options/observations tasks are substituted with a
    long-running stand-in here too, asserted `cancelled()` from inside the
    running loop."""
    real_resolve_variable = _pipeline_mod.resolve_variable

    async def _resolve_variable_second_leg_infra_error(variable, *args, **kwargs):
        if variable == "GDP growth":
            raise GraphInfraError("simulated graph transport failure")
        return await real_resolve_variable(variable, *args, **kwargs)

    monkeypatch.setattr(
        "search.engine.fanout.resolve_variable", _resolve_variable_second_leg_infra_error
    )

    slot_tasks: list[asyncio.Task] = []
    obs_tasks: list[asyncio.Task] = []

    def _stalling_launch_slot_options(*args, **kwargs):
        task = asyncio.create_task(asyncio.sleep(5.0))
        slot_tasks.append(task)
        return task

    def _stalling_launch_observations(*args, **kwargs):
        task = asyncio.create_task(asyncio.sleep(5.0))
        obs_tasks.append(task)
        return task

    monkeypatch.setattr(_relaunch_mod, "_launch_slot_options_task", _stalling_launch_slot_options)
    monkeypatch.setattr(
        _observations_dispatch_mod, "_launch_observations_task", _stalling_launch_observations
    )

    entries = [
        ResubmitParsedEntry(
            part=ParsedPart(variable_text="fertility rate", entities=[ParsedEntity(text="Kenya")])
        ),
        ResubmitParsedEntry(
            part=ParsedPart(variable_text="GDP growth", entities=[ParsedEntity(text="Kenya")])
        ),
    ]
    request = ResolveRequest(
        input=DatasetResubmitInput(entries=entries),
        response_options=ResolveOptions(include_option_coverage=True, include_observations=True),
    )

    async def _drain() -> None:
        graph = FakeGraph()
        raised: BaseException | None = None
        try:
            async for _ev in resolve_events(
                request,
                graph=graph,
                resolver=make_resolver(graph),
                llm=FakeLLM(),
                stream_slot_options=True,
                sdmx=FakeSdmxClient(),
            ):
                pass
        except GraphInfraError as exc:
            raised = exc
        assert raised is not None, "expected the injected graph infra error to fail the request"
        assert slot_tasks, "expected the completed parsed leg to launch a slot-options probe"
        assert obs_tasks, "expected the completed parsed leg to launch an observations probe"
        for task in (*slot_tasks, *obs_tasks):
            assert task.cancelled(), f"trailing task leaked past the failing request: {task}"

    asyncio.run(_drain())


@pytest.mark.unrecorded_requests
def test_request_scope_cleanup_survives_a_self_cancellation_landing_on_the_first_await(
    monkeypatch,
):
    """The request's own concurrency scope (`search.engine.concurrency`,
    opened by `core.resolve_events`) cancels every registered task in one
    synchronous pass, then awaits them in a second, not interleaved (see
    `scoped_concurrency`'s own docstring). An interleaved cancel-then-await
    loop (cancel task N, await task N, cancel task N+1, await task N+1, ...)
    composes fine against an ordinary raise, but not against the one exit
    the request controls least: the client disconnecting mid-stream, which
    cancels the task driving this generator. That cancellation is delivered
    at the very next `await` -- which, in an interleaved loop, is the
    awaiting half of the first task's own cleanup. It lands there, aborts
    the loop, and every later task leaks.

    Two parsed entries both resolve `definite` here (no stall needed), so
    both regions carry LIVE slot_options_task/observations_task handles,
    each registered with the request scope the moment it was created,
    before the door ever reaches its combiner. `_stream_combined_regions` --
    the next thing `handle` calls -- is replaced with a stand-in that
    self-cancels the current task (simulating the disconnect) and then
    raises synchronously, so the cancellation is pending, undelivered, at
    the moment the scope's own cleanup starts: the very scenario this test
    exercises.

    Against an interleaved loop, the pending cancellation lands inside the
    first task's own awaiting half, leaving the other three still running.
    The cancel-first pass is pure `task.cancel()` calls with no `await` in
    it at all, so it cannot be interrupted -- every one of the four stand-in
    tasks is told to stop before the (now separately interruptible) awaiting
    pass even starts.

    Asserted from inside the running loop, after draining the self-delivered
    cancellation and giving the event loop one more turn to let the
    now-cancelled stand-in tasks actually finish transitioning to
    `cancelled()`: a `finally`/except cleanup that fires during this
    generator's own teardown is invisible once `asyncio.run` has already
    returned and torn the loop down."""
    slot_tasks: list[asyncio.Task] = []
    obs_tasks: list[asyncio.Task] = []

    def _stalling_launch_slot_options(*args, **kwargs):
        task = create_owned_task(asyncio.sleep(5.0))
        slot_tasks.append(task)
        return task

    def _stalling_launch_observations(*args, **kwargs):
        task = create_owned_task(asyncio.sleep(5.0))
        obs_tasks.append(task)
        return task

    monkeypatch.setattr(_relaunch_mod, "_launch_slot_options_task", _stalling_launch_slot_options)
    monkeypatch.setattr(
        _observations_dispatch_mod, "_launch_observations_task", _stalling_launch_observations
    )

    def _boom_stream_combined_regions(*args, **kwargs):
        # Simulates a client disconnect landing right here: the current
        # task's own cancellation becomes pending (no CancelledError is
        # thrown until the next await), then this raises synchronously, so
        # the door-level `except BaseException` block starts its cleanup
        # with that cancellation still undelivered.
        asyncio.current_task().cancel()
        raise RuntimeError("boom")

    monkeypatch.setattr(
        "search.engine.entry.dataset_resubmit._stream_combined_regions",
        _boom_stream_combined_regions,
    )

    entries = [
        ResubmitParsedEntry(
            part=ParsedPart(variable_text="fertility rate", entities=[ParsedEntity(text="Kenya")])
        ),
        ResubmitParsedEntry(
            part=ParsedPart(variable_text="population", entities=[ParsedEntity(text="Kenya")])
        ),
    ]
    request = ResolveRequest(
        input=DatasetResubmitInput(entries=entries),
        response_options=ResolveOptions(include_option_coverage=True, include_observations=True),
    )

    async def _drain() -> None:
        graph = FakeGraph()
        raised: BaseException | None = None
        try:
            async for _ev in resolve_events(
                request,
                graph=graph,
                resolver=make_resolver(graph),
                llm=FakeLLM(),
                stream_slot_options=True,
                sdmx=FakeSdmxClient(),
            ):
                pass
        except asyncio.CancelledError as exc:
            raised = exc
        assert raised is not None, "expected the self-cancellation to surface as CancelledError"
        assert len(slot_tasks) == 2, f"expected two slot-options probes, got {len(slot_tasks)}"
        assert len(obs_tasks) == 2, f"expected two observations probes, got {len(obs_tasks)}"
        # One extra turn of the event loop for the now-`cancel()`ed stand-in
        # tasks to actually finish raising CancelledError internally and
        # reach `cancelled() == True` -- calling `.cancel()` only requests
        # it.
        await asyncio.gather(*slot_tasks, *obs_tasks, return_exceptions=True)
        for task in (*slot_tasks, *obs_tasks):
            assert task.cancelled(), f"trailing task leaked past the failing request: {task}"

    asyncio.run(_drain())


# ---------------------------------------------------------------------------
# 10. The candidates shrink rule's one-survivor and zero-survivor branches
#     (dataset_entries._build_candidates_region), otherwise uncovered.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("repeats", "resolves"),
    [
        pytest.param(
            0, "2 candidates were posted; 1 still resolves to a definite answer", id="one"
        ),
        pytest.param(
            1, "3 candidates were posted; 2 still resolve and rebuild to 1 reading", id="repeated"
        ),
    ],
)
def test_build_candidates_region_one_survivor_collapses_to_definite(repeats: int, resolves: str):
    """A candidate set where exactly one reading survives to `definite`
    collapses to that survivor's own RegionResult, re-stamped with the
    entry's own variable_text/timing/earliest_index -- not a one-element
    `candidates` part, which CandidateSet.specs' own min_length=2 has no
    wire shape for. A repeat of the survivor counts as resolving and
    rebuilds to the same reading."""
    ethiopia = offline_resolve(
        make_request("population and birth rate in Ethiopia"),
        llm=KeyedBindReplyLLM(BIRTH_RATE_CANDIDATES),
    )
    birth_rate_part = ethiopia.parts[1]
    survivor_spec = _spec_with_sv(birth_rate_part, _CRUDE_BIRTH_RATE)
    other_spec = _spec_with_sv(birth_rate_part, _FERTILITY_RATE)
    entry = _candidates_entry_from_part(
        birth_rate_part,
        candidates=[survivor_spec] * (1 + repeats) + [other_spec],
        variable_text="birth rate",
        separate=True,
    )
    survivor = RegionResult(
        variable_text="ignored",
        status="definite",
        specs=(survivor_spec,),
        no_data_reason=None,
        warnings=(),
        timing_by_step={"recall": 5},
    )
    promoted = [
        survivor,
        *[dataclasses.replace(survivor, timing_by_step={}) for _ in range(repeats)],
        RegionResult(
            variable_text="ignored",
            status="no_data",
            specs=(),
            no_data_reason="no_observations",
            warnings=(),
            timing_by_step={"recall": 3},
        ),
    ]

    region = _build_candidates_region(entry, promoted, index=2)

    assert region.status == "definite"
    assert region.spec is survivor_spec
    assert region.variable_text == "birth rate"
    assert region.earliest_index == 2
    assert region.separate is True
    assert region.timing_by_step == {"recall": 8}
    warning = next(w for w in region.warnings if w.code == "candidates_collapsed_definite")
    assert warning.message == (
        f"{resolves}, so the part is reported as that answer and not as the posted candidate set."
    )


def test_build_candidates_region_zero_survivors_collapses_to_no_data():
    """A candidate set where every promote fails to `definite` collapses to
    `no_data`, preferring the first (posted-order) promote's own reason when
    it has one -- CandidateSet.specs' own min_length=2 has no
    "unsupported candidates" wire shape to fall back to either, so a mix of
    no_data/unsupported promotes collapses the same way regardless of which
    reason(s) the individual promotes carried."""
    ethiopia = offline_resolve(
        make_request("population and birth rate in Ethiopia"),
        llm=KeyedBindReplyLLM(BIRTH_RATE_CANDIDATES),
    )
    birth_rate_part = ethiopia.parts[1]
    spec_a = _spec_with_sv(birth_rate_part, _CRUDE_BIRTH_RATE)
    spec_b = _spec_with_sv(birth_rate_part, _FERTILITY_RATE)
    entry = _candidates_entry_from_part(
        birth_rate_part,
        candidates=[spec_a, spec_b],
        variable_text="birth rate",
        separate=True,
    )
    promoted = [
        RegionResult(
            variable_text="ignored",
            status="no_data",
            specs=(),
            no_data_reason="no_observations",
            warnings=(),
            timing_by_step={},
        ),
        RegionResult(
            variable_text="ignored",
            status="unsupported",
            specs=(),
            no_data_reason=None,
            warnings=(),
            timing_by_step={},
            unsupported=Unsupported(
                code="unit_request_unsupported", pointer="", detail="x", definition={}
            ),
        ),
    ]

    region = _build_candidates_region(entry, promoted, index=1)

    assert region.status == "no_data"
    assert region.no_data_reason == "no_observations"
    assert region.specs == ()
    assert region.unsupported is None
    assert region.separate is True
    warning = next(w for w in region.warnings if w.code == "candidates_collapsed_no_data")
    assert warning.message == (
        "2 candidates were posted; none resolves, so the part is reported as no_data "
        "and not as the posted candidate set."
    )


@pytest.mark.unrecorded_requests
def test_identity_resubmit_round_trips_comparison_without_folding():
    """A parsed comparison stays separate through identity-only resubmits."""
    initial = offline_resolve(
        ResolveRequest(
            input=ParsedInput(
                parts=[
                    ParsedPart(
                        variable_text="fertility rate",
                        entities=[ParsedEntity(text="Kenya")],
                    ),
                    ParsedPart(
                        variable_text="fertility rate",
                        entities=[ParsedEntity(text="Nigeria")],
                        separate=True,
                    ),
                ]
            )
        )
    )
    assert [part.separate for part in initial.parts] == [False, True]

    def resubmit(parts):
        return offline_resolve(
            ResolveRequest(
                input=DatasetResubmitInput(
                    entries=[
                        _identity_entry_from_spec(
                            part.spec, part.variable_text, separate=part.separate
                        )
                        for part in parts
                    ]
                )
            )
        )

    first = resubmit(initial.parts)
    second = resubmit(first.parts)
    for result in (first, second):
        assert [part.separate for part in result.parts] == [False, True]
        assert result.parts[0].spec.spec_id != result.parts[1].spec.spec_id
        assert not any(
            warning.code == "binding_folded" for part in result.parts for warning in part.warnings
        )


# ---------------------------------------------------------------------------
# Scope entries: an identity entry posting scope_entries compiles them over its base.
# ---------------------------------------------------------------------------

_SOURCE_A = "sector:1"
_SOURCE_B = "sector:2"


def _activity_source_rule(base, record, facts) -> AxisTable:
    """A synthetic axis over the GDP identity's activitySource slot, which the base holds
    on its first option."""
    return AxisTable(
        axis="sector",
        rule="test_activity_source",
        options=tuple(
            Option(
                option_id=option_id,
                label=option_id,
                binding=SlotBinding(slot="activitySource", state="value", refs=[ref]),
                availability=OptionAvailability(status="available"),
            )
            for option_id, ref in ((_SOURCE_A, "SourceA"), (_SOURCE_B, "SourceB"))
        ),
        preference=(_SOURCE_A, _SOURCE_B),
        held=_SOURCE_A,
    )


def _scope_entry(option_id: str, *, choice_id: str = "c") -> ScopeEntry:
    return ScopeEntry(
        axis="sector",
        option_id=option_id,
        provenance="explicit",
        origin=ChoiceOrigin(choice_id=choice_id),
    )


def _scoped_entry(identity, scope_entries) -> ResubmitIdentityEntry:
    return ResubmitIdentityEntry(
        variable_text="GDP", identity=identity, scope_entries=scope_entries
    )


@pytest.mark.parametrize("live_axis", [False, True], ids=["no-live-axis", "synthetic-axis"])
def test_a_served_identity_resubmitted_with_its_own_scope_entries_keeps_its_spec_id(
    live_axis, monkeypatch
):
    """The no-op switch: a served identity posted as its own base with the entries that
    settled it resolves to the same spec_id, and the part states its scope."""
    if live_axis:
        monkeypatch.setattr(scope_rules, "AXIS_RULES", {"sector": _activity_source_rule})
    first = offline_resolve(
        ResolveRequest(input=DatasetResubmitInput(entries=[_gdp_entry(_INDIA, "GDP")]))
    )
    served = first.parts[0].spec
    entries = [_scope_entry(_SOURCE_A, choice_id=f"{served.spec_id}:sector")] if live_axis else []

    again = offline_resolve(
        ResolveRequest(
            input=DatasetResubmitInput(entries=[_scoped_entry(served.definition.identity, entries)])
        )
    )

    spec = again.parts[0].spec
    scope = spec.resolved.scope_choices
    assert served.resolved.scope_choices is None
    assert spec.spec_id == served.spec_id
    assert scope is not None
    assert (scope.id, scope.base.spec_id, scope.compiled.spec_id) == (served.spec_id,) * 3
    assert [(c.axis, c.active, c.provenance) for c in scope.choices] == (
        [("sector", _SOURCE_A, "explicit")] if live_axis else []
    )
    assert scope.entries == entries


@pytest.mark.parametrize(
    ("live_axis", "scope_entries", "code", "pointer"),
    [
        pytest.param(
            False,
            [_scope_entry("sector:120+130")],
            "requested_option_unavailable",
            "/input/entries/0/scope_entries/0",
            id="axis-not-settled",
        ),
        pytest.param(
            True,
            [_scope_entry("sector:3")],
            "requested_option_unavailable",
            "/input/entries/0/scope_entries/0",
            id="option-not-offered",
        ),
        pytest.param(
            True,
            [_scope_entry(_SOURCE_A), _scope_entry(_SOURCE_B)],
            "conflicting_requested_options",
            "/input/entries/0/scope_entries/1",
            id="two-options-on-one-axis",
        ),
    ],
)
def test_scope_entries_a_part_cannot_take_refuse_422_at_the_entry(
    live_axis, scope_entries, code, pointer, monkeypatch
):
    if live_axis:
        monkeypatch.setattr(scope_rules, "AXIS_RULES", {"sector": _activity_source_rule})
    identity = _gdp_entry(_INDIA, "GDP").identity

    resp = _client().post(
        "/api/search/resolve", json=_wire_request([_scoped_entry(identity, scope_entries)])
    )

    body = resp.json()
    assert (resp.status_code, body["code"], body["pointer"]) == (422, code, pointer)
    assert "parts" not in body
