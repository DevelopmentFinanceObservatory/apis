"""Integration tests for per-variable parts, fan-out, observations, and timing."""

from __future__ import annotations

import threading
from datetime import date
from unittest.mock import patch

from fastapi.testclient import TestClient

from search.engine.assemble import OBSERVATIONS_OMITTED
from search.engine.conjoin import BINDING_FOLDED, NESTED_SHAPE_STEPS
from search.engine.extract import _EXTRACTION_SYSTEM_PROMPT, Extraction
from search.engine.quantity_read import QuantityOutput
from search.engine.stated_words_read import StatedWordsOutput
from search.models import (
    CandidatesPart,
    DatasetResubmitInput,
    DefinitePart,
    NoDataPart,
    RawTextInput,
    ResolveOptions,
    ResolveRequest,
    ResubmitIdentityEntry,
)
from tests.engine._axis_doubles import TOTAL_POPULATION_INDIA_FAMILY, axis_confirming_reader
from tests.engine._binders import BIRTH_RATE_CANDIDATES, KeyedBindReplyLLM, elected_reply
from tests.engine._fake_sdmx import FakeSdmxClient
from tests.engine._harness import (
    PINNED_DATE,
    make_request,
    materialise_double_from_obs,
    offline_resolve,
    parse_sse,
)
from tests.engine._quantity import NO_QUANTITY, NO_RATING
from tests.engine._search_app import search_app
from tests.engine.test_availability import TestMemberProbeRunsNearestReal
from tests.fixtures import FakeGraph, FakeLLM, _load_json, make_resolver
from tests.fixtures.keys import fixture_key
from tests.materialise.conftest import FakeGraphReader


def _extract_key(query: str) -> str:
    return fixture_key(Extraction, _EXTRACTION_SYSTEM_PROMPT, query)


def _llm_with_extraction(query: str, *, variables: list[str], entities: list[str]) -> FakeLLM:
    """Add an extraction fixture while retaining recorded bind and confirm replies."""
    return FakeLLM(
        responses={
            **_load_json("llm_responses.json"),
            _extract_key(query): {
                "kind": "resolution",
                "variables": variables,
                "entities": [{"text": e} for e in entities],
                "dates": [],
            },
        }
    )


# ---------------------------------------------------------------------------
# Three-variable query, middle variable fails
# ---------------------------------------------------------------------------

_MIDDLE_FAILS_QUERY = "population and left-handedness and birth rate in Ethiopia"
_MIDDLE_FAILS_VARIABLES = ["population", "left-handedness", "birth rate"]


def test_three_variable_query_middle_fails():
    """The missing detect entry makes the middle leg fail independently."""
    fake_llm = _llm_with_extraction(
        _MIDDLE_FAILS_QUERY, variables=_MIDDLE_FAILS_VARIABLES, entities=["Ethiopia"]
    )
    result = offline_resolve(make_request(_MIDDLE_FAILS_QUERY), llm=fake_llm)

    assert result.status == "mixed"
    assert len(result.parts) == 3
    assert [part.variable_text for part in result.parts] == _MIDDLE_FAILS_VARIABLES

    population, left_handedness, birth_rate = result.parts
    assert population.status == "definite"
    assert isinstance(left_handedness, NoDataPart)
    assert left_handedness.no_data.reason == "variable_not_resolved"
    assert left_handedness.rendered_sentence is not None
    assert isinstance(left_handedness.warnings, list)
    # Birth rate in Ethiopia answers candidates on the shared fixture bank. The test checks
    # only that the part surfaced, whatever its exact status.
    assert birth_rate.status in ("candidates", "definite", "no_data", "unsupported")


def test_clamp_truncates_parts_to_the_unique_clamped_list():
    """Seven variables clamp to six parts in input order."""
    query = "test clamp parts seven variables in Brazil"
    seven = ["v0", "v1", "v2", "v3", "v4", "v5", "v6"]
    fake_llm = FakeLLM(
        responses={
            _extract_key(query): {
                "kind": "resolution",
                "variables": seven,
                "entities": [{"text": "Brazil"}],
                "dates": [],
            }
        }
    )
    fake_graph = FakeGraph(detect={}, resolve={"Brazil": "country/BRA"})
    result = offline_resolve(make_request(query), graph=fake_graph, llm=fake_llm)

    assert len(result.parts) == 6
    assert [part.variable_text for part in result.parts] == seven[:6]


# ---------------------------------------------------------------------------
# Zero-variable extraction: one no_data part, variable_text is the whole query
# ---------------------------------------------------------------------------


def test_no_variable_extracted_yields_one_part_with_the_whole_query_as_variable_text():
    """A resolution naming an operation and no variable produces one `no_data` part with
    the query text. With no operation either it would answer as a catalogue."""
    query = "rank asdfghjkl qwertyuiop no variable here"
    fake_llm = FakeLLM(
        responses={
            _extract_key(query): {
                "kind": "resolution",
                "variables": [],
                "entities": [],
                "operations": [{"verb": "rank", "variable": 0, "phrase": "rank", "over": None}],
                "dates": [],
            }
        }
    )
    result = offline_resolve(make_request(query), llm=fake_llm)

    assert result.status == "no_data"
    assert len(result.parts) == 1
    part = result.parts[0]
    assert isinstance(part, NoDataPart)
    assert part.no_data.reason == "variable_not_resolved"
    assert part.variable_text == query.strip()


# ---------------------------------------------------------------------------
# Same-shape collapse across two distinct variable phrasings, through the
# real fan-out (not combine_regions called directly)
# ---------------------------------------------------------------------------

_SAME_SHAPE_QUERY = "population and total population in Brazil"
_SAME_SHAPE_VARIABLES = ["population", "total population"]


class _FixedShapeBindLLM:
    """Extraction from a small scripted dict; elects the sole candidate shape
    with no constraint bindings for the universal choose+bind call, exactly
    the trick test_conjunction.py's own _DedupLLM uses. Count_Person carries
    no constraint slots, so a fixed zero-binding election is always correct
    here regardless of which of the two identical-shape variables is asked."""

    def __init__(self, extraction_responses: dict):
        self._fake = FakeLLM(responses=extraction_responses)

    async def generate_structured(self, *, prompt, system, schema):
        if schema.__name__ == "_UniversalBindOutput":
            return elected_reply(0, also_plausible=None), None
        if schema is QuantityOutput:
            return NO_QUANTITY
        if schema is StatedWordsOutput:
            return NO_RATING
        return await self._fake.generate_structured(prompt=prompt, system=system, schema=schema)


def test_two_identical_shapes_remain_unfolded_when_collapse_refuses_them_through_fanout():
    """Distinct phrases resolve to one shape but remain separate without differing values."""
    fake_llm = _FixedShapeBindLLM(
        {
            _extract_key(_SAME_SHAPE_QUERY): {
                "kind": "resolution",
                "variables": _SAME_SHAPE_VARIABLES,
                "entities": [{"text": "Brazil"}],
                "dates": [],
            }
        }
    )
    fake_graph = FakeGraph(
        detect={
            "population": {"svs": ["Count_Person"]},
            "total population": {"svs": ["Count_Person"]},
        },
        resolve={"Brazil": "country/BRA"},
    )
    result = offline_resolve(make_request(_SAME_SHAPE_QUERY), graph=fake_graph, llm=fake_llm)

    assert result.status == "definite"
    assert len(result.parts) == 2
    first, second = result.parts
    assert isinstance(first, DefinitePart)
    assert isinstance(second, DefinitePart)
    assert first.spec.spec_id == second.spec.spec_id

    first_codes = [w.code for w in first.warnings]
    second_codes = [w.code for w in second.warnings]
    assert BINDING_FOLDED not in first_codes
    assert BINDING_FOLDED not in second_codes


# ---------------------------------------------------------------------------
# Candidates + observations_omitted alongside a definite part carrying real
# observations, in one mixed response
# ---------------------------------------------------------------------------

_MIXED_OBS_QUERY = "population and birth rate in Ethiopia"

_ETHIOPIA_POP_OBS = {
    "Count_Person|country/ETH": [
        {
            "earliestDate": "2020",
            "latestDate": "2022",
            "obsCount": 3,
            "provenanceId": "dc/base/WorldPopulationProspects",
            "observations": [
                {"date": "2020", "value": 210000000.0},
                {"date": "2021", "value": 211000000.0},
                {"date": "2022", "value": 212000000.0},
            ],
        }
    ],
}


def test_candidates_part_carries_observations_omitted_alongside_definite_with_observations():
    """The definite part has observations; the ambiguous part reports `observations_omitted`."""
    default_obs = _load_json("graph_obs.json")
    graph = FakeGraph(obs={**default_obs, **_ETHIOPIA_POP_OBS})
    materialise_graph, materialise_sdmx = materialise_double_from_obs(_ETHIOPIA_POP_OBS)

    request = ResolveRequest(
        input=RawTextInput(query=_MIXED_OBS_QUERY),
        response_options=ResolveOptions(include_observations=True),
    )
    result = offline_resolve(
        request,
        graph=graph,
        llm=KeyedBindReplyLLM(BIRTH_RATE_CANDIDATES),
        materialise_graph=materialise_graph,
        materialise_sdmx=materialise_sdmx,
    )

    assert result.status == "mixed"
    population, birth_rate = result.parts
    assert isinstance(population, DefinitePart)
    assert population.spec.resolved.observations is not None
    assert population.spec.resolved.observations.total_obs == 3

    assert isinstance(birth_rate, CandidatesPart)
    birth_rate_codes = [w.code for w in birth_rate.warnings]
    assert OBSERVATIONS_OMITTED in birth_rate_codes
    for spec in birth_rate.candidates.specs:
        assert spec.resolved.observations is None


def _mixed_obs_app(graph, materialise_graph, materialise_sdmx):
    return search_app(
        graph=graph,
        resolver=make_resolver(graph),
        llm=KeyedBindReplyLLM(
            BIRTH_RATE_CANDIDATES,
            delegate=FakeLLM(
                responses={
                    **_load_json("llm_responses.json"),
                    _extract_key(_MIXED_OBS_QUERY): {
                        "kind": "resolution",
                        "variables": ["population", "birth rate"],
                        "entities": [{"text": "Ethiopia"}],
                        "dates": [],
                    },
                }
            ),
        ),
        sdmx=FakeSdmxClient(observations={}, availability={}),
        materialise_graph=materialise_graph,
        materialise_sdmx=materialise_sdmx,
    )


def _post_mixed_sse(app) -> list[tuple[str, dict]]:
    client = TestClient(app, raise_server_exceptions=False)
    with patch("search.engine.extract.date") as mock_date:
        mock_date.today.return_value = PINNED_DATE
        mock_date.side_effect = lambda *a, **kw: date(*a, **kw)
        resp = client.post(
            "/api/search/resolve",
            json={
                "input": {"kind": "raw_text", "query": _MIXED_OBS_QUERY},
                "response_options": {"include_observations": True},
            },
            headers={"Accept": "text/event-stream"},
        )
    assert resp.status_code == 200
    return parse_sse(resp.text)


def test_sse_candidates_part_carries_observations_omitted_alongside_definite_with_observations():
    """#588: SSE retains the candidates warning while streaming the definite observations."""
    default_obs = _load_json("graph_obs.json")
    graph = FakeGraph(obs={**default_obs, **_ETHIOPIA_POP_OBS})
    materialise_graph, materialise_sdmx = materialise_double_from_obs(_ETHIOPIA_POP_OBS)
    frames = _post_mixed_sse(_mixed_obs_app(graph, materialise_graph, materialise_sdmx))

    event_types = [ev for ev, _ in frames]
    response_idx = event_types.index("response")
    done_idx = event_types.index("done")
    obs_idxs = [i for i, ev in enumerate(event_types) if ev == "observations"]
    assert obs_idxs, "expected the definite part's trailing observations event"
    assert all(response_idx < i < done_idx for i in obs_idxs), (
        f"observations events must fall strictly between response ({response_idx}) "
        f"and done ({done_idx}); got observations at {obs_idxs}"
    )

    response_payload = frames[response_idx][1]["response"]
    parts = response_payload["parts"]
    population, birth_rate = parts
    assert population["status"] == "definite"
    assert birth_rate["status"] == "candidates"
    birth_rate_codes = [w["code"] for w in birth_rate["warnings"]]
    assert OBSERVATIONS_OMITTED in birth_rate_codes

    # The candidates part's specs carry no preview on either transport, and the
    # definite part's own `response` payload defers it -- both payloads agree
    # with the buffered body's split.
    for spec in birth_rate["candidates"]["specs"]:
        assert "observations" not in spec["resolved"]
    assert "observations" not in population["spec"]["resolved"]

    # The trailing event for the definite part carries the real series.
    populated = [frames[i][1] for i in obs_idxs if frames[i][1]["observations"]["series"]]
    assert populated, "expected a trailing observations event carrying populated series"
    observations = populated[0]["observations"]
    assert observations["total_obs"] == 3
    assert observations["truncated"] is False
    assert observations["series"][0]["points"][0]["value"] == 210000000.0
    assert frames[obs_idxs[0]][1]["spec_id"] == population["spec"]["spec_id"]


# ---------------------------------------------------------------------------
# nearest_real: the relaxed-spec observation fetch runs only on the buffered
# path. SSE has no trailing observations event for a nearest_real spec, so
# asking for a preview must not put a graph/SDMX read inline mid-stream.
# ---------------------------------------------------------------------------


class _CountingReader(FakeGraphReader):
    """Record the preview's facet-policy reads by StatVar and axis across worker threads.
    The preview reads its plan's candidates, so a policy read marks a dispatched pair."""

    def __init__(self, reader: FakeGraphReader) -> None:
        super().__init__(
            facets=reader._facets,
            policies=reader._policies,
            labels=reader._labels,
            arcs=reader._arcs,
            subjects=reader._subjects,
        )
        self.observation_probes: list[tuple[str, str]] = []
        self._probe_lock = threading.Lock()

    def default_facet_policy(self, *, stat_var, axis):
        with self._probe_lock:
            self.observation_probes.append((stat_var.removeprefix("dcid:"), axis))
        return super().default_facet_policy(stat_var=stat_var, axis=axis)


class _ExtractThenGenderLLM:
    """Extract from recordings and bind gender to exercise nearest-real relaxation."""

    def __init__(self, query: str) -> None:
        self.bound_schemas: list[str] = []
        self._inner = FakeLLM(
            responses={
                **_load_json("llm_responses.json"),
                _extract_key(query): {
                    "kind": "resolution",
                    "variables": ["population"],
                    "entities": [{"text": "Zeta"}],
                    "dates": [],
                },
            }
        )
        self._gender = TestMemberProbeRunsNearestReal._GenderBoundLLM()

    async def generate_structured(self, *, prompt, system, schema):
        if schema.__name__ == "Extraction":
            return await self._inner.generate_structured(
                prompt=prompt, system=system, schema=schema
            )
        self.bound_schemas.append(schema.__name__)
        return await self._gender.generate_structured(prompt=prompt, system=system, schema=schema)


def _no_obs_app_and_reader(query: str):
    """Build an app whose only observation is the ungendered aggregate.

    Return its reader so tests can detect relaxed-spec preview probes.
    """
    # Resolve and the preview read the same facet, as both read one graph.
    obs = {
        f"{TestMemberProbeRunsNearestReal._AGGREGATE_SV}|"
        f"{TestMemberProbeRunsNearestReal._RECIPIENT}": [
            {
                "facetId": "f0",
                "earliestDate": "2020",
                "latestDate": "2020",
                "obsCount": 1,
                "provenanceId": "dc/base/TestNearestRealProvenance",
                "observations": [{"date": "2020", "value": 137.0}],
            }
        ]
    }
    graph = TestMemberProbeRunsNearestReal._graph(obs=obs)
    reader, materialise_sdmx = materialise_double_from_obs(obs)
    counting = _CountingReader(reader)
    llm = _ExtractThenGenderLLM(query)
    app = search_app(
        graph=graph,
        resolver=make_resolver(graph),
        llm=llm,
        sdmx=FakeSdmxClient(observations={}, availability={}),
        materialise_graph=counting,
        materialise_sdmx=materialise_sdmx,
    )
    return app, counting, llm


def _post_no_obs(app, query: str, *, sse: bool):
    client = TestClient(app, raise_server_exceptions=False)
    return client.post(
        "/api/search/resolve",
        json={
            "input": {"kind": "raw_text", "query": query},
            "response_options": {"include_observations": True},
        },
        headers={"Accept": "text/event-stream"} if sse else {},
    )


def test_nearest_real_observation_fetch_is_buffered_only():
    """#588: only buffered responses fetch observations for `nearest_real`."""
    query = "female population of zeta"

    sse_app, sse_reader, sse_llm = _no_obs_app_and_reader(query)
    sse_resp = _post_no_obs(sse_app, query, sse=True)
    assert sse_resp.status_code == 200
    assert "_UniversalBindOutput" in sse_llm.bound_schemas, (
        "expected the bind stage to have run, so the member-probe branch and its "
        f"relaxation probe were genuinely reached; saw {sse_llm.bound_schemas}"
    )
    sse_frames = parse_sse(sse_resp.text)
    sse_part = next(
        payload["response"]["parts"][0] for ev, payload in sse_frames if ev == "response"
    )
    assert sse_part["status"] == "no_data"
    assert sse_part["no_data"]["nearest_real"], (
        "expected the relaxation to find the aggregate, so the SSE leg has a "
        "nearest_real spec whose fetch would have run"
    )
    _aggregate = TestMemberProbeRunsNearestReal._AGGREGATE_SV
    sse_reads = [probe for probe in sse_reader.observation_probes if probe[0] == _aggregate]
    assert sse_reads == [], (
        f"expected no observation probe for the relaxed spec's own stat var under "
        f"SSE; got {sse_reads} out of every probe {sse_reader.observation_probes}"
    )

    buffered_app, buffered_reader, _buffered_llm = _no_obs_app_and_reader(query)
    buffered_resp = _post_no_obs(buffered_app, query, sse=False)
    assert buffered_resp.status_code == 200
    buffered_part = buffered_resp.json()["parts"][0]
    assert buffered_part["status"] == "no_data"
    assert buffered_part["no_data"]["nearest_real"]
    buffered_reads = [
        probe for probe in buffered_reader.observation_probes if probe[0] == _aggregate
    ]
    assert buffered_reads, (
        f"expected the buffered transport to probe the relaxed spec's own stat var; "
        f"saw every probe {buffered_reader.observation_probes}"
    )
    assert buffered_part["no_data"]["nearest_real"][0]["resolved"]["observations"]["total_obs"] == 1


# ---------------------------------------------------------------------------
# N=1: parts[0] carries the envelope's own fields directly; envelope status
# == part status
# ---------------------------------------------------------------------------


def test_single_variable_parts0_equals_envelope():
    result = offline_resolve(make_request("total population India"))
    assert len(result.parts) == 1
    part = result.parts[0]
    assert result.status == part.status
    assert isinstance(part, DefinitePart)
    assert part.variable_text == "total population"


# ---------------------------------------------------------------------------
# a resubmitted identity: one part, the entry's own variable_text echoed
# ---------------------------------------------------------------------------


_IND_OBS = {
    "Count_Person|country/IND": [
        {
            "earliestDate": "2020",
            "latestDate": "2022",
            "obsCount": 3,
            "provenanceId": "dc/base/WorldPopulationProspects",
            "observations": [
                {"date": "2020", "value": 100.0},
                {"date": "2021", "value": 101.0},
                {"date": "2022", "value": 102.0},
            ],
        }
    ],
}


def test_resubmit_part_echoes_the_entry_variable_text():
    """Axis confirmation uses the total-population India family."""
    graph = FakeGraph(obs=_IND_OBS)
    inner_graph, materialise_sdmx = materialise_double_from_obs(_IND_OBS)
    materialise_graph = axis_confirming_reader(*TOTAL_POPULATION_INDIA_FAMILY, inner=inner_graph)

    base = offline_resolve(
        make_request("total population India"),
        graph=graph,
        materialise_graph=materialise_graph,
        materialise_sdmx=materialise_sdmx,
    ).parts[0]
    assert base.status == "definite"
    spec = base.spec

    resubmit_req = ResolveRequest(
        input=DatasetResubmitInput(
            entries=[
                ResubmitIdentityEntry(
                    variable_text="total population of India",
                    identity=spec.definition.identity,
                )
            ]
        ),
    )
    result = offline_resolve(
        resubmit_req,
        graph=graph,
        materialise_graph=materialise_graph,
        materialise_sdmx=materialise_sdmx,
    )

    assert len(result.parts) == 1
    assert result.parts[0].variable_text == "total population of India", (
        "expected the identity entry's posted variable_text to be echoed onto "
        f"the rebuilt part; got {result.parts[0].variable_text!r}"
    )


# ---------------------------------------------------------------------------
# Per-part timing_ms.by_step vs the envelope's shared-step timing_ms
# ---------------------------------------------------------------------------


def test_per_part_timing_by_step_excludes_the_envelopes_shared_step():
    fake_llm = _llm_with_extraction(
        _MIDDLE_FAILS_QUERY, variables=_MIDDLE_FAILS_VARIABLES, entities=["Ethiopia"]
    )
    result = offline_resolve(make_request(_MIDDLE_FAILS_QUERY), llm=fake_llm)

    assert result.diagnostics.timing_ms is not None
    assert result.diagnostics.timing_ms.by_step is not None
    assert set(result.diagnostics.timing_ms.by_step) == {"coverage", "extract"}
    assert result.diagnostics.timing_ms.by_step["extract"] >= 0
    assert result.diagnostics.timing_ms.total >= 0

    for part in result.parts:
        assert part.timing_ms is not None
        assert part.timing_ms.by_step
        assert "extract" not in part.timing_ms.by_step
        assert part.timing_ms.total == sum(
            ms for step, ms in part.timing_ms.by_step.items() if step not in NESTED_SHAPE_STEPS
        )
