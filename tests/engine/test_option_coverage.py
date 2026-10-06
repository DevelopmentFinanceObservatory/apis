"""Tests option-coverage probes, request wiring, and trailing SSE enrichment."""

from __future__ import annotations

import dataclasses
import json
import time

import pytest
from fastapi.testclient import TestClient

from materialise.facets import ElectedDataset
from materialise.graph import Facet as MaterialiseFacet
from search.engine.bind import SlotBindingDraft
from search.engine.extract import DateRequest
from search.engine.retrieval.standard import standard_option_coverage
from search.engine.shape import ShapeDraft, SlotKeyDraft
from search.models import (
    CoverageBare,
    CoverageExact,
    DatasetResubmitInput,
    GraphRef,
    RawTextInput,
    ResolveOptions,
    ResolveRequest,
    ResubmitIdentityEntry,
    SlotKey,
    SlotOption,
    SlotOptions,
    SlotValue,
    TimeWindow,
)
from tests.engine._fake_sdmx import FakeSdmxClient
from tests.engine._harness import (
    PINNED_DATE,
    materialise_double_from_obs,
    offline_resolve,
    with_member_arcs,
)
from tests.engine._search_app import search_app
from tests.fixtures import FakeGraph, FakeLLM, _load_json, make_resolver
from tests.materialise.conftest import FakeGraphReader


def _facet(*, obs_count: int, earliest: str, latest: str) -> MaterialiseFacet:
    """Builds a materialise facet from its summary fields, from one import, since the graph
    names a provenance on every facet it serves."""
    return MaterialiseFacet(
        facet_id=f"f{earliest}",
        provenance_id="dc/base/WorldDevelopmentIndicators",
        measurement_method=None,
        unit=None,
        obs_count=obs_count,
        earliest_date=earliest,
        latest_date=latest,
    )


# FertilityRate_Person_Female|country/KEN's own graph_obs.json facts, reused
# (not duplicated) so the materialise-backed probe and FakeGraph's own
# offline_resolve default agree on the same underlying data.
_FERTILITY_KEN_KEY = "FertilityRate_Person_Female|country/KEN"


def _fertility_kenya_materialise():
    obs = _load_json("graph_obs.json")
    return materialise_double_from_obs({_FERTILITY_KEN_KEY: obs[_FERTILITY_KEN_KEY]})


def _by_dcid(options: list[SlotOption]) -> dict[str, SlotOption]:
    """Indexes options by dcid and asserts every fixture has a reference."""
    result: dict[str, SlotOption] = {}
    for opt in options:
        assert opt.value.ref is not None
        result[opt.value.ref.dcid] = opt
    return result


# ---------------------------------------------------------------------------
# standard_option_coverage() probes sibling members from sv_arc_facts
# ---------------------------------------------------------------------------

_MALE_SV = "UnemploymentRate_Person_Male"
_FEMALE_SV = "UnemploymentRate_Person_Female"
_RECIPIENT = "country/USA"


def _gender_arcs(gender: str) -> dict:
    return {
        "populationType": {"nodes": [{"dcid": "Person"}]},
        "measuredProperty": {"nodes": [{"dcid": "unemploymentRate"}]},
        "statType": {"nodes": [{"dcid": "measuredValue"}]},
        "constraintProperties": {"nodes": [{"dcid": "gender"}]},
        "gender": {"nodes": [{"dcid": gender}]},
    }


def _gender_shape() -> ShapeDraft:
    return ShapeDraft(
        shape_id="test_unemployment_shape",
        label="test shape",
        pop_type_dcid="Person",
        meas_prop_dcid="unemploymentRate",
        stat_type_dcid="measuredValue",
        meas_qual_dcid=None,
        meas_denom_dcid=None,
        slot_keys=(SlotKeyDraft(axis="how", property_dcid="gender", label="gender"),),
        sv_arc_facts={_MALE_SV: _gender_arcs("Male"), _FEMALE_SV: _gender_arcs("Female")},
        representative_score=0.9,
    )


def _gender_slot_options(*genders: str) -> SlotOptions:
    return SlotOptions(
        key=SlotKey(axis="how", property=GraphRef(dcid="gender", label="gender"), label="gender"),
        options=[
            SlotOption(
                value=SlotValue(
                    ref=GraphRef(dcid=g, label=g),
                    value_kind="enum_value",
                )
            )
            for g in genders
        ],
        truncated=False,
        source="observed",
    )


def _gender_slot_options_with_relation(*genders: str) -> SlotOptions:
    """Like `_gender_slot_options`, but every option carries `relation`/
    `member_count` too -- the level-scoping fields the coverage rewrap knows
    nothing about and must carry forward unchanged."""
    return SlotOptions(
        key=SlotKey(axis="how", property=GraphRef(dcid="gender", label="gender"), label="gender"),
        options=[
            SlotOption(
                value=SlotValue(ref=GraphRef(dcid=g, label=g), value_kind="enum_value"),
                relation="child",
                member_count=idx + 1,
            )
            for idx, g in enumerate(genders)
        ],
        truncated=False,
        source="observed",
    )


# Two slots (gender, ageGroup) whose realising StatVars overlap completely.
# Both SVs below carry both constraint properties, so gender's own all_svs
# and ageGroup's own all_svs are the identical two-element set. A per-slot
# probe would read each (SV, entity) pair twice. The union probe reads it
# once.
_SV_MALE_ADULT = "UnemploymentRate_Person_Male_Adult"
_SV_FEMALE_ADULT = "UnemploymentRate_Person_Female_Adult"


def _gender_age_arcs(gender: str, age_group: str) -> dict:
    return {
        "populationType": {"nodes": [{"dcid": "Person"}]},
        "measuredProperty": {"nodes": [{"dcid": "unemploymentRate"}]},
        "statType": {"nodes": [{"dcid": "measuredValue"}]},
        "constraintProperties": {"nodes": [{"dcid": "gender"}, {"dcid": "ageGroup"}]},
        "gender": {"nodes": [{"dcid": gender}]},
        "ageGroup": {"nodes": [{"dcid": age_group}]},
    }


def _gender_age_shape() -> ShapeDraft:
    return ShapeDraft(
        shape_id="test_gender_age_shape",
        label="test shape",
        pop_type_dcid="Person",
        meas_prop_dcid="unemploymentRate",
        stat_type_dcid="measuredValue",
        meas_qual_dcid=None,
        meas_denom_dcid=None,
        slot_keys=(),
        sv_arc_facts={
            _SV_MALE_ADULT: _gender_age_arcs("Male", "Adult"),
            _SV_FEMALE_ADULT: _gender_age_arcs("Female", "Adult"),
        },
        representative_score=0.9,
    )


def _slot_options(prop_dcid: str, *values: str) -> SlotOptions:
    return SlotOptions(
        key=SlotKey(
            axis="how", property=GraphRef(dcid=prop_dcid, label=prop_dcid), label=prop_dcid
        ),
        options=[
            SlotOption(value=SlotValue(ref=GraphRef(dcid=v, label=v), value_kind="enum_value"))
            for v in values
        ],
        truncated=False,
        source="observed",
    )


class TestStandardOptionCoverage:
    def test_each_option_gets_its_own_sibling_members_coverage(self):
        materialise_graph = FakeGraphReader(
            facets={
                (_MALE_SV, _RECIPIENT): [_facet(obs_count=2, earliest="2020", latest="2021")],
                (_FEMALE_SV, _RECIPIENT): [_facet(obs_count=5, earliest="2019", latest="2021")],
            }
        )
        shape = _gender_shape()
        slot_options = [_gender_slot_options("Male", "Female")]

        result = standard_option_coverage(
            shape=shape,
            slot_options=slot_options,
            entities={"observationAbout": [_RECIPIENT]},
            materialise_graph=materialise_graph,
            date_request=None,
        )

        assert len(result) == 1
        by_dcid = _by_dcid(result[0].options)
        assert isinstance(by_dcid["Male"].coverage, CoverageExact)
        assert by_dcid["Male"].coverage.observation_count == 2
        assert isinstance(by_dcid["Female"].coverage, CoverageExact)
        assert by_dcid["Female"].coverage.observation_count == 5

    def test_an_option_counts_only_its_members_elected_dataset(self):
        """The male member holds two facets and its family elected the dataset of the
        smaller one, so the option counts that facet's points, as an answer serves it. The
        female member has no elected dataset and counts every facet."""
        elected = dataclasses.replace(
            _facet(obs_count=2, earliest="2020", latest="2021"), provenance_id="provenance/Elected"
        )
        materialise_graph = FakeGraphReader(
            facets={
                (_MALE_SV, _RECIPIENT): [
                    _facet(obs_count=9, earliest="2010", latest="2021"),
                    elected,
                ],
                (_FEMALE_SV, _RECIPIENT): [_facet(obs_count=5, earliest="2019", latest="2021")],
            }
        )

        result = standard_option_coverage(
            shape=_gender_shape(),
            slot_options=[_gender_slot_options("Male", "Female")],
            entities={"observationAbout": [_RECIPIENT]},
            materialise_graph=materialise_graph,
            date_request=None,
            datasets={
                _MALE_SV: ElectedDataset(
                    provenance="provenance/Elected",
                    member_provenances=(
                        "dc/base/WorldDevelopmentIndicators",
                        "provenance/Elected",
                    ),
                )
            },
        )

        by_dcid = _by_dcid(result[0].options)
        assert isinstance(by_dcid["Male"].coverage, CoverageExact)
        assert by_dcid["Male"].coverage.observation_count == 2
        assert isinstance(by_dcid["Female"].coverage, CoverageExact)
        assert by_dcid["Female"].coverage.observation_count == 5

    def test_option_with_no_sibling_member_stays_uncovered(self):
        """Unrealizable members have no coverage; probed members with no data use `CoverageBare`."""
        materialise_graph = FakeGraphReader()
        shape = _gender_shape()  # only Male/Female members
        slot_options = [_gender_slot_options("Male", "Nonbinary")]  # no Nonbinary member

        result = standard_option_coverage(
            shape=shape,
            slot_options=slot_options,
            entities={"observationAbout": [_RECIPIENT]},
            materialise_graph=materialise_graph,
            date_request=None,
        )

        by_dcid = _by_dcid(result[0].options)
        assert by_dcid["Nonbinary"].coverage is None
        # Male exists as a member but has zero observations at this place -> an
        # honest "probed, no data" CoverageBare, not None.
        assert isinstance(by_dcid["Male"].coverage, CoverageBare)
        assert by_dcid["Male"].coverage.has_data is False

    def test_no_entity_no_probe_returns_slot_options_unchanged(self):
        """An empty entity set produces no coverage probe."""
        materialise_graph = FakeGraphReader()
        shape = _gender_shape()
        slot_options = [_gender_slot_options("Male", "Female")]

        result = standard_option_coverage(
            shape=shape,
            slot_options=slot_options,
            entities={},
            materialise_graph=materialise_graph,
            date_request=None,
        )

        assert all(o.coverage is None for so in result for o in so.options)

    def test_empty_slot_options_returns_unchanged(self):
        materialise_graph = FakeGraphReader()
        result = standard_option_coverage(
            shape=_gender_shape(),
            slot_options=[],
            entities={"observationAbout": [_RECIPIENT]},
            materialise_graph=materialise_graph,
            date_request=None,
        )
        assert result == []

    def test_rewrap_preserves_relation_and_member_count_on_a_covered_option(self):
        """The coverage rewrap preserves the option's `relation` and `member_count`."""
        materialise_graph = FakeGraphReader(
            facets={(_MALE_SV, _RECIPIENT): [_facet(obs_count=2, earliest="2020", latest="2021")]}
        )
        shape = _gender_shape()
        slot_options = [_gender_slot_options_with_relation("Male")]

        result = standard_option_coverage(
            shape=shape,
            slot_options=slot_options,
            entities={"observationAbout": [_RECIPIENT]},
            materialise_graph=materialise_graph,
            date_request=None,
        )

        by_dcid = _by_dcid(result[0].options)
        assert by_dcid["Male"].coverage is not None
        assert by_dcid["Male"].relation == "child"
        assert by_dcid["Male"].member_count == 1

    def test_rewrap_preserves_relation_and_member_count_on_an_uncovered_option(self):
        """Options without sibling members keep their metadata and have no coverage."""
        materialise_graph = FakeGraphReader()
        shape = _gender_shape()  # only Male/Female members
        slot_options = [_gender_slot_options_with_relation("Male", "Nonbinary")]

        result = standard_option_coverage(
            shape=shape,
            slot_options=slot_options,
            entities={"observationAbout": [_RECIPIENT]},
            materialise_graph=materialise_graph,
            date_request=None,
        )

        by_dcid = _by_dcid(result[0].options)
        assert by_dcid["Nonbinary"].coverage is None
        assert by_dcid["Nonbinary"].relation == "child"
        assert by_dcid["Nonbinary"].member_count == 2

    def test_overlapping_slots_probe_the_union_in_one_call(self):
        """Overlapping slots probe their shared StatVar set once, then slice results per slot."""
        materialise_graph = FakeGraphReader(
            facets={
                (_SV_MALE_ADULT, _RECIPIENT): [_facet(obs_count=3, earliest="2020", latest="2021")],
                (_SV_FEMALE_ADULT, _RECIPIENT): [
                    _facet(obs_count=7, earliest="2019", latest="2021")
                ],
            }
        )
        shape = _gender_age_shape()
        gender_slot = _slot_options("gender", "Male", "Female")
        age_slot = _slot_options("ageGroup", "Adult")

        result = standard_option_coverage(
            shape=shape,
            slot_options=[gender_slot, age_slot],
            entities={"observationAbout": [_RECIPIENT]},
            materialise_graph=materialise_graph,
            date_request=None,
        )

        assert materialise_graph.calls["observation_facets_batch_many"] == 1

        assert len(result) == 2
        gender_by_dcid = _by_dcid(result[0].options)
        assert isinstance(gender_by_dcid["Male"].coverage, CoverageExact)
        assert gender_by_dcid["Male"].coverage.observation_count == 3
        assert isinstance(gender_by_dcid["Female"].coverage, CoverageExact)
        assert gender_by_dcid["Female"].coverage.observation_count == 7

        age_by_dcid = _by_dcid(result[1].options)
        assert isinstance(age_by_dcid["Adult"].coverage, CoverageExact)
        # Adult is realised by both SVs, so its combined facet set, and
        # observation_count, sums both: 10 across the two SVs' facets.
        assert age_by_dcid["Adult"].coverage.observation_count == 10


# ---------------------------------------------------------------------------
# End-to-end: ResolveOptions.include_option_coverage wiring
# ---------------------------------------------------------------------------


def test_raw_text_include_option_coverage_populates_gender():
    """With the include_option_coverage flag on, the currently-bound gender
    option's own coverage is populated (reconstructing the same SV the resolve
    already confirmed)."""
    materialise_graph, materialise_sdmx = _fertility_kenya_materialise()
    req = ResolveRequest(
        input=RawTextInput(query="fertility rate in Kenya"),
        response_options=ResolveOptions(include_option_coverage=True),
    )
    r = offline_resolve(
        req,
        sdmx=FakeSdmxClient(),
        materialise_graph=materialise_graph,
        materialise_sdmx=materialise_sdmx,
    ).parts[0]
    assert r.status == "definite"
    spec = r.spec
    gender_opts = next(
        so
        for so in spec.resolved.slot_options
        if so.key.property and so.key.property.dcid == "gender"
    )
    covered = [o for o in gender_opts.options if o.coverage is not None]
    assert covered, "expected at least the bound gender value to get real coverage"


def test_raw_text_default_leaves_slot_options_uncovered():
    """Without the flag, slot_options coverage stays None -- zero extra graph
    calls, maintaining the opt-in default (regression guard)."""
    req = ResolveRequest(input=RawTextInput(query="fertility rate in Kenya"))
    r = offline_resolve(req, sdmx=FakeSdmxClient()).parts[0]
    assert r.status == "definite"
    spec = r.spec
    assert spec.resolved.slot_options  # sanity: options exist
    assert all(o.coverage is None for so in spec.resolved.slot_options for o in so.options)


def test_available_at_place_false_never_probes_coverage(monkeypatch):
    """A flagged-unavailable part returns no_data and omits coverage options, even with opt-in."""

    def _fail_if_called(*args, **kwargs):
        raise AssertionError("standard_option_coverage must not be reached")

    monkeypatch.setattr(
        "search.engine.region.option_coverage.standard_option_coverage", _fail_if_called
    )

    def _fail_if_facet_called(*args, **kwargs):
        raise AssertionError("facet_options probe must not be reached")

    monkeypatch.setattr(
        "search.engine.region.option_coverage._facet_options", _fail_if_facet_called
    )

    req = ResolveRequest(
        input=RawTextInput(query="teacher count in Nauru"),
        response_options=ResolveOptions(include_option_coverage=True),
    )
    result = offline_resolve(req, sdmx=FakeSdmxClient()).parts[0]
    assert result.status == "no_data"
    assert result.no_data.reason == "no_observations"
    assert result.no_data.flagged_match is not None


def test_resubmit_include_option_coverage_populates_coverage():
    """On-demand popover refresh via a resubmit gets the same computation
    without a full raw_text re-resolve."""
    materialise_graph, materialise_sdmx = _fertility_kenya_materialise()
    # First resolve normally to get a real shape_id + slots to resubmit.
    base = offline_resolve(
        ResolveRequest(input=RawTextInput(query="fertility rate in Kenya")),
        sdmx=FakeSdmxClient(),
    ).parts[0]
    assert base.status == "definite"
    spec = base.spec

    resubmit_req = ResolveRequest(
        input=DatasetResubmitInput(
            entries=[
                ResubmitIdentityEntry(
                    variable_text="a resubmitted identity",
                    identity=spec.definition.identity,
                )
            ]
        ),
        response_options=ResolveOptions(include_option_coverage=True),
    )
    r = offline_resolve(
        resubmit_req,
        sdmx=FakeSdmxClient(),
        materialise_graph=with_member_arcs(materialise_graph, FakeGraph()),
        materialise_sdmx=materialise_sdmx,
    ).parts[0]
    assert r.status == "definite"
    covered = [
        o for so in r.spec.resolved.slot_options for o in so.options if o.coverage is not None
    ]
    assert covered, "expected a resubmit with include_option_coverage=True to populate coverage"


# ---------------------------------------------------------------------------
# SSE end-to-end: trailing `slot_options` events (lazy enrichment)
# ---------------------------------------------------------------------------


def _parse_sse(body: str) -> list[tuple[str, dict]]:
    """Parse an SSE body into [(event_type, data_dict), ...], skipping `: ping` comments."""
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


def test_sse_streams_trailing_slot_options_after_response_before_done():
    """SSE coverage arrives in trailing `slot_options` events.

    The response payload keeps its own uncovered options."""
    from datetime import date
    from unittest.mock import patch

    materialise_graph, materialise_sdmx = _fertility_kenya_materialise()
    app = search_app(
        graph=FakeGraph(),
        resolver=make_resolver(FakeGraph()),
        llm=FakeLLM(),
        sdmx=FakeSdmxClient(),
        materialise_graph=materialise_graph,
        materialise_sdmx=materialise_sdmx,
    )
    client = TestClient(app, raise_server_exceptions=False)
    with patch("search.engine.extract.date") as mock_date:
        mock_date.today.return_value = PINNED_DATE
        mock_date.side_effect = lambda *args, **kwargs: date(*args, **kwargs)
        resp = client.post(
            "/api/search/resolve",
            json={
                "input": {"kind": "raw_text", "query": "fertility rate in Kenya"},
                "response_options": {"include_option_coverage": True},
            },
            headers={"Accept": "text/event-stream"},
        )
    assert resp.status_code == 200
    frames = _parse_sse(resp.text)
    event_types = [ev for ev, _ in frames]

    assert "response" in event_types
    assert "slot_options" in event_types
    response_idx = event_types.index("response")
    done_idx = event_types.index("done")
    slot_options_idxs = [i for i, ev in enumerate(event_types) if ev == "slot_options"]
    assert slot_options_idxs, "expected at least one trailing slot_options event"
    assert all(response_idx < i < done_idx for i in slot_options_idxs), (
        f"slot_options events must fall strictly between response ({response_idx}) "
        f"and done ({done_idx}); got slot_options at {slot_options_idxs}"
    )

    # The event carries probe results.
    slot_options_payload = frames[slot_options_idxs[0]][1]
    covered = [o for o in slot_options_payload["options"] if o.get("coverage") is not None]
    assert covered, "trailing slot_options event must carry real coverage"

    # response's own Spec.resolved.slot_options stays untouched -- byte-identical
    # to the buffered JSON body regardless of the trailing enrichment.
    response_payload = frames[response_idx][1]["response"]
    spec = response_payload["parts"][0]["spec"]
    resolved = spec["resolved"]
    assert all(
        o.get("coverage") is None for so in resolved["slot_options"] for o in so["options"]
    ), "response event's Spec.resolved.slot_options must stay uncovered"


def test_response_event_arrives_before_slow_probe_completes(monkeypatch):
    """The response event arrives before the one-second coverage probe completes.

    A completion flag avoids runner-dependent timing thresholds."""
    import asyncio
    import threading
    from datetime import date
    from unittest.mock import patch

    from search.engine.core import resolve_events

    real_option_coverage = standard_option_coverage
    probe_delay = 1.0
    probe_finished = threading.Event()

    def _slow_option_coverage(*args, **kwargs):
        time.sleep(probe_delay)  # Synchronous probe work incurs this delay.
        result = real_option_coverage(*args, **kwargs)
        probe_finished.set()
        return result

    monkeypatch.setattr(
        "search.engine.region.option_coverage.standard_option_coverage", _slow_option_coverage
    )

    async def _body():
        t0 = time.monotonic()
        response_seen = False
        probe_finished_at_response: bool | None = None
        slot_options_at: float | None = None
        with patch("search.engine.extract.date") as mock_date:
            mock_date.today.return_value = PINNED_DATE
            mock_date.side_effect = lambda *args, **kwargs: date(*args, **kwargs)
            async for ev in resolve_events(
                ResolveRequest(
                    input=RawTextInput(query="fertility rate in Kenya"),
                    response_options=ResolveOptions(include_option_coverage=True),
                ),
                graph=FakeGraph(),
                llm=FakeLLM(),
                stream_slot_options=True,
                sdmx=FakeSdmxClient(),
            ):
                if ev.kind == "response" and not response_seen:
                    response_seen = True
                    probe_finished_at_response = probe_finished.is_set()
                if ev.kind == "slot_options" and slot_options_at is None:
                    slot_options_at = time.monotonic() - t0

        assert response_seen, "expected a response event"
        assert slot_options_at is not None, "expected the slow probe to still complete"
        assert probe_finished_at_response is False, (
            "the probe had already completed by the time `response` arrived -- "
            "resolve_variable appears to be awaiting the probe before emitting it"
        )
        assert slot_options_at >= probe_delay, (
            f"slot_options arrived at {slot_options_at:.2f}s, before the "
            f"{probe_delay}s probe could have finished"
        )

    asyncio.run(_body())


def test_buffered_probe_does_not_block_a_concurrent_plain_request(monkeypatch):
    """The plain request must finish while the buffered coverage probe runs.

    The probe waits on an event from that request, proving it runs off-loop without a timing
    threshold."""
    import asyncio
    import threading
    from datetime import date
    from unittest.mock import patch

    from search.engine.core import resolve_async

    real_option_coverage = standard_option_coverage
    plain_done = threading.Event()
    observed: dict[str, bool] = {}

    def _blocking_option_coverage(*args, **kwargs):
        # Blocking models synchronous probe cost. The timeout bounds deadlock.
        observed["plain_finished_during_probe"] = plain_done.wait(timeout=10.0)
        return real_option_coverage(*args, **kwargs)

    monkeypatch.setattr(
        "search.engine.region.option_coverage.standard_option_coverage", _blocking_option_coverage
    )

    async def _body():
        async def _run_plain() -> None:
            await resolve_async(
                ResolveRequest(input=RawTextInput(query="total population India")),
                graph=FakeGraph(),
                llm=FakeLLM(),
                sdmx=FakeSdmxClient(observations={}, availability={}),
            )
            plain_done.set()

        slow_task = asyncio.create_task(
            resolve_async(
                ResolveRequest(
                    input=RawTextInput(query="fertility rate in Kenya"),
                    response_options=ResolveOptions(include_option_coverage=True),
                ),
                graph=FakeGraph(),
                llm=FakeLLM(),
                sdmx=FakeSdmxClient(),
            )
        )
        plain_task = asyncio.create_task(_run_plain())
        await asyncio.gather(slow_task, plain_task)

        assert observed.get("plain_finished_during_probe") is True, (
            "the plain request could not finish while the probe was running -- "
            "the buffered probe appears to be blocking the event loop"
        )

    with patch("search.engine.extract.date") as mock_date:
        mock_date.today.return_value = PINNED_DATE
        mock_date.side_effect = lambda *args, **kwargs: date(*args, **kwargs)
        asyncio.run(_body())


def test_sse_without_include_option_coverage_never_probes():
    """SSE coverage probes run only when `include_option_coverage` is true."""
    app = search_app(
        graph=FakeGraph(), resolver=make_resolver(FakeGraph()), llm=FakeLLM(), sdmx=FakeSdmxClient()
    )
    client = TestClient(app, raise_server_exceptions=False)
    resp = client.post(
        "/api/search/resolve",
        json={"input": {"kind": "raw_text", "query": "health ODA grants to Ethiopia"}},
        headers={"Accept": "text/event-stream"},
    )
    assert resp.status_code == 200
    event_types = [ev for ev, _ in _parse_sse(resp.text)]
    assert "response" in event_types
    assert "slot_options" not in event_types


def test_deadline_cancelled_probe_preserves_primary_completion(monkeypatch, search_settings):
    """A deadline during optional enrichment leaves the completed answer untimed out.
    The stalled probe is cancelled."""
    import asyncio
    from datetime import date
    from unittest.mock import patch

    from search.engine.core import resolve_events
    from search.engine.region import OPTION_COVERAGE_TIMED_OUT

    real_option_coverage = standard_option_coverage

    def _hanging_option_coverage(*args, **kwargs):
        # The worker sleeps 3 seconds, beyond the 1-second deadline.
        # Cancellation cannot interrupt the thread, so the sleep stays short.
        time.sleep(3)
        return real_option_coverage(*args, **kwargs)  # pragma: no cover

    monkeypatch.setattr(
        "search.engine.region.option_coverage.standard_option_coverage", _hanging_option_coverage
    )
    # The 1-second timeout leaves the ~200 ms fake fan-out time to finish on loaded CI.
    # The coverage probe then exceeds the deadline.
    search_settings(request_timeout_s=1.0)

    async def _body():
        events = [
            ev
            async for ev in resolve_events(
                ResolveRequest(
                    input=RawTextInput(query="fertility rate in Kenya"),
                    response_options=ResolveOptions(include_option_coverage=True),
                ),
                graph=FakeGraph(),
                llm=FakeLLM(),
                stream_slot_options=True,
                sdmx=FakeSdmxClient(),
            )
        ]

        kinds = [e.kind for e in events]
        assert kinds.count("done") == 1
        assert kinds.count("response") == 1
        done = next(e for e in events if e.kind == "done")
        assert done.timed_out is False, (
            "a stalled enrichment probe must never mark the overall response as timed out"
        )
        codes = [warning.code for warning in done.warnings]
        assert OPTION_COVERAGE_TIMED_OUT in codes, (
            "a deadline-cancelled probe must surface a distinct warning rather "
            "than look like an honest empty probe"
        )

        # Cancellation removes the probe task from the event loop that ran the request.
        await asyncio.sleep(0.1)  # let the cancellation finalize
        current = asyncio.current_task()
        pending = {t for t in asyncio.all_tasks() if t is not current and not t.done()}
        assert not pending, f"Orphaned tasks remain: {pending}"

    with patch("search.engine.extract.date") as mock_date:
        mock_date.today.return_value = PINNED_DATE
        mock_date.side_effect = lambda *args, **kwargs: date(*args, **kwargs)
        asyncio.run(_body())


# ---------------------------------------------------------------------------
# Probe results must be distinguishable: honest-empty, probe-failed, or
# deadline-cancelled must each be represented differently
# ---------------------------------------------------------------------------


def test_probe_failure_still_streams_the_event_and_surfaces_a_warning(monkeypatch):
    """A failed probe still streams uncovered `slot_options` and warns on `done`.

    The response event has already shipped when the probe fails."""
    import asyncio
    from datetime import date
    from unittest.mock import patch

    from search.engine.core import resolve_events
    from search.engine.errors import GraphInfraError
    from search.engine.region import OPTION_COVERAGE_PROBE_FAILED

    def _raising_option_coverage(*args, **kwargs):
        raise GraphInfraError("boom")

    monkeypatch.setattr(
        "search.engine.region.option_coverage.standard_option_coverage", _raising_option_coverage
    )

    async def _body():
        events = [
            ev
            async for ev in resolve_events(
                ResolveRequest(
                    input=RawTextInput(query="fertility rate in Kenya"),
                    response_options=ResolveOptions(include_option_coverage=True),
                ),
                graph=FakeGraph(),
                llm=FakeLLM(),
                stream_slot_options=True,
                sdmx=FakeSdmxClient(),
            )
        ]

        slot_options_events = [e for e in events if e.kind == "slot_options"]
        assert slot_options_events, (
            "a probe failure must fail open to the original (uncovered) slot "
            "options and still stream the event, not vanish it"
        )
        assert all(o.coverage is None for ev in slot_options_events for o in ev.options), (
            "the failed probe must not fabricate coverage"
        )

        done = next(e for e in events if e.kind == "done")
        codes = [w.code for w in done.warnings]
        assert OPTION_COVERAGE_PROBE_FAILED in codes, (
            "a probe failure must be distinguishable from an honest "
            "'probed, nothing found' via a warning on the terminal event"
        )

    with patch("search.engine.extract.date") as mock_date:
        mock_date.today.return_value = PINNED_DATE
        mock_date.side_effect = lambda *args, **kwargs: date(*args, **kwargs)
        asyncio.run(_body())


def test_probe_non_infra_error_propagates_instead_of_degrading(monkeypatch):
    """`GraphInfraError` becomes a warning; a resolver bug propagates unchanged."""
    from search.engine.region.option_coverage import _maybe_enrich_slot_options

    class _ResolverBug(Exception):
        pass

    def _raising_option_coverage(*args, **kwargs):
        raise _ResolverBug("bug in the probe, not a graph error")

    monkeypatch.setattr(
        "search.engine.region.option_coverage.standard_option_coverage", _raising_option_coverage
    )

    with pytest.raises(_ResolverBug):
        _maybe_enrich_slot_options(
            _gender_shape(),
            [
                SlotBindingDraft(
                    axis="how", property_dcid="gender", kind="value", value_dcids=["Male"]
                )
            ],
            [_gender_slot_options("Male", "Female")],
            entities={"observationAbout": [_RECIPIENT]},
            date_request=None,
            materialise_graph=FakeGraphReader(),
            include_option_coverage=True,
            facets_pinned=False,
            open_axes=frozenset(),
        )


def test_real_materialise_transport_failure_degrades_through_the_real_translation(monkeypatch):
    """A real `GraphReadError` becomes a coverage warning through the bridge."""
    from materialise.graph import GraphReadError
    from search.engine.region.option_coverage import _maybe_enrich_slot_options

    class _RaisingGraphReader(FakeGraphReader):
        def observation_facets_batch(self, *, stat_var, entities, deadline: float | None = None):
            raise GraphReadError("simulated transport error")

        def observation_facets_batch_many(
            self, *, stat_vars, entities, deadline: float | None = None
        ):
            raise GraphReadError("simulated transport error")

    enriched, warn = _maybe_enrich_slot_options(
        _gender_shape(),
        [SlotBindingDraft(axis="how", property_dcid="gender", kind="value", value_dcids=["Male"])],
        [_gender_slot_options("Male", "Female")],
        entities={"observationAbout": [_RECIPIENT]},
        date_request=None,
        materialise_graph=_RaisingGraphReader(),
        include_option_coverage=True,
        facets_pinned=False,
        open_axes=frozenset(),
    )
    assert all(o.coverage is None for so in enriched for o in so.options)
    assert warn is not None
    assert warn.code == "option_coverage_probe_failed"


def test_an_option_with_no_point_in_the_window_shows_no_data() -> None:
    """Under "since 2021", the Male option's member holds 2010 to 2012 alone and shows no
    data; the Female option's member holds 2019 to 2023 and states the years from 2021."""
    graph = FakeGraphReader(
        facets={
            (_MALE_SV, _RECIPIENT): [_facet(obs_count=3, earliest="2010", latest="2012")],
            (_FEMALE_SV, _RECIPIENT): [_facet(obs_count=5, earliest="2019", latest="2023")],
        }
    )
    result = standard_option_coverage(
        shape=_gender_shape(),
        slot_options=[_gender_slot_options("Male", "Female")],
        entities={"observationAbout": [_RECIPIENT]},
        materialise_graph=graph,
        date_request=DateRequest(window=TimeWindow(start_year=2021, end_year=None), latest=False),
    )
    by_dcid = _by_dcid(result[0].options)
    assert by_dcid["Male"].coverage.has_data is False
    assert by_dcid["Female"].coverage.window.start_year == 2021


# ---------------------------------------------------------------------------
# The trailing probe counts the series a pick of each option serves
# ---------------------------------------------------------------------------

# GDP of one country has the total and two sector parts, a family the shape holds exactly, as
# a resubmit's elected family is held. The answer binds the activity and leaves the sector
# open, so it serves the total alone.
_GDP_TOTAL = "Amount_EconomicActivity_GDP"
_GDP_AGRICULTURE = "Amount_EconomicActivity_GDP_Agriculture"
_GDP_MINING = "Amount_EconomicActivity_GDP_Mining"


def _gdp_arcs(**constraints: str) -> dict:
    return {
        "populationType": {"nodes": [{"dcid": "EconomicActivity"}]},
        "measuredProperty": {"nodes": [{"dcid": "amount"}]},
        "statType": {"nodes": [{"dcid": "measuredValue"}]},
        "constraintProperties": {"nodes": [{"dcid": p} for p in constraints]},
        **{p: {"nodes": [{"dcid": v}]} for p, v in constraints.items()},
    }


def _gdp_options(prop: str, *dcids: str) -> SlotOptions:
    return SlotOptions(
        key=SlotKey(axis="how", property=GraphRef(dcid=prop, label=prop), label=prop),
        options=[
            SlotOption(value=SlotValue(ref=GraphRef(dcid=d, label=d), value_kind="enum_value"))
            for d in dcids
        ],
        truncated=True,
        source="observed",
    )


def test_the_probe_counts_the_series_a_pick_serves_and_drops_a_value_no_member_carries():
    """The activity menu's one option is the answer itself, which serves the total alone,
    its 3 rows where counting every member carrying the activity would state 9. Each sector
    serves its part's 3 rows. Construction, which the menu lists but no member of the held
    family carries, serves nothing and is dropped."""
    from search.engine.region.option_coverage import _maybe_enrich_slot_options

    shape = ShapeDraft(
        shape_id="gdp",
        label="gdp",
        pop_type_dcid="EconomicActivity",
        meas_prop_dcid="amount",
        stat_type_dcid="measuredValue",
        meas_qual_dcid=None,
        meas_denom_dcid=None,
        slot_keys=(
            SlotKeyDraft(axis="how", property_dcid="activitySource", label="activity"),
            SlotKeyDraft(axis="how", property_dcid="economicSector", label="sector"),
        ),
        sv_arc_facts={
            _GDP_TOTAL: _gdp_arcs(activitySource="GDP"),
            _GDP_AGRICULTURE: _gdp_arcs(activitySource="GDP", economicSector="Agriculture"),
            _GDP_MINING: _gdp_arcs(activitySource="GDP", economicSector="Mining"),
        },
        sv_arc_facts_complete=True,
    )
    graph = FakeGraphReader(
        facets={
            (sv, _RECIPIENT): [_facet(obs_count=3, earliest="2020", latest="2022")]
            for sv in (_GDP_TOTAL, _GDP_AGRICULTURE, _GDP_MINING)
        }
    )

    enriched, warn = _maybe_enrich_slot_options(
        shape,
        [
            SlotBindingDraft(
                axis="how", property_dcid="activitySource", kind="value", value_dcids=["GDP"]
            ),
            SlotBindingDraft(axis="how", property_dcid="economicSector", kind="unbound"),
        ],
        [
            _gdp_options("activitySource", "GDP"),
            _gdp_options("economicSector", "Agriculture", "Mining", "Construction"),
        ],
        entities={"observationAbout": [_RECIPIENT]},
        date_request=None,
        materialise_graph=graph,
        include_option_coverage=True,
        facets_pinned=False,
        open_axes=frozenset(),
    )

    assert warn is None
    counts = {
        so.key.property.dcid: {
            o.value.ref.dcid: (o.served_rows, o.served_rows_exact) for o in so.options
        }
        for so in enriched
    }
    assert counts == {
        "activitySource": {"GDP": (3, True)},
        "economicSector": {"Agriculture": (3, True), "Mining": (3, True)},
    }
