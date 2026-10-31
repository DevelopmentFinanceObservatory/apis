"""Recall's two resolves: ours and the UN System Data Commons', each admitted by the request's
catalogue before its own `SEARCH_DETECT_K` cut and kept apart, with `un_system_unavailable`
when UN reads name places its front door refuses."""

from __future__ import annotations

import json
import logging
import threading
import time
import types
from collections.abc import Callable

import httpx
import pytest

import dc_graph
import dc_graph.client as dc_graph_client
from code_index.query import CodeIndex
from code_index.types import IndexStamp, StatVarRow
from origin_catalogue import BASE, DFO, UN_SYSTEM, note_un_blocked_places
from search.engine.clients import graph_client
from search.engine.clients.graph import DetectedSV, DetectLegs, admit_legs
from search.engine.clients.graph_client import _UN_RESOLVE_WORKERS, LiveGraphClient
from search.engine.errors import GraphDeadlineFailure, GraphInfraError
from search.engine.response_cache_keys import is_storable
from search.models import DefinitePart
from tests.engine._harness import make_request, member_walk_reader, offline_resolve
from tests.fixtures import _FIXTURES_DIR, FakeGraph
from tests.fixtures.record import _RecordingGraph
from tests.origin_catalogue.fake_catalogue import (
    FakeCatalogue,
    provenance_row,
    restrictive_catalogue,
    un_mirror,
    un_variable,
)

_OURS_HOST = "ours.example.org"
_UN_HOST = "un.example.org"

_WDI = "dc/base/WorldDevelopmentIndicators"
_WHO = "undata/p/WHO"
_CRS_VARIABLE = "Amount_FinancialFlow_Gross_Disbursement_Nominal_ODA"
_BASE_MORTALITY = "Count_Death_Upto5Years_AsFractionOf_Count_BirthEvent_LiveBirth"
_UN_MORTALITY = "undata/who/MORT_U5_RATE"
_UN_NOT_HELD = "undata/ilo/EMP_TEMP"
_WITHHELD = "Count_Death_Upto5Years_WithheldSource"


def _catalogue(*, un_system_enabled: bool = True):
    """Admits the CRS variable (DFO), the base mortality variable (WDI) and the WHO variable
    its UN tables hold. `_WITHHELD` is withheld and the UN tables hold no ILO variable."""
    return restrictive_catalogue(
        provenances=[
            provenance_row("provenance/OECDCreditorReportingSystem", DFO),
            provenance_row(_WDI, BASE),
            provenance_row(_WHO, UN_SYSTEM, segment="who"),
        ],
        dfo_variables=[_CRS_VARIABLE],
        base_variables={_BASE_MORTALITY: ([_WDI], [])},
        withheld_variables=[_WITHHELD],
        un_mirror=un_mirror(
            {_UN_MORTALITY: un_variable("undata/who/SERIES-MORT_U5_RATE", name="Mortality")}
        ),
        un_system_enabled=un_system_enabled,
    )


def _sv(dcid: str, score: float) -> DetectedSV:
    return DetectedSV(dcid=dcid, score=score, type_of="StatisticalVariable", sentence=dcid)


def _body(*candidates: tuple[str, float]) -> dict:
    return {
        "entities": [
            {
                "candidates": [
                    {
                        "dcid": dcid,
                        "typeOf": ["StatisticalVariable"],
                        "metadata": {"score": str(score), "sentence": dcid},
                    }
                    for dcid, score in candidates
                ]
            }
        ]
    }


# -- admission and the split --


def test_our_leg_is_cut_at_detect_k_and_holds_no_un_candidate() -> None:
    legs = DetectLegs(
        ours=(_sv(_CRS_VARIABLE, 0.5), _sv(_BASE_MORTALITY, 0.7)),
        un=(_sv(_UN_MORTALITY, 0.95),),
    )

    detection = admit_legs(legs, catalogue=_catalogue(), detect_k=1)

    assert [c.dcid for c in detection.candidates] == [_BASE_MORTALITY]
    assert [c.dcid for c in detection.un.result() or ()] == [_UN_MORTALITY]


def test_a_refused_candidate_on_our_leg_leaves_its_slot_to_an_admitted_one() -> None:
    legs = DetectLegs(ours=(_sv(_WITHHELD, 0.9), _sv(_BASE_MORTALITY, 0.7)), un=())

    detection = admit_legs(legs, catalogue=_catalogue(), detect_k=1)

    assert [c.dcid for c in detection.candidates] == [_BASE_MORTALITY]


def test_the_un_leg_keeps_admitted_un_candidates_before_its_own_cut() -> None:
    legs = DetectLegs(
        ours=(_sv(_BASE_MORTALITY, 0.7),),
        un=(_sv(_UN_NOT_HELD, 0.95), _sv(_CRS_VARIABLE, 0.9), _sv(_UN_MORTALITY, 0.8)),
    )

    detection = admit_legs(legs, catalogue=_catalogue(), detect_k=1)

    assert [c.dcid for c in detection.un.result() or ()] == [_UN_MORTALITY]
    assert [c.dcid for c in detection.candidates] == [_BASE_MORTALITY]


def test_a_failed_un_leg_leaves_no_un_candidates_and_marks_the_detection() -> None:
    legs = DetectLegs(ours=(_sv(_BASE_MORTALITY, 0.7),), un=None)

    detection = admit_legs(legs, catalogue=_catalogue(), detect_k=80)

    assert [c.dcid for c in detection.candidates] == [_BASE_MORTALITY]
    assert detection.un.result() is None


# -- rival index families --

_GE = "GrantEquivalent_FinancialFlow_Disbursement"
_CRS_GE = tuple(f"{_GE}_CRSPurposeCode{code}_Nominal" for code in (110, 120, 130))
_DAC1_GE_BILATERAL = f"{_GE}_DAC1AidType1015_DACTypeOfFinanceGrants_Nominal"
_DAC1_GE_TOTAL = f"{_GE}_DAC1AidType1010_Nominal"
_DAC2B_GE = f"{_GE}_DAC2bAidType972_Nominal"


def _ge_row(dcid: str, provenance: str, roles: tuple[str, ...], **codes: str) -> StatVarRow:
    return StatVarRow(
        dcid=dcid,
        population_type="FinancialFlow",
        measured_property="grantEquivalent",
        stat_type="measuredValue",
        measurement_qualifier="Nominal",
        measurement_denominator=None,
        observation_properties=roles,
        constraint_properties={"financialFlowType": "Disbursement", **codes},
        provenances=(provenance,),
    )


def _grant_equivalent_index() -> CodeIndex:
    """One grant-equivalent five-tuple published by three families: CRS by donor and
    recipient, DAC1 by donor, its narrower row first in index order, and DAC2b."""
    crs = "provenance/OECDCreditorReportingSystem"
    pair = ("financingProvider", "financingRecipient")
    return CodeIndex(
        stamp=IndexStamp(
            graph_endpoint="https://fixture.example.org",
            exported_at="2026-10-09T00:00:00Z",
            code_counts={},
            statvar_shape_counts={},
            content_hash="sha256:fixture",
        ),
        declarations=[],
        codes=[],
        statvars=[
            *(_ge_row(dcid, crs, pair, crsPurpose=dcid) for dcid in _CRS_GE),
            _ge_row(
                _DAC1_GE_BILATERAL,
                "provenance/OECDDAC1",
                ("financingProvider",),
                dac1AidType="DAC1AidType1015",
                dacTypeOfFinance="DACTypeOfFinanceGrants",
            ),
            _ge_row(
                _DAC1_GE_TOTAL,
                "provenance/OECDDAC1",
                ("financingProvider",),
                dac1AidType="DAC1AidType1010",
            ),
            _ge_row(_DAC2B_GE, "provenance/OECDDAC2B", pair, dac2bAidType="DAC2bAidType972"),
        ],
    )


@pytest.mark.parametrize(
    ("ours", "admitted"),
    [
        pytest.param(
            (_sv(_CRS_GE[0], 0.83), _sv(_CRS_GE[1], 0.80), _sv(_CRS_GE[2], 0.79)),
            [(_CRS_GE[0], 0.83), (_DAC1_GE_TOTAL, 0.83), (_CRS_GE[1], 0.80)],
            id="parts-gain-the-whole",
        ),
        pytest.param(
            (_sv(_DAC1_GE_BILATERAL, 0.7),),
            [(_DAC1_GE_BILATERAL, 0.7)],
            id="the-whole-gains-no-parts",
        ),
    ],
)
def test_the_cut_gains_the_family_holding_the_totals_of_its_hits(
    ours: tuple[DetectedSV, ...], admitted: list[tuple[str, float]]
) -> None:
    """CRS hits, by donor and recipient, fill the cut, and DAC1, by donor alone, joins
    through its broadest row at the best CRS score. DAC2b publishes on the CRS roles and adds
    nothing, and a DAC1 hit adds neither family on more roles."""
    detection = admit_legs(
        DetectLegs(ours=ours, un=()),
        catalogue=restrictive_catalogue(),
        detect_k=2,
        index=_grant_equivalent_index(),
    )

    assert [(c.dcid, c.score) for c in detection.candidates] == admitted
    assert detection.hits == detection.candidates


# -- the live client --


class _Resolves:
    """Answers `/v2/resolve` per deployment through a patched `dc_graph.GraphClient.post`."""

    def __init__(
        self,
        *,
        ours: dict | Exception,
        un: dict | Exception,
        on_post: Callable[[str], None] | None = None,
    ) -> None:
        self.ours = ours
        self.un = un
        self.on_post = on_post
        self.calls: list[tuple[str, dict, float | None]] = []

    def post(self, client: dc_graph.GraphClient, path: str, payload: dict, *, deadline=None):
        host = client._client.base_url.host
        if self.on_post is not None:
            self.on_post(host)
        self.calls.append((host, payload, deadline))
        assert path == "/v2/resolve"
        answer = self.ours if host == _OURS_HOST else self.un
        if isinstance(answer, Exception):
            raise answer
        return answer

    def hosts(self) -> list[str]:
        return [host for host, _, _ in self.calls]


def _live(monkeypatch, resolves: _Resolves, **kwargs) -> LiveGraphClient:
    monkeypatch.setattr(
        dc_graph.GraphClient,
        "post",
        lambda self, path, payload, *, deadline=None: resolves.post(
            self, path, payload, deadline=deadline
        ),
    )
    return LiveGraphClient(
        base=f"https://{_OURS_HOST}",
        un_base_url=f"https://{_UN_HOST}/core/api",
        **kwargs,
    )


def test_each_leg_is_admitted_apart(monkeypatch, bind_catalogue) -> None:
    bind_catalogue(_catalogue())
    resolves = _Resolves(
        ours=_body((_BASE_MORTALITY, 0.7)),
        un=_body((_UN_MORTALITY, 0.8), (_UN_NOT_HELD, 0.9)),
    )
    client = _live(monkeypatch, resolves)

    detection = client.detect_svs("under-five mortality")

    assert [c.dcid for c in detection.candidates] == [_BASE_MORTALITY]
    assert [c.dcid for c in detection.un.result() or ()] == [_UN_MORTALITY]
    assert sorted(resolves.hosts()) == [_OURS_HOST, _UN_HOST]
    targets = {host: payload["target"] for host, payload, _ in resolves.calls}
    assert targets == {_OURS_HOST: "base_and_custom", _UN_HOST: "custom_only"}


def test_the_un_resolve_is_not_sent_with_the_un_off(monkeypatch, bind_catalogue) -> None:
    bind_catalogue(_catalogue(un_system_enabled=False))
    resolves = _Resolves(ours=_body((_BASE_MORTALITY, 0.7)), un=_body((_UN_MORTALITY, 0.9)))
    client = _live(monkeypatch, resolves)

    detection = client.detect_svs("under-five mortality")

    assert [c.dcid for c in detection.candidates] == [_BASE_MORTALITY]
    assert detection.un.result() == ()
    assert resolves.hosts() == [_OURS_HOST]


def test_both_resolves_run_at_once(monkeypatch, bind_catalogue) -> None:
    bind_catalogue(_catalogue())
    un_started = threading.Event()

    def on_post(host: str) -> None:
        if host == _UN_HOST:
            un_started.set()
        else:
            assert un_started.wait(5), "our resolve waited for the UN resolve to finish"

    resolves = _Resolves(
        ours=_body((_BASE_MORTALITY, 0.7)), un=_body((_UN_MORTALITY, 0.8)), on_post=on_post
    )
    client = _live(monkeypatch, resolves)

    detection = client.detect_svs("under-five mortality")

    assert [c.dcid for c in detection.un.result() or ()] == [_UN_MORTALITY]


def test_a_un_timeout_answers_from_ours_and_is_not_cached(monkeypatch, bind_catalogue) -> None:
    bind_catalogue(_catalogue())
    resolves = _Resolves(
        ours=_body((_BASE_MORTALITY, 0.7)), un=GraphDeadlineFailure("deadline exceeded")
    )
    client = _live(monkeypatch, resolves, un_resolve_timeout_s=1.5)

    before = time.monotonic()
    first = client.detect_svs("under-five mortality")
    second = client.detect_svs("under-five mortality")

    assert first.un.result() is None and second.un.result() is None
    assert [c.dcid for c in first.candidates] == [_BASE_MORTALITY]
    assert resolves.hosts().count(_OURS_HOST) == 2, "a detection with a failed UN leg is cached"
    un_deadline = next(d for host, _, d in resolves.calls if host == _UN_HOST)
    ours_deadline = next(d for host, _, d in resolves.calls if host == _OURS_HOST)
    assert un_deadline is not None and before < un_deadline <= time.monotonic() + 1.5
    assert ours_deadline is None

    resolves.un = _body((_UN_MORTALITY, 0.8))
    recovered = client.detect_svs("under-five mortality")
    assert recovered.un.result() is not None
    client.detect_svs("under-five mortality")

    assert resolves.hosts().count(_OURS_HOST) == 3


def test_detect_returns_before_the_un_resolve_and_caches_once_it_succeeds(
    monkeypatch, bind_catalogue
) -> None:
    bind_catalogue(_catalogue())
    release = threading.Event()

    def on_post(host: str) -> None:
        if host == _UN_HOST:
            assert release.wait(5), "the test never released the UN resolve"

    resolves = _Resolves(
        ours=_body((_BASE_MORTALITY, 0.7)), un=_body((_UN_MORTALITY, 0.8)), on_post=on_post
    )
    client = _live(monkeypatch, resolves)

    detection = client.detect_svs("under-five mortality")

    assert [c.dcid for c in detection.candidates] == [_BASE_MORTALITY]
    assert not detection.un.done()
    release.set()
    assert [c.dcid for c in detection.un.result(timeout=5) or ()] == [_UN_MORTALITY]
    assert client.detect_svs("under-five mortality") is detection
    assert resolves.hosts().count(_OURS_HOST) == 1


def test_a_failed_resolve_of_ours_raises_and_leaves_the_un_resolve_unobserved(
    monkeypatch, bind_catalogue, caplog
) -> None:
    bind_catalogue(_catalogue())
    release = threading.Event()

    def on_post(host: str) -> None:
        if host == _UN_HOST:
            assert release.wait(5), "the test never released the UN resolve"

    resolves = _Resolves(
        ours=GraphInfraError("our mixer is down"), un=_body((_UN_MORTALITY, 0.8)), on_post=on_post
    )
    client = _live(monkeypatch, resolves)

    with caplog.at_level(logging.DEBUG), pytest.raises(GraphInfraError):
        client.detect_svs("under-five mortality")
    release.set()
    deadline = time.monotonic() + 5
    while client.stats()["un_resolves_in_flight"]:
        assert time.monotonic() < deadline, "the UN resolve never finished"
        time.sleep(0.01)

    assert _UN_HOST in resolves.hosts()
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert client.stats()["detect_entries"] == 0


def test_concurrent_detections_of_one_query_share_one_un_resolve(
    monkeypatch, bind_catalogue
) -> None:
    bind_catalogue(_catalogue())
    release = threading.Event()

    def on_post(host: str) -> None:
        if host == _UN_HOST:
            assert release.wait(5), "the test never released the UN resolve"

    resolves = _Resolves(
        ours=_body((_BASE_MORTALITY, 0.7)), un=_body((_UN_MORTALITY, 0.8)), on_post=on_post
    )
    client = _live(monkeypatch, resolves)

    first = client.detect_svs("under-five mortality")
    second = client.detect_svs("under-five mortality")
    release.set()

    assert [c.dcid for c in first.un.result(timeout=5) or ()] == [_UN_MORTALITY]
    assert [c.dcid for c in second.un.result(timeout=5) or ()] == [_UN_MORTALITY]
    assert resolves.hosts().count(_UN_HOST) == 1
    assert resolves.hosts().count(_OURS_HOST) == 2


def test_un_resolves_past_the_pool_bound_queue_on_its_threads(monkeypatch, bind_catalogue) -> None:
    bind_catalogue(_catalogue())
    release = threading.Event()
    un_started = threading.Semaphore(0)

    def on_post(host: str) -> None:
        if host == _UN_HOST:
            un_started.release()
            assert release.wait(5), "the test never released the UN resolves"

    resolves = _Resolves(
        ours=_body((_BASE_MORTALITY, 0.7)), un=_body((_UN_MORTALITY, 0.8)), on_post=on_post
    )
    client = _live(monkeypatch, resolves)

    detections = [
        client.detect_svs(f"under-five mortality {n}") for n in range(_UN_RESOLVE_WORKERS + 4)
    ]
    for _ in range(_UN_RESOLVE_WORKERS):
        assert un_started.acquire(timeout=5), "a worker never started its UN resolve"
    assert not un_started.acquire(timeout=0.05), "a UN resolve started past the pool bound"
    detect_threads = [t for t in threading.enumerate() if t.name.startswith("detect-un")]
    assert len(detect_threads) <= _UN_RESOLVE_WORKERS
    release.set()

    for detection in detections:
        assert [c.dcid for c in detection.un.result(timeout=5) or ()] == [_UN_MORTALITY]
    assert resolves.hosts().count(_UN_HOST) == _UN_RESOLVE_WORKERS + 4


class _FrozenMonotonic:
    """A `time.monotonic` that reads `now` until a test moves it."""

    def __init__(self) -> None:
        self.now = time.monotonic()

    def time_module(self) -> types.SimpleNamespace:
        """The `time` module with this clock as its `monotonic`."""
        return types.SimpleNamespace(
            monotonic=lambda: self.now,
            sleep=time.sleep,
            time=time.time,
            perf_counter=time.perf_counter,
        )


def test_a_un_resolve_that_expires_in_the_queue_sends_no_request(
    monkeypatch, bind_catalogue
) -> None:
    """Patched at the HTTP client, below `dc_graph`'s pre-send deadline check."""
    bind_catalogue(_catalogue())
    release = threading.Event()
    un_started = threading.Semaphore(0)
    un_queries: list[str] = []

    def post(self: httpx.Client, url: str, *, json: dict, timeout: float | None = None):
        if self.base_url.host != _UN_HOST:
            return httpx.Response(200, json=_body((_BASE_MORTALITY, 0.7)))
        un_queries.append(json["nodes"][0])
        un_started.release()
        assert release.wait(5), "the test never released the UN resolves"
        return httpx.Response(200, json=_body((_UN_MORTALITY, 0.8)))

    monkeypatch.setattr(httpx.Client, "post", post)
    # The deadline reads `time.monotonic` in the client that sets it and the one that checks it.
    # Both read a clock that moves only when the test moves it, so the running resolves cannot
    # expire before they send, however slowly their workers start.
    clock = _FrozenMonotonic()
    for module in (graph_client, dc_graph_client):
        monkeypatch.setattr(module, "time", clock.time_module())
    client = LiveGraphClient(
        base=f"https://{_OURS_HOST}",
        un_base_url=f"https://{_UN_HOST}/core/api",
        un_resolve_timeout_s=0.3,
    )

    running = [client.detect_svs(f"under-five mortality {n}") for n in range(_UN_RESOLVE_WORKERS)]
    for _ in range(_UN_RESOLVE_WORKERS):
        assert un_started.acquire(timeout=5), "a worker never started its UN resolve"
    queued = client.detect_svs("expires in the queue")
    clock.now += 0.4
    release.set()

    assert queued.un.result(timeout=5) is None
    assert "expires in the queue" not in un_queries
    for detection in running:
        assert [c.dcid for c in detection.un.result(timeout=5) or ()] == [_UN_MORTALITY]
    assert len(un_queries) == _UN_RESOLVE_WORKERS


def test_the_detect_cache_keys_on_the_catalogue(monkeypatch, bind_catalogue) -> None:
    resolves = _Resolves(ours=_body((_BASE_MORTALITY, 0.7)), un=_body((_UN_MORTALITY, 0.8)))
    client = _live(monkeypatch, resolves)

    bind_catalogue(_catalogue())
    with_un = client.detect_svs("under-five mortality")
    bind_catalogue(_catalogue(un_system_enabled=False))
    without_un = client.detect_svs("under-five mortality")

    assert [c.dcid for c in with_un.un.result() or ()] == [_UN_MORTALITY]
    assert without_un.un.result() == ()


# -- the answer --

_QUERY = "fertility rate in Kenya"


def _recorded() -> dict[str, dict]:
    recorded = json.loads((_FIXTURES_DIR / "graph_detect.json").read_text())
    return {query: entry for query, entry in recorded.items() if isinstance(entry, dict)}


def _codes(response) -> list[str]:
    return [w.code for part in response.parts for w in part.warnings]


def test_a_failed_un_resolve_leaves_an_answer_unwarned_and_storable() -> None:
    """Our pass answers, so the UN pool, which only fills a no-data gap, could not have
    changed the answer: it neither warns nor stays out of the response cache."""
    baseline = offline_resolve(make_request(_QUERY), materialise_graph=member_walk_reader())
    failed_un = FakeGraph(un_detect=dict.fromkeys(_recorded()))

    response = offline_resolve(
        make_request(_QUERY), graph=failed_un, materialise_graph=member_walk_reader(failed_un)
    )

    assert response.status == baseline.status == "definite"
    assert "un_system_unavailable" not in _codes(response)
    assert isinstance(answer := response.parts[0], DefinitePart)
    assert isinstance(expected := baseline.parts[0], DefinitePart)
    assert answer.spec.spec_id == expected.spec.spec_id
    assert is_storable(response, timed_out=False, degraded_leg=False, refused_at_build=frozenset())


class _RefusingUnGraph(FakeGraph):
    """Reports the UN front door refusing Greece on every facet read, as the origin client
    does when a UN read names it."""

    def observation_facets_batch(
        self, stat_vars, entities, needs_dates=False, *, max_bytes=None, deadline=None
    ):
        note_un_blocked_places(["country/GRC"])
        return super().observation_facets_batch(
            stat_vars, entities, needs_dates, max_bytes=max_bytes, deadline=deadline
        )


def test_an_answer_whose_un_reads_met_a_refused_place_names_it() -> None:
    response = offline_resolve(make_request(_QUERY), graph=_RefusingUnGraph())

    warnings = [w for p in response.parts for w in p.warnings if w.code == "un_system_unavailable"]
    assert len(warnings) == 1
    assert "country/GRC" in warnings[0].message


# -- fixture replay and recording per leg --


def test_fake_graph_replays_each_resolve_and_admits_them() -> None:
    graph = FakeGraph(
        detect={"q": {"svs": [_BASE_MORTALITY], "cosine_scores": [0.7]}},
        un_detect={"q": {"svs": [_UN_MORTALITY], "cosine_scores": [0.8]}},
    )

    legs = graph.detect_legs("q")
    detection = graph.detect_svs("q")

    assert [c.dcid for c in legs.ours] == [_BASE_MORTALITY]
    assert legs.un is not None and [c.dcid for c in legs.un] == [_UN_MORTALITY]
    assert [c.dcid for c in detection.candidates] == [_BASE_MORTALITY]
    assert [c.dcid for c in detection.un.result() or ()] == [_UN_MORTALITY]


def test_fake_graph_replays_a_failed_un_resolve() -> None:
    graph = FakeGraph(
        detect={"q": {"svs": [_BASE_MORTALITY], "cosine_scores": [0.7]}}, un_detect={"q": None}
    )

    detection = graph.detect_svs("q")

    assert graph.detect_legs("q").un is None
    assert detection.un.result() is None
    assert [c.dcid for c in detection.candidates] == [_BASE_MORTALITY]


def test_fake_graph_skips_the_un_resolve_with_the_un_off(bind_catalogue) -> None:
    bind_catalogue(FakeCatalogue(un_system_enabled=False))
    graph = FakeGraph(
        detect={"q": {"svs": [_BASE_MORTALITY], "cosine_scores": [0.7]}}, un_detect={"q": None}
    )

    assert graph.detect_legs("q").un == ()
    assert graph.detect_svs("q").un.result() is not None


def test_the_recorded_fixture_holds_each_resolve() -> None:
    entries = list(_recorded().values())

    assert entries
    assert all(set(entry) == {"ours", "un"} for entry in entries)


@pytest.mark.parametrize("un_entry", [None, {"svs": [_UN_MORTALITY], "cosine_scores": [0.8]}])
def test_the_recorder_writes_each_resolve_separately(monkeypatch, un_entry) -> None:
    monkeypatch.setattr("tests.fixtures.record._RECORD_THROTTLE_S", 0)
    live = FakeGraph(
        detect={"q": {"svs": [_BASE_MORTALITY], "cosine_scores": [0.7]}},
        un_detect={"q": un_entry},
    )
    store: dict = {}
    recorder = _RecordingGraph(live, nodes={}, obs={}, detect=store, resolve={})

    detection = recorder.detect_svs("q")
    replayed = FakeGraph(
        detect={"q": store["q"]["ours"]}, un_detect={"q": store["q"]["un"]}
    ).detect_svs("q")

    assert store["q"]["ours"]["svs"] == [_BASE_MORTALITY]
    assert (store["q"]["un"] is None) == (un_entry is None)
    assert replayed.candidates == detection.candidates
    assert replayed.un.result() == detection.un.result()
