"""Tests for `GET /api/graph/schema/search`: its legs, their order, the switches, failing,
late and refused legs and label reads, validation and the answer cache."""

from __future__ import annotations

import asyncio
import threading
import time
from collections.abc import Mapping, Sequence

import pytest

from dc_graph import ArcTarget
from graph_api.doors import schema_search
from graph_api.resolve_models import ResolveResponse, ResolveResult
from graph_generation import FakeGeneration
from materialise.graph import GraphReadError
from tests.graph_release.support import release_at_revision
from tests.origin_catalogue.fake_catalogue import FakeCatalogue

from .conftest import FakeGraphNodeReader, make_client, make_resolver
from .test_origin_usage import _serving_releases, _SwappableReleases

_ROUTE = "/api/graph/schema/search"
_SV = "StatisticalVariable"


def _node(name: str, type_of: str, provenance: str) -> dict[str, tuple[ArcTarget, ...]]:
    return {
        "name": (ArcTarget(value=name, literal=True),),
        "typeOf": (ArcTarget(value=type_of, provenances=(provenance,)),),
    }


_MIXER = {
    "country/KEN": _node("Kenya", "Country", "dc/base/Places"),
    "dfo/OOP": _node("Out-of-pocket spending", _SV, "dfo/WHO"),
    "Count_Person": _node("Population", _SV, "dc/base/Census"),
    "dc/topic/Live": _node("Health topic", "Topic", "dc/base/Topics"),
    "dfo/DescOnly": {"description": (ArcTarget(value="Only a description", literal=True),)},
    **{
        dcid: _node(dcid, _SV, "dfo/WHO")
        for dcid in ("sv/untyped", "sv/second", "sv/0", "sv/1", "sv/2")
    },
}
_UN = {"undata/sdg/SH_OOP": _node("Out-of-pocket share", _SV, "undata/sdg")}


class _Reader(FakeGraphNodeReader):
    """A node view reader over `arc_targets` that raises a read error for any batch naming a
    dcid in `failing`, and records every batch it is asked for in `batches`. Like the graph
    client, it answers `None` for a node holding none of the properties a read names, and its
    `->*` read takes a byte cap."""

    def __init__(
        self,
        arc_targets: Mapping[str, Mapping[str, Sequence[ArcTarget]]],
        *,
        failing: frozenset[str] = frozenset(),
    ) -> None:
        super().__init__(arc_targets=arc_targets)
        self._failing = failing
        self.batches: list[tuple[str, ...]] = []

    def node_arc_targets_many(self, dcids, **kwargs):
        self.batches.append(tuple(dcids))
        if self._failing & set(dcids):
            raise GraphReadError("read failed", upstream_status=503)
        entries = super().node_arc_targets_many(dcids, **kwargs)
        return {
            dcid: entry if entry and any(entry.values()) else None
            for dcid, entry in entries.items()
        }

    def node_all_arc_targets_many(self, dcids, *, max_bytes=None, deadline=None):
        return super().node_all_arc_targets_many(dcids, deadline=deadline)


type _Answer = Sequence[
    tuple[str, str | list[str] | None] | tuple[str, str | list[str] | None, float] | dict
]


class _Indicators:
    """Indicator resolves answered from `answers`, keyed by `(target, un)`, each a list of
    `(dcid, typeOf)` scored 0.9 or `(dcid, typeOf, score)`, where a `None` typeOf leaves the
    member out, or a raw candidate. A key in `failing` raises its exception, and a key in `slow`
    sleeps that many seconds first. `calls` records `(query, target, un)`."""

    def __init__(
        self,
        answers: Mapping[tuple[str, bool], _Answer],
        *,
        failing: Mapping[tuple[str, bool], Exception] | None = None,
        slow: Mapping[tuple[str, bool], float] | None = None,
    ) -> None:
        self._answers = answers
        self._failing = failing or {}
        self._slow = slow or {}
        self._lock = threading.Lock()
        self.calls: list[tuple[str, str, bool]] = []

    def resolve_indicators(
        self, query: str, *, target: str, un: bool = False, deadline: float | None = None
    ) -> dict:
        with self._lock:
            self.calls.append((query, target, un))
        time.sleep(self._slow.get((target, un), 0.0))
        if (target, un) in self._failing:
            raise self._failing[(target, un)]
        candidates = []
        for entry in self._answers.get((target, un), ()):
            if isinstance(entry, dict):
                candidates.append(entry)
                continue
            dcid, type_of, score = entry if len(entry) == 3 else (*entry, 0.9)
            candidate: dict = {"dcid": dcid, "metadata": {"score": str(score), "sentence": "s"}}
            if type_of is not None:
                candidate["typeOf"] = type_of if isinstance(type_of, list) else [type_of]
            candidates.append(candidate)
        return {"entities": [{"node": query, "candidates": candidates}]}


_ANSWERS: dict[tuple[str, bool], _Answer] = {
    ("custom_only", False): [("dfo/OOP", _SV), ("dc/topic/Health", "Topic")],
    ("base_only", False): [("Count_Person", _SV)],
    ("custom_only", True): [("undata/sdg/SH_OOP", _SV), ("dfo/OOP", _SV)],
}


def _client(
    shared_resolver,
    indicators: _Indicators,
    *,
    mixer: _Reader | None = None,
    un: _Reader | None = None,
    un_generation: FakeGeneration | None = None,
):
    graph = FakeGraphNodeReader(nodes={"country/KEN": {"typeOf": ("Country",), "name": ("Kenya",)}})
    return make_client(
        graph=graph,
        resolver=make_resolver(shared_resolver, graph=graph),
        view_graph=mixer if mixer is not None else _Reader(_MIXER),
        un_view_graph=un if un is not None else _Reader(_UN),
        un_generation=un_generation,
        indicators=indicators,
    )


def test_hits_come_dcid_then_name_then_indicator_each_once_and_are_cached(
    serve_catalogue, shared_resolver
):
    serve_catalogue(FakeCatalogue())
    indicators = _Indicators(_ANSWERS)

    with _client(shared_resolver, indicators) as client:
        response = client.get(_ROUTE, params={"q": " Kenya ", "un": 1})
        calls = len(indicators.calls)
        repeat = client.get(_ROUTE, params={"q": "Kenya", "un": 1})

    assert response.status_code == 200, response.text
    assert response.json() == {
        "query": "Kenya",
        "hits": [
            {
                "ref": "country/KEN",
                "label": "Kenya",
                "types": ["Country"],
                "declared_by": "base",
                "via": "name",
                "score": None,
            },
            {
                "ref": "dfo/OOP",
                "label": "Out-of-pocket spending",
                "types": [_SV],
                "declared_by": "dfo",
                "via": "indicator",
                "score": 0.9,
            },
            {
                "ref": "undata/sdg/SH_OOP",
                "label": "Out-of-pocket share",
                "types": [_SV],
                "declared_by": "un_system",
                "via": "indicator",
                "score": 0.9,
            },
        ],
        "unavailable": [],
    }
    assert repeat.json() == response.json()
    assert len(indicators.calls) == calls == 2


@pytest.mark.parametrize(
    ("query", "un", "first", "read"),
    [
        ("dfo/OOP", False, ("dfo/OOP", "dfo"), True),
        ("undata/sdg/SH_OOP", True, ("undata/sdg/SH_OOP", "un_system"), True),
        ("undata/sdg/SH_OOP", False, None, False),
        ("NoSuchNode", True, None, True),
        ("dfo/DescOnly", False, ("dfo/DescOnly", None), True),
        ("dfo/OOP?", False, None, False),
    ],
    ids=["mixer", "un", "un-off", "absent", "description-only", "not-dcid-syntax"],
)
def test_a_whole_dcid_is_the_first_hit_where_a_read_holds_it(
    serve_catalogue,
    shared_resolver,
    query: str,
    un: bool,
    first: tuple[str, str] | None,
    read: bool,
):
    serve_catalogue(FakeCatalogue())
    mixer, un_reader = _Reader(_MIXER), _Reader(_UN)

    with _client(shared_resolver, _Indicators({}), mixer=mixer, un=un_reader) as client:
        response = client.get(_ROUTE, params={"q": query, "un": int(un)})

    assert response.status_code == 200, response.text
    dcid_hits = [
        (hit["ref"], hit["declared_by"], hit["score"])
        for hit in response.json()["hits"]
        if hit["via"] == "dcid"
    ]
    assert dcid_hits == ([] if first is None else [(*first, None)])
    assert ((query,) in mixer.batches + un_reader.batches) == read


@pytest.mark.parametrize(
    ("base", "un", "un_served", "sent"),
    [
        (False, False, True, {("custom_only", False)}),
        (True, False, True, {("custom_only", False), ("base_only", False)}),
        (False, True, True, {("custom_only", False), ("custom_only", True)}),
        (False, True, False, {("custom_only", False)}),
    ],
    ids=["dfo", "base", "un", "un-not-served"],
)
def test_the_switches_choose_the_indicator_resolves_and_the_un_reads(
    serve_catalogue,
    shared_resolver,
    base: bool,
    un: bool,
    un_served: bool,
    sent: set[tuple[str, bool]],
):
    serve_catalogue(FakeCatalogue(un_system_enabled=un_served))
    indicators = _Indicators(_ANSWERS)
    un_reader = _Reader(_UN)

    with _client(shared_resolver, indicators, un=un_reader) as client:
        response = client.get(
            _ROUTE, params={"q": "out of pocket", "base": int(base), "un": int(un)}
        )

    assert response.status_code == 200, response.text
    assert {(target, on_un) for _, target, on_un in indicators.calls} == sent
    assert len(indicators.calls) == len(sent)
    assert bool(un_reader.batches) == (un and un_served)


def test_an_indicator_leg_keeps_its_first_valid_statvars_and_untyped_candidates(
    serve_catalogue, shared_resolver, monkeypatch
):
    serve_catalogue(FakeCatalogue())
    monkeypatch.setattr(schema_search, "INDICATOR_CANDIDATES", 3)
    answers: dict[tuple[str, bool], _Answer] = {
        ("custom_only", False): [
            ("dc/topic/T", "Topic"),
            {"dcid": "sv/no-score", "typeOf": [_SV], "metadata": {}},
            ("sv/untyped", None),
            ("sv/second", ["Thing", _SV]),
            *((f"sv/{i}", _SV) for i in range(3)),
        ]
    }

    with _client(shared_resolver, _Indicators(answers)) as client:
        response = client.get(_ROUTE, params={"q": "out of pocket"})

    assert [(hit["ref"], hit["types"]) for hit in response.json()["hits"]] == [
        ("sv/untyped", [_SV]),
        ("sv/second", [_SV]),
        ("sv/0", [_SV]),
    ]


def test_indicator_hits_merge_by_score_above_the_floor(serve_catalogue, shared_resolver):
    """Every indicator leg's hits merge by descending score after the name hit, a tie keeping
    leg order, a candidate under the floor is dropped, and a ref keeps its first place."""
    serve_catalogue(FakeCatalogue())
    answers: dict[tuple[str, bool], _Answer] = {
        ("custom_only", False): [("sv/0", _SV, 0.5), ("sv/1", _SV, 0.2), ("sv/2", _SV, 0.45)],
        ("base_only", False): [("Count_Person", _SV, 0.8), ("sv/second", _SV, 0.5)],
        ("custom_only", True): [("undata/sdg/SH_OOP", _SV, 0.6), ("sv/0", _SV, 0.4)],
    }

    with _client(shared_resolver, _Indicators(answers)) as client:
        response = client.get(_ROUTE, params={"q": "Kenya", "base": 1, "un": 1})

    assert response.status_code == 200, response.text
    assert [(hit["ref"], hit["score"]) for hit in response.json()["hits"]] == [
        ("country/KEN", None),
        ("Count_Person", 0.8),
        ("undata/sdg/SH_OOP", 0.6),
        ("sv/0", 0.5),
        ("sv/second", 0.5),
        ("sv/2", 0.45),
    ]


@pytest.mark.parametrize(
    ("base", "failing", "dcid_fails", "unavailable"),
    [
        (False, ("custom_only", True), False, ["un_system"]),
        (True, ("base_only", False), False, ["base"]),
        (False, None, True, ["dcid"]),
    ],
    ids=["un", "mixer", "dcid"],
)
def test_a_failing_leg_is_named_and_the_rest_answer(
    serve_catalogue,
    shared_resolver,
    base: bool,
    failing: tuple[str, bool] | None,
    dcid_fails: bool,
    unavailable: list[str],
):
    serve_catalogue(FakeCatalogue())
    indicators = _Indicators(
        _ANSWERS, failing={} if failing is None else {failing: RuntimeError("resolve failed")}
    )
    mixer = _Reader(_MIXER, failing=frozenset({"Kenya"} if dcid_fails else ()))

    with _client(shared_resolver, indicators, mixer=mixer) as client:
        response = client.get(_ROUTE, params={"q": "Kenya", "base": int(base), "un": 1})
        calls = len(indicators.calls)
        client.get(_ROUTE, params={"q": "Kenya", "base": int(base), "un": 1})

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["unavailable"] == unavailable
    assert body["hits"][0]["ref"] == "country/KEN"
    assert ("dfo/OOP" in [hit["ref"] for hit in body["hits"]]) == ("dfo" not in unavailable)
    # A partial answer is not kept, so the repeat resolves again.
    assert len(indicators.calls) == 2 * calls


def test_indicator_hits_the_graph_does_not_hold_as_statvars_are_dropped(
    serve_catalogue, shared_resolver
):
    """A candidate the label read does not find is dropped, and so is an untyped one the graph
    types other than StatisticalVariable. The answer is complete and kept."""
    serve_catalogue(FakeCatalogue())
    answers: dict[tuple[str, bool], _Answer] = {
        ("custom_only", False): [
            ("dead/SV", _SV),
            ("dc/topic/Live", None),
            ("sv/untyped", None),
            ("dfo/OOP", _SV),
        ]
    }
    indicators = _Indicators(answers)

    with _client(shared_resolver, indicators) as client:
        response = client.get(_ROUTE, params={"q": "out of pocket"})
        client.get(_ROUTE, params={"q": "out of pocket"})

    body = response.json()
    assert [hit["ref"] for hit in body["hits"]] == ["sv/untyped", "dfo/OOP"]
    assert body["unavailable"] == []
    assert len(indicators.calls) == 1


def test_a_new_graph_release_misses_the_cache(serve_catalogue, shared_resolver):
    """The entity resolver admits by the bound release's code index, so an answer is kept per
    release."""
    serve_catalogue(FakeCatalogue())
    indicators = _Indicators(_ANSWERS)
    releases = _SwappableReleases(release_at_revision("rev-1"))

    with _serving_releases(releases), _client(shared_resolver, indicators) as client:
        client.get(_ROUTE, params={"q": "out of pocket"})
        client.get(_ROUTE, params={"q": "out of pocket"})
        cached = len(indicators.calls)
        releases.release = release_at_revision("rev-2")
        client.get(_ROUTE, params={"q": "out of pocket"})

    assert cached == 1
    assert len(indicators.calls) == 2


async def _no_names(deps, body, **kwargs):
    return ResolveResponse(results=[ResolveResult(input=body.texts[0], status="not_found")])


def test_names_legs_run_one_at_a_time_without_near_matches(
    serve_catalogue, shared_resolver, monkeypatch
):
    """Two searches at once share the shared resolver one names leg at a time, and each asks
    for no near matches."""
    serve_catalogue(FakeCatalogue())
    running = 0
    most = 0
    asked: list[bool] = []

    async def counting(deps, body, **kwargs):
        nonlocal running, most
        asked.append(kwargs["near_matches"])
        running += 1
        most = max(most, running)
        await asyncio.sleep(0.2)
        running -= 1
        return await _no_names(deps, body)

    monkeypatch.setattr(schema_search, "run_resolve", counting)

    with _client(shared_resolver, _Indicators({})) as client:
        threads = [
            threading.Thread(target=client.get, args=(_ROUTE,), kwargs={"params": {"q": q}})
            for q in ("first query", "second query")
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=5.0)

    assert asked == [False, False]
    assert most == 1


def test_a_late_leg_and_a_refused_leg_are_named(serve_catalogue, shared_resolver, monkeypatch):
    """With one worker the mixer resolve, submitted first, holds it, so the UN resolve is
    refused. With a short leg cap, a mixer resolve slower than the cap is late. The names leg
    answers at once, so the cap reaches only the legs under test."""
    serve_catalogue(FakeCatalogue())
    monkeypatch.setattr(schema_search, "run_resolve", _no_names)
    monkeypatch.setattr(schema_search, "SCHEMA_SEARCH_WORKERS", 1)
    monkeypatch.setattr(schema_search, "SCHEMA_SEARCH_LEG_SECONDS", 0.3)
    indicators = _Indicators(_ANSWERS, slow={("custom_only", False): 0.6})

    with _client(shared_resolver, indicators) as client:
        response = client.get(_ROUTE, params={"q": "out of pocket", "un": 1})

    assert response.status_code == 200, response.text
    assert response.json()["unavailable"] == ["dfo", "un_system"]


def test_a_failed_label_read_keeps_the_other_reads_facts_and_is_not_cached(
    serve_catalogue, shared_resolver
):
    serve_catalogue(FakeCatalogue())
    indicators = _Indicators(_ANSWERS)
    un_reader = _Reader(_UN, failing=frozenset({"undata/sdg/SH_OOP"}))

    with _client(shared_resolver, indicators, un=un_reader) as client:
        response = client.get(_ROUTE, params={"q": "out of pocket", "un": 1})
        calls = len(indicators.calls)
        client.get(_ROUTE, params={"q": "out of pocket", "un": 1})

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["unavailable"] == ["labels"]
    assert [
        (hit["ref"], hit["label"], hit["types"], hit["declared_by"]) for hit in body["hits"]
    ] == [
        ("dfo/OOP", "Out-of-pocket spending", [_SV], "dfo"),
        ("undata/sdg/SH_OOP", None, [_SV], None),
    ]
    assert len(indicators.calls) == 2 * calls


def test_a_new_un_generation_misses_the_cache(serve_catalogue, shared_resolver):
    serve_catalogue(FakeCatalogue())
    indicators = _Indicators(_ANSWERS)
    un_generation = FakeGeneration(token="un-1")

    with _client(shared_resolver, indicators, un_generation=un_generation) as client:
        client.get(_ROUTE, params={"q": "out of pocket", "un": 1})
        client.get(_ROUTE, params={"q": "out of pocket", "un": 1})
        cached = len(indicators.calls)
        un_generation.advance("un-2")
        client.get(_ROUTE, params={"q": "out of pocket", "un": 1})

    assert cached == 2
    assert len(indicators.calls) == 4


def _read_error(status: int, retry_after: float | None) -> GraphReadError:
    error = GraphReadError("resolve failed", upstream_status=status)
    error.retry_after = retry_after
    return error


@pytest.mark.parametrize(
    ("status", "waits", "retry_after", "cause"),
    [
        (503, (None, None), None, None),
        (429, (7.2, None), "8", "upstream_rate_limited"),
        (429, (5.0, 30.0), "30", "upstream_rate_limited"),
        (429, (None, None), "5", "upstream_rate_limited"),
    ],
    ids=["failing", "rate-limited", "rate-limited-longest", "rate-limited-no-wait"],
)
def test_every_leg_failing_answers_503(
    serve_catalogue,
    shared_resolver,
    status: int,
    waits: tuple[float | None, float | None],
    retry_after: str | None,
    cause: str | None,
):
    """A query with whitespace past the resolver's 200 characters runs only the indicator
    legs. A rate-limited refusal waits the longest `Retry-After` the reads named, and the
    schema routes' 5 s where none named one."""
    serve_catalogue(FakeCatalogue())
    failing = {
        key: _read_error(status, wait)
        for key, wait in zip((("custom_only", False), ("custom_only", True)), waits, strict=True)
    }

    with _client(shared_resolver, _Indicators({}, failing=failing)) as client:
        response = client.get(_ROUTE, params={"q": "a " * 110, "un": 1})

    assert response.status_code == 503, response.text
    body = response.json()
    assert (body["code"], body["retryable"]) == ("service_unavailable", True)
    assert (body.get("cause") or {}).get("code") == cause
    assert response.headers.get("Retry-After") == retry_after


@pytest.mark.parametrize(
    "params", [{"q": "   "}, {"q": "x" * 257}, {}], ids=["blank", "long", "none"]
)
def test_a_blank_or_long_query_is_refused(serve_catalogue, shared_resolver, params):
    serve_catalogue(FakeCatalogue())
    indicators = _Indicators({})

    with _client(shared_resolver, indicators) as client:
        response = client.get(_ROUTE, params=params)

    assert response.status_code == 400, response.text
    assert response.json()["code"] == "request_invalid"
    assert indicators.calls == []
