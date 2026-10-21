"""The default place scope Search and the datasets mint share (`materialise.place_scope`): every
place a StatVar's availability lists that the graph knows, settled once per graph generation,
else its countries with data.

Each branch of the rule runs against fakes whose reads the test releases, so no assertion depends
on how long a read takes: the request's wait (`EVERY_PLACE_SECONDS`) is as long as a read may run
(`SETTLE_DEADLINE_SECONDS`) unless a test sets it to zero to cut a read, and background reads are
drained before the memo is checked."""

from __future__ import annotations

import contextvars
import threading
import time
from collections.abc import Collection, Iterator, Mapping, Sequence
from concurrent.futures import Executor, Future, ThreadPoolExecutor, wait

import httpx
import pytest

from dc_graph import GraphDeadlineExceededError, GraphReadError
from dc_sdmx import (
    EntityFilter,
    SdmxCapExceededError,
    SdmxClient,
    SdmxHttpError,
    SdmxTransportError,
)
from dfo_platform.request_deadline import bounded_deadline, request_deadline
from graph_generation import FakeGeneration, GenerationStamp
from materialise import place_scope
from materialise.errors import MaterialisationUsageError
from materialise.place_scope import (
    AVAILABILITY_WORKERS,
    MAX_EVERY_PLACE,
    MAX_EVERY_PLACE_BYTES,
    NO_DEFAULT_SCOPE,
    SETTLE_DEADLINE_SECONDS,
    SETTLE_SECONDS,
    AvailabilityReader,
    DefaultPlaceScope,
    DefaultPresence,
    DefaultScopeUnknownError,
    PlaceScopeDeadlineError,
    PlaceScopeReadError,
    clear_memo,
    default_place_scope,
    is_worldwide,
    memo_stats,
    read_default_presence,
    scope_of_places,
    scope_places,
)
from origin_catalogue import bound_catalogue, current_catalogue
from tests.materialise._scope_availability import ScopeAvailability

_KEN, _TGO, _UGA = "country/KEN", "country/TGO", "country/UGA"
_NAIROBI, _AFRICA = "wikidataId/Q3870", "africa"
_COUNTRIES = (_UGA, _KEN, _TGO)
_UN_VARIABLE = "undata/who/MORT_U5_RATE"
_QUEUE_BOUND = place_scope.MAX_QUEUED_SETTLES
_CALLER: contextvars.ContextVar[str | None] = contextvars.ContextVar("caller", default=None)


@pytest.fixture(autouse=True)
def _fresh_memo(monkeypatch) -> Iterator[None]:
    # A request waits for its read to end, so a read slowed past the production window by a
    # loaded machine still answers. A test that cuts a read sets the window to zero.
    monkeypatch.setattr(place_scope, "EVERY_PLACE_SECONDS", SETTLE_DEADLINE_SECONDS)
    _drain()
    clear_memo()
    yield
    _drain()
    clear_memo()


def _drain() -> None:
    """Wait for every availability read in flight, including those a read queued. A flight
    leaves `_flights` before it resolves, so a finished flight still listed is a leak."""
    deadline = time.monotonic() + 10
    while flights := list(place_scope._flights.values()):
        pending = [flight for flight in flights if not flight.done()]
        assert pending, f"finished flights left in _flights: {flights}"
        assert time.monotonic() < deadline, "availability reads in flight never ended"
        wait(pending, timeout=1)


class _Availability:
    """Answers each unfiltered read with the places `data` lists for the StatVar, raising
    `SdmxCapExceededError` past `max_values`. A StatVar in `errors` raises its error; one in
    `blocked` waits for its event first. `source` names its backend. `reads` records each
    read's StatVars and keywords."""

    def __init__(
        self,
        *,
        data: Collection[tuple[str, str]] = (),
        errors: Mapping[str, Exception] | None = None,
        blocked: Mapping[str, threading.Event] | None = None,
        source: str = "fake-availability",
    ) -> None:
        self._source = source
        self._data = tuple(data)
        self._errors = dict(errors or {})
        self._blocked = dict(blocked or {})
        self._lock = threading.Lock()
        self.reads: list[tuple[tuple[str, ...], dict[str, object]]] = []

    def source_key(self) -> str:
        return self._source

    def availability_values(
        self,
        *,
        variables: Sequence[str],
        component: str,
        filters: Mapping[str, EntityFilter] | None = None,
        max_values: int | None = None,
        max_bytes: int | None = None,
        deadline: float | None = None,
        read_seconds: float | None = None,
    ) -> list[str]:
        assert component == "observationAbout"
        assert filters is None
        with self._lock:
            self.reads.append(
                (
                    tuple(variables),
                    {
                        "max_values": max_values,
                        "max_bytes": max_bytes,
                        "deadline": deadline,
                        "read_seconds": read_seconds,
                    },
                )
            )
        (stat_var,) = variables
        if (event := self._blocked.get(stat_var)) is not None:
            assert event.wait(timeout=30)
        if (error := self._errors.get(stat_var)) is not None:
            raise error
        places = [place for sv, place in self._data if sv == stat_var]
        if max_values is not None and len(places) > max_values:
            raise SdmxCapExceededError(
                "too many places", cap_kind="values", cap=max_values, observed=len(places)
            )
        return places

    def read_stat_vars(self) -> list[str]:
        with self._lock:
            return [sv for variables, _ in self.reads for sv in variables]


class _Graph:
    """`admitted` holds the `(StatVar, place)` pairs the countries fallback finds admitted data at,
    and `countries` Earth's countries. `nodes` names the places the graph holds a node for, every
    place by default. `node_error` makes the node check raise it, and each read kind in `expire`
    waits out its deadline and raises `GraphDeadlineExceededError`. `source` names its backend.
    `reads` records each read's kind: `"countries"`, `"presence"` with its StatVars, or `"nodes"`
    with its places. `deadlines` records the deadline each countries-fallback read receives."""

    def __init__(
        self,
        *,
        admitted: Collection[tuple[str, str]] = (),
        countries: Sequence[str] = _COUNTRIES,
        nodes: Collection[str] | None = None,
        node_error: Exception | None = None,
        un_availability: AvailabilityReader | None = None,
        generation: FakeGeneration | None = None,
        expire: Collection[str] = (),
        source: str = "fake-graph",
    ) -> None:
        self._source = source
        self._admitted = frozenset(admitted)
        self._countries = list(countries)
        self._nodes = None if nodes is None else frozenset(nodes)
        self.node_error = node_error
        self._un_availability = un_availability
        self.generation = generation if generation is not None else FakeGeneration(token="r1")
        self.reads: list[tuple[str, tuple[str, ...]]] = []
        self._expire = frozenset(expire)
        self.deadlines: dict[str, float | None] = {}

    def _fallback_read(self, kind: str, deadline: float | None) -> None:
        self.deadlines[kind] = deadline
        if kind in self._expire:
            assert deadline is not None
            time.sleep(max(0.0, deadline - time.monotonic()))
            raise GraphDeadlineExceededError(f"the {kind} read ran past its deadline")

    def source_key(self) -> str:
        return self._source

    def generation_stamp(self) -> GenerationStamp:
        return self.generation.stamp()

    def un_availability(self) -> AvailabilityReader | None:
        return self._un_availability

    def node_arcs_many(
        self, dcids: Sequence[str], *, properties: Sequence[str], deadline: float | None = None
    ) -> Mapping[str, Mapping[str, tuple[str, ...]] | None]:
        assert tuple(properties) == ("typeOf",)
        self.reads.append(("nodes", tuple(dcids)))
        if self.node_error is not None:
            raise self.node_error
        return {
            dcid: {"typeOf": ("Place",)} if self._nodes is None or dcid in self._nodes else None
            for dcid in dcids
        }

    def child_places_batch(
        self, parent_dcids: list[str], child_type: str, *, deadline: float | None = None
    ) -> dict[str, list[str]]:
        assert (parent_dcids, child_type) == (["Earth"], "Country")
        self.reads.append(("countries", ()))
        self._fallback_read("countries", deadline)
        return {"Earth": list(self._countries)}

    def entities_with_data_by_variable(
        self,
        variable_dcids: list[str],
        entity_dcids: list[str],
        *,
        deadline: float | None = None,
    ) -> dict[str, list[str]]:
        assert entity_dcids == [*self._countries, "Earth"]
        self.reads.append(("presence", tuple(variable_dcids)))
        self._fallback_read("presence", deadline)
        return {
            sv: [entity for entity in entity_dcids if (sv, entity) in self._admitted]
            for sv in variable_dcids
        }

    def kinds(self) -> list[str]:
        return [kind for kind, _ in self.reads]


_EVERY = [("A", _TGO), ("A", "Earth"), ("A", _AFRICA), ("A", _NAIROBI), ("A", _KEN)]
_ADMITTED = [("A", _KEN), ("A", "Earth"), ("D", _UGA), ("D", _KEN), ("W", "Earth")]


def _countries_of_d() -> DefaultPlaceScope:
    return DefaultPlaceScope(places=(_KEN, _UGA), every_place=False)


def _fallback_logs(caplog: pytest.LogCaptureFixture) -> list[tuple[object, object]]:
    """The StatVar and outcome of each countries fallback an availability read logged."""
    return [
        (record.args[0], record.args[1])
        for record in caplog.records
        if record.name == place_scope.__name__
        and record.levelname == "WARNING"
        and isinstance(record.args, tuple)
    ]


# -- every place --


def test_the_scope_is_every_place_availability_lists_by_dcid_with_earth_last():
    availability, graph = _Availability(data=_EVERY), _Graph()
    submitted = time.monotonic()

    presence = read_default_presence(graph=graph, availability=availability, stat_vars=["A"])

    returned = time.monotonic()
    assert presence.scope("A") == DefaultPlaceScope(
        places=(_AFRICA, _KEN, _TGO, _NAIROBI, "Earth"), every_place=True
    )
    assert presence.unsettled == frozenset()
    assert presence.cold_reads == 1
    # No countries read: nothing fell back.
    assert presence.countries == ()
    assert graph.reads == [("nodes", (_AFRICA, _KEN, _TGO, _NAIROBI, "Earth"))]
    ((variables, keywords),) = availability.reads
    deadline = keywords.pop("deadline")
    assert variables == ("A",)
    assert keywords == {
        "max_values": MAX_EVERY_PLACE,
        "max_bytes": MAX_EVERY_PLACE_BYTES,
        "read_seconds": SETTLE_SECONDS,
    }
    # The read's absolute bound runs from its submission, inside the request.
    assert isinstance(deadline, float)
    assert submitted + SETTLE_DEADLINE_SECONDS <= deadline <= returned + SETTLE_DEADLINE_SECONDS


def test_one_node_check_covers_every_list_and_drops_places_the_graph_does_not_know():
    availability = _Availability(
        data=[*_EVERY, ("B", "country/SouthAfrica"), ("B", "country/ZAF"), ("B", _KEN)]
    )
    graph = _Graph(nodes={_AFRICA, _KEN, _TGO, "country/ZAF", "Earth"})

    presence = read_default_presence(graph=graph, availability=availability, stat_vars=["A", "B"])

    assert presence.scope("A").places == (_AFRICA, _KEN, _TGO, "Earth")
    assert presence.scope("B").places == (_KEN, "country/ZAF")
    assert graph.kinds() == ["nodes"]
    assert sorted(graph.reads[0][1]) == sorted(
        {_AFRICA, _KEN, _TGO, _NAIROBI, "Earth", "country/SouthAfrica", "country/ZAF"}
    )


def test_a_stat_var_availability_lists_nowhere_has_an_empty_every_place_scope():
    availability, graph = _Availability(), _Graph()

    scope = default_place_scope("X", graph=graph, availability=availability)

    assert scope == NO_DEFAULT_SCOPE
    assert graph.reads == []


# -- the memo --


def test_a_settled_scope_is_read_once_per_generation():
    availability, graph = _Availability(data=_EVERY), _Graph()
    first = read_default_presence(graph=graph, availability=availability, stat_vars=["A"])

    again = read_default_presence(graph=graph, availability=availability, stat_vars=["A"])

    assert again.scopes == first.scopes
    assert again.cold_reads == 0
    assert availability.read_stat_vars() == ["A"]
    assert graph.kinds() == ["nodes"]


def test_the_memo_is_shared_by_every_graph_client_on_one_generation():
    availability = _Availability(data=_EVERY)
    read_default_presence(graph=_Graph(), availability=availability, stat_vars=["A"])

    other = _Graph()
    read_default_presence(graph=other, availability=availability, stat_vars=["A"])

    assert availability.read_stat_vars() == ["A"]
    assert other.reads == []


@pytest.mark.parametrize(
    ("availability_source", "graph_source", "shared"),
    [
        pytest.param("mixer-a", "graph-a", True, id="same-backends"),
        pytest.param("mixer-b", "graph-a", False, id="other-availability"),
        pytest.param("mixer-a", "graph-b", False, id="other-graph"),
    ],
)
def test_the_memo_is_shared_only_by_readers_of_the_same_backends(
    availability_source: str, graph_source: str, shared: bool
):
    read_default_presence(
        graph=_Graph(source="graph-a"),
        availability=_Availability(data=[("A", _KEN)], source="mixer-a"),
        stat_vars=["A"],
    )
    availability = _Availability(data=[("A", _UGA)], source=availability_source)

    scope = default_place_scope("A", graph=_Graph(source=graph_source), availability=availability)

    assert scope.places == ((_KEN,) if shared else (_UGA,))
    assert availability.read_stat_vars() == ([] if shared else ["A"])


def test_a_source_scope_reads_and_memoises_apart_from_the_unscoped_request():
    """The countries fallback counts admitted facets, which a source scope narrows, so a
    fallback memoised under one scope never answers a request under another."""
    availability = _Availability(errors={"D": SdmxHttpError("server error", upstream_status=500)})
    graph = _Graph(admitted=_ADMITTED)
    with bound_catalogue(current_catalogue().scoped(frozenset({"provenance/X"}))):
        read_default_presence(graph=graph, availability=availability, stat_vars=["D"])

    read_default_presence(graph=graph, availability=availability, stat_vars=["D"])

    assert availability.read_stat_vars() == ["D", "D"]


def test_a_generation_change_clears_the_memo_and_reads_the_scope_again():
    availability, graph = _Availability(data=_EVERY), _Graph()
    graph.generation.register("default_place_scope", stats=memo_stats, clear=clear_memo)
    read_default_presence(graph=graph, availability=availability, stat_vars=["A"])
    assert memo_stats()["entries"] == 1

    graph.generation.advance("r2")

    assert memo_stats()["entries"] == 0
    read_default_presence(graph=graph, availability=availability, stat_vars=["A"])
    assert availability.read_stat_vars() == ["A", "A"]


def test_a_new_stamp_misses_the_memo_without_a_clear():
    availability = _Availability(data=_EVERY)
    read_default_presence(graph=_Graph(), availability=availability, stat_vars=["A"])

    read_default_presence(
        graph=_Graph(generation=FakeGeneration(token="r2")),
        availability=availability,
        stat_vars=["A"],
    )

    assert availability.read_stat_vars() == ["A", "A"]


# -- the countries fallback --


@pytest.mark.parametrize(
    "error",
    [
        SdmxCapExceededError("values", cap_kind="values", cap=MAX_EVERY_PLACE, observed=9_000),
        SdmxCapExceededError("bytes", cap_kind="bytes", cap=MAX_EVERY_PLACE_BYTES, observed=1),
        SdmxHttpError("server error", upstream_status=500),
    ],
    ids=["value-cap", "byte-cap", "http-500"],
)
def test_a_cap_or_server_error_falls_back_to_countries_and_is_memoised(error, caplog):
    availability = _Availability(errors={"D": error})
    graph = _Graph(admitted=_ADMITTED)

    presence = read_default_presence(graph=graph, availability=availability, stat_vars=["D"])

    assert _fallback_logs(caplog) == [("D", "definite")]

    assert presence.scope("D") == _countries_of_d()
    assert presence.countries == _COUNTRIES
    assert presence.unsettled == frozenset()
    assert graph.reads == [("countries", ()), ("presence", ("D",))]

    again = read_default_presence(graph=graph, availability=availability, stat_vars=["D"])

    assert again.scope("D") == _countries_of_d()
    assert availability.read_stat_vars() == ["D"]
    assert graph.kinds() == ["countries", "presence"]


def test_the_value_cap_is_counted_on_the_list():
    availability = _Availability(data=[("D", f"geoId/{n:05}") for n in range(MAX_EVERY_PLACE + 1)])

    scope = default_place_scope("D", graph=_Graph(admitted=_ADMITTED), availability=availability)

    assert scope == _countries_of_d()


@pytest.mark.parametrize(
    "error",
    [
        SdmxHttpError("bad gateway", upstream_status=502),
        SdmxHttpError("unavailable", upstream_status=503),
        SdmxHttpError("gateway timeout", upstream_status=504),
        TimeoutError("past the settle bound"),
    ],
    ids=["502", "503", "504", "settle-bound"],
)
def test_a_gateway_status_or_the_settle_bound_falls_back_for_the_request_only(error, caplog):
    availability = _Availability(errors={"D": error})
    graph = _Graph(admitted=_ADMITTED)

    presence = read_default_presence(graph=graph, availability=availability, stat_vars=["D"])

    assert presence.scope("D") == _countries_of_d()
    assert presence.unsettled == {"D"}
    assert _fallback_logs(caplog) == [("D", "transient")]

    read_default_presence(graph=graph, availability=availability, stat_vars=["D"])

    assert availability.read_stat_vars() == ["D", "D"]
    assert graph.kinds() == ["countries", "presence", "countries", "presence"]


def test_stat_vars_falling_back_together_share_one_presence_read():
    cap = SdmxCapExceededError("values", cap_kind="values", cap=1, observed=2)
    availability = _Availability(errors={"D": cap, "W": cap, "X": cap})
    graph = _Graph(admitted=_ADMITTED)

    presence = read_default_presence(
        graph=graph, availability=availability, stat_vars=["D", "W", "X"]
    )

    assert presence.scope("D") == _countries_of_d()
    # Earth alone when no country holds admitted data, else nothing.
    assert presence.scope("W") == DefaultPlaceScope(places=("Earth",), every_place=False)
    assert is_worldwide(presence.scope("W").places)
    assert presence.scope("X") == DefaultPlaceScope(places=(), every_place=False)
    assert graph.reads == [("countries", ()), ("presence", ("D", "W", "X"))]


def test_a_known_presence_lends_its_countries_to_the_fallback():
    cap = SdmxCapExceededError("values", cap_kind="values", cap=1, observed=2)
    availability = _Availability(errors={"D": cap})
    graph = _Graph(admitted=_ADMITTED)
    known = DefaultPresence(countries=_COUNTRIES, scopes={})

    presence = read_default_presence(
        graph=graph, availability=availability, stat_vars=["D"], known=known
    )

    assert presence.scope("D") == _countries_of_d()
    assert graph.kinds() == ["presence"]


def test_an_earth_listing_naming_no_country_refuses():
    """Earth holds every country, so a listing naming none failed. Taken as "no country", it
    would leave Earth's worldwide total in place of every country's data."""
    cap = SdmxCapExceededError("values", cap_kind="values", cap=1, observed=2)
    graph = _Graph(admitted=_ADMITTED, countries=())

    with pytest.raises(DefaultScopeUnknownError, match="'Earth' named no 'Country'"):
        read_default_presence(
            graph=graph, availability=_Availability(errors={"D": cap}), stat_vars=["D"]
        )

    assert graph.kinds() == ["countries"]


@pytest.mark.parametrize(
    ("error", "status"),
    [
        (SdmxHttpError("bad query", upstream_status=400), 400),
        (SdmxTransportError("connection reset"), None),
        (ValueError("malformed body"), None),
    ],
    ids=["http-400", "transport", "malformed"],
)
def test_any_other_availability_error_raises_one_place_scope_read_error(error, status, caplog):
    """Reads are single-flight across doors, so the error a joiner gets carries no reader's
    own type: each door refuses `PlaceScopeReadError`, a `GraphReadError`."""
    availability = _Availability(errors={"A": error})

    with pytest.raises(PlaceScopeReadError) as raised:
        read_default_presence(graph=_Graph(), availability=availability, stat_vars=["A"])

    assert isinstance(raised.value, GraphReadError)
    assert raised.value.upstream_status == status
    assert raised.value.__cause__ is error
    assert memo_stats()["entries"] == 0
    (record,) = [r for r in caplog.records if r.name == place_scope.__name__]
    assert (record.levelname, record.args[0]) == ("WARNING", "A")


@pytest.mark.parametrize(
    "error", [TypeError("bad argument"), AssertionError("broken invariant")], ids=["type", "assert"]
)
def test_a_defect_in_the_availability_read_propagates_as_itself(error):
    availability = _Availability(errors={"A": error})

    with pytest.raises(type(error)) as raised:
        read_default_presence(graph=_Graph(), availability=availability, stat_vars=["A"])

    assert raised.value is error
    assert memo_stats()["entries"] == 0


def test_a_server_error_through_the_sdmx_client_is_memoised():
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(500, json={"message": "internal"})

    client = SdmxClient(
        base_url="https://sdmx.example.org/core/api", transport=httpx.MockTransport(handler)
    )
    graph = _Graph(admitted=_ADMITTED)

    assert default_place_scope("D", graph=graph, availability=client) == _countries_of_d()
    sent = len(requests)
    assert default_place_scope("D", graph=graph, availability=client) == _countries_of_d()

    assert sent >= 1
    assert len(requests) == sent
    assert requests[0].url.params["c[variableMeasured]"] == "D"


@pytest.mark.parametrize(
    ("second_root", "shared"),
    [
        pytest.param("https://a.example.org/core/api", True, id="same-root"),
        pytest.param("https://b.example.org/core/api", False, id="other-root"),
    ],
)
def test_sdmx_clients_share_a_scope_only_on_the_same_api_root(second_root: str, shared: bool):
    def client(root: str, availability: ScopeAvailability) -> SdmxClient:
        transport = httpx.MockTransport(lambda request: availability.respond(request))
        return SdmxClient(base_url=root, transport=transport)

    graph = _Graph()
    default_place_scope(
        "A",
        graph=graph,
        availability=client(
            "https://a.example.org/core/api", ScopeAvailability(data=[("A", _KEN)])
        ),
    )
    second = ScopeAvailability(data=[("A", _UGA)])

    scope = default_place_scope("A", graph=graph, availability=client(second_root, second))

    assert scope.places == ((_KEN,) if shared else (_UGA,))
    assert second.reads == ([] if shared else [("A",)])


# -- the node check --


@pytest.mark.parametrize(
    "error",
    [GraphDeadlineExceededError("node check window"), GraphReadError("down", upstream_status=503)],
    ids=["window", "server-error"],
)
def test_a_node_check_past_its_window_or_failing_falls_back_for_the_request_only(error, caplog):
    availability = _Availability(data=[("D", _KEN), ("D", _AFRICA)])
    graph = _Graph(admitted=_ADMITTED, node_error=error)

    presence = read_default_presence(graph=graph, availability=availability, stat_vars=["D"])

    assert presence.scope("D") == _countries_of_d()
    assert presence.unsettled == {"D"}
    (record,) = [r for r in caplog.records if r.name == place_scope.__name__]
    assert (record.levelname, record.args[1]) == ("WARNING", ["D"])

    # The list stays memoised, so the next request checks its places again and reads no
    # availability.
    graph.node_error = None
    again = read_default_presence(graph=graph, availability=availability, stat_vars=["D"])

    assert again.scope("D") == DefaultPlaceScope(places=(_AFRICA, _KEN), every_place=True)
    assert again.unsettled == frozenset()
    assert availability.read_stat_vars() == ["D"]
    assert graph.kinds() == ["nodes", "countries", "presence", "nodes"]


def test_a_node_check_failing_on_the_client_side_propagates():
    graph = _Graph(node_error=GraphReadError("bad request", upstream_status=400))

    with pytest.raises(GraphReadError, match="bad request"):
        read_default_presence(graph=graph, availability=_Availability(data=_EVERY), stat_vars=["A"])


def test_a_later_check_reads_only_the_places_no_earlier_check_answered():
    """A place's answer, held or not, is memoised, so a variable whose list overlaps an earlier
    one's reads only its new places."""
    availability = _Availability(data=[*_EVERY, ("B", _KEN), ("B", _NAIROBI), ("B", "country/ZAF")])
    graph = _Graph(nodes={_AFRICA, _KEN, _TGO, "country/ZAF", "Earth"})

    read_default_presence(graph=graph, availability=availability, stat_vars=["A"])
    later = read_default_presence(graph=graph, availability=availability, stat_vars=["B"])

    assert later.scope("B").places == (_KEN, "country/ZAF")
    assert [places for kind, places in graph.reads if kind == "nodes"][1:] == [("country/ZAF",)]


class _HeldNodesGraph(_Graph):
    """Holds the first node check until `release` is set, then answers it or raises
    `first_error`, and signals `started` when it begins. With `first_error` set to
    `"deadline"`, it raises `GraphDeadlineExceededError` as a read its deadline ended does."""

    def __init__(self, *, first_error: Exception | str | None, **kwargs) -> None:
        super().__init__(**kwargs)
        self.started, self.release = threading.Event(), threading.Event()
        self._first_error = first_error
        self._held = True

    def node_arcs_many(self, dcids, *, properties, deadline=None):
        if not self._held:
            return super().node_arcs_many(dcids, properties=properties, deadline=deadline)
        self._held = False
        self.started.set()
        assert self.release.wait(timeout=5)
        if self._first_error is None:
            return super().node_arcs_many(dcids, properties=properties, deadline=deadline)
        self.reads.append(("nodes", tuple(dcids)))
        if self._first_error == "deadline":
            assert deadline is not None and time.monotonic() >= deadline
            raise GraphDeadlineExceededError("the node check ran past its deadline")
        raise self._first_error

    def node_reads(self) -> list[tuple[str, ...]]:
        return [places for kind, places in self.reads if kind == "nodes"]


_FIRST_PLACES = sorted({_TGO, "Earth", _AFRICA, _NAIROBI, _KEN})


@pytest.mark.parametrize(
    ("first_error", "reads"),
    [
        pytest.param(None, [_FIRST_PLACES, ["country/ZAF"]], id="answers"),
        pytest.param(
            GraphReadError("down", upstream_status=503),
            [_FIRST_PLACES, ["country/ZAF"]],
            id="fails",
        ),
        # The first check's own deadline ends the read it shares; the second, with time left,
        # reads the shared place itself.
        pytest.param(
            "deadline",
            [_FIRST_PLACES, ["country/KEN"], ["country/ZAF"]],
            id="owner-deadline",
        ),
    ],
)
def test_a_check_waits_for_the_places_another_check_is_reading(first_error, reads):
    """Concurrent checks read each place once: the second reads only the places the first is
    not reading and takes the first's answers. When that read fails it falls back for the
    request, as the first does, unless the first's own deadline ended it: then it reads those
    places once under its own deadline and answers."""
    availability = _Availability(data=[*_EVERY, ("D", _KEN), ("D", "country/ZAF")])
    graph = _HeldNodesGraph(admitted=_ADMITTED, first_error=first_error)
    found: dict[str, DefaultPresence | Exception] = {}
    first_deadline = time.monotonic() + 0.5 if first_error == "deadline" else None

    def presence(stat_var: str, deadline: float | None) -> None:
        try:
            found[stat_var] = read_default_presence(
                graph=graph, availability=availability, stat_vars=[stat_var], deadline=deadline
            )
        except Exception as exc:
            found[stat_var] = exc

    first = threading.Thread(target=presence, args=("A", first_deadline))
    first.start()
    assert graph.started.wait(timeout=5)
    second = threading.Thread(target=presence, args=("D", time.monotonic() + 8))
    second.start()
    # The second check's own read lands while the first is held, so it has claimed its places.
    wait_until = time.monotonic() + 5
    while not graph.node_reads() and time.monotonic() < wait_until:
        time.sleep(0.01)
    if first_deadline is not None:
        time.sleep(max(0.0, first_deadline - time.monotonic()))
    graph.release.set()
    first.join(timeout=5)
    second.join(timeout=5)

    assert sorted(sorted(places) for places in graph.node_reads()) == reads
    second_found = found["D"]
    assert isinstance(second_found, DefaultPresence), second_found
    if isinstance(first_error, Exception):
        assert second_found.unsettled == {"D"}
        assert second_found.scope("D") == _countries_of_d()
    else:
        assert second_found.unsettled == frozenset()
        assert second_found.scope("D") == DefaultPlaceScope(
            places=(_KEN, "country/ZAF"), every_place=True
        )


# -- settling in the background --


def test_a_read_past_the_requests_wait_settles_in_the_background(monkeypatch):
    monkeypatch.setattr(place_scope, "EVERY_PLACE_SECONDS", 0.0)
    release = threading.Event()
    availability = _Availability(data=[("D", _AFRICA), ("D", _KEN)], blocked={"D": release})
    graph = _Graph(admitted=_ADMITTED)

    cut = read_default_presence(graph=graph, availability=availability, stat_vars=["D"])

    assert cut.scope("D") == _countries_of_d()
    assert cut.unsettled == {"D"}
    assert memo_stats()["in_flight"] == 1

    release.set()
    _drain()
    settled = read_default_presence(graph=graph, availability=availability, stat_vars=["D"])

    assert settled.scope("D") == DefaultPlaceScope(places=(_AFRICA, _KEN), every_place=True)
    assert settled.unsettled == frozenset()
    assert settled.cold_reads == 0
    assert availability.read_stat_vars() == ["D"]
    # The first request's countries, then the settled list's node check, then nothing.
    read_default_presence(graph=graph, availability=availability, stat_vars=["D"])
    assert graph.kinds() == ["countries", "presence", "nodes"]


def test_a_cut_read_that_ends_in_a_gateway_status_memoises_nothing(monkeypatch):
    monkeypatch.setattr(place_scope, "EVERY_PLACE_SECONDS", 0.0)
    release = threading.Event()
    availability = _Availability(
        errors={"D": SdmxHttpError("unavailable", upstream_status=503)}, blocked={"D": release}
    )
    graph = _Graph(admitted=_ADMITTED)
    read_default_presence(graph=graph, availability=availability, stat_vars=["D"])

    release.set()
    _drain()

    assert memo_stats()["entries"] == 0


def test_requests_for_a_stat_var_in_flight_share_its_one_read(monkeypatch):
    monkeypatch.setattr(place_scope, "EVERY_PLACE_SECONDS", 0.0)
    release = threading.Event()
    availability = _Availability(data=[("D", _KEN)], blocked={"D": release})
    graph = _Graph(admitted=_ADMITTED)

    first = read_default_presence(graph=graph, availability=availability, stat_vars=["D"])
    second = read_default_presence(graph=graph, availability=availability, stat_vars=["D"])

    assert first.unsettled == second.unsettled == {"D"}
    # The second request joined the flight: it is no cold read of its own.
    assert (first.cold_reads, second.cold_reads) == (1, 0)
    release.set()
    _drain()
    assert availability.read_stat_vars() == ["D"]


def test_a_request_on_another_backend_starts_its_own_read(monkeypatch):
    monkeypatch.setattr(place_scope, "EVERY_PLACE_SECONDS", 0.0)
    release = threading.Event()
    graph = _Graph(admitted=_ADMITTED)
    first = _Availability(data=[("D", _KEN)], blocked={"D": release}, source="mixer-a")
    second = _Availability(data=[("D", _UGA)], source="mixer-b")
    read_default_presence(graph=graph, availability=first, stat_vars=["D"])

    presence = read_default_presence(graph=graph, availability=second, stat_vars=["D"])

    release.set()
    _drain()
    assert presence.cold_reads == 1
    assert second.read_stat_vars() == ["D"]


def test_a_read_that_ends_inside_the_requests_wait_answers_it():
    release = threading.Event()
    availability = _Availability(data=[("D", _AFRICA)], blocked={"D": release})
    threading.Timer(0.05, release.set).start()

    scope = default_place_scope("D", graph=_Graph(), availability=availability)

    assert scope == DefaultPlaceScope(places=(_AFRICA,), every_place=True)


def test_a_queued_read_past_its_settle_deadline_ends_unread_and_unmemoised(monkeypatch):
    """A read that waited past `SETTLE_DEADLINE_SECONDS` for a settle thread frees it at once."""
    pool = _one_settle_thread(monkeypatch)
    release = threading.Event()
    availability = _Availability(data=[("D", _AFRICA)], blocked={"A": release})
    graph = _Graph(admitted=_ADMITTED)
    read_default_presence(graph=graph, availability=availability, stat_vars=["A"])
    # `D` is queued behind `A` with a bound already past.
    monkeypatch.setattr(place_scope, "SETTLE_DEADLINE_SECONDS", 0.0)
    read_default_presence(graph=graph, availability=availability, stat_vars=["D"])

    release.set()
    _drain()

    assert availability.read_stat_vars() == ["A"]
    # Only `A`'s empty list is memoised; `D` ended unsettled.
    assert memo_stats()["entries"] == 1
    pool.shutdown()


def test_a_read_past_its_settle_deadline_is_not_memoised():
    availability = _Availability(errors={"D": TimeoutError("absolute caller deadline")})
    graph = _Graph(admitted=_ADMITTED)

    presence = read_default_presence(graph=graph, availability=availability, stat_vars=["D"])

    assert presence.unsettled == {"D"}
    assert memo_stats()["entries"] == 0


def _one_settle_thread(monkeypatch) -> ThreadPoolExecutor:
    pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="place-scope-test")
    monkeypatch.setattr(place_scope, "_pool", pool)
    monkeypatch.setattr(place_scope, "AVAILABILITY_WORKERS", 1)
    monkeypatch.setattr(place_scope, "EVERY_PLACE_SECONDS", 0.0)
    return pool


def test_a_saturated_pool_gives_countries_for_the_request_and_queues_the_read(monkeypatch):
    pool = _one_settle_thread(monkeypatch)
    release = threading.Event()
    availability = _Availability(data=[("D", _AFRICA)], blocked={"A": release})
    graph = _Graph(admitted=_ADMITTED)
    read_default_presence(graph=graph, availability=availability, stat_vars=["A"])

    presence = read_default_presence(graph=graph, availability=availability, stat_vars=["D"])

    assert presence.scope("D") == _countries_of_d()
    assert presence.unsettled == {"D"}
    assert presence.cold_reads == 0
    # `D` waits behind `A` and has not been read.
    assert availability.read_stat_vars() == ["A"]
    assert memo_stats()["in_flight"] == 2

    release.set()
    _drain()
    settled = read_default_presence(graph=graph, availability=availability, stat_vars=["D"])

    assert settled.scope("D") == DefaultPlaceScope(places=(_AFRICA,), every_place=True)
    assert availability.read_stat_vars() == ["A", "D"]
    pool.shutdown()


def test_a_queued_read_runs_in_the_context_of_the_call_that_queued_it(monkeypatch):
    """A read is submitted by whichever thread frees, often one running another request's
    read, yet it runs in its own call's context, which carries that request's trace id and
    timings."""
    pool = _one_settle_thread(monkeypatch)
    release = threading.Event()
    callers: dict[str, str | None] = {}

    class _RecordingAvailability(_Availability):
        def availability_values(self, **kwargs) -> list[str]:  # type: ignore[override]
            (stat_var,) = kwargs["variables"]
            callers[stat_var] = _CALLER.get()
            return super().availability_values(**kwargs)

    availability = _RecordingAvailability(data=[("Z", _KEN)], blocked={"W0": release})
    graph = _Graph(admitted=_ADMITTED)

    def call(caller: str, stat_vars: list[str]) -> None:
        _CALLER.set(caller)
        read_default_presence(graph=graph, availability=availability, stat_vars=stat_vars)

    contextvars.Context().run(call, "A", ["W0", "W1"])
    contextvars.Context().run(call, "B", ["Z"])
    release.set()
    _drain()

    # `W0`'s thread submits `Z`, and `Z`'s thread submits `W1`.
    assert callers == {"W0": "A", "Z": "B", "W1": "A"}
    pool.shutdown()


def test_a_read_two_lanes_hold_runs_once(monkeypatch):
    """A joined flight sits in two lanes. Once one lane's read has ended the flight, the other
    lane's entry is skipped even before the flight resolves."""
    pool = _one_settle_thread(monkeypatch)
    release = threading.Event()
    availability = _Availability(data=[("W1", _KEN)], blocked={"W0": release})
    graph = _Graph(admitted=_ADMITTED)
    ended, resolve = threading.Event(), threading.Event()
    end_flight = place_scope._end_flight

    def end_and_hold(key, flight, *, outcome=None):
        end_flight(key, flight, outcome=outcome)
        if key.stat_var == "W1" and outcome is not None:
            ended.set()
            assert resolve.wait(timeout=5)

    monkeypatch.setattr(place_scope, "_end_flight", end_and_hold)
    read_default_presence(graph=graph, availability=availability, stat_vars=["W0", "W1"])
    read_default_presence(graph=graph, availability=availability, stat_vars=["W1"])
    release.set()
    assert ended.wait(timeout=5)

    # `W1` has ended its flight and not yet resolved it; a call arriving now dispatches.
    place_scope._dispatch()
    resolve.set()
    pool.shutdown(wait=True)

    assert availability.read_stat_vars() == ["W0", "W1"]


def test_a_read_arriving_at_a_full_queue_is_dropped(monkeypatch):
    pool = _one_settle_thread(monkeypatch)
    monkeypatch.setattr(place_scope, "MAX_QUEUED_SETTLES", 0)
    release = threading.Event()
    availability = _Availability(data=[("D", _AFRICA)], blocked={"A": release})
    graph = _Graph(admitted=_ADMITTED)
    read_default_presence(graph=graph, availability=availability, stat_vars=["A"])

    presence = read_default_presence(graph=graph, availability=availability, stat_vars=["D"])

    assert presence.unsettled == {"D"}
    assert memo_stats()["in_flight"] == 1
    release.set()
    _drain()
    assert availability.read_stat_vars() == ["A"]
    pool.shutdown()


def test_a_lone_request_reads_on_every_settle_thread(monkeypatch):
    """A request queueing more reads than the pool has threads runs one on each thread, and
    the rest as threads free."""
    monkeypatch.setattr(place_scope, "EVERY_PLACE_SECONDS", 0.0)
    release = threading.Event()
    wide = [f"W{i:02}" for i in range(40)]
    availability = _Availability(data=[], blocked=dict.fromkeys(wide, release))

    presence = read_default_presence(
        graph=_Graph(admitted=_ADMITTED), availability=availability, stat_vars=wide
    )
    deadline = time.monotonic() + 5
    while len(availability.reads) < AVAILABILITY_WORKERS and time.monotonic() < deadline:
        time.sleep(0.01)

    assert presence.cold_reads == AVAILABILITY_WORKERS
    assert availability.read_stat_vars() == wide[:AVAILABILITY_WORKERS]
    release.set()
    _drain()
    assert sorted(availability.read_stat_vars()) == wide


def test_a_request_arriving_beside_a_queue_takes_the_next_free_thread(monkeypatch):
    """One request queues 60 reads past the pool's threads. Another request's cold StatVar
    is queued too, as every thread runs a read, and the first thread to free reads it ahead
    of the first request's queue, so it settles while that queue still waits."""
    monkeypatch.setattr(place_scope, "EVERY_PLACE_SECONDS", 0.0)
    wide = [f"W{i:02}" for i in range(60)]
    released = {sv: threading.Event() for sv in wide}
    availability = _Availability(data=[("Z", _KEN)], blocked=released)
    graph = _Graph(admitted=_ADMITTED)
    read_default_presence(graph=graph, availability=availability, stat_vars=wide)

    second = read_default_presence(graph=graph, availability=availability, stat_vars=["Z"])
    (z_flight,) = (f for key, f in place_scope._flights.items() if key.stat_var == "Z")
    released[wide[0]].set()
    z_flight.result(timeout=5)

    assert second.unsettled == {"Z"}
    # `Z` took the freed thread; the thread `Z` then frees reads the first request's next.
    assert availability.read_stat_vars()[: AVAILABILITY_WORKERS + 1] == [
        *wide[:AVAILABILITY_WORKERS],
        "Z",
    ]
    settled = read_default_presence(graph=graph, availability=availability, stat_vars=["Z"])
    assert settled.scope("Z") == DefaultPlaceScope(places=(_KEN,), every_place=True)
    assert settled.cold_reads == 0
    for event in released.values():
        event.set()
    _drain()
    assert sorted(availability.read_stat_vars()) == sorted([*wide, "Z"])


def test_a_request_joining_a_read_deep_in_another_queue_runs_it_in_its_own_turn(monkeypatch):
    """One request queues 60 reads past the pool's threads. Another request wants one of
    them, deep in that queue: it joins the flight and puts it in its own lane, so the first
    thread to free reads it ahead of the queue, and the first request's lane drops it."""
    monkeypatch.setattr(place_scope, "EVERY_PLACE_SECONDS", 0.0)
    wide = [f"W{i:02}" for i in range(60)]
    released = {sv: threading.Event() for sv in wide}
    availability = _Availability(data=[("W40", _KEN)], blocked=released)
    graph = _Graph(admitted=_ADMITTED)
    read_default_presence(graph=graph, availability=availability, stat_vars=wide)
    released["W40"].set()

    second = read_default_presence(graph=graph, availability=availability, stat_vars=["W40"])
    (flight,) = (f for key, f in place_scope._flights.items() if key.stat_var == "W40")
    released[wide[0]].set()
    flight.result(timeout=5)

    assert (second.unsettled, second.cold_reads) == ({"W40"}, 0)
    assert availability.read_stat_vars()[: AVAILABILITY_WORKERS + 1] == [
        *wide[:AVAILABILITY_WORKERS],
        "W40",
    ]
    settled = read_default_presence(graph=graph, availability=availability, stat_vars=["W40"])
    assert settled.scope("W40") == DefaultPlaceScope(places=(_KEN,), every_place=True)
    for event in released.values():
        event.set()
    _drain()
    assert sorted(availability.read_stat_vars()) == wide


def test_stat_vars_past_the_cold_read_cap_take_countries_and_settle(monkeypatch):
    monkeypatch.setattr(place_scope, "MAX_COLD_READS_PER_REQUEST", 1)
    availability = _Availability(data=[("A", _KEN), ("D", _AFRICA)])
    graph = _Graph(admitted=_ADMITTED)

    presence = read_default_presence(graph=graph, availability=availability, stat_vars=["A", "D"])

    assert presence.scope("A") == DefaultPlaceScope(places=(_KEN,), every_place=True)
    assert presence.scope("D") == _countries_of_d()
    assert presence.unsettled == {"D"}
    assert presence.cold_reads == 1

    _drain()
    settled = read_default_presence(graph=graph, availability=availability, stat_vars=["D"])

    assert settled.scope("D") == DefaultPlaceScope(places=(_AFRICA,), every_place=True)
    assert sorted(availability.read_stat_vars()) == ["A", "D"]


def test_the_cold_read_cap_counts_across_a_requests_known_presence(monkeypatch):
    monkeypatch.setattr(place_scope, "MAX_COLD_READS_PER_REQUEST", 1)
    availability = _Availability(data=[("A", _KEN), ("D", _AFRICA)])
    graph = _Graph(admitted=_ADMITTED)
    known = read_default_presence(graph=graph, availability=availability, stat_vars=["A"])

    presence = read_default_presence(
        graph=graph, availability=availability, stat_vars=["D"], known=known
    )

    assert presence.unsettled == {"D"}
    assert presence.cold_reads == 1


class _InlineExecutor(Executor):
    """Runs each submitted call on the submitting thread before `submit` returns, as the
    engine's virtual clock does."""

    def submit(self, fn, /, *args, **kwargs):
        future: Future[object] = Future()
        try:
            future.set_result(fn(*args, **kwargs))
        except BaseException as exc:
            future.set_exception(exc)
        return future


def test_an_inline_executor_runs_the_read_without_deadlocking(monkeypatch):
    """The read is submitted after the memo lock is released, so a pool that runs it at once,
    on the requesting thread, takes the lock freely."""
    monkeypatch.setattr(place_scope, "_pool", _InlineExecutor())
    availability = _Availability(data=[("D", _AFRICA)])
    result: list[DefaultPlaceScope] = []
    worker = threading.Thread(
        target=lambda: result.append(
            default_place_scope("D", graph=_Graph(), availability=availability)
        ),
        daemon=True,
    )

    worker.start()
    worker.join(timeout=5)

    assert not worker.is_alive(), "the read deadlocked on the memo lock"
    assert result == [DefaultPlaceScope(places=(_AFRICA,), every_place=True)]
    assert memo_stats()["in_flight"] == 0


def test_a_failed_submission_ends_the_flight_with_its_error(monkeypatch):
    class _ShutDown(Executor):
        def submit(self, fn, /, *args, **kwargs):
            raise RuntimeError("cannot schedule new futures after shutdown")

    monkeypatch.setattr(place_scope, "_pool", _ShutDown())

    with pytest.raises(RuntimeError, match="after shutdown"):
        default_place_scope("D", graph=_Graph(), availability=_Availability(data=[("D", _KEN)]))

    assert memo_stats()["in_flight"] == 0


def test_a_pool_task_cancelled_before_it_runs_ends_its_flight(monkeypatch):
    """A pool shutdown cancels queued tasks. Their flights leave `_flights`, and the request
    waiting on one takes its countries for the request."""

    class _Cancelling(Executor):
        def submit(self, fn, /, *args, **kwargs):
            task: Future[object] = Future()
            task.cancel()
            return task

    monkeypatch.setattr(place_scope, "_pool", _Cancelling())
    availability = _Availability(data=[("D", _AFRICA)])
    result: list[DefaultPresence] = []
    # A settling read waits up to `SETTLE_DEADLINE_SECONDS`; it sees the cancellation at once.
    worker = threading.Thread(
        target=lambda: result.append(
            read_default_presence(
                graph=_Graph(admitted=_ADMITTED),
                availability=availability,
                stat_vars=["D"],
                settle=True,
            )
        ),
        daemon=True,
    )

    worker.start()
    worker.join(timeout=5)

    assert not worker.is_alive(), "the wait missed the cancelled flight"
    (presence,) = result
    assert presence.scope("D") == _countries_of_d()
    assert presence.unsettled == {"D"}
    assert memo_stats()["in_flight"] == 0
    assert availability.reads == []


def test_a_cancelled_flight_leaves_at_once_and_its_read_still_memoises(monkeypatch):
    monkeypatch.setattr(place_scope, "EVERY_PLACE_SECONDS", 0.0)
    release = threading.Event()
    availability = _Availability(data=[("D", _AFRICA)], blocked={"D": release})
    graph = _Graph(admitted=_ADMITTED)
    read_default_presence(graph=graph, availability=availability, stat_vars=["D"])
    (flight,) = place_scope._flights.values()
    # The pool task still holds the read; the flight is cancelled from outside, as a caller
    # cancelling the futures it waits on would.

    assert flight.cancel()

    assert memo_stats()["in_flight"] == 0
    release.set()
    # The cancelled flight no longer shows in `_flights`, so poll the memo for its read.
    deadline = time.monotonic() + 5
    while memo_stats()["entries"] == 0 and time.monotonic() < deadline:
        time.sleep(0.01)
    assert memo_stats()["entries"] == 1
    settled = default_place_scope("D", graph=graph, availability=availability)
    assert settled == DefaultPlaceScope(places=(_AFRICA,), every_place=True)
    assert availability.read_stat_vars() == ["D"]


# -- settling reads --


def test_a_flight_started_inside_a_request_runs_detached_from_its_deadline():
    """The settle flight is shared with other requests and the mint, so the starting
    request's deadline must not narrow its read."""
    request_bounds: list[float | None] = []

    class _RecordingAvailability(_Availability):
        def availability_values(self, **kwargs) -> list[str]:  # type: ignore[override]
            request_bounds.append(bounded_deadline(None))
            return super().availability_values(**kwargs)

    with request_deadline(time.monotonic() + 1):
        read_default_presence(
            graph=_Graph(), availability=_RecordingAvailability(data=_EVERY), stat_vars=["A"]
        )

    assert request_bounds == [None]


def test_a_settling_read_waits_past_the_requests_window(monkeypatch):
    monkeypatch.setattr(place_scope, "EVERY_PLACE_SECONDS", 0.0)
    release = threading.Event()
    availability = _Availability(data=[("D", _AFRICA)], blocked={"D": release})
    threading.Timer(0.05, release.set).start()

    presence = read_default_presence(
        graph=_Graph(admitted=_ADMITTED), availability=availability, stat_vars=["D"], settle=True
    )

    assert presence.scope("D") == DefaultPlaceScope(places=(_AFRICA,), every_place=True)
    assert presence.unsettled == frozenset()


def test_a_settling_read_joins_a_flight_another_request_cut(monkeypatch):
    monkeypatch.setattr(place_scope, "EVERY_PLACE_SECONDS", 0.0)
    release = threading.Event()
    availability = _Availability(data=[("D", _AFRICA)], blocked={"D": release})
    graph = _Graph(admitted=_ADMITTED)
    assert read_default_presence(graph=graph, availability=availability, stat_vars=["D"]).unsettled
    threading.Timer(0.05, release.set).start()

    settled = default_place_scope("D", graph=graph, availability=availability, settle=True)

    assert settled == DefaultPlaceScope(places=(_AFRICA,), every_place=True)
    assert availability.read_stat_vars() == ["D"]


@pytest.mark.parametrize("queue", [_QUEUE_BOUND, 0], ids=["queue", "full-queue"])
def test_a_settling_read_waits_out_a_saturated_pool(monkeypatch, queue):
    pool = _one_settle_thread(monkeypatch)
    monkeypatch.setattr(place_scope, "MAX_QUEUED_SETTLES", queue)
    release = threading.Event()
    availability = _Availability(data=[("D", _AFRICA)], blocked={"A": release})
    graph = _Graph(admitted=_ADMITTED)
    read_default_presence(graph=graph, availability=availability, stat_vars=["A"])
    threading.Timer(0.05, release.set).start()

    presence = read_default_presence(
        graph=graph, availability=availability, stat_vars=["D"], settle=True
    )

    assert presence.scope("D") == DefaultPlaceScope(places=(_AFRICA,), every_place=True)
    assert presence.unsettled == frozenset()
    pool.shutdown()


def test_a_settling_read_takes_no_per_request_cap(monkeypatch):
    monkeypatch.setattr(place_scope, "MAX_COLD_READS_PER_REQUEST", 0)
    availability = _Availability(data=[("A", _KEN), ("D", _AFRICA)])

    presence = read_default_presence(
        graph=_Graph(), availability=availability, stat_vars=["A", "D"], settle=True
    )

    assert presence.scope("A") == DefaultPlaceScope(places=(_KEN,), every_place=True)
    assert presence.scope("D") == DefaultPlaceScope(places=(_AFRICA,), every_place=True)
    assert presence.unsettled == frozenset()
    assert presence.cold_reads == 2


def test_a_settling_read_memoises_as_a_request_does():
    availability = _Availability(data=[("D", _AFRICA)])
    graph = _Graph()
    default_place_scope("D", graph=graph, availability=availability, settle=True)

    again = read_default_presence(graph=graph, availability=availability, stat_vars=["D"])

    assert again.scope("D") == DefaultPlaceScope(places=(_AFRICA,), every_place=True)
    assert availability.read_stat_vars() == ["D"]


def test_a_settling_read_takes_countries_on_a_definite_fallback_and_memoises_it():
    cap = SdmxCapExceededError("values", cap_kind="values", cap=1, observed=2)
    availability = _Availability(errors={"D": cap})
    graph = _Graph(admitted=_ADMITTED)

    presence = read_default_presence(
        graph=graph, availability=availability, stat_vars=["D"], settle=True
    )

    assert presence.scope("D") == _countries_of_d()
    assert presence.unsettled == frozenset()
    default_place_scope("D", graph=graph, availability=availability, settle=True)
    assert availability.read_stat_vars() == ["D"]
    assert graph.kinds() == ["countries", "presence"]


@pytest.mark.parametrize(
    "error",
    [SdmxHttpError("unavailable", upstream_status=503), TimeoutError("settle bound")],
    ids=["503", "settle-bound"],
)
def test_a_settling_read_takes_countries_on_a_transient_outcome_and_marks_it(error):
    availability = _Availability(errors={"D": error})

    presence = read_default_presence(
        graph=_Graph(admitted=_ADMITTED), availability=availability, stat_vars=["D"], settle=True
    )

    assert presence.scope("D") == _countries_of_d()
    assert presence.unsettled == {"D"}
    assert memo_stats()["entries"] == 0


def test_a_settling_read_past_its_settle_deadline_is_transient(monkeypatch):
    monkeypatch.setattr(place_scope, "SETTLE_DEADLINE_SECONDS", 0.0)
    availability = _Availability(data=[("D", _AFRICA)])

    presence = read_default_presence(
        graph=_Graph(admitted=_ADMITTED), availability=availability, stat_vars=["D"], settle=True
    )

    assert presence.scope("D") == _countries_of_d()
    assert presence.unsettled == {"D"}
    _drain()
    assert availability.reads == []


def test_a_settling_read_refuses_when_the_requests_deadline_ends_its_wait():
    release = threading.Event()
    availability = _Availability(data=[("D", _AFRICA)], blocked={"D": release})

    with pytest.raises(PlaceScopeDeadlineError):
        read_default_presence(
            graph=_Graph(),
            availability=availability,
            stat_vars=["D"],
            deadline=time.monotonic() + 0.05,
            settle=True,
        )

    release.set()


# -- the request's deadline --


def test_a_deadline_already_past_refuses_before_any_read():
    availability, graph = _Availability(data=_EVERY), _Graph()

    with pytest.raises(PlaceScopeDeadlineError) as raised:
        read_default_presence(
            graph=graph,
            availability=availability,
            stat_vars=["A", "D"],
            deadline=time.monotonic() - 1,
        )

    assert raised.value.stat_vars == ("A", "D")
    assert availability.reads == []
    assert graph.reads == []


def test_a_deadline_that_ends_the_wait_refuses_and_the_read_settles():
    release = threading.Event()
    availability = _Availability(data=[("D", _AFRICA)], blocked={"D": release})
    graph = _Graph()

    with pytest.raises(PlaceScopeDeadlineError) as raised:
        read_default_presence(
            graph=graph,
            availability=availability,
            stat_vars=["D"],
            deadline=time.monotonic() + 0.05,
        )

    assert raised.value.stat_vars == ("D",)
    release.set()
    _drain()
    assert default_place_scope("D", graph=graph, availability=availability).every_place
    assert availability.read_stat_vars() == ["D"]


def test_a_node_check_the_deadline_ends_refuses():
    class _ExpiringGraph(_Graph):
        def node_arcs_many(
            self,
            dcids: Sequence[str],
            *,
            properties: Sequence[str],
            deadline: float | None = None,
        ) -> Mapping[str, Mapping[str, tuple[str, ...]] | None]:
            assert deadline is not None
            time.sleep(max(0.0, deadline - time.monotonic()))
            raise GraphDeadlineExceededError("deadline")

    with pytest.raises(PlaceScopeDeadlineError):
        read_default_presence(
            graph=_ExpiringGraph(),
            availability=_Availability(data=_EVERY),
            stat_vars=["A"],
            deadline=time.monotonic() + 0.05,
        )


def test_the_countries_fallback_reads_take_the_requests_deadline():
    cap = SdmxCapExceededError("values", cap_kind="values", cap=1, observed=2)
    graph = _Graph(admitted=_ADMITTED)
    deadline = time.monotonic() + 30.0

    read_default_presence(
        graph=graph,
        availability=_Availability(errors={"D": cap}),
        stat_vars=["D"],
        deadline=deadline,
    )

    assert graph.deadlines == {"countries": deadline, "presence": deadline}


@pytest.mark.parametrize("read", ["countries", "presence"])
def test_a_countries_fallback_read_the_deadline_ends_refuses(read):
    cap = SdmxCapExceededError("values", cap_kind="values", cap=1, observed=2)

    with pytest.raises(PlaceScopeDeadlineError) as raised:
        read_default_presence(
            graph=_Graph(admitted=_ADMITTED, expire=[read]),
            availability=_Availability(errors={"D": cap}),
            stat_vars=["D"],
            deadline=time.monotonic() + 0.5,
        )

    assert raised.value.stat_vars == ("D",)


# -- UN variables --


def test_a_un_variable_reads_the_un_deployments_availability():
    ours = _Availability(data=[(_UN_VARIABLE, _KEN)])
    un = _Availability(data=[(_UN_VARIABLE, "undata-geo/G00134000"), (_UN_VARIABLE, _TGO)])

    scope = default_place_scope(_UN_VARIABLE, graph=_Graph(un_availability=un), availability=ours)

    assert scope == DefaultPlaceScope(places=(_TGO, "undata-geo/G00134000"), every_place=True)
    assert ours.reads == []
    assert un.read_stat_vars() == [_UN_VARIABLE]


def test_a_un_variable_without_a_un_deployment_refuses_before_any_read():
    availability, graph = _Availability(data=_EVERY), _Graph()

    with pytest.raises(MaterialisationUsageError, match="reaches none"):
        read_default_presence(graph=graph, availability=availability, stat_vars=["A", _UN_VARIABLE])

    assert availability.reads == []
    assert graph.reads == []


# -- the presence --


def test_no_stat_var_makes_no_read():
    availability, graph = _Availability(data=_EVERY), _Graph()

    assert read_default_presence(graph=graph, availability=availability, stat_vars=[]) == (
        DefaultPresence(countries=(), scopes={})
    )
    assert availability.reads == []
    assert graph.reads == []


def test_a_known_presence_covering_every_stat_var_is_returned_as_it_is():
    availability, graph = _Availability(data=_EVERY), _Graph()
    known = read_default_presence(graph=graph, availability=availability, stat_vars=["A"])

    assert (
        read_default_presence(graph=graph, availability=availability, stat_vars=["A"], known=known)
        is known
    )


def test_a_known_presence_is_topped_up_and_keeps_its_unsettled_stat_vars():
    known = DefaultPresence(
        countries=_COUNTRIES,
        scopes={"D": _countries_of_d()},
        unsettled=frozenset({"D"}),
        cold_read=frozenset({"X", "Y", "Z"}),
    )

    presence = read_default_presence(
        graph=_Graph(), availability=_Availability(data=_EVERY), stat_vars=["A", "D"], known=known
    )

    assert presence.scopes.keys() == {"A", "D"}
    assert presence.scope("D") == _countries_of_d()
    assert presence.unsettled == {"D"}
    assert presence.cold_reads == 4
    assert presence.countries == _COUNTRIES


def test_the_scope_of_a_stat_var_the_read_did_not_cover_raises():
    presence = DefaultPresence(countries=(), scopes={})

    with pytest.raises(AssertionError, match="no default-scope presence was read for 'B'"):
        presence.scope("B")


def test_stat_vars_read_together_are_read_at_the_union_of_their_own_scopes():
    presence = DefaultPresence(
        countries=_COUNTRIES,
        scopes={
            "A": DefaultPlaceScope(places=(_AFRICA, _KEN), every_place=True),
            "D": _countries_of_d(),
            "W": DefaultPlaceScope(places=("Earth",), every_place=True),
        },
    )

    assert presence.facet_entities(["W", "A", "W"]) == [_AFRICA, _KEN, "Earth"]
    assert presence.union_scope(["A", "W"]) == DefaultPlaceScope(
        places=(_AFRICA, _KEN, "Earth"), every_place=True
    )
    # A union holding a countries fallback is not every place.
    assert presence.union_scope(["A", "D"]) == DefaultPlaceScope(
        places=(_AFRICA, _KEN, _UGA), every_place=False
    )
    assert presence.facet_entities([]) == []
    assert presence.union_scope([]) == NO_DEFAULT_SCOPE


def test_merged_keeps_this_presences_scopes_and_adds_the_others():
    mine = DefaultPresence(
        countries=(),
        scopes={"A": DefaultPlaceScope(places=(_KEN,), every_place=True)},
        cold_read=frozenset({"A", "B"}),
    )
    other = DefaultPresence(
        countries=_COUNTRIES,
        scopes={"A": _countries_of_d(), "D": _countries_of_d()},
        unsettled=frozenset({"A", "D"}),
        cold_read=frozenset({"D"}),
    )

    merged = mine.merged(other)

    assert merged.scope("A") == mine.scope("A")
    assert merged.scope("D") == _countries_of_d()
    assert merged.unsettled == {"D"}
    assert merged.countries == _COUNTRIES
    assert merged.cold_reads == 3


def test_merging_a_presence_read_over_this_one_counts_each_cold_read_once():
    graph, availability = _Graph(), _Availability(data=_EVERY)
    known = read_default_presence(graph=graph, availability=availability, stat_vars=["A"])

    presence = read_default_presence(
        graph=graph, availability=availability, stat_vars=["W"], known=known
    )

    assert (known.cold_reads, presence.cold_reads, known.merged(presence).cold_reads) == (1, 2, 2)


def test_scope_places_orders_by_dcid_with_earth_last():
    assert scope_places(["Earth", _TGO, _AFRICA, _TGO]) == (_AFRICA, _TGO, "Earth")
    assert scope_of_places((), every_place=True) == NO_DEFAULT_SCOPE
    assert scope_of_places(["Earth", _KEN], every_place=False) == DefaultPlaceScope(
        places=(_KEN, "Earth"), every_place=False
    )
    assert not is_worldwide([_KEN, "Earth"])
    assert is_worldwide(["Earth"])


def test_an_empty_scope_keeps_whether_it_is_a_countries_fallback():
    """A countries fallback that finds no country and no Earth is empty, and still reads as a
    fallback, alone and in a union."""
    assert scope_of_places((), every_place=False) == DefaultPlaceScope(places=(), every_place=False)
    presence = DefaultPresence(
        countries=_COUNTRIES,
        scopes={
            "X": DefaultPlaceScope(places=(), every_place=False),
            "Y": DefaultPlaceScope(places=(), every_place=True),
        },
    )

    assert presence.union_scope(["X"]) == DefaultPlaceScope(places=(), every_place=False)
    assert presence.union_scope(["X", "Y"]) == DefaultPlaceScope(places=(), every_place=False)
    assert presence.union_scope(["Y"]) == NO_DEFAULT_SCOPE
