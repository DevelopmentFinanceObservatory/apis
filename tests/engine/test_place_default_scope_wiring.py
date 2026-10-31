"""Search's wiring of the default place scope (`search.engine.place`): its SDMX client reads
our variables' availability, its graph client the UN deployment's, and each failure class
reaches the refusal Search answers it with."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from concurrent.futures import wait

import pytest

from datasets_api.doors.mint_errors import classify_mint_exception
from dc_sdmx import SdmxCapExceededError, SdmxClient, SdmxHttpError, SdmxTransportError
from graph_generation import UNKNOWN_GENERATION, GenerationStamp
from materialise import place_scope
from materialise.place_scope import (
    DefaultPlaceScope,
    PlaceScopeDeadlineError,
    PlaceScopeReadError,
)
from search.engine.app import classify_search_exception
from search.engine.clients.graph_client import LiveGraphClient
from search.engine.errors import GraphInfraError, GraphReadFailure
from search.engine.place import default_place_scope_message, read_default_presence

_SV = "Count_Person"
_UN_SV = "undata/who/MORT_U5_RATE"
_KEN, _AFRICA = "country/KEN", "africa"


@pytest.fixture(autouse=True)
def _fresh_memo() -> Iterator[None]:
    yield
    while flights := list(place_scope._flights.values()):
        wait(flights)
    place_scope.clear_memo()


class _Availability:
    """Answers each read with `places`, or raises `error`; records the StatVars read."""

    def __init__(self, places: Sequence[str] = (), *, error: Exception | None = None) -> None:
        self._places = list(places)
        self._error = error
        self.reads: list[tuple[str, ...]] = []

    def source_key(self) -> str:
        return "fake-availability"

    def availability_values(self, *, variables: Sequence[str], **_: object) -> list[str]:
        self.reads.append(tuple(variables))
        if self._error is not None:
            raise self._error
        return self._places


class _Graph:
    """Holds a node for each of `nodes`; Earth contains `countries`. Each instance keys the
    memo on its own stamp, so no test reads another's settled scope."""

    def __init__(
        self,
        *,
        nodes: Sequence[str] = (),
        countries: Sequence[str] = (),
        un: _Availability | None = None,
    ) -> None:
        self._nodes = frozenset(nodes)
        self._countries = list(countries)
        self._un = un
        self._stamp = GenerationStamp(token=f"wiring-{id(self)}")

    def source_key(self) -> str:
        return "fake-graph"

    def generation_stamp(self) -> GenerationStamp:
        return self._stamp

    def un_availability(self) -> _Availability | None:
        return self._un

    def node_arcs_many(
        self, dcids: Sequence[str], *, properties: Sequence[str], deadline: float | None = None
    ) -> Mapping[str, Mapping[str, tuple[str, ...]] | None]:
        return {dcid: {"typeOf": ("Place",)} if dcid in self._nodes else None for dcid in dcids}

    def child_places_batch(
        self, parent_dcids: list[str], child_type: str, *, deadline: float | None = None
    ) -> dict[str, list[str]]:
        return {parent: list(self._countries) for parent in parent_dcids}

    def entities_with_data_by_variable(
        self, variable_dcids: list[str], entity_dcids: list[str], *, deadline: float | None = None
    ) -> dict[str, list[str]]:
        return {sv: list(entity_dcids) for sv in variable_dcids}


def _read(graph: _Graph, sdmx: _Availability, stat_vars: Sequence[str]):
    # The doubles implement only the reads the default scope takes, not Search's full clients.
    return read_default_presence(
        graph=graph,  # ty: ignore[invalid-argument-type]
        sdmx=sdmx,  # ty: ignore[invalid-argument-type]
        stat_vars=stat_vars,
    )


@pytest.mark.parametrize(
    ("scope", "message"),
    [
        (
            DefaultPlaceScope(places=("Earth",), every_place=True),
            "no place was named; the answer was resolved as a worldwide total.",
        ),
        (
            DefaultPlaceScope(places=(_AFRICA, _KEN), every_place=True),
            "no place was named; the answer covers every place with data, at every level.",
        ),
        (
            DefaultPlaceScope(places=(_KEN,), every_place=False),
            "no place was named; the answer covers the countries with data, because the full "
            "list of places was too large or unavailable.",
        ),
    ],
)
def test_the_defaulted_message_tells_every_place_from_the_countries_fallback(scope, message):
    assert default_place_scope_message(scope) == message


def test_search_sdmx_reads_our_variables_availability():
    sdmx = _Availability([_AFRICA, _KEN])

    presence = _read(_Graph(nodes=[_AFRICA, _KEN]), sdmx, [_SV])

    assert sdmx.reads == [(_SV,)]
    assert presence.consulted_scope(_SV) == DefaultPlaceScope(
        places=(_AFRICA, _KEN), every_place=True
    )


def test_the_graph_client_reads_a_un_variables_availability():
    sdmx, un = _Availability(), _Availability([_KEN])

    presence = _read(_Graph(nodes=[_KEN], un=un), sdmx, [_UN_SV])

    assert (sdmx.reads, un.reads) == ([], [(_UN_SV,)])
    assert presence.consulted_scope(_UN_SV).places == (_KEN,)


_READ_FAILURES = pytest.mark.parametrize(
    ("error", "status", "search_problem"),
    [
        (SdmxHttpError("bad filter", upstream_status=400), 400, (502, "upstream_error")),
        (SdmxTransportError("connection reset"), None, (503, "service_unavailable")),
        # Search's own SDMX client translates a failed read to `GraphInfraError`.
        (GraphInfraError("connection reset"), None, (503, "service_unavailable")),
    ],
    ids=["sdmx-4xx", "sdmx-transport", "search-client"],
)


@_READ_FAILURES
@pytest.mark.parametrize("variable", [_SV, _UN_SV], ids=["ours", "un"])
def test_an_availability_read_failure_refuses_as_a_retryable_infra_failure(
    variable, error, status, search_problem
):
    sdmx = _Availability(error=error)

    with pytest.raises(GraphReadFailure) as exc_info:
        _read(_Graph(un=sdmx), sdmx, [variable])

    assert exc_info.value.upstream_status == status
    problem = classify_search_exception(exc_info.value)
    assert problem is not None
    assert (problem.status, problem.code, problem.retryable) == (*search_problem, True)


@_READ_FAILURES
def test_a_read_search_started_fails_a_datasets_joiner_as_a_retryable_upstream_error(
    error, status, search_problem
):
    """Reads are single-flight across doors, so a mint can wait on a read Search's SDMX
    client started. Its error carries neither door's type."""
    with pytest.raises(PlaceScopeReadError) as exc_info:
        place_scope.read_default_presence(
            graph=_Graph(),
            availability=_Availability(error=error),
            stat_vars=[_SV],
        )

    assert exc_info.value.upstream_status == status
    problem = classify_mint_exception(exc_info.value)
    assert problem is not None
    assert (problem.status, problem.code, problem.retryable) == (502, "upstream_error", True)


def test_an_earth_listing_naming_no_country_refuses_as_a_graph_read_failure():
    cap = SdmxCapExceededError("too many places", cap_kind="values", cap=5000, observed=5001)

    with pytest.raises(GraphReadFailure):
        _read(_Graph(countries=[]), _Availability(error=cap), [_SV])


def test_a_scope_the_deadline_cuts_refuses_as_a_timeout():
    problem = classify_search_exception(PlaceScopeDeadlineError([_SV]))

    assert problem is not None
    assert (problem.status, problem.code, problem.retryable) == (504, "timeout", True)


def test_the_live_graph_client_serves_the_presence_reader_accessors():
    with LiveGraphClient(base="http://fake", generation=UNKNOWN_GENERATION) as graph:
        assert isinstance(graph.un_availability(), SdmxClient)
        assert graph.generation_stamp() == UNKNOWN_GENERATION.stamp()
