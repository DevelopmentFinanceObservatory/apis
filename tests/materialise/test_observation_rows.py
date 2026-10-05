"""Rows for UN System Data Commons variables: read through the origin graph client, laid out
as the SDMX service lays out the same series, loud on failure, and silent only for places the
UN front door refuses, which the origin client reports."""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator, Sequence
from typing import Literal

import httpx
import pyarrow as pa
import pytest

from dc_graph import Facet, GraphDeadlineExceededError
from dc_sdmx import (
    SdmxCapExceededError,
    SdmxClient,
    SdmxHttpError,
    SdmxResponseError,
    SdmxTransportError,
)
from materialise import (
    EnrichmentLookup,
    MaterialisationRequest,
    columnar,
    materialise_collection,
    materialise_group,
    materialise_preview,
    materialise_spec,
    observation_rows,
)
from materialise.errors import MaterialisationUsageError
from materialise.facets import FacetBinding, PlaceAdmission, probe_facets
from materialise.generation import request_batch_key
from materialise.observation_rows import (
    UN_READ_MAX_BYTES,
    stream_observation_batches,
    un_observation_batches,
)
from materialise.revision import RevisionSnapshot
from materialise.window import RequestedWindow
from origin_catalogue import recording_un_blocked_places
from origin_catalogue.graph import OriginGraphClient, build_graph_client

from ._scope_availability import NO_SCOPE_READ
from .conftest import FakeGraphReader, FakeRevisionReader
from .un_deployment import (
    OURS_API,
    UN_API,
    UN_FACET_ID,
    UN_PROVENANCE,
    UN_SECOND_VARIABLE,
    UN_UNIT,
    UN_VARIABLE,
    UnDeployments,
)

_SDMX_HEADER = (
    "STRUCTURE,STRUCTURE_ID,ACTION,variableMeasured,observationAbout,unit,measurementMethod,"
    "observationPeriod,provenance,TIME_PERIOD,OBS_VALUE,scalingFactor,facetId\r\n"
)


def _sdmx_row(place: str, date: str, value: str) -> str:
    return (
        f"dataflow,DC:DF_OBS(1.0.0),I,{UN_VARIABLE},{place},{UN_UNIT},NotApplicable,P1Y,"
        f"{UN_PROVENANCE},{date},{value},,{UN_FACET_ID}\r\n"
    )


# The same series as the SDMX service lays it out: series sorted by variable, place and facet
# id, newest period first.
_SDMX_BODY = (
    _SDMX_HEADER
    + _sdmx_row("country/GHA", "2014", "88.0")
    + _sdmx_row("country/GHA", "2008", "110.5")
    + _sdmx_row("country/KEN", "2014", "56.70805")
    + _sdmx_row("country/KEN", "2008", "97.53082")
    + _sdmx_row("country/KEN", "2003", "148.88945")
    + _sdmx_row("country/KEN", "1998", "136.17859")
    + _sdmx_row("country/KEN", "1993", "129.32726")
)


@pytest.fixture
def deployments() -> UnDeployments:
    return UnDeployments()


@pytest.fixture
def graph(deployments: UnDeployments) -> Iterator[OriginGraphClient]:
    client = build_graph_client(
        base_url=OURS_API,
        token=None,
        un_base_url=UN_API,
        un_availability=SdmxClient(
            base_url=UN_API, transport=httpx.MockTransport(deployments.handler)
        ),
        transport=httpx.MockTransport(deployments.handler),
        max_attempts=1,
    )
    yield client
    client.close()


def _sdmx_serving(body: str) -> SdmxClient:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, text=body, headers={"content-type": "application/vnd.sdmx.data+csv;version=2.0.0"}
        )

    return SdmxClient(base_url=OURS_API, transport=httpx.MockTransport(handler))


def _sdmx_never_called() -> SdmxClient:
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError(f"a UN variable reached the SDMX service: {request.url}")

    return SdmxClient(base_url=OURS_API, transport=httpx.MockTransport(handler))


def _table(
    graph: OriginGraphClient,
    places: list[str],
    *,
    facet_ids: tuple[str, ...] = (UN_FACET_ID,),
    time_period: Literal["LATEST"] | None = None,
    max_rows: int | None = None,
) -> tuple[tuple[str, ...], pa.Table]:
    with stream_observation_batches(
        sdmx=_sdmx_never_called(),
        graph=graph,
        variables=[UN_VARIABLE],
        entities={"observationAbout": places},
        facet_ids=facet_ids,
        time_period=time_period,
        max_rows=max_rows,
    ) as (roles, batches):
        return roles, pa.Table.from_batches(list(batches))


def test_un_rows_match_the_sdmx_layout_of_the_same_series(graph: OriginGraphClient) -> None:
    un_roles, un_table = _table(graph, ["country/KEN", "country/GHA"])
    with _sdmx_serving(_SDMX_BODY).stream_data_batches(
        variables=[UN_VARIABLE],
        entities={"observationAbout": ["country/KEN", "country/GHA"]},
        facet_ids=(UN_FACET_ID,),
    ) as (sdmx_roles, sdmx_batches):
        sdmx_table = pa.Table.from_batches(list(sdmx_batches))

    assert un_roles == sdmx_roles
    assert un_table.schema == sdmx_table.schema
    assert un_table.to_pylist() == sdmx_table.to_pylist()


def test_latest_keeps_each_series_newest_observation(graph: OriginGraphClient) -> None:
    _roles, table = _table(graph, ["country/KEN", "country/GHA"], time_period="LATEST")

    assert [(row["observationAbout"], row["TIME_PERIOD"]) for row in table.to_pylist()] == [
        ("country/GHA", "2014"),
        ("country/KEN", "2014"),
    ]


def test_only_the_dispatched_facets_are_kept(graph: OriginGraphClient) -> None:
    _roles, table = _table(graph, ["country/KEN"], facet_ids=("another-facet",))

    assert table.num_rows == 0


def test_rows_past_the_cap_raise_the_sdmx_cap_error_before_any_batch_is_built(
    graph: OriginGraphClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    built: list[str] = []
    build = observation_rows._variable_batch

    def recording_build(variable, series, *, latest):
        built.append(variable)
        return build(variable, series, latest=latest)

    monkeypatch.setattr(observation_rows, "_variable_batch", recording_build)

    with pytest.raises(SdmxCapExceededError) as excinfo:
        _table(graph, ["country/KEN", "country/GHA"], max_rows=6)

    assert (excinfo.value.cap_kind, excinfo.value.cap, excinfo.value.observed) == ("rows", 6, 7)
    assert built == []


@pytest.mark.parametrize(
    ("time_period", "facet_ids", "max_rows"),
    [
        pytest.param(None, (UN_FACET_ID,), 7, id="every-row"),
        pytest.param("LATEST", (UN_FACET_ID,), 2, id="latest"),
        pytest.param(None, ("another-facet",), 0, id="facet-filter"),
    ],
)
def test_the_cap_counts_only_the_rows_the_read_keeps(
    graph: OriginGraphClient,
    time_period: Literal["LATEST"] | None,
    facet_ids: tuple[str, ...],
    max_rows: int,
) -> None:
    _roles, table = _table(
        graph,
        ["country/KEN", "country/GHA"],
        facet_ids=facet_ids,
        time_period=time_period,
        max_rows=max_rows,
    )

    assert table.num_rows == max_rows


def test_a_body_past_the_byte_cap_raises_the_sdmx_cap_error_while_reading(
    graph: OriginGraphClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(observation_rows, "UN_READ_MAX_BYTES", 512)
    with pytest.raises(SdmxCapExceededError) as excinfo:
        _table(graph, ["country/KEN", "country/GHA"])

    assert (excinfo.value.cap_kind, excinfo.value.cap) == ("bytes", 512)


def test_a_series_read_short_of_its_observation_count_refuses(
    graph: OriginGraphClient, deployments: UnDeployments
) -> None:
    deployments.short_by = 1

    with pytest.raises(SdmxResponseError, match="5 of 6 dated observations"):
        _table(graph, ["country/KEN"])
    _roles, table = _table(graph, ["country/KEN"], facet_ids=("another-facet",))
    assert table.num_rows == 0


class _Reader:
    """A UN reader that records its read's bounds and answers with `answer`."""

    def __init__(self, answer: Callable[[], dict]) -> None:
        self.answer = answer
        self.calls: list[dict] = []

    def observation_facets_batch(
        self,
        stat_vars: Sequence[str],
        entities: Sequence[str],
        needs_dates: bool = False,
        *,
        max_bytes: int | None = None,
        deadline: float | None = None,
    ) -> dict[str, dict[str, list[Facet]]]:
        self.calls.append(
            {"needs_dates": needs_dates, "max_bytes": max_bytes, "deadline": deadline}
        )
        return self.answer()


def test_the_read_carries_the_deadline_and_an_expired_one_raises_timeout() -> None:
    deadline = time.monotonic() + 60.0
    reader = _Reader(dict)
    list(
        un_observation_batches(
            reader,
            variables=[UN_VARIABLE],
            entities={"observationAbout": ["country/KEN"]},
            facet_ids=None,
            latest=False,
            max_rows=10,
            deadline=deadline,
        )
    )
    assert reader.calls == [
        {"needs_dates": True, "max_bytes": UN_READ_MAX_BYTES, "deadline": deadline}
    ]

    def expired() -> dict:
        raise GraphDeadlineExceededError("the read ran past its deadline")

    with pytest.raises(TimeoutError):
        un_observation_batches(
            _Reader(expired),
            variables=[UN_VARIABLE],
            entities={"observationAbout": ["country/KEN"]},
            facet_ids=None,
            latest=False,
            max_rows=None,
            deadline=deadline,
        )


def test_a_failed_un_read_raises_the_sdmx_upstream_error(
    graph: OriginGraphClient, deployments: UnDeployments
) -> None:
    deployments.status = 503

    with pytest.raises(SdmxHttpError) as excinfo:
        _table(graph, ["country/KEN"])

    assert excinfo.value.upstream_status == 503


def test_an_unreachable_un_deployment_raises_the_sdmx_transport_error(
    graph: OriginGraphClient, deployments: UnDeployments
) -> None:
    deployments.unreachable = True

    with pytest.raises(SdmxTransportError):
        _table(graph, ["country/KEN"])


def test_places_the_un_front_door_refuses_are_left_out_and_reported(
    graph: OriginGraphClient, deployments: UnDeployments
) -> None:
    deployments.refused = {"country/GRC"}

    with recording_un_blocked_places() as record:
        _roles, table = _table(graph, ["country/KEN", "country/GRC"])

    assert set(table.column("observationAbout").to_pylist()) == {"country/KEN"}
    assert table.num_rows == 5
    assert record.places() == ("country/GRC",)
    assert graph.un_blocked_places() == {"country/GRC"}


def test_a_fetch_mixing_un_and_other_variables_refuses(graph: OriginGraphClient) -> None:
    with (
        pytest.raises(MaterialisationUsageError),
        stream_observation_batches(
            sdmx=_sdmx_never_called(),
            graph=graph,
            variables=[UN_VARIABLE, "Count_Person"],
            entities={"observationAbout": ["country/KEN"]},
        ),
    ):
        pass


def test_un_rows_need_the_origin_graph_client() -> None:
    with (
        pytest.raises(MaterialisationUsageError),
        stream_observation_batches(
            sdmx=_sdmx_never_called(),
            graph=FakeGraphReader(),
            variables=[UN_VARIABLE],
            entities={"observationAbout": ["country/KEN"]},
        ),
    ):
        pass


def _request(stat_var: str, *, window: RequestedWindow | None = None) -> MaterialisationRequest:
    return MaterialisationRequest(
        stat_var=f"dcid:{stat_var}",
        role_bindings={"observationAbout": ["country/KEN"]},
        declared_roles=("observationAbout",),
        facet_bindings={"provenance": FacetBinding(state="bound", ref=UN_PROVENANCE)},
        window=window or RequestedWindow(kind="all"),
    )


def test_un_and_other_variables_never_share_a_fetch() -> None:
    assert request_batch_key(_request(UN_VARIABLE)) != request_batch_key(_request("Count_Person"))
    assert request_batch_key(_request(UN_VARIABLE)) == request_batch_key(
        _request("undata/who/MORT_U5_RATE")
    )


def test_un_rows_carry_the_facet_and_place_the_admission_mask_admits(
    graph: OriginGraphClient,
) -> None:
    places = ["country/GHA", "country/KEN"]
    probe = probe_facets(
        availability=NO_SCOPE_READ,
        stat_var=UN_VARIABLE,
        bound_entities=places,
        settle=True,
        graph=graph,
    )
    _roles, table = _table(graph, places)

    assert probe.admission == PlaceAdmission(places_by_facet={UN_FACET_ID: frozenset(places)})
    assert table.num_rows == 7
    assert columnar.admission_mask(table, admission=probe.admission).to_pylist() == [True] * 7


def test_a_preview_of_a_un_variable_serves_its_rows_without_sdmx(
    graph: OriginGraphClient,
) -> None:
    result = materialise_preview(
        _request(UN_VARIABLE), sdmx=_sdmx_never_called(), graph=graph, limit=10
    )

    assert result.total_rows == 5
    index = result.frame.columns.index("time_period")
    assert sorted(str(row[index]) for row in result.frame.rows) == [
        "1993",
        "1998",
        "2003",
        "2008",
        "2014",
    ]


def test_a_preview_refuses_loudly_when_the_un_is_down(
    graph: OriginGraphClient, deployments: UnDeployments
) -> None:
    # The facet probe succeeds and the row read fails: the preview refuses and never serves an
    # empty frame.
    deployments.status = 503
    deployments.fail_rows_only = True

    with pytest.raises(SdmxHttpError):
        materialise_preview(_request(UN_VARIABLE), sdmx=_sdmx_never_called(), graph=graph, limit=10)
    assert any("value" not in r.url.params.get_list("select") for r in deployments.un_requests)


_SNAPSHOT = RevisionSnapshot(graph_revision="graph-rev", source_revisions=())


def _mint_kwargs(graph: OriginGraphClient) -> dict:
    return {
        "sdmx": _sdmx_never_called(),
        "graph": graph,
        "revision_reader": FakeRevisionReader([_SNAPSHOT, _SNAPSHOT]),
        "materialiser_build": "build-1",
        "capability_version": "cap-1",
        "enrichment": EnrichmentLookup(graph),
    }


def test_a_mint_of_a_un_variable_serves_its_rows(graph: OriginGraphClient) -> None:
    result = materialise_spec(_request(UN_VARIABLE), **_mint_kwargs(graph))

    assert sum(len(frame.rows) for frame in result.frame_chunks()) == 5


def test_a_group_of_un_variables_reads_them_in_one_un_fetch(
    graph: OriginGraphClient, deployments: UnDeployments
) -> None:
    result = materialise_group(
        [_request(UN_VARIABLE), _request(UN_SECOND_VARIABLE)],
        member_concurrency=1,
        **_mint_kwargs(graph),
    )

    assert sum(len(frame.rows) for frame in result.frame_chunks()) == 10
    dated = [r for r in deployments.un_requests if "value" in r.url.params.get_list("select")]
    assert [sorted(r.url.params.get_list("variable.dcids")) for r in dated] == [
        [UN_VARIABLE, UN_SECOND_VARIABLE]
    ]


def test_a_collection_of_un_variables_serves_each_member(graph: OriginGraphClient) -> None:
    result = materialise_collection(
        [
            (f"spec_{1:032x}", _request(UN_VARIABLE)),
            (f"spec_{2:032x}", _request(UN_SECOND_VARIABLE)),
        ],
        **_mint_kwargs(graph),
    )

    assert [
        sum(len(frame.rows) for frame in member.frame_chunks()) for _, member in result.members
    ] == [5, 5]
