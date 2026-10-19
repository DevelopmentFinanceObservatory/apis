"""Checks `c[TIME_PERIOD]=LATEST` across single, collection, group, and preview fetch paths,
and the read of the year a `latest` window elects.

Latest and all-window members use separate chunks. A blank latest value yields no row on every
covered path.
"""

from __future__ import annotations

import httpx
import pytest

from dc_sdmx import SdmxClient
from materialise import (
    EnrichmentLookup,
    MaterialisationRequest,
    materialise_group,
    materialise_preview,
    materialise_spec,
)
from materialise.facets import FacetBinding
from materialise.graph import Facet
from materialise.window import RequestedWindow
from series_coverage.index import CoverageIndex
from series_coverage.schema import FacetMetadata, RoleNames
from tests.engine._fake_coverage import build_coverage_index

from .conftest import FakeGraphReader, only_chunk

_BASE_URL = "https://datacommons.one.org/core/api"
_STAT_VAR_A = "dcid:LatestWindowMemberA"
_STAT_VAR_B = "dcid:LatestWindowMemberB"
_BARE_A = "LatestWindowMemberA"
_BARE_B = "LatestWindowMemberB"


_BOUND_FACETS = {
    "provenance": FacetBinding(state="bound", ref="OECD_DAC2A"),
    "measurementMethod": FacetBinding(state="bound", ref="GrantEquivalent"),
    "unit": FacetBinding(state="bound", ref="USD"),
}

_CSV_HEADER = (
    "STRUCTURE,STRUCTURE_ID,ACTION,variableMeasured,recipient,unit,measurementMethod,"
    "observationPeriod,provenance,TIME_PERIOD,OBS_VALUE,scalingFactor,facetId\r\n"
)


def _facet(facet_id: str) -> Facet:
    return Facet(
        facet_id=facet_id,
        provenance_id="OECD_DAC2A",
        measurement_method="GrantEquivalent",
        unit="USD",
        obs_count=1,
        earliest_date="2015",
        latest_date="2017",
    )


def _row(*, stat_var: str, recipient: str, year: str, value: str, facet_id: str = "facet1") -> str:
    return (
        f"dataflow,DC:DF_OBS(1.0.0),I,{stat_var},{recipient},USD,GrantEquivalent,"
        f"NotApplicable,OECD_DAC2A,{year},{value},,{facet_id}\r\n"
    )


def _request(*, stat_var: str, recipient: str, window: RequestedWindow) -> MaterialisationRequest:
    return MaterialisationRequest(
        stat_var=stat_var,
        role_bindings={"recipient": [recipient]},
        declared_roles=("recipient",),
        facet_bindings=_BOUND_FACETS,
        window=window,
    )


def _is_data_fetch(request: httpx.Request) -> bool:
    """Facet probes omit `c[facetId]`; data fetches include it."""
    return "c[facetId]" in request.url.params


def _coverage_for(entries: list[tuple[str, str]], *, facet_id: str = "facet1") -> CoverageIndex:
    """Builds coverage rows for each variable-recipient pair under `facet_id`."""
    bare_stat_vars = {bare for bare, _ in entries}
    role_names = {bare: RoleNames(donor="donor", recipient="recipient") for bare in bare_stat_vars}
    facet_metadata = {
        facet_id: FacetMetadata(
            provenance_id="OECD_DAC2A", measurement_method="GrantEquivalent", unit="USD"
        )
    }
    rows = [
        (bare, facet_id, "country/DONOR", recipient, [2015, 2016, 2017])
        for bare, recipient in entries
    ]
    return build_coverage_index(rows, role_names=role_names, facet_metadata=facet_metadata)


def _recording_sdmx(csv_body: str) -> tuple[SdmxClient, list[httpx.Request]]:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            text=csv_body,
            headers={"content-type": "application/vnd.sdmx.data+csv;version=2.0.0"},
        )

    return SdmxClient(base_url=_BASE_URL, transport=httpx.MockTransport(handler)), requests


# --- Single-member path (`materialise_spec` -> `_generate_member`) ---------


def test_single_member_latest_window_sends_time_period_latest():
    csv_body = _CSV_HEADER + _row(
        stat_var=_BARE_A, recipient="country/TGO", year="2017", value="1000"
    )
    sdmx, requests = _recording_sdmx(csv_body)
    graph = FakeGraphReader(facets={(_BARE_A, "country/TGO"): [_facet("facet1")]})
    coverage = _coverage_for([(_BARE_A, "country/TGO")])

    materialise_spec(
        _request(
            stat_var=_STAT_VAR_A, recipient="country/TGO", window=RequestedWindow(kind="latest")
        ),
        sdmx=sdmx,
        graph=graph,
        enrichment=EnrichmentLookup(graph, parties=lambda: {}),
        materialiser_build="build-123",
        capability_version="cap-v1",
        coverage=coverage,
    )

    assert len(requests) == 1
    fetch_requests = [r for r in requests if _is_data_fetch(r)]
    assert len(fetch_requests) == 1
    assert fetch_requests[0].url.params.get("c[TIME_PERIOD]") == "LATEST"


def test_single_member_non_latest_window_omits_time_period():
    csv_body = _CSV_HEADER + _row(
        stat_var=_BARE_A, recipient="country/TGO", year="2017", value="1000"
    )
    sdmx, requests = _recording_sdmx(csv_body)
    graph = FakeGraphReader(facets={(_BARE_A, "country/TGO"): [_facet("facet1")]})
    coverage = _coverage_for([(_BARE_A, "country/TGO")])

    materialise_spec(
        _request(stat_var=_STAT_VAR_A, recipient="country/TGO", window=RequestedWindow(kind="all")),
        sdmx=sdmx,
        graph=graph,
        enrichment=EnrichmentLookup(graph, parties=lambda: {}),
        materialiser_build="build-123",
        capability_version="cap-v1",
        coverage=coverage,
    )

    assert len(requests) == 1
    fetch_requests = [r for r in requests if _is_data_fetch(r)]
    assert len(fetch_requests) == 1
    assert "c[TIME_PERIOD]" not in fetch_requests[0].url.params


def test_single_member_latest_window_blank_value_at_latest_period_serves_no_row():
    # The reduction selects 2017 before value filtering; the mock returns full history.
    csv_body = (
        _CSV_HEADER
        + _row(stat_var=_BARE_A, recipient="country/TGO", year="2015", value="1000")
        + _row(stat_var=_BARE_A, recipient="country/TGO", year="2016", value="2000")
        + _row(stat_var=_BARE_A, recipient="country/TGO", year="2017", value="")
    )
    sdmx, requests = _recording_sdmx(csv_body)
    graph = FakeGraphReader(facets={(_BARE_A, "country/TGO"): [_facet("facet1")]})
    coverage = _coverage_for([(_BARE_A, "country/TGO")])

    result = materialise_spec(
        _request(
            stat_var=_STAT_VAR_A, recipient="country/TGO", window=RequestedWindow(kind="latest")
        ),
        sdmx=sdmx,
        graph=graph,
        enrichment=EnrichmentLookup(graph, parties=lambda: {}),
        materialiser_build="build-123",
        capability_version="cap-v1",
        coverage=coverage,
    )

    assert requests[0].url.params.get("c[TIME_PERIOD]") == "LATEST"
    assert only_chunk(result).rows == ()
    assert result.coverage.rows == 0
    assert {w["code"] for w in result.warnings} == {"empty_result"}


def test_group_member_time_period_matches_each_members_own_window_kind():
    # The group uses separate chunks so each member gets its own window filter.
    def handler(request: httpx.Request) -> httpx.Response:
        variables = str(request.url.params.get("c[variableMeasured]"))
        if variables == _BARE_A:
            body = _CSV_HEADER + _row(
                stat_var=_BARE_A, recipient="country/KEN", year="2017", value="1000"
            )
        elif variables == _BARE_B:
            body = _CSV_HEADER + _row(
                stat_var=_BARE_B, recipient="country/TGO", year="2016", value="2000"
            )
        else:
            raise AssertionError(f"unexpected c[variableMeasured]={variables!r}")
        return httpx.Response(
            200, text=body, headers={"content-type": "application/vnd.sdmx.data+csv;version=2.0.0"}
        )

    requests: list[httpx.Request] = []

    def recording_handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return handler(request)

    sdmx = SdmxClient(base_url=_BASE_URL, transport=httpx.MockTransport(recording_handler))
    graph = FakeGraphReader(
        facets={
            (_BARE_A, "country/KEN"): [_facet("facet1")],
            (_BARE_B, "country/TGO"): [_facet("facet1")],
        }
    )
    coverage = _coverage_for([(_BARE_A, "country/KEN"), (_BARE_B, "country/TGO")])

    materialise_group(
        [
            _request(
                stat_var=_STAT_VAR_A, recipient="country/KEN", window=RequestedWindow(kind="latest")
            ),
            _request(
                stat_var=_STAT_VAR_B, recipient="country/TGO", window=RequestedWindow(kind="all")
            ),
        ],
        sdmx=sdmx,
        graph=graph,
        enrichment=EnrichmentLookup(graph, parties=lambda: {}),
        materialiser_build="build-123",
        capability_version="cap-v1",
        coverage=coverage,
    )

    assert len(requests) == 2
    fetch_requests = [r for r in requests if _is_data_fetch(r)]
    assert len(fetch_requests) == 2
    by_variables = {str(r.url.params.get("c[variableMeasured]")): r for r in fetch_requests}
    assert by_variables[_BARE_A].url.params.get("c[TIME_PERIOD]") == "LATEST"
    assert "c[TIME_PERIOD]" not in by_variables[_BARE_B].url.params


def test_preview_time_period_matches_the_requests_own_window_kind():
    csv_body = _CSV_HEADER + _row(
        stat_var=_BARE_A, recipient="country/TGO", year="2017", value="1000"
    )

    coverage = _coverage_for([(_BARE_A, "country/TGO")])

    sdmx_latest, requests_latest = _recording_sdmx(csv_body)
    graph_latest = FakeGraphReader(facets={(_BARE_A, "country/TGO"): [_facet("facet1")]})
    materialise_preview(
        _request(
            stat_var=_STAT_VAR_A, recipient="country/TGO", window=RequestedWindow(kind="latest")
        ),
        sdmx=sdmx_latest,
        graph=graph_latest,
        coverage=coverage,
        limit=100,
    )
    fetch_latest = [r for r in requests_latest if _is_data_fetch(r)]
    assert fetch_latest[0].url.params.get("c[TIME_PERIOD]") == "LATEST"

    sdmx_all, requests_all = _recording_sdmx(csv_body)
    graph_all = FakeGraphReader(facets={(_BARE_A, "country/TGO"): [_facet("facet1")]})
    materialise_preview(
        _request(stat_var=_STAT_VAR_A, recipient="country/TGO", window=RequestedWindow(kind="all")),
        sdmx=sdmx_all,
        graph=graph_all,
        coverage=coverage,
        limit=100,
    )
    fetch_all = [r for r in requests_all if _is_data_fetch(r)]
    assert "c[TIME_PERIOD]" not in fetch_all[0].url.params


# --- The year a `latest` window elects, on the member, group and preview paths ---------

_RECIPIENTS = tuple(f"country/R{i}" for i in range(1, 6))
_TO_2017 = ("2015", "2016", "2017")
_TO_2018 = (*_TO_2017, "2018")
# Two of the five series reach 2018, under half of the five holding 2017, which they elect.
_TWO_REACH_2018 = dict(
    zip(_RECIPIENTS, (_TO_2018, _TO_2018, _TO_2017, _TO_2017, _TO_2017), strict=True)
)


def _held_sdmx(
    held: dict[str, dict[str, tuple[str, ...]]],
) -> tuple[SdmxClient, list[tuple[str, str | None]]]:
    """An SDMX double over `held`, each StatVar's periods by recipient. A read returns the
    rows of the variables and recipients it names, each series' newest alone under `LATEST`
    and those at the periods it names otherwise. Records each read's variables and period."""
    reads: list[tuple[str, str | None]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        params = request.url.params
        period = params.get("c[TIME_PERIOD]")
        reads.append((params["c[variableMeasured]"], period))
        body = _CSV_HEADER
        for stat_var in params["c[variableMeasured]"].split(","):
            for recipient in params["c[recipient]"].split(","):
                periods = held[stat_var].get(recipient, ())
                if period == "LATEST":
                    periods = periods[-1:]
                elif period is not None:
                    periods = tuple(p for p in periods if p in period.split(","))
                for served in periods:
                    body += _row(stat_var=stat_var, recipient=recipient, year=served, value="1")
        return httpx.Response(
            200, text=body, headers={"content-type": "application/vnd.sdmx.data+csv;version=2.0.0"}
        )

    return SdmxClient(base_url=_BASE_URL, transport=httpx.MockTransport(handler)), reads


def _served_latest(
    path: str,
    *,
    indexed: dict[str, tuple[str, ...]],
    live: dict[str, tuple[str, ...]],
    observation_period: str = "P1Y",
    build_year: int | None = None,
    undispatched: tuple[str, ...] = (),
) -> tuple[
    list[tuple[str, str]], list[tuple[str, str | None]], tuple[int | None, int | None, bool] | None
]:
    """A `latest` read of member A at five recipients the index dates by `indexed` and SDMX
    holds at `live`, as a mint of A alone (`member`), a mint of A beside member B, which the
    index and SDMX both hold at the first three periods of `live` and read in one chunk
    with A (`group`), or a preview of A (`preview`). The index also dates A at every
    recipient by `undispatched` under a facet of another source, which the request's bound
    provenance leaves out. Returns A's `(recipient, period)` rows, the SDMX reads, and the
    year A elected, the year it served and whether it fell back (`LatestYear`), None for a
    preview."""
    beside = {r: periods[:3] for r, periods in live.items()} if path == "group" else {}
    members = {_BARE_A: indexed} | ({_BARE_B: beside} if beside else {})
    coverage = build_coverage_index(
        [
            (bare, "facet1", "country/DONOR", recipient, [int(period[:4]) for period in periods])
            for bare, held in members.items()
            for recipient, periods in held.items()
        ]
        + [
            (_BARE_A, "facet2", "country/DONOR", recipient, [int(p) for p in undispatched])
            for recipient in _RECIPIENTS
            if undispatched
        ],
        role_names=dict.fromkeys(members, RoleNames(donor="donor", recipient="recipient")),
        facet_metadata={
            "facet1": FacetMetadata(
                provenance_id="OECD_DAC2A",
                measurement_method="GrantEquivalent",
                unit="USD",
                observation_period=observation_period,
            ),
            "facet2": FacetMetadata(
                provenance_id="OTHER_SOURCE", measurement_method="GrantEquivalent", unit="USD"
            ),
        },
        build_year=build_year,
    )
    sdmx, reads = _held_sdmx({_BARE_A: live, _BARE_B: beside})
    graph = FakeGraphReader(
        facets={(bare, r): [_facet("facet1")] for bare in members for r in _RECIPIENTS}
    )
    requests = [
        MaterialisationRequest(
            stat_var=f"dcid:{bare}",
            role_bindings={"recipient": list(_RECIPIENTS)},
            declared_roles=("recipient",),
            facet_bindings=_BOUND_FACETS,
            window=RequestedWindow(kind="latest"),
        )
        for bare in members
    ]
    if path == "preview":
        preview = materialise_preview(
            requests[0], sdmx=sdmx, graph=graph, coverage=coverage, limit=100
        )
        frames, year = [preview.frame], None
    else:
        result = materialise_group(
            requests,
            sdmx=sdmx,
            graph=graph,
            enrichment=EnrichmentLookup(graph, parties=lambda: {}),
            materialiser_build="build-123",
            capability_version="cap-v1",
            coverage=coverage,
            member_concurrency=1,
        )
        latest = result.latest_years[_STAT_VAR_A]
        assert (latest.facet_ids, latest.omitted) == (("facet1",), ())
        frames, year = (
            list(result.frame_chunks()),
            (latest.elected, latest.served, latest.fell_back),
        )
        assert result.coverage.rows == sum(len(frame.rows) for frame in frames)
    served: list[tuple[str, str]] = []
    for frame in frames:
        stat_var, recipient, period = (
            frame.columns.index(name) for name in ("stat_var", "recipient", "time_period")
        )
        served.extend(
            (str(row[recipient]), str(row[period]))
            for row in frame.rows
            if row[stat_var] == _STAT_VAR_A
        )
    return sorted(served), reads, year


_PATHS = pytest.mark.parametrize("path", ["member", "group", "preview"])


def _variables(path: str) -> str:
    """The variables the `LATEST` read of `path` names."""
    return f"{_BARE_A},{_BARE_B}" if path == "group" else _BARE_A


@_PATHS
def test_latest_makes_one_read_when_every_series_ends_at_the_elected_year(path: str):
    """Five series the index and SDMX end in 2017 elect it, and the `LATEST` read holds
    every row of it, in one request of five rows."""
    held = dict.fromkeys(_RECIPIENTS, _TO_2017)

    served, reads, year = _served_latest(path, indexed=held, live=held)

    assert reads == [(_variables(path), "LATEST")]
    assert served == [(recipient, "2017") for recipient in _RECIPIENTS]
    assert year in (None, (2017, 2017, False))


@_PATHS
def test_latest_reads_the_elected_year_again_for_a_member_ending_after_it(path: str):
    """Two of five series reach 2018 and the five elect 2017. The `LATEST` read holds the
    two at 2018, so one more read names member A at 2017 and all five serve it. The newest
    period alone would serve two rows of 2018."""
    served, reads, year = _served_latest(path, indexed=_TWO_REACH_2018, live=_TWO_REACH_2018)

    assert reads == [(_variables(path), "LATEST"), (_BARE_A, "2017")]
    assert served == [(recipient, "2017") for recipient in _RECIPIENTS]
    assert year in (None, (2017, 2017, False))


@_PATHS
def test_latest_elects_over_the_facets_the_read_dispatches(path: str):
    """Another source's facet holds 2018 at all five recipients, and the request binds the
    provenance whose facet two of five reach 2018 under. The year is counted over the
    dispatched facet's series, so it is 2017, as with that facet alone. Counted over every
    facet it would be 2018, served from two."""
    served, reads, year = _served_latest(
        path, indexed=_TWO_REACH_2018, live=_TWO_REACH_2018, undispatched=_TO_2018
    )

    assert reads == [(_variables(path), "LATEST"), (_BARE_A, "2017")]
    assert served == [(recipient, "2017") for recipient in _RECIPIENTS]
    assert year in (None, (2017, 2017, False))


@_PATHS
def test_latest_serves_the_newest_period_when_no_row_holds_the_elected_year(path: str):
    """The index dates five series to 2018, which they elect, and SDMX ends them in 2017 or
    2016. No row holds 2018, so the read serves the newest period SDMX holds from the
    three series holding it, with no second request, and records that it fell back."""
    indexed = dict.fromkeys(_RECIPIENTS, _TO_2018)
    live = dict(
        zip(_RECIPIENTS, (_TO_2017, _TO_2017, _TO_2017, _TO_2017[:2], _TO_2017[:2]), strict=True)
    )

    served, reads, year = _served_latest(path, indexed=indexed, live=live)

    assert reads == [(_variables(path), "LATEST")]
    assert served == [(recipient, "2017") for recipient in _RECIPIENTS[:3]]
    assert year in (None, (2018, 2017, True))


@_PATHS
def test_latest_serves_the_elected_year_from_the_members_holding_it(path: str):
    """The index dates all five series at 2017, which they elect. SDMX holds 2017 for three
    and ends two in 2016. The three serve 2017, the two serve nothing, and the result
    counts the three rows it holds."""
    live = dict(
        zip(_RECIPIENTS, (_TO_2018, _TO_2018, _TO_2017, _TO_2017[:2], _TO_2017[:2]), strict=True)
    )

    served, reads, year = _served_latest(path, indexed=_TWO_REACH_2018, live=live)

    assert reads == [(_variables(path), "LATEST"), (_BARE_A, "2017")]
    assert served == [(recipient, "2017") for recipient in _RECIPIENTS[:3]]
    assert year in (None, (2017, 2017, False))


@_PATHS
@pytest.mark.parametrize(
    ("observation_period", "build_year", "live", "period"),
    [
        pytest.param(
            "P1M",
            None,
            {
                recipient: tuple(f"{year}-06" for year in periods)
                for recipient, periods in _TWO_REACH_2018.items()
            },
            "2018-06",
            id="sub-annual",
        ),
        pytest.param("P1Y", 2014, _TWO_REACH_2018, "2018", id="every-year-after-the-build"),
    ],
)
def test_latest_keeps_the_newest_period_for_a_member_that_elects_no_year(
    path: str,
    observation_period: str,
    build_year: int | None,
    live: dict[str, tuple[str, ...]],
    period: str,
):
    """Two of five series reach 2018, counts that elect 2017. A facet stating months, or an
    index built before every year it holds, elects no year, so one `LATEST` read serves the
    two series at the newest period."""
    served, reads, year = _served_latest(
        path,
        indexed=_TWO_REACH_2018,
        live=live,
        observation_period=observation_period,
        build_year=build_year,
    )

    assert reads == [(_variables(path), "LATEST")]
    assert served == [(recipient, period) for recipient in _RECIPIENTS[:2]]
    assert year in (None, (None, 2018, False))
