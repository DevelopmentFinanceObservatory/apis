"""Test SDMX facet grouping, request construction, caps, and error translation offline."""

from __future__ import annotations

import time
from urllib.parse import parse_qs, urlparse

import httpx
import pytest

import search.engine.clients.sdmx as sdmx_module
from dc_sdmx.errors import SdmxQueryError
from dc_sdmx.results import SdmxObservation
from search.engine.clients.sdmx import (
    LiveSdmxClient,
    _facets_from_sdmx_rows,
)
from search.engine.errors import GraphInfraError, RoleNotDeclaredError, RoleSignatureError

_BASE = "https://fake.example.org"

_CSV_HEADER = (
    "STRUCTURE,STRUCTURE_ID,ACTION,variableMeasured,provider,recipient,"
    "unit,measurementMethod,observationPeriod,provenance,TIME_PERIOD,OBS_VALUE,"
    "scalingFactor,facetId\r\n"
)


def _csv_row(
    *,
    variable: str = "Count_Person",
    provider: str = "country/USA",
    recipient: str = "country/KEN",
    unit: str = "USDollar",
    method: str = "EstimatedByOECD",
    period: str = "P1Y",
    provenance: str = "provenance/OECDDACCreditorReportingSystem",
    time_period: str = "2020",
    value: str = "123.4",
    facet_id: str = "facet1",
) -> str:
    return (
        f"dataflow,DC:DF_OBS(1.0.0),I,{variable},{provider},{recipient},"
        f"{unit},{method},{period},{provenance},{time_period},{value},,{facet_id}\r\n"
    )


def _csv_transport(
    rows: list[str], *, content_type: str = "application/vnd.sdmx.data+csv;version=2.0.0"
):
    body = (_CSV_HEADER + "".join(rows)).encode("utf-8")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=body, headers={"content-type": content_type})

    return httpx.MockTransport(handler)


def _error_transport(*, status_code: int, message: str) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status_code, json={"message": message, "code": 3})

    return httpx.MockTransport(handler)


def _client(transport: httpx.BaseTransport | None = None) -> LiveSdmxClient:
    return LiveSdmxClient(base=_BASE, transport=transport)


# Group rows into facets.


def _row(**overrides: object) -> SdmxObservation:
    defaults = {
        "variable": "Count_Person",
        "role_values": {"provider": "country/USA", "recipient": "country/KEN"},
        "unit": "USDollar",
        "measurement_method": "EstimatedByOECD",
        "observation_period": "P1Y",
        "provenance": "provenance/OECDDACCreditorReportingSystem",
        "time_period": "2020",
        "value": "123.4",
        "scaling_factor": None,
        "facet_id": "facet1",
    }
    defaults.update(overrides)
    return SdmxObservation(**defaults)


def test_facets_from_rows_single_facet_groups_all_rows_together():
    rows = [_row(time_period="2019", value="10"), _row(time_period="2020", value="20")]
    facets = _facets_from_sdmx_rows(rows)
    assert len(facets) == 1
    facet = facets[0]
    assert facet.dates == ("2019", "2020")
    assert facet.values == (10.0, 20.0)
    assert facet.obs_count == 2
    assert facet.earliest_date == "2019"
    assert facet.latest_date == "2020"
    assert facet.provenance_id == "provenance/OECDDACCreditorReportingSystem"
    assert facet.measurement_method == "EstimatedByOECD"
    assert facet.unit == "USDollar"
    assert facet.scaling_factor is None
    assert facet.facet_id == "facet1"


def test_facets_from_rows_splits_by_distinct_facet_id():
    rows = [
        _row(facet_id="facet1", time_period="2020"),
        _row(facet_id="facet2", time_period="2020", unit="EUR"),
    ]
    facets = _facets_from_sdmx_rows(rows)
    assert len(facets) == 2
    assert {f.unit for f in facets} == {"USDollar", "EUR"}
    # role_series filters by each Facet's id, on the wire and in the fake.
    assert {f.facet_id for f in facets} == {"facet1", "facet2"}


def test_facets_from_rows_same_facet_id_disagreeing_unit_raises():
    rows = [
        _row(facet_id="facet1", unit="USDollar"),
        _row(facet_id="facet1", unit="EUR"),
    ]
    with pytest.raises(GraphInfraError, match="facet1"):
        _facets_from_sdmx_rows(rows)


def test_facets_from_rows_empty_rows_returns_empty_list():
    assert _facets_from_sdmx_rows([]) == []


def test_facets_from_rows_carries_scaling_factor_through():
    rows = [_row(scaling_factor="100"), _row(scaling_factor="100", time_period="2021")]
    facets = _facets_from_sdmx_rows(rows)
    assert len(facets) == 1
    assert facets[0].scaling_factor == "100"


def test_facets_from_rows_same_facet_id_disagreeing_scaling_factor_raises():
    rows = [
        _row(facet_id="facet1", scaling_factor="100"),
        _row(facet_id="facet1", scaling_factor="1000"),
    ]
    with pytest.raises(GraphInfraError, match="facet1"):
        _facets_from_sdmx_rows(rows)


# LiveSdmxClient.role_series request shape, guards, and caps.


def test_role_series_request_shape_explicit_time_period():
    captured: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        body = (_CSV_HEADER + _csv_row()).encode("utf-8")
        return httpx.Response(
            200,
            content=body,
            headers={"content-type": "application/vnd.sdmx.data+csv;version=2.0.0"},
        )

    client = _client(httpx.MockTransport(handler))
    result = client.role_series(
        variables=["Count_Person"],
        entities={"provider": ["country/USA"], "recipient": ["country/KEN"]},
        facet_ids={"facet1"},
        time_period=["2020"],
        max_rows=100,
        max_bytes=100_000,
    )

    assert len(captured) == 1
    query = parse_qs(urlparse(str(captured[0].url)).query)
    assert query["c[facetId]"] == ["facet1"]
    assert query["c[TIME_PERIOD]"] == ["2020"]
    assert not any("lastNObservations" in key for key in query)

    role_tuple = (("provider", "country/USA"), ("recipient", "country/KEN"))
    facets = result["Count_Person"][role_tuple]
    assert len(facets) == 1
    assert facets[0].obs_count == 1
    assert facets[0].facet_id == "facet1"


def test_role_series_comma_joins_several_facet_ids_into_one_request():
    """A comma-joined c[facetId] OR-list uses one request.

    Each result keeps its row's facet id.
    """
    captured: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        body = (
            _CSV_HEADER
            + _csv_row(facet_id="facet1", time_period="2020", value="10")
            + _csv_row(facet_id="facet2", time_period="2020", value="20")
        ).encode("utf-8")
        return httpx.Response(
            200,
            content=body,
            headers={"content-type": "application/vnd.sdmx.data+csv;version=2.0.0"},
        )

    client = _client(httpx.MockTransport(handler))
    result = client.role_series(
        variables=["Count_Person"],
        entities={"provider": ["country/USA"], "recipient": ["country/KEN"]},
        facet_ids={"facet1", "facet2"},
        time_period=["2020"],
        max_rows=100,
        max_bytes=100_000,
    )

    assert len(captured) == 1
    query = parse_qs(urlparse(str(captured[0].url)).query)
    assert query["c[facetId]"] == ["facet1,facet2"]

    role_tuple = (("provider", "country/USA"), ("recipient", "country/KEN"))
    by_facet = {f.facet_id: f for f in result["Count_Person"][role_tuple]}
    assert set(by_facet) == {"facet1", "facet2"}
    assert by_facet["facet1"].values == (10.0,)
    assert by_facet["facet2"].values == (20.0,)


def test_role_series_raises_when_wire_returns_a_different_facet_id():
    """The server can ignore c[facetId].

    An unexpected row id raises before it contaminates a facet.
    """
    rows = [_csv_row(facet_id="facet1"), _csv_row(facet_id="facet2", recipient="country/NGA")]
    client = _client(_csv_transport(rows))
    with pytest.raises(GraphInfraError, match="facet1"):
        client.role_series(
            variables=["Count_Person"],
            entities={"provider": ["country/USA"], "recipient": ["country/KEN", "country/NGA"]},
            facet_ids={"facet1"},
            time_period=None,
            max_rows=100,
            max_bytes=100_000,
        )


def test_role_series_splits_across_requests_when_facet_ids_exceed_url_budget(monkeypatch):
    """Find a budget that splits two facet ids, then check their results merge.

    Search the range because joint query encoding adds parameter separators.
    """
    entities = {"provider": ["country/USA"], "recipient": ["country/KEN"]}
    time_period = ["2020"]
    variables = ["Count_Person"]
    facet_ids = {"facet1", "facet2"}

    def _chunk_count(candidate_budget: int) -> int | None:
        try:
            chunks = sdmx_module._split_facet_ids_for_request(
                facet_ids,
                variables=variables,
                entities=entities,
                time_period=time_period,
                budget=candidate_budget,
            )
        except SdmxQueryError:
            return None  # No facet-id chunk fits this budget.
        return len(chunks)

    split_budgets = [candidate for candidate in range(40, 500) if _chunk_count(candidate) == 2]
    assert split_budgets, "no budget in the search range split facet_ids into two chunks"
    # The midpoint leaves room above the split boundary for the real /data call.
    budget = split_budgets[len(split_budgets) // 2]

    captured: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        [facet_id] = parse_qs(urlparse(str(request.url)).query)["c[facetId]"][0].split(",")
        body = (_CSV_HEADER + _csv_row(facet_id=facet_id, time_period="2020")).encode("utf-8")
        return httpx.Response(
            200,
            content=body,
            headers={"content-type": "application/vnd.sdmx.data+csv;version=2.0.0"},
        )

    client = _client(httpx.MockTransport(handler))
    monkeypatch.setattr(client._client, "_url_budget_bytes", budget)
    result = client.role_series(
        variables=variables,
        entities=entities,
        facet_ids=facet_ids,
        time_period=time_period,
        max_rows=100,
        max_bytes=100_000,
    )

    assert len(captured) == 2
    requested_facet_ids = {parse_qs(urlparse(str(r.url)).query)["c[facetId]"][0] for r in captured}
    assert requested_facet_ids == {"facet1", "facet2"}

    role_tuple = (("provider", "country/USA"), ("recipient", "country/KEN"))
    merged_facet_ids = {f.facet_id for f in result["Count_Person"][role_tuple]}
    assert merged_facet_ids == {"facet1", "facet2"}


def test_role_series_large_variables_with_one_facet_id_does_not_raise():
    """SdmxClient.data chunks 400 variables within one facet id.

    Variables that alone exceed the URL budget must not make the one-facet split look
    unservable. Each variable is requested once.
    """
    variables = [
        f"dfo/statvar/crsAidReceived_{i:04d}_LongMeasureQualifierSuffix" for i in range(400)
    ]
    requested_variables: set[str] = set()
    request_count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal request_count
        request_count += 1
        query = parse_qs(urlparse(str(request.url)).query)
        chunk_vars = query["c[variableMeasured]"][0].split(",")
        requested_variables.update(chunk_vars)
        rows = "".join(_csv_row(variable=v, facet_id="facet1") for v in chunk_vars)
        body = (_CSV_HEADER + rows).encode("utf-8")
        return httpx.Response(
            200,
            content=body,
            headers={"content-type": "application/vnd.sdmx.data+csv;version=2.0.0"},
        )

    client = _client(httpx.MockTransport(handler))
    result = client.role_series(
        variables=variables,
        entities={"provider": ["country/USA"], "recipient": ["country/KEN"]},
        facet_ids={"facet1"},
        time_period=None,
        max_rows=100_000,
        max_bytes=100_000_000,
    )

    assert request_count > 1  # the variable list alone forces >1 wire request
    assert requested_variables == set(variables)
    assert set(result.keys()) == set(variables)


def test_role_series_large_variables_x_many_facet_ids_covers_every_pair():
    """Request every variable/facet pair exactly once across wire chunks."""
    variables = [
        f"dfo/statvar/crsAidReceived_{i:03d}_LongMeasureQualifierSuffix" for i in range(120)
    ]
    facet_ids = {f"{17650800455313554206 + i}" for i in range(40)}
    seen_pairs: set[tuple[str, str]] = set()

    def handler(request: httpx.Request) -> httpx.Response:
        query = parse_qs(urlparse(str(request.url)).query)
        chunk_vars = query["c[variableMeasured]"][0].split(",")
        chunk_facets = query["c[facetId]"][0].split(",")
        rows = "".join(_csv_row(variable=v, facet_id=f) for v in chunk_vars for f in chunk_facets)
        for v in chunk_vars:
            for f in chunk_facets:
                assert (v, f) not in seen_pairs, f"({v!r}, {f!r}) requested more than once"
                seen_pairs.add((v, f))
        body = (_CSV_HEADER + rows).encode("utf-8")
        return httpx.Response(
            200,
            content=body,
            headers={"content-type": "application/vnd.sdmx.data+csv;version=2.0.0"},
        )

    client = _client(httpx.MockTransport(handler))
    result = client.role_series(
        variables=variables,
        entities={"provider": ["country/USA"], "recipient": ["country/KEN"]},
        facet_ids=facet_ids,
        time_period=None,
        max_rows=1_000_000,
        max_bytes=100_000_000,
    )

    assert seen_pairs == {(v, f) for v in variables for f in facet_ids}
    role_tuple = (("provider", "country/USA"), ("recipient", "country/KEN"))
    for variable in variables:
        got_facets = {f.facet_id for f in result[variable][role_tuple]}
        assert got_facets == facet_ids


def test_role_series_facet_id_alone_past_budget_raises_naming_a_facet_id(monkeypatch):
    """An unservable facet id raises before HTTP.

    The 300-character id exceeds the patched 200-byte budget.
    """
    long_facet_id = "9" * 300
    client = _client()
    monkeypatch.setattr(client._client, "_url_budget_bytes", 200)
    with pytest.raises(SdmxQueryError, match=r"facet id.*" + long_facet_id):
        client.role_series(
            variables=["Count_Person"],
            entities={"provider": ["country/USA"], "recipient": ["country/KEN"]},
            facet_ids={long_facet_id},
            time_period=None,
            max_rows=100,
            max_bytes=100_000,
        )


def test_role_series_facet_budget_boundary_matches_real_joint_query_cost(monkeypatch):
    """Joint query separators count toward the URL budget.

    One byte below the encoded length fails. The exact length serves one facet.
    """
    entities = {"provider": ["country/USA"], "recipient": ["country/KEN"]}
    time_period = ["2020"]
    variable = "Count_Person"
    facet_id = "9"

    def real_joint_len(candidate_facet_id: str) -> int:
        params = {
            "c[variableMeasured]": sdmx_module._or_list("variableMeasured", [variable]),
            "c[facetId]": sdmx_module._or_list("facetId", [candidate_facet_id]),
            **sdmx_module._entity_params(entities),
            "c[TIME_PERIOD]": sdmx_module._time_period_param(time_period),
        }
        return len(str(httpx.QueryParams(params)))

    exact_len = real_joint_len(facet_id)

    with pytest.raises(SdmxQueryError, match=r"facet id.*" + facet_id):
        sdmx_module._split_facet_ids_for_request(
            {facet_id},
            variables=[variable],
            entities=entities,
            time_period=time_period,
            budget=exact_len - 1,
        )

    chunks = sdmx_module._split_facet_ids_for_request(
        {facet_id},
        variables=[variable],
        entities=entities,
        time_period=time_period,
        budget=exact_len,
    )
    assert chunks == [(facet_id,)]

    captured: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        body = (
            _CSV_HEADER + _csv_row(variable=variable, facet_id=facet_id, time_period="2020")
        ).encode("utf-8")
        return httpx.Response(
            200,
            content=body,
            headers={"content-type": "application/vnd.sdmx.data+csv;version=2.0.0"},
        )

    client = _client(httpx.MockTransport(handler))
    monkeypatch.setattr(client._client, "_url_budget_bytes", exact_len)

    result = client.role_series(
        variables=[variable],
        entities=entities,
        facet_ids={facet_id},
        time_period=time_period,
        max_rows=100,
        max_bytes=100_000,
    )

    assert len(captured) == 1
    role_tuple = (("provider", "country/USA"), ("recipient", "country/KEN"))
    assert {f.facet_id for f in result[variable][role_tuple]} == {facet_id}


def test_role_series_long_facet_fits_with_longest_variable_not_with_half_budget_reservation():
    """A single facet that fits with the longest variable is servable.

    This holds even when the packer's reservation is larger.
    """
    entities = {"provider": ["country/USA"], "recipient": ["country/KEN"]}
    time_period = ["2020"]
    variables = ["Count_Person"]
    budget = 8000

    fixed_params = dict(sdmx_module._entity_params(entities))
    fixed_params["c[TIME_PERIOD]"] = sdmx_module._time_period_param(time_period)
    fixed_len = len(str(httpx.QueryParams(fixed_params)))
    a = budget - fixed_len
    longest_variable_len = len(
        str(
            httpx.QueryParams(
                {"c[variableMeasured]": sdmx_module._or_list("variableMeasured", variables)}
            )
        )
    )
    reservation = max(longest_variable_len, a // 2)

    def facet_chunk_len(facet_id: str) -> int:
        return len(str(httpx.QueryParams({"c[facetId]": facet_id, **fixed_params})))

    long_facet_id = None
    for n in range(1, 6000):
        candidate = "9" * n
        length = facet_chunk_len(candidate)
        if budget - reservation < length <= budget - longest_variable_len:
            long_facet_id = candidate
            break
    assert long_facet_id is not None, "no facet id in the search range hit the target window"

    chunks = sdmx_module._split_facet_ids_for_request(
        {long_facet_id},
        variables=variables,
        entities=entities,
        time_period=time_period,
        budget=budget,
    )

    assert chunks == [(long_facet_id,)]


def test_role_series_facet_ids_within_half_budget_of_headroom_pack_into_one_chunk():
    """Facet ids within half the budget headroom fit in one chunk."""
    entities = {"provider": ["country/USA"], "recipient": ["country/KEN"]}
    time_period = ["2020"]
    variables = ["Count_Person"]
    budget = 8000
    facet_ids = {"facet-alpha", "facet-beta", "facet-gamma"}

    fixed_params = dict(sdmx_module._entity_params(entities))
    fixed_params["c[TIME_PERIOD]"] = sdmx_module._time_period_param(time_period)
    fixed_len = len(str(httpx.QueryParams(fixed_params)))
    a = budget - fixed_len
    c = len(sdmx_module._or_list("facetId", sorted(facet_ids)))
    assert c <= a // 2, "test fixture must exercise the c <= a/2 regime"

    chunks = sdmx_module._split_facet_ids_for_request(
        facet_ids, variables=variables, entities=entities, time_period=time_period, budget=budget
    )

    assert chunks == [tuple(sorted(facet_ids))]


def test_role_series_duplicate_variable_is_refused_before_any_request():
    """#850 rejects duplicate variables before HTTP.

    Cache settings produce the same refusal.
    """
    captured: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        body = (_CSV_HEADER + _csv_row(time_period="2020")).encode("utf-8")
        return httpx.Response(
            200,
            content=body,
            headers={"content-type": "application/vnd.sdmx.data+csv;version=2.0.0"},
        )

    client = _client(httpx.MockTransport(handler))
    with pytest.raises(ValueError, match=r"repeats 'Count_Person' \(2 times\)"):
        client.role_series(
            variables=["Count_Person", "Count_Person"],
            entities={"provider": ["country/USA"], "recipient": ["country/KEN"]},
            facet_ids={"facet1"},
            time_period=["2020"],
            max_rows=100,
            max_bytes=100_000,
        )

    assert captured == []


def test_role_series_singleton_facet_with_large_variables_matches_bulk_role_series_call_shape():
    """A singleton facet id with 250 variables serves without raising."""
    variables = [f"dfo/statvar/crsAidReceivedConstant2023USD_{i:04d}" for i in range(250)]

    def handler(request: httpx.Request) -> httpx.Response:
        query = parse_qs(urlparse(str(request.url)).query)
        chunk_vars = query["c[variableMeasured]"][0].split(",")
        rows = "".join(_csv_row(variable=v, facet_id="17650800455313554206") for v in chunk_vars)
        body = (_CSV_HEADER + rows).encode("utf-8")
        return httpx.Response(
            200,
            content=body,
            headers={"content-type": "application/vnd.sdmx.data+csv;version=2.0.0"},
        )

    client = _client(httpx.MockTransport(handler))
    result = client.role_series(
        variables=variables,
        entities={"provider": ["country/USA"], "recipient": ["country/KEN"]},
        facet_ids={"17650800455313554206"},
        time_period=None,
        max_rows=1_000_000,
        max_bytes=100_000_000,
    )

    assert set(result.keys()) == set(variables)


def test_role_series_time_period_none_omits_time_period_param():
    captured: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        body = (_CSV_HEADER + _csv_row()).encode("utf-8")
        return httpx.Response(
            200,
            content=body,
            headers={"content-type": "application/vnd.sdmx.data+csv;version=2.0.0"},
        )

    client = _client(httpx.MockTransport(handler))
    client.role_series(
        variables=["Count_Person"],
        entities={"provider": ["country/USA"], "recipient": ["country/KEN"]},
        facet_ids={"facet1"},
        time_period=None,
        max_rows=100,
        max_bytes=100_000,
    )
    query = parse_qs(urlparse(str(captured[0].url)).query)
    assert "c[TIME_PERIOD]" not in query
    assert query["c[facetId]"] == ["facet1"]


def test_role_series_sends_latest_as_the_time_period_filter():
    """`LATEST` reaches the wire whole, so the mixer returns each series' latest point."""
    captured: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        body = (_CSV_HEADER + _csv_row()).encode("utf-8")
        return httpx.Response(
            200,
            content=body,
            headers={"content-type": "application/vnd.sdmx.data+csv;version=2.0.0"},
        )

    client = _client(httpx.MockTransport(handler))
    client.role_series(
        variables=["Count_Person"],
        entities={"provider": ["country/USA"], "recipient": ["country/KEN"]},
        facet_ids={"facet1"},
        time_period="LATEST",
        max_rows=100,
        max_bytes=100_000,
    )
    query = parse_qs(urlparse(str(captured[0].url)).query)
    assert query["c[TIME_PERIOD]"] == ["LATEST"]


def test_role_series_rejects_other_bare_string_time_period():
    """A bare string would split into characters, so reject it."""
    client = _client()
    with pytest.raises(ValueError, match="bare string"):
        client.role_series(
            variables=["Count_Person"],
            entities={"provider": ["country/USA"], "recipient": ["country/KEN"]},
            facet_ids={"facet1"},
            time_period="2020",
            max_rows=100,
            max_bytes=100_000,
        )


def test_role_series_rejects_empty_facet_ids():
    client = _client()
    with pytest.raises(ValueError, match="facet_ids"):
        client.role_series(
            variables=["Count_Person"],
            entities={"provider": ["country/USA"], "recipient": ["country/KEN"]},
            facet_ids=set(),
            time_period=None,
            max_rows=100,
            max_bytes=100_000,
        )


def test_role_series_max_rows_cap_propagates_uncaught():
    from dc_sdmx.errors import SdmxCapExceededError

    rows = [_csv_row(time_period=str(2018 + i)) for i in range(3)]
    client = _client(_csv_transport(rows))
    with pytest.raises(SdmxCapExceededError):
        client.role_series(
            variables=["Count_Person"],
            entities={"provider": ["country/USA"], "recipient": ["country/KEN"]},
            facet_ids={"facet1"},
            time_period=None,
            max_rows=1,
            max_bytes=100_000,
        )


# LiveSdmxClient.availability_values request shape.


def test_availability_values_request_shape_and_parsing():
    availability_body = {
        "data": {
            "dataConstraints": [
                {
                    "cubeRegions": [
                        {
                            "keyValues": [
                                {"id": "recipient", "values": ["country/KEN", "country/NGA"]},
                            ]
                        }
                    ]
                }
            ]
        }
    }
    captured: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(200, json=availability_body)

    client = _client(httpx.MockTransport(handler))
    values = client.availability_values(
        variables=["Count_Person"],
        component="recipient",
        filters={"provider": ["country/USA"]},
    )

    assert values == ["country/KEN", "country/NGA"]
    query = parse_qs(urlparse(str(captured[0].url)).query)
    assert query["c[variableMeasured]"] == ["Count_Person"]
    assert query["c[provider]"] == ["country/USA"]
    assert "/availability/dataflow/DC/DF_OBS/1.0.0/*/recipient" in str(captured[0].url)


def test_availability_values_unknown_component_returns_empty_list():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"data": {"dataConstraints": []}})

    client = _client(httpx.MockTransport(handler))
    values = client.availability_values(variables=["Count_Person"], component="recipient")
    assert values == []


def test_availability_values_time_period_request_shape():
    availability_body = {
        "data": {
            "dataConstraints": [
                {"cubeRegions": [{"keyValues": [{"id": "recipient", "values": ["country/KEN"]}]}]}
            ]
        }
    }
    captured: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(200, json=availability_body)

    client = _client(httpx.MockTransport(handler))
    values = client.availability_values(
        variables=["Count_Person"],
        component="recipient",
        time_period=["2020", "2021"],
    )

    assert values == ["country/KEN"]
    query = parse_qs(urlparse(str(captured[0].url)).query)
    assert query["c[TIME_PERIOD]"] == ["2020,2021"]


def test_availability_values_passes_absolute_deadline_to_sdmx_client():
    availability_body = {"data": {"dataConstraints": []}}
    captured: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(200, json=availability_body)

    client = _client(httpx.MockTransport(handler))
    deadline = time.monotonic() + 5
    assert (
        client.availability_values(
            variables=["Count_Person"], component="recipient", deadline=deadline
        )
        == []
    )

    assert 0 < captured[0].extensions["timeout"]["read"] <= 5


def test_availability_values_max_values_cap_propagates_uncaught():
    from dc_sdmx.errors import SdmxCapExceededError

    availability_body = {
        "data": {
            "dataConstraints": [
                {
                    "cubeRegions": [
                        {
                            "keyValues": [
                                {
                                    "id": "recipient",
                                    "values": ["country/KEN", "country/NGA", "country/USA"],
                                }
                            ]
                        }
                    ]
                }
            ]
        }
    }

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=availability_body)

    client = _client(httpx.MockTransport(handler))
    with pytest.raises(SdmxCapExceededError):
        client.availability_values(variables=["Count_Person"], component="recipient", max_values=1)


def test_availability_values_unsupported_time_period_raises_role_not_declared():
    # LATEST has no availability endpoint and maps to RoleNotDeclaredError.
    client = _client()
    with pytest.raises(RoleNotDeclaredError):
        client.availability_values(
            variables=["Count_Person"], component="recipient", time_period="LATEST"
        )


# Exception translation table


def test_missing_filter_error_raises_role_not_declared():
    client = _client(
        _error_transport(status_code=400, message="missing required SDMX component filter")
    )
    with pytest.raises(RoleNotDeclaredError):
        client.role_series(
            variables=["v"],
            entities={"provider": ["country/USA"], "recipient": ["country/KEN"]},
            facet_ids={"facet1"},
            max_rows=1000,
            max_bytes=1000,
        )


def test_unsupported_filter_error_raises_role_not_declared():
    client = _client(_error_transport(status_code=400, message="unsupported SDMX component filter"))
    with pytest.raises(RoleNotDeclaredError):
        client.role_series(
            variables=["v"],
            entities={"provider": ["country/USA"], "recipient": ["country/KEN"]},
            facet_ids={"facet1"},
            max_rows=1000,
            max_bytes=1000,
        )


def test_containment_filter_error_raises_graph_infra_error():
    client = _client(
        _error_transport(status_code=400, message="require containedInPlace+ and typeOf")
    )
    with pytest.raises(GraphInfraError):
        client.role_series(
            variables=["v"],
            entities={"provider": ["country/USA"], "recipient": ["country/KEN"]},
            facet_ids={"facet1"},
            max_rows=1000,
            max_bytes=1000,
        )


def test_incompatible_variables_error_raises_role_signature_error():
    client = _client(
        _error_transport(status_code=400, message="incompatible observationProperties")
    )
    with pytest.raises(RoleSignatureError):
        client.role_series(
            variables=["v"],
            entities={"provider": ["country/USA"], "recipient": ["country/KEN"]},
            facet_ids={"facet1"},
            max_rows=1000,
            max_bytes=1000,
        )


def test_role_ceiling_error_raises_role_signature_error():
    client = _client(_error_transport(status_code=400, message="max supported entity slots is 3"))
    with pytest.raises(RoleSignatureError):
        client.role_series(
            variables=["v"],
            entities={"provider": ["country/USA"], "recipient": ["country/KEN"]},
            facet_ids={"facet1"},
            max_rows=1000,
            max_bytes=1000,
        )


def test_unmatched_http_error_raises_graph_infra_error():
    client = _client(_error_transport(status_code=500, message="internal server error"))
    with pytest.raises(GraphInfraError):
        client.role_series(
            variables=["v"],
            entities={"provider": ["country/USA"], "recipient": ["country/KEN"]},
            facet_ids={"facet1"},
            max_rows=1000,
            max_bytes=1000,
        )


def test_bad_content_type_response_error_raises_graph_infra_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"{}", headers={"content-type": "application/json"})

    client = _client(httpx.MockTransport(handler))
    with pytest.raises(GraphInfraError):
        client.role_series(
            variables=["v"],
            entities={"provider": ["country/USA"], "recipient": ["country/KEN"]},
            facet_ids={"facet1"},
            max_rows=1000,
            max_bytes=1000,
        )


def test_transport_error_raises_graph_infra_error():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    client = _client(httpx.MockTransport(handler))
    with pytest.raises(GraphInfraError):
        client.role_series(
            variables=["v"],
            entities={"provider": ["country/USA"], "recipient": ["country/KEN"]},
            facet_ids={"facet1"},
            max_rows=1000,
            max_bytes=1000,
        )


def test_availability_unsupported_component_raises_role_not_declared():
    client = _client(
        _error_transport(status_code=400, message="unsupported SDMX availability component")
    )
    with pytest.raises(RoleNotDeclaredError):
        client.availability_values(variables=["v"], component="TIME_PERIOD")


# Role-map construction with empty lists and None values.


def test_role_series_empty_entity_list_raises_sdmx_query_error_uncaught():
    from dc_sdmx.errors import SdmxQueryError

    client = _client()
    with pytest.raises(SdmxQueryError):
        client.role_series(
            variables=["v"],
            entities={"provider": []},
            facet_ids={"facet1"},
            max_rows=1000,
            max_bytes=1000,
        )


def test_role_series_none_entity_value_raises_type_error_uncaught():
    client = _client()
    with pytest.raises(TypeError):
        client.role_series(
            variables=["v"],
            entities={"provider": [None]},  # ty: ignore[invalid-argument-type]
            facet_ids={"facet1"},
            max_rows=1000,
            max_bytes=1000,
        )


# LiveSdmxClient.subject_observations wire assembly, LATEST, facet ids, and caps.

_SUBJECT_CSV_HEADER = (
    "STRUCTURE,STRUCTURE_ID,ACTION,variableMeasured,observationAbout,"
    "unit,measurementMethod,observationPeriod,provenance,TIME_PERIOD,OBS_VALUE,"
    "scalingFactor,facetId\r\n"
)


def _subject_csv_row(
    *,
    variable: str = "Count_Person",
    entity: str = "country/KEN",
    unit: str = "USDollar",
    method: str = "EstimatedByOECD",
    period: str = "P1Y",
    provenance: str = "provenance/OECDDACCreditorReportingSystem",
    time_period: str = "2020",
    value: str = "123.4",
    facet_id: str = "facet1",
) -> str:
    return (
        f"dataflow,DC:DF_OBS(1.0.0),I,{variable},{entity},"
        f"{unit},{method},{period},{provenance},{time_period},{value},,{facet_id}\r\n"
    )


def test_subject_observations_request_shape():
    captured: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        body = (_SUBJECT_CSV_HEADER + _subject_csv_row()).encode("utf-8")
        return httpx.Response(
            200,
            content=body,
            headers={"content-type": "application/vnd.sdmx.data+csv;version=2.0.0"},
        )

    client = _client(httpx.MockTransport(handler))
    result = client.subject_observations(
        variables=["Count_Person"],
        places=["country/KEN", "Earth"],
        facet_ids=["facet1"],
    )

    assert len(captured) == 1
    query = parse_qs(urlparse(str(captured[0].url)).query)
    assert query["c[variableMeasured]"] == ["Count_Person"]
    assert query["c[observationAbout]"] == ["country/KEN,Earth"]
    assert query["c[facetId]"] == ["facet1"]
    assert query["c[TIME_PERIOD]"] == ["LATEST"]

    facets = result["Count_Person"]["country/KEN"]
    assert len(facets) == 1
    assert facets[0].facet_id == "facet1"


def test_subject_observations_latest_only_false_omits_time_period_param():
    captured: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        body = (_SUBJECT_CSV_HEADER + _subject_csv_row()).encode("utf-8")
        return httpx.Response(
            200,
            content=body,
            headers={"content-type": "application/vnd.sdmx.data+csv;version=2.0.0"},
        )

    client = _client(httpx.MockTransport(handler))
    client.subject_observations(
        variables=["Count_Person"], places=["country/KEN"], latest_only=False
    )

    query = parse_qs(urlparse(str(captured[0].url)).query)
    assert "c[TIME_PERIOD]" not in query


def _subject_csv_transport(
    rows: list[str], *, content_type: str = "application/vnd.sdmx.data+csv;version=2.0.0"
) -> httpx.MockTransport:
    body = (_SUBJECT_CSV_HEADER + "".join(rows)).encode("utf-8")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=body, headers={"content-type": content_type})

    return httpx.MockTransport(handler)


def test_subject_observations_max_rows_cap_propagates_uncaught():
    from dc_sdmx.errors import SdmxCapExceededError

    rows = [_subject_csv_row(entity=f"country/E{i}") for i in range(3)]
    client = _client(_subject_csv_transport(rows))
    with pytest.raises(SdmxCapExceededError):
        client.subject_observations(variables=["Count_Person"], places=["country/KEN"], max_rows=1)


def test_subject_observations_groups_rows_by_variable_and_entity():
    rows = [
        _subject_csv_row(entity="country/E1", facet_id="f1"),
        _subject_csv_row(entity="country/E2", facet_id="f1"),
    ]
    client = _client(_subject_csv_transport(rows))
    result = client.subject_observations(
        variables=["Count_Person"], places=["country/E1", "country/E2"]
    )
    assert set(result["Count_Person"]) == {"country/E1", "country/E2"}
    assert result["Count_Person"]["country/E1"][0].facet_id == "f1"


def test_subject_observations_reads_a_long_place_list_in_url_budget_slices():
    # 2,000 places cannot ride one GET. Each slice is its own read, and the row cap counts the
    # rows of every slice together.
    places = [f"wikidataId/Q{index:07d}" for index in range(2_000)]
    named: list[list[str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        assert len(str(request.url.params)) <= 7_800
        slice_places = str(request.url.params["c[observationAbout]"]).split(",")
        named.append(slice_places)
        body = (
            _SUBJECT_CSV_HEADER
            + "".join(_subject_csv_row(entity=place) for place in slice_places[:1])
        ).encode("utf-8")
        return httpx.Response(
            200,
            content=body,
            headers={"content-type": "application/vnd.sdmx.data+csv;version=2.0.0"},
        )

    client = _client(httpx.MockTransport(handler))
    result = client.subject_observations(variables=["Count_Person"], places=places)

    assert len(named) > 1
    assert [place for slice_places in named for place in slice_places] == places
    assert set(result["Count_Person"]) == {slice_places[0] for slice_places in named}

    from dc_sdmx.errors import SdmxCapExceededError

    with pytest.raises(SdmxCapExceededError):
        client.subject_observations(
            variables=["Count_Person"], places=places, max_rows=len(named) - 1
        )
