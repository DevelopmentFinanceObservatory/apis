"""Tests role-series filtering, stub handling, availability keys, and deadline recording."""

from __future__ import annotations

import pytest

from search.engine.clients.graph import Facet
from tests.engine._fake_sdmx import FakeSdmxClient, StubSeriesReadError

_ROLE_TUPLE = (("provider", "country/USA"), ("recipient", "country/KEN"))

_FACET_1 = Facet(
    earliest_date="2019",
    latest_date="2021",
    obs_count=3,
    dates=["2019", "2020", "2021"],
    values=[10.0, 20.0, 30.0],
    provenance_id="provenance/OECDDACCreditorReportingSystem",
    unit="USDollar",
    facet_id="facet1",
)
_FACET_2 = Facet(
    earliest_date="2020",
    latest_date="2020",
    obs_count=1,
    dates=["2020"],
    values=[99.0],
    unit="EUR",
    facet_id="facet2",
)


def _client() -> FakeSdmxClient:
    return FakeSdmxClient(observations={"Count_Person": {_ROLE_TUPLE: [_FACET_1, _FACET_2]}})


# role_series: honours facet_id and time_period, rejects malformed calls


def test_role_series_filters_to_requested_facet_id():
    client = _client()
    result = client.role_series(
        variables=["Count_Person"],
        entities={"provider": ["country/USA"], "recipient": ["country/KEN"]},
        facet_ids={"facet1"},
        max_rows=100,
        max_bytes=100_000,
    )
    facets = result["Count_Person"][_ROLE_TUPLE]
    assert len(facets) == 1
    assert facets[0].facet_id == "facet1"
    # facet2 (a real, distinct facet on the same role tuple) never appears --
    # a fake that ignored facet_id would return both and let a caller relying
    # on narrowing pass against a client that never narrowed.
    assert all(f.facet_id != "facet2" for f in facets)


def test_role_series_time_period_none_returns_full_history():
    client = _client()
    result = client.role_series(
        variables=["Count_Person"],
        entities={"provider": ["country/USA"], "recipient": ["country/KEN"]},
        facet_ids={"facet1"},
        time_period=None,
        max_rows=100,
        max_bytes=100_000,
    )
    facet = result["Count_Person"][_ROLE_TUPLE][0]
    assert facet.dates == ["2019", "2020", "2021"]
    assert facet.obs_count == 3


def test_role_series_narrows_to_requested_time_period():
    client = _client()
    result = client.role_series(
        variables=["Count_Person"],
        entities={"provider": ["country/USA"], "recipient": ["country/KEN"]},
        facet_ids={"facet1"},
        time_period=["2020", "2021"],
        max_rows=100,
        max_bytes=100_000,
    )
    facet = result["Count_Person"][_ROLE_TUPLE][0]
    assert facet.dates == ["2020", "2021"]
    assert facet.values == [20.0, 30.0]
    assert facet.obs_count == 2
    assert facet.earliest_date == "2020"
    assert facet.latest_date == "2021"


def test_role_series_rejects_empty_facet_ids():
    client = _client()
    with pytest.raises(ValueError, match="facet_ids"):
        client.role_series(
            variables=["Count_Person"],
            entities={"provider": ["country/USA"], "recipient": ["country/KEN"]},
            facet_ids=set(),
            max_rows=100,
            max_bytes=100_000,
        )


def test_role_series_latest_keeps_each_series_latest_point():
    client = _client()
    result = client.role_series(
        variables=["Count_Person"],
        entities={"provider": ["country/USA"], "recipient": ["country/KEN"]},
        facet_ids={"facet1"},
        time_period="LATEST",
        max_rows=100,
        max_bytes=100_000,
    )
    facet = result["Count_Person"][_ROLE_TUPLE][0]
    assert facet.dates == ["2021"]
    assert facet.values == [30.0]
    assert client.series_calls[0]["time_period"] == "LATEST"


def test_role_series_call_recorded_with_facet_ids_and_time_period():
    client = _client()
    client.role_series(
        variables=["Count_Person"],
        entities={"provider": ["country/USA"], "recipient": ["country/KEN"]},
        facet_ids={"facet1"},
        time_period=["2020"],
        max_rows=100,
        max_bytes=100_000,
    )
    assert len(client.series_calls) == 1
    assert client.series_calls[0]["facet_ids"] == {"facet1"}
    assert client.series_calls[0]["time_period"] == ["2020"]


# stub_keys: role_series refuses a stub row

_STUB_FACET = Facet(
    earliest_date="2021",
    latest_date="2021",
    obs_count=1,
    dates=["2021"],
    values=[5.0],
    facet_id="facet_stub",
)


def _client_with_a_stub() -> FakeSdmxClient:
    return FakeSdmxClient(
        observations={"Count_Person": {_ROLE_TUPLE: [_STUB_FACET]}},
        stub_keys={("Count_Person", _ROLE_TUPLE, "facet_stub")},
    )


def test_role_series_raises_on_a_stub_row():
    """A requested stub key raises `StubSeriesReadError`."""
    client = _client_with_a_stub()
    with pytest.raises(StubSeriesReadError) as exc_info:
        client.role_series(
            variables=["Count_Person"],
            entities={"provider": ["country/USA"], "recipient": ["country/KEN"]},
            facet_ids={"facet_stub"},
            max_rows=100,
            max_bytes=100_000,
        )
    assert "Count_Person" in str(exc_info.value)
    assert str(_ROLE_TUPLE) in str(exc_info.value)


def test_role_series_unaffected_by_a_stub_row_for_a_different_facet_id():
    """A stub row does not block a different facet id on the same role tuple."""
    real_facet = Facet(
        earliest_date="2019",
        latest_date="2021",
        obs_count=3,
        dates=["2019", "2020", "2021"],
        values=[1.0, 2.0, 3.0],
        facet_id="facet_real",
    )
    client = FakeSdmxClient(
        observations={"Count_Person": {_ROLE_TUPLE: [_STUB_FACET, real_facet]}},
        stub_keys={("Count_Person", _ROLE_TUPLE, "facet_stub")},
    )
    result = client.role_series(
        variables=["Count_Person"],
        entities={"provider": ["country/USA"], "recipient": ["country/KEN"]},
        facet_ids={"facet_real"},
        max_rows=100,
        max_bytes=100_000,
    )
    facets = result["Count_Person"][_ROLE_TUPLE]
    assert len(facets) == 1
    assert facets[0].facet_id == "facet_real"


def test_stub_keys_defaults_to_empty_for_an_explicit_observations_override():
    """Explicit observations have no stub keys unless the caller supplies them."""
    client = FakeSdmxClient(observations={"Count_Person": {_ROLE_TUPLE: [_STUB_FACET]}})
    result = client.role_series(
        variables=["Count_Person"],
        entities={"provider": ["country/USA"], "recipient": ["country/KEN"]},
        facet_ids={"facet_stub"},
        max_rows=100,
        max_bytes=100_000,
    )
    assert result["Count_Person"][_ROLE_TUPLE][0].facet_id == "facet_stub"


# availability_values: keying on component, variables, filters, and time_period


def test_availability_values_keys_on_component_variables_filters_and_time_period():
    client = FakeSdmxClient(
        observations={},
        availability={
            "recipient|variables=Count_Person": ["country/KEN", "country/NGA"],
            "recipient|variables=Count_Person|provider=country/USA": ["country/KEN"],
            "recipient|variables=Count_Person@time_period=2020": ["country/NGA"],
            "recipient|variables=Count_Person|provider=country/USA@time_period=2020,2021": [
                "country/KEN"
            ],
        },
    )
    # Base key returns the full member set.
    assert client.availability_values(variables=["Count_Person"], component="recipient") == [
        "country/KEN",
        "country/NGA",
    ]
    # A provider filter selects a separate fixture key.
    assert client.availability_values(
        variables=["Count_Person"],
        component="recipient",
        filters={"provider": ["country/USA"]},
    ) == ["country/KEN"]
    # A time period selects another fixture key.
    assert client.availability_values(
        variables=["Count_Person"], component="recipient", time_period="2020"
    ) == ["country/NGA"]
    # Filters and period combine in the lookup key.
    assert client.availability_values(
        variables=["Count_Person"],
        component="recipient",
        filters={"provider": ["country/USA"]},
        time_period=["2020", "2021"],
    ) == ["country/KEN"]


def test_availability_values_distinguishes_variables_but_normalizes_order_and_duplicates():
    client = FakeSdmxClient(
        observations={},
        availability={
            "recipient|variables=Count_Person,Count_Worker": ["country/KEN"],
            "recipient|variables=Count_Person": ["country/NGA"],
        },
    )

    assert client.availability_values(
        variables=["Count_Worker", "Count_Person", "Count_Worker"], component="recipient"
    ) == ["country/KEN"]
    assert client.availability_values(variables=["Count_Person"], component="recipient") == [
        "country/NGA"
    ]


def test_role_reads_record_the_callers_deadline():
    client = _client()
    client.role_series(
        variables=["Count_Person"],
        entities={"provider": ["country/USA"], "recipient": ["country/KEN"]},
        facet_ids={"facet1"},
        max_rows=10,
        max_bytes=1000,
        deadline=123.0,
    )

    assert client.series_calls[0]["deadline"] == 123.0
